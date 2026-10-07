"""SingleShotAgent: a Harbor ``BaseAgent`` that makes exactly ONE chat-completion call and
writes the reply verbatim to an artifact path. Used for the ``music`` domain only (1 turn,
no terminal interaction needed -- see ``strl/tasks/mimo/__init__.py``); the other four domains
run the stock ``terminus-2`` agent.

Loadable via Harbor's ``AgentConfig(import_path="strl.tasks.mimo.agents:SingleShotAgent")``
(see ``harbor.agents.factory.AgentFactory.create_agent_from_config``).

Goes through the exact same LLM route Terminus-2 uses -- Harbor's ``LiteLLM`` wrapper
(``harbor.llms.lite_llm.LiteLLM``) fed the agent's ``model_name``/``api_base``/``llm_kwargs``/
``model_info``/``collect_rollout_details`` -- and accumulates rollout details through
``harbor.llms.chat.Chat`` exactly as ``Terminus2.run`` does (see
``assets/overlay/harbor/agents/terminus_2/terminus_2.py::_run_agent_loop`` /
``::run``), so SkyRL gets the same token-id + logprob contract from a one-turn rollout that it
gets from a multi-turn one. ``context.metadata["n_episodes"] = 1`` always (one LLM call is one
episode, by definition).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.llms.base import LLMBackend
from harbor.llms.chat import Chat
from harbor.models.agent.context import AgentContext


class SingleShotAgent(BaseAgent):
    SUPPORTS_ATIF = False
    SUPPORTS_WINDOWS = True  # no terminal session is started; nothing Linux-specific happens.

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        artifact_path: str = "/app/answer.abc",
        api_base: str | None = None,
        temperature: float | None = None,
        model_info: dict | None = None,
        llm_kwargs: dict | None = None,
        collect_rollout_details: bool = False,
        session_id: str | None = None,
        llm_backend: LLMBackend | str = LLMBackend.LITELLM,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir, model_name, *(), **kwargs)
        if model_name is None:
            raise ValueError("model_name is required for SingleShotAgent")
        self._artifact_path = artifact_path
        self._collect_rollout_details = collect_rollout_details
        self._llm = self._build_llm(
            llm_backend=llm_backend,
            model_name=model_name,
            api_base=api_base,
            temperature=temperature,
            model_info=model_info,
            llm_kwargs=llm_kwargs,
            session_id=session_id,
            collect_rollout_details=collect_rollout_details,
        )

    @staticmethod
    def _build_llm(
        *,
        llm_backend: LLMBackend | str,
        model_name: str,
        api_base: str | None,
        temperature: float | None,
        model_info: dict | None,
        llm_kwargs: dict | None,
        session_id: str | None,
        collect_rollout_details: bool,
    ):
        """Construct the same ``LiteLLM`` wrapper Terminus-2 constructs (see
        ``Terminus2._init_llm``), trimmed to the kwargs SingleShotAgent actually exposes."""
        backend_value = llm_backend.value if isinstance(llm_backend, LLMBackend) else llm_backend
        if backend_value != LLMBackend.LITELLM.value:
            raise ValueError(f"SingleShotAgent only supports the litellm backend, got {backend_value!r}")
        from harbor.llms.lite_llm import LiteLLM

        constructor_kwargs = dict(llm_kwargs or {})
        if temperature is not None:
            constructor_kwargs["temperature"] = temperature
        return LiteLLM(
            model_name=model_name,
            api_base=api_base,
            collect_rollout_details=collect_rollout_details,
            session_id=session_id,
            model_info=model_info,
            **constructor_kwargs,
        )

    @staticmethod
    def name() -> str:
        return "single-shot"

    def version(self) -> str | None:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        """No terminal session: one LLM call, one file write. Nothing to set up."""

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        chat = Chat(self._llm)
        write_error: str | None = None
        try:
            response = await chat.chat(instruction)
            reply_text = response.content or ""
            try:
                await self._write_artifact(environment, reply_text)
            except Exception as e:  # noqa: BLE001
                write_error = f"{type(e).__name__}: {e}"
        finally:
            context.rollout_details = chat.rollout_details
            context.n_input_tokens = chat.total_input_tokens
            context.n_output_tokens = chat.total_output_tokens
            context.n_cache_tokens = chat.total_cache_tokens
            context.cost_usd = chat.total_cost if chat.total_cost > 0 else None
            context.metadata = {"n_episodes": 1, "artifact_path": self._artifact_path}
            if write_error:
                context.metadata["artifact_write_error"] = write_error

    async def _write_artifact(self, environment: BaseEnvironment, text: str) -> None:
        remote_dir = str(Path(self._artifact_path).parent)
        await environment.exec(f"mkdir -p {remote_dir}")
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", delete=False) as f:
            f.write(text)
            local_path = f.name
        try:
            await environment.upload_file(local_path, self._artifact_path)
        finally:
            Path(local_path).unlink(missing_ok=True)
