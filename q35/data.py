from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

from .answers import extract_reference


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_indices(size: int, validation_size: int, seed: int) -> tuple[list[int], list[int]]:
    if not 0 < validation_size < size:
        raise ValueError("validation_size must be between zero and dataset size")
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    validation = sorted(indices[:validation_size])
    training = sorted(indices[validation_size:])
    return training, validation


def make_record(example: dict[str, Any], *, split: str, index: int, prompt_suffix: str) -> dict[str, Any]:
    question = str(example["question"])
    reference = extract_reference(str(example["answer"]))
    sample_id = f"gsm8k-{split}-{index:05d}"
    return {
        "data_source": "openai/gsm8k",
        "prompt": [{"role": "user", "content": f"{question}\n\n{prompt_suffix}"}],
        "env_class": "gsm8k",
        "reward_spec": {"method": "rule", "ground_truth": reference},
        "extra_info": {
            "sample_id": sample_id,
            "split": split,
            "source_index": index,
            "question": question,
            "ground_truth": reference,
        },
    }


def prepare_dataset(cfg: dict[str, Any], *, dataset: Any | None = None) -> dict[str, Any]:
    if dataset is None:
        from datasets import load_dataset

        dataset = load_dataset(
            cfg["dataset"]["id"],
            cfg["dataset"]["config"],
            revision=cfg["dataset"]["revision"],
        )
    train_source = dataset["train"]
    test_source = dataset["test"]
    train_idx, validation_idx = split_indices(
        len(train_source), cfg["dataset"]["validation_size"], cfg["dataset"]["split_seed"]
    )
    suffix = cfg["dataset"]["prompt_suffix"]
    splits = {
        "train": [make_record(train_source[i], split="train", index=i, prompt_suffix=suffix) for i in train_idx],
        "validation": [
            make_record(train_source[i], split="validation", index=i, prompt_suffix=suffix) for i in validation_idx
        ],
        "test": [make_record(test_source[i], split="test", index=i, prompt_suffix=suffix) for i in range(len(test_source))],
    }
    output_dir = Path(cfg["run"]["data_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    for split, rows in splits.items():
        path = output_dir / f"{split}.parquet"
        from datasets import Dataset

        Dataset.from_list(rows).to_parquet(path)
        files[split] = {"path": str(path), "rows": len(rows), "sha256": _sha256(path)}
    canary_path = output_dir / "canary.parquet"
    from datasets import Dataset

    qualitative_count = min(cfg["evaluation"]["fixed_qualitative_samples"], len(splits["validation"]))
    Dataset.from_list(splits["validation"][:qualitative_count]).to_parquet(canary_path)
    files["canary"] = {"path": str(canary_path), "rows": qualitative_count, "sha256": _sha256(canary_path)}
    qualitative = output_dir / "qualitative_ids.json"
    qualitative_ids = [row["extra_info"]["sample_id"] for row in splits["validation"][:qualitative_count]]
    qualitative.write_text(
        json.dumps(qualitative_ids, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "dataset": cfg["dataset"]["id"],
        "revision": cfg["dataset"]["revision"],
        "split_seed": cfg["dataset"]["split_seed"],
        "files": files,
        "qualitative_ids": {
            "path": str(qualitative),
            "rows": len(qualitative_ids),
            "sha256": _sha256(qualitative),
        },
        "source_indices": {
            "train": train_idx,
            "validation": validation_idx,
            "test": list(range(len(test_source))),
        },
        "official_test_untouched": True,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest


def assert_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    config_root = Path(cfg.get("_config_path", ".")).resolve().parent
    root = Path(cfg["run"]["data_dir"])
    if not root.is_absolute():
        root = config_root / root
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("dataset") != cfg["dataset"]["id"]
        or manifest.get("revision") != cfg["dataset"]["revision"]
        or manifest.get("split_seed") != cfg["dataset"]["split_seed"]
        or not manifest.get("official_test_untouched")
    ):
        raise RuntimeError("dataset manifest does not match the configured immutable split contract")
    for item in manifest["files"].values():
        path = Path(item["path"])
        if not path.is_absolute():
            path = config_root / path
        if _sha256(path) != item["sha256"]:
            raise RuntimeError(f"data hash mismatch: {path}")
    qualitative = manifest.get("qualitative_ids") or {}
    qualitative_path = Path(str(qualitative.get("path", "")))
    if not qualitative_path.is_absolute():
        qualitative_path = config_root / qualitative_path
    if (
        int(qualitative.get("rows", -1))
        != min(cfg["evaluation"]["fixed_qualitative_samples"], cfg["dataset"]["validation_size"])
        or not qualitative_path.is_file()
        or _sha256(qualitative_path) != qualitative.get("sha256")
    ):
        raise RuntimeError("fixed qualitative sample manifest differs from the configured contract")
    train = set(manifest["source_indices"]["train"])
    validation = set(manifest["source_indices"]["validation"])
    if (
        len(validation) != cfg["dataset"]["validation_size"]
        or len(train) != int(manifest["files"]["train"]["rows"])
        or train & validation
        or train | validation != set(range(len(train) + len(validation)))
    ):
        raise RuntimeError("validation split size drifted")
    test = manifest["source_indices"]["test"]
    if test != list(range(int(manifest["files"]["test"]["rows"]))):
        raise RuntimeError("official test split indices drifted")
    return manifest


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
