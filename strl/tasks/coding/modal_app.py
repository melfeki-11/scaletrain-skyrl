"""The deployed Modal app that executes candidate programs: ``strl-code-exec``.

    python -m strl executor deploy          # from the Mac (or automatically by the worker on a version mismatch)

Containers: python 3.12 slim, 2 CPU, 2 GiB, no network, no Modal API access; up to 32 containers x 4 concurrent programs.
No ``min_containers``: a warm pool that outlives a crashed worker would bill forever; the worker pre-warms with a burst instead.
"""
from __future__ import annotations

import modal

from . import harness

APP_NAME = 'strl-code-exec'
FUNCTION = 'run_tests'

image = (modal.Image.debian_slim(python_version='3.12')
         .add_local_python_source('strl', ignore=['**/__pycache__/**', '**/*.pyc']))
app = modal.App(APP_NAME)


@app.function(image=image, cpu=2.0, memory=2048, timeout=120, max_containers=32, scaledown_window=300, buffer_containers=2,
              block_network=True, restrict_modal_access=True)
@modal.concurrent(max_inputs=4)
def run_tests(job: dict) -> dict:
    if job.get('ping'):
        return {'version': harness.VERSION}
    return harness.run_program(job, cpu_seconds=int(job.get('cpu_seconds', 6)), wall_seconds=float(job.get('wall_seconds', 12)))
