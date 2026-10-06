import json
import subprocess
import sys
from pathlib import Path
from .config import load
from .skyrl import overrides


def main():
    from skyrl.train.config import SkyRLTrainConfig
    from skyrl.train.utils import validate_cfg
    cfg=load()
    name=__import__('os').environ['GRPO_RUN_NAME']
    values=overrides(cfg,stage='train',run_name=name)
    validated=SkyRLTrainConfig.from_cli_overrides(values)
    validate_cfg(validated)
    from .runtime import validate_generator
    validate_generator(validated)
    Path('artifacts/schema.json').write_text(json.dumps({'validated':True,'overrides':values},indent=2))
    subprocess.run([sys.executable,'-m','q35.entrypoint',*values],check=True)

if __name__=='__main__':main()
