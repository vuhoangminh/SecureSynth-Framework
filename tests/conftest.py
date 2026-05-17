import os
import subprocess
import sys
import textwrap
from pathlib import Path

_REPO = Path(__file__).parent.parent

_CLINICAL_CSV = _REPO / "database" / "raw" / "clinical.csv"
_TVAE_CKPT_DIR = _REPO / "database" / "prepared" / "clinical" / "tvae"
_TVAE_REQUIRED = ["model.pt", "encoder.pt", "decoder.pt", "train_z.npy"]

_ENV = {**os.environ, "PYTHONPATH": str(_REPO)}

# ---------------------------------------------------------------------------
# TabSyn TVAE checkpoint bootstrap
# The TVAE hard-codes 4000 training epochs; without pre-built checkpoints the
# smoke test times out.  We exec the TVAE source with num_epochs=3 to create
# structurally-valid checkpoints in < 10 s so that the real test run hits the
# early-exit shortcut ("checkpoints already exist → skip training").
# ---------------------------------------------------------------------------

_TVAE_BOOTSTRAP_SCRIPT = textwrap.dedent("""
    import sys
    sys.path.insert(0, 'models/tabsyn')

    with open('models/tabsyn/tabsyn/vae/main.py') as f:
        src = f.read()

    # Patch the hardcoded 4000-epoch count down to 3 for a quick bootstrap
    src = src.replace('    num_epochs = 4000', '    num_epochs = 3')

    import types
    _mod = types.ModuleType('_vae_bootstrap')
    _mod.__file__ = 'models/tabsyn/tabsyn/vae/main.py'
    exec(compile(src, 'models/tabsyn/tabsyn/vae/main.py', 'exec'), _mod.__dict__)

    import torch
    sys.argv = ['', '--method', 'vae', '--dataname', 'clinical', '--batch_size_tvae', '512']
    from utils import get_args
    args = get_args()
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if not args.save_path:
        args.save_path = f'synthetic/{args.dataname}/{args.method}.csv'
    _mod.main(args)
""").strip()


def pytest_sessionstart(session):
    """Create minimal TabSyn TVAE checkpoints before the test suite runs."""
    if not _CLINICAL_CSV.exists():
        return  # data/clinical.csv absent → pipeline tests will skip themselves
    if all((_TVAE_CKPT_DIR / f).exists() for f in _TVAE_REQUIRED):
        return  # checkpoints already present

    _TVAE_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-W", "ignore", "-c", _TVAE_BOOTSTRAP_SCRIPT],
        check=True,
        env=_ENV,
        cwd=str(_REPO),
        timeout=120,
    )
