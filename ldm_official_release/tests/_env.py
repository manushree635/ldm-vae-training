"""Shared setup for the ldm_official test suite.

ROOT is derived from this file, so the same tests/ directory works unchanged when
copied into ldm_official_release/. Everything here is CPU-only and stdlib unittest
(neither venv ships pytest).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Same three entries scripts/env.sh / activate_ldm_env.sh put on PYTHONPATH.
# latent-diffusion/ is there so the YAMLs' `main.DataModuleFromConfig` resolves.
for p in (os.path.join(ROOT, "taming-transformers"),
          os.path.join(ROOT, "latent-diffusion"),
          ROOT):
    # Remove-then-insert, not "if absent": the venv's editable-install .pth already lists
    # both vendored repos, taming first, and BOTH have a top-level main.py. The YAMLs mean
    # latent-diffusion's.
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

# The release configs read these through ${oc.env:...}; harmless for ldm_official.
os.environ.setdefault("LDM_ROOT", ROOT)
os.environ.setdefault("LDM_DATA", os.path.join(ROOT, "data"))

#: Directories that are not this repo's own code.
SKIP_DIRS = {"latent-diffusion", "taming-transformers", "CLIP", "pylibs", "logs", "wandb",
             "__pycache__", "setup_logs", "models", "ckpt_snapshots"}


def own_files(*exts):
    """Every file under ROOT with one of `exts`, skipping vendored code and venvs."""
    out = []
    for d, dirs, files in os.walk(ROOT):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS and not x.startswith("venv")]
        out.extend(os.path.join(d, f) for f in files if f.endswith(exts))
    return sorted(out)


def rel(path):
    return os.path.relpath(path, ROOT)
