"""Resolve a config and check that every file it points at exists -- before a GPU is touched.

    python scripts/check_config_paths.py configs/04_steerable_graded.yaml

The configs reach the filesystem only through ${oc.env:LDM_ROOT} and
${oc.env:LDM_DATA} (set by scripts/env.sh). A missing input otherwise surfaces
minutes into a job, per DDP rank, as a traceback from inside a dataset class;
here it is one line that names the script which builds the missing piece.
"""
import os
import sys

from omegaconf import OmegaConf

# path-valued keys -> what creates the thing they point at
HINTS = {
    "ckpt_path": "bash scripts/prepare_weights.sh",
    "images_list_file": "bash scripts/prepare_data.sh   (or scripts/make_filelists*.py for your own set)",
    "teacher_dir": "bash scripts/prepare_data.sh   (scripts/precompute_teacher_recons.py)",
    "latents_file": "scripts/precompute_latents.py --config <this config> --split train|validation",
    "image_path": "any image; it is only the pinned picture the wandb panel reconstructs",
}


def walk(node, trail=()):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, trail + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, trail + (str(i),))
    else:
        yield trail, node


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    for var in ("LDM_ROOT", "LDM_DATA"):
        if var not in os.environ:
            raise SystemExit("{} is not set -- run `source scripts/env.sh` first".format(var))
    cfg = OmegaConf.to_container(OmegaConf.load(sys.argv[1]), resolve=True)

    missing = []
    for trail, value in walk(cfg):
        if trail and trail[-1] in HINTS and isinstance(value, str) and value:
            ok = os.path.exists(value)
            print("  {:<4} {:<17} {}".format("ok" if ok else "MISS", trail[-1], value))
            if not ok:
                missing.append((trail[-1], value))
    if missing:
        print("\nCONFIG CHECK FAILED -- {} input(s) missing:".format(len(missing)), file=sys.stderr)
        for key, value in missing:
            print("  {}\n      build with: {}".format(value, HINTS[key]), file=sys.stderr)
        return 1
    print("config check: all inputs present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
