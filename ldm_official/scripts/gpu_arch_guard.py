"""Fail in seconds, not minutes, if the job landed on a GPU this build can't run.

The pinned stack is torch 1.11.0+cu113, whose libtorch_cuda_cu.so ships SASS for
sm_35..sm_86 and *no PTX at all* -- verified with `cuobjdump --list-ptx`, which
returns nothing. With no PTX there is no JIT fallback, so a newer device does
not degrade to a slow path; it dies on the first kernel launch with a bare
"no kernel image is available for execution on the device".

The qsub allow-list (`-l gpu_type=(A40|A6000|A100)`) is the real defence. This
is the backstop for when that list goes stale because the cluster gained a type.
"""

import sys

import torch


def main():
    if not torch.cuda.is_available():
        print("GPU GUARD: CUDA is not available", file=sys.stderr)
        return 1

    supported = set(torch.cuda.get_arch_list())
    print(f"GPU GUARD: torch {torch.__version__} supports {sorted(supported)}")

    bad = []
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        arch = f"sm_{major}{minor}"
        name = torch.cuda.get_device_name(i)
        print(f"  cuda:{i}  {name}  {arch}"
              f"  {'ok' if arch in supported else 'UNSUPPORTED'}")
        if arch not in supported:
            bad.append((i, name, arch))

    if bad:
        print("", file=sys.stderr)
        print("GPU GUARD: aborting -- this torch build has no kernels for:",
              file=sys.stderr)
        for i, name, arch in bad:
            print(f"  cuda:{i} {name} ({arch})", file=sys.stderr)
        print("Resubmit onto a supported type, e.g.\n"
              "  qsub -l 'gpu_type=(A40|A6000|A100)' scripts/run_ldm_ft_20k_scc.sh",
              file=sys.stderr)
        return 1

    print("GPU GUARD: all visible devices are supported")
    return 0


if __name__ == "__main__":
    sys.exit(main())
