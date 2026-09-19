"""Fail in seconds, not minutes, if the job landed on a GPU this build can't run.

The pinned stack is torch 1.11.0+cu113, whose libtorch_cuda_cu.so ships SASS for
sm_35..sm_86 and *no PTX at all* -- verified with `cuobjdump --list-ptx`, which
returns nothing. With no PTX there is no JIT fallback, so a newer device does
not degrade to a slow path; it dies on the first kernel launch with a bare
"no kernel image is available for execution on the device".

The qsub allow-list (`-l gpu_type=(A40|A6000|A100)`) is the real defence. This
is the backstop for when that list goes stale because the cluster gained a type.

A device counts as supported under CUDA's binary-compatibility rule, not exact match:
SASS built for sm_XY runs on any sm_XZ with Z >= Y (same major), and PTX for compute_XY
JIT-compiles on anything >= XY. The exact-match version refused L40S (sm_89) under
torch 2.4 cu121, whose sm_86 SASS runs there -- the reason the stock wheels work on
L40S / RTX 4090 (nmgG/nmgH died here in 14 s, 2026-09-15). Arch-specific targets
(sm_90a) are not forward compatible and only match exactly.
"""

import sys

import torch


def runnable(major, minor, arch_list):
    """(ok, how): can a build with `arch_list` run on an sm_{major}{minor} device?"""
    if f"sm_{major}{minor}" in arch_list:
        return True, "exact"
    for a in arch_list:
        kind, _, v = a.partition("_")
        if not v.isdigit():                  # sm_90a etc: arch-specific, exact only
            if kind == "sm" and v[:-1] == f"{major}{minor}":
                return True, f"exact ({a})"
            continue
        am, an = int(v[:-1]), int(v[-1])
        if kind == "sm" and am == major and an <= minor:
            return True, f"via {a} SASS"
        if kind == "compute" and (am, an) <= (major, minor):
            return True, f"via {a} PTX (JIT)"
    return False, "UNSUPPORTED"


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
        ok, how = runnable(major, minor, supported)
        print(f"  cuda:{i}  {name}  {arch}  {'ok (' + how + ')' if ok else how}")
        if not ok:
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
