"""Prove NCCL collectives actually complete before committing to a long run.

The single-GPU smoke test cannot catch a collective deadlock -- DDP with one
process performs no real all-reduce. This spawns the same world size the run
will use and times a few all-reduces, so a hang costs ~60s instead of hours.
"""

import datetime
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

TIMEOUT_S = 60


def _worker(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29577")
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size,
        timeout=datetime.timedelta(seconds=TIMEOUT_S),
    )
    torch.cuda.set_device(rank)
    for numel in (1, 1 << 20, 1 << 24):
        t = torch.ones(numel, device=f"cuda:{rank}")
        dist.all_reduce(t)
        expected = float(world_size)
        assert abs(t[0].item() - expected) < 1e-5, (numel, t[0].item(), expected)
    dist.barrier()
    if rank == 0:
        print(f"DDP PREFLIGHT: {world_size}-rank NCCL all-reduce OK")
    dist.destroy_process_group()


def main():
    world_size = int(sys.argv[1]) if len(sys.argv) > 1 else torch.cuda.device_count()
    if world_size < 2:
        print(f"DDP PREFLIGHT: skipped (world_size={world_size})")
        return 0
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_worker, args=(r, world_size)) for r in range(world_size)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=TIMEOUT_S + 60)
    for r, p in enumerate(procs):
        if p.is_alive():
            p.terminate()
            print(f"DDP PREFLIGHT: FAILED -- rank {r} hung. NCCL collectives are "
                  f"deadlocking on this node.", file=sys.stderr)
            return 1
        if p.exitcode != 0:
            print(f"DDP PREFLIGHT: FAILED -- rank {r} exited {p.exitcode}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
