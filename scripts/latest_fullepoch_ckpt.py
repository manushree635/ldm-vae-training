"""Print the newest FULL-EPOCH checkpoint in an LDM run directory.

    python scripts/latest_fullepoch_ckpt.py <run_dir>   ->  <run_dir>/checkpoints/epoch=NNNNNN.ckpt

Used by the 1M run scripts when RESUME_DIR is a directory, so a queued resume
picks its checkpoint when it STARTS rather than when it was submitted (a run can
finish another epoch between the two).

Deliberately ignores last.ckpt: LDM's main.py `melk` signal handler writes a
last.ckpt at whatever batch the job is on when it is signalled, i.e. mid-epoch.
Only `epoch=*.ckpt` files are candidates, newest first, and the first one whose
fit loop says `is_last_batch=True` wins -- resume from it continues at the next
epoch with no replayed or skipped data. Exits 1 if there is none.
"""
import glob, os, sys
import torch

run_dir = sys.argv[1].rstrip("/")
cands = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "epoch=*.ckpt")),
               key=os.path.getmtime, reverse=True)
for p in cands:
    try:
        ck = torch.load(p, map_location="cpu", mmap=True, weights_only=False)
        bp = ck["loops"]["fit_loop"]["epoch_loop.batch_progress"]
        full = bool(bp.get("is_last_batch"))
        print(f"[fullepoch] {os.path.basename(p)} epoch={ck['epoch']} gs={ck['global_step']} "
              f"batches={bp['current']['completed']} is_last_batch={full}", file=sys.stderr)
    except Exception as e:                      # truncated / still being written
        print(f"[fullepoch] {os.path.basename(p)} unreadable: {e}", file=sys.stderr)
        continue
    if full:
        print(p)
        sys.exit(0)
print(f"[fullepoch] no full-epoch checkpoint in {run_dir}", file=sys.stderr)
sys.exit(1)
