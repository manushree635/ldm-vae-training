"""Dump a run's scalar metrics from its tfevents file to JSON.

TestTube's metrics.csv lags badly (it buffers), so read the tfevents stream
instead. Cadence is Lightning's log_every_n_steps=50 against a global_step that
advances once per `accumulate_grad_batches` batches -- so with accum=2 a point
lands every 100 batches.

    python scripts/dump_metrics.py --run logs/2026-...-_ldm_ftema_20k -o metrics.json
"""

import argparse
import glob
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory under logs/")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    from tensorboard.backend.event_processing import event_accumulator

    pattern = os.path.join(args.run, "testtube", "version_*", "tf", "events.out.tfevents*")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no tfevents under {pattern}")

    series = {}
    for f in files:
        ea = event_accumulator.EventAccumulator(
            f, size_guidance={event_accumulator.SCALARS: 0})
        ea.Reload()
        for tag in ea.Tags()["scalars"]:
            series.setdefault(tag, [])
            series[tag].extend([[e.step, e.value] for e in ea.Scalars(tag)])
    for tag in series:
        series[tag] = sorted(set(map(tuple, series[tag])))
        series[tag] = [list(p) for p in series[tag]]

    out = args.out or os.path.join(args.run, "metrics.json")
    with open(out, "w") as fh:
        json.dump({"run": os.path.basename(args.run.rstrip("/")), "series": series}, fh)

    print(f"{os.path.basename(args.run.rstrip('/'))} -> {out}")
    for tag, pts in sorted(series.items()):
        if not pts:
            continue
        print(f"  {tag:<22} n={len(pts):<4} steps {pts[0][0]}..{pts[-1][0]}  "
              f"first={pts[0][1]:.5g} last={pts[-1][1]:.5g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
