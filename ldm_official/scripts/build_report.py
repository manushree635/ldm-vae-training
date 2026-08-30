"""Render a self-contained HTML status page for an LDM finetune run.

Reads the run's tfevents, its SGE job log, and its logged image grids, and emits
one HTML file with the data inlined (no external requests beyond fonts). Re-run
it any time and re-publish the same file to refresh the page.

    python scripts/build_report.py --run logs/2026-..._ldm_ftema_20k \
        --job-log logs/ldm_ftema_20k.o7364029 -o report.html
"""

import argparse
import base64
import glob
import io
import json
import os
import re
from datetime import datetime

MAX_EPOCHS = 50


def read_scalars(run):
    from tensorboard.backend.event_processing import event_accumulator
    series = {}
    for f in sorted(glob.glob(os.path.join(run, "testtube", "version_*", "tf", "events.out.tfevents*"))):
        ea = event_accumulator.EventAccumulator(f, size_guidance={event_accumulator.SCALARS: 0})
        ea.Reload()
        for tag in ea.Tags()["scalars"]:
            series.setdefault(tag, [])
            series[tag].extend([(e.step, e.value) for e in ea.Scalars(tag)])
    return {t: [list(p) for p in sorted(set(v))] for t, v in series.items()}


def parse_job_log(path):
    info = {"host": None, "epoch_times": [], "peak_mem": [], "val": [], "batches": None, "last_batch": None}
    if not path or not os.path.exists(path):
        return info
    with open(path, "rb") as fh:
        raw = fh.read().decode("utf-8", "replace")
    text = raw.replace("\r", "\n")
    m = re.search(r"^host\s+(\S+)", text, re.M)
    if m:
        info["host"] = m.group(1)
    info["epoch_times"] = [float(x) for x in re.findall(r"Average Epoch time: ([\d.]+) seconds", text)]
    info["peak_mem"] = [float(x) for x in re.findall(r"Average Peak memory ([\d.]+)MiB", text)]
    for ep, step, val, best in re.findall(
            r"Epoch (\d+), global step (\d+): val/rec_loss reached ([\d.]+) \(best ([\d.]+)\)", text):
        info["val"].append({"epoch": int(ep), "step": int(step),
                            "value": float(val), "best": float(best)})
    bars = re.findall(r"Epoch (\d+):\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)", text)
    if bars:
        info["last_epoch"], info["last_batch"], info["batches"] = (int(bars[-1][0]), int(bars[-1][1]), int(bars[-1][2]))
    return info


def embed_image(path, width=880, quality=82):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def latest_pair(run):
    d = os.path.join(run, "images", "train")
    recons = sorted(glob.glob(os.path.join(d, "reconstructions_gs-*.png")))
    if not recons:
        return None
    rec = recons[-1]
    inp = rec.replace("reconstructions_", "inputs_")
    if not os.path.exists(inp):
        return None
    gs = re.search(r"gs-(\d+)", os.path.basename(rec)).group(1)
    return {"step": int(gs), "input": embed_image(inp), "recon": embed_image(rec)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--job-log", default=None)
    ap.add_argument("--job-id", default="")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--template", default=None)
    args = ap.parse_args()

    run = args.run.rstrip("/")
    data = {
        "run": os.path.basename(run),
        "run_path": run,
        "job_id": args.job_id,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "max_epochs": MAX_EPOCHS,
        "series": read_scalars(run),
        "log": parse_job_log(args.job_log),
        "checkpoints": sorted(os.path.basename(p) for p in glob.glob(os.path.join(run, "checkpoints", "*.ckpt"))),
        "pair": latest_pair(run),
    }

    tpl = args.template or os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_template.html")
    with open(tpl) as fh:
        html = fh.read()
    html = html.replace("/*__DATA__*/null", json.dumps(data))
    with open(args.out, "w") as fh:
        fh.write(html)
    size = os.path.getsize(args.out)
    print(f"wrote {args.out}  ({size/1e6:.2f} MB)")
    print(f"  epochs done : {len(data['log']['epoch_times'])}/{MAX_EPOCHS}")
    print(f"  rec_loss pts: {len(data['series'].get('train/rec_loss', []))}")
    print(f"  val points  : {len(data['log']['val'])}")
    print(f"  checkpoints : {data['checkpoints']}")


if __name__ == "__main__":
    main()
