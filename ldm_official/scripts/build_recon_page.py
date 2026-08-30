"""Compose the reconstruction comparison into one self-contained HTML page.

Emits every <img> tag server-side. The page runs with JavaScript disabled --
for a static side-by-side there is nothing to gain from client-side rendering,
and plenty to lose if a script fails.
"""
import base64, html, io, os, re, sys
from PIL import Image

ARMS = [
    ("00_original",      "Original",      "ground truth, 256px"),
    ("01_sd-vae",        "sd-vae",        "SD-1.5 VAE, un-finetuned"),
    ("02_sd-vae-ft-ema", "sd-vae-ft-ema", "Stability, L1 + LPIPS"),
    ("03_sd-vae-ft-mse", "sd-vae-ft-mse", "Stability, MSE + 0.1 LPIPS"),
    ("04_strpy-1M",      "strpy-1M",      "your distillation, 1M images"),
    ("05_nerc-vgg-ep10", "nerc-vgg ep10", "your plain finetune, VGG LPIPS"),
    ("06_ldm-ftema-ep4", "ldm-ftema ep4", "official LDM, ft-EMA recipe"),
]
CROPS = [("face & headdress", (104, 14, 168, 78)),
         ("gold embroidery",  (94, 140, 158, 204))]


def durl(im, scale=1):
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    b = io.BytesIO(); im.save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def figure(name, note, psnr, src, ref, best):
    cls = "cell ref" if ref else "cell"
    p = (f'<span class="p">PSNR {psnr}{" <b>best</b>" if best else ""}</span>'
         if psnr else "")
    return (f'<figure class="{cls}">'
            f'<img src="{src}" alt="{html.escape(name)}">'
            f'<figcaption class="cap"><span class="n">{html.escape(name)}</span>'
            f'<span class="d">{html.escape(note)}</span>{p}</figcaption></figure>')


def main():
    src_dir, psnr_log, out = sys.argv[1], sys.argv[2], sys.argv[3]
    psnr = dict(re.findall(r"^(\S+)\s+PSNR=\s*([\d.]+)", open(psnr_log).read(), re.M))

    items = []
    for stem, name, note in ARMS:
        p = os.path.join(src_dir, stem + ".png")
        if not os.path.exists(p):
            print(f"  missing {p}"); continue
        items.append((stem, name, note, Image.open(p).convert("RGB"),
                      psnr.get(stem.split("_", 1)[1])))
    vals = [float(v) for *_, v in items if v]
    top = max(vals) if vals else None

    strips = {}
    strips["full"] = "".join(
        figure(n, d, pv, durl(im, 2), n == "Original", bool(pv) and float(pv) == top)
        for _, n, d, im, pv in items)
    for key, box in CROPS:
        strips[key] = "".join(
            figure(n, d, pv, durl(im.crop(box), 8), n == "Original",
                   bool(pv) and float(pv) == top)
            for _, n, d, im, pv in items)

    tpl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recon_template.html")
    page = open(tpl).read()
    page = (page.replace("<!--FULL-->", strips["full"])
                .replace("<!--CROP_FACE-->", strips["face & headdress"])
                .replace("<!--CROP_EMB-->", strips["gold embroidery"])
                .replace("<!--COUNT-->", str(len(items))))
    open(out, "w").write(page)
    print(f"wrote {out} ({os.path.getsize(out)/1e6:.2f} MB), {len(items)} arms, "
          f"{page.count('<img ')} img tags")


if __name__ == "__main__":
    main()
