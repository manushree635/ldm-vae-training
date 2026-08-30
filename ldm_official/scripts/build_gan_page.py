"""GAN vs no-GAN vs init, three subjects, native 256px 1:1.

Each subject gets its own grid -- the arms are few enough to read side by side,
so no toggling. Pure CSS, no JavaScript.
"""
import base64, glob, html, io, os, re, sys
from PIL import Image

R = "/projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/ldm_official"
SUBJECTS = [("gan_blue_camo", "Man in blue camo", "Second-Phase14396.jpg"),
            ("gan_gown",      "Red gown",         "Second-Phase17674.jpg"),
            ("gan_child",     "Child portrait",   "200K_for_200Kids…jpg")]
ARMS = [("00_original",         "Original",       "ground truth", None),
        ("01_init-sd-vae",      "init · sd-vae",  "un-finetuned start", "10.481"),
        ("02_noGAN-1ep",        "no GAN · 1 ep",  "same run, before disc_start", "11.755"),
        ("03_GAN-2ep",          "GAN · 2 ep",     "one epoch of adversarial training", "9.983"),
        ("07_stability-ft-ema", "sd-vae-ft-ema",  "Stability, 313k steps", "9.356")]

# Laplacian-variance texture retention, measured on the blue-camo subject.
TEX = {"01_init-sd-vae": "90%", "02_noGAN-1ep": "59%", "03_GAN-2ep": "104%",
       "07_stability-ft-ema": "103%", "00_original": "100%"}


def durl(p):
    b = io.BytesIO(); Image.open(p).convert("RGB").save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def main():
    out = sys.argv[1]
    psnr = dict(re.findall(r"^(\S+)\s+PSNR=\s*([\d.]+)",
                           open(f"{R}/logs/recon_gan_trace.log").read(), re.M))
    blocks = []
    for d, title, fname in SUBJECTS:
        src = f"{R}/recon_compare/{d}"
        if not os.path.isdir(src):
            print(f"  skip {d}"); continue
        cells = []
        for stem, name, note, rfid in ARMS:
            p = os.path.join(src, stem + ".png")
            if not os.path.exists(p):
                continue
            arm = stem.split("_", 1)[1]
            bits = []
            if psnr.get(arm): bits.append(f"PSNR {psnr[arm]}")
            if rfid: bits.append(f"rFID {rfid}")
            if TEX.get(stem): bits.append(f"texture {TEX[stem]}")
            cls = "cell hero" if stem == "03_GAN-2ep" else "cell"
            cells.append(
                f'<figure class="{cls}"><img src="{durl(p)}" width="256" height="256" '
                f'alt="{html.escape(name)}"><figcaption>'
                f'<span class="n">{html.escape(name)}</span>'
                f'<span class="d">{html.escape(note)}</span>'
                f'<span class="m">{" · ".join(bits)}</span></figcaption></figure>')
        blocks.append(f'<section><h2>{html.escape(title)}</h2>'
                      f'<p class="sub">{html.escape(fname)}</p>'
                      f'<div class="grid">{"".join(cells)}</div></section>')
    page = open(f"{R}/scripts/gan_template.html").read().replace("<!--BLOCKS-->", "".join(blocks))
    open(out, "w").write(page)
    print(f"wrote {out} ({os.path.getsize(out)/1e6:.2f} MB), {len(blocks)} subjects")


if __name__ == "__main__":
    main()
