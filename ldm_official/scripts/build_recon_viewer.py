"""Reconstruction comparison: 1:1 flicker A/B, plus a three-up side-by-side.

Every image is emitted ONCE as a CSS custom property (--img-aN) and painted with
background-image, so the four stages share one copy of each PNG instead of
duplicating a ~75KB data URI per pane.

Pure CSS throughout -- radio :checked + sibling combinators. No JavaScript, and
the radio groups are keyboard-driven, so arrow keys flick between arms in place.
"""
import base64, glob, html, io, os, re, sys
from PIL import Image

def discover(src):
    """Arms are whatever recon_compare.py wrote: NN_name.png, already in order."""
    out = []
    for p in sorted(glob.glob(os.path.join(src, "[0-9][0-9]_*.png"))):
        stem = os.path.basename(p)[:-4]
        name = stem.split("_", 1)[1]
        note = NOTES.get(name, "")
        out.append((stem, name, note))
    return out


NOTES = {
    "original":      "ground truth",
    "init-sd-vae":   "the run's starting weights",
    "sd-vae":        "SD-1.5, un-finetuned",
    "sd-vae-ft-ema": "Stability, L1 + LPIPS",
    "sd-vae-ft-mse": "Stability, MSE + 0.1 LPIPS",
    "strpy-1M":      "your distillation, 1M images",
}



def durl(im):
    b = io.BytesIO(); im.save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def main():
    src, log, out = sys.argv[1], sys.argv[2], sys.argv[3]
    txt = open(log).read()
    sec = txt.split("=== 256px ===", 1)[-1].split("=== 512px ===", 1)[0]
    psnr = dict(re.findall(r"^(\S+)\s+PSNR=\s*([\d.]+)", sec, re.M))

    arms = []
    for i, (stem, name, note) in enumerate(discover(src)):
        p = os.path.join(src, stem + ".png")
        if not os.path.exists(p):
            print(f"  missing {stem}"); continue
        im = Image.open(p).convert("RGB")
        assert im.size == (256, 256), f"{stem} is {im.size}"
        arms.append({"i": len(arms), "name": name, "note": note, "url": durl(im),
                     "psnr": psnr.get(stem.split("_", 1)[1])})

    vars_css = "\n".join(f'  --img-a{a["i"]}: url("{a["url"]}");' for a in arms)
    rules, radios_ab, tabs_ab = [], [], []

    # --- section 1: single stage, flicker A/B ---------------------------------
    for a in arms:
        i = a["i"]
        label = (f'<b>{html.escape(a["name"])}</b> &middot; {html.escape(a["note"])}'
                 + (f' &middot; PSNR {a["psnr"]}' if a["psnr"] else ''))
        rules += [
            f'#ab{i}:checked ~ .ab .stage{{background-image:var(--img-a{i})}}',
            f'#ab{i}:checked ~ .ab .readout::after{{content:"{a["name"]}'
            + (f'  \\00b7  PSNR {a["psnr"]}"}}' if a["psnr"] else '"}'),
            f'#ab{i}:checked ~ .ab label[for="ab{i}"]'
            '{background:var(--accent);color:#fff;border-color:var(--accent)}',
        ]
        radios_ab.append(f'<input type="radio" name="ab" id="ab{i}"'
                         f'{" checked" if i == 0 else ""}>')
        tabs_ab.append(f'<label for="ab{i}">{html.escape(a["name"])}</label>')

    # --- section 2: three independent panes -----------------------------------
    radios_3, panes = [], []
    n = len(arms)
    pane_defaults = [0, max(0, n // 2), n - 1]   # first | middle | last epoch
    for pane, default in enumerate(pane_defaults, start=1):
        for a in arms:
            i = a["i"]
            rules += [
                f'#p{pane}a{i}:checked ~ .three .s{pane}'
                f'{{background-image:var(--img-a{i})}}',
                f'#p{pane}a{i}:checked ~ .three .r{pane}::after'
                f'{{content:"{a["name"]}'
                + (f'  \\00b7  {a["psnr"]}"}}' if a["psnr"] else '"}'),
                f'#p{pane}a{i}:checked ~ .three .pane{pane} label[for="p{pane}a{i}"]'
                '{background:var(--accent);color:#fff;border-color:var(--accent)}',
            ]
            radios_3.append(f'<input type="radio" name="p{pane}" id="p{pane}a{i}"'
                            f'{" checked" if i == default else ""}>')
        opts = "".join(f'<label for="p{pane}a{a["i"]}">{html.escape(a["name"])}</label>'
                       for a in arms)
        panes.append(
            f'<div class="pane pane{pane}">'
            f'<div class="stage s{pane}" role="img" aria-label="reconstruction {pane}"></div>'
            f'<div class="readout r{pane}"></div>'
            f'<div class="tabs tiny">{opts}</div></div>')

    tpl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer_template.html")
    page = (open(tpl).read()
            .replace("/*__VARS__*/", vars_css)
            .replace("/*__RULES__*/", "\n".join(rules))
            .replace("<!--RADIOS_AB-->", "".join(radios_ab))
            .replace("<!--TABS_AB-->", "".join(tabs_ab))
            .replace("<!--RADIOS_3-->", "".join(radios_3))
            .replace("<!--PANES-->", "".join(panes)))
    open(out, "w").write(page)
    print(f"wrote {out} ({os.path.getsize(out)/1e6:.2f} MB), {len(arms)} arms, "
          f"{len(rules)} css rules, image copies: {page.count('base64,')}")


if __name__ == "__main__":
    main()
