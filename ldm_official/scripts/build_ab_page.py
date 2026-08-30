"""Framework A/B page: LDM vs distillation_nerc at MATCHED epochs, one subject.

One epoch selector drives both stages, so a click puts both trainers at the same
amount of training. Native 256px, 1:1, no scaling. Pure CSS -- no JavaScript.

    build_ab_page.py <recon_dir> <trace_log> <out.html> <title> <subject-line>
"""
import base64, glob, html, io, os, re, sys
from PIL import Image

# COCO rFID, n=2000, keyed by epochs COMPLETED. Measured by eval_rfid_coco.py.
RFID = {
    "ldm":  {1: "11.758", 2: "11.815", 8: "12.089", 10: "12.089", 13: "12.150"},
    "nerc": {2: "11.562", 4: "11.598", 6: "11.749", 8: "11.769"},
}
INIT_RFID = "10.481"


def durl(p):
    b = io.BytesIO(); Image.open(p).convert("RGB").save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def main():
    src, trace, out, title, subject = sys.argv[1:6]
    psnr = dict(re.findall(r"^(\S+)\s+PSNR=\s*([\d.]+)", open(trace).read(), re.M))
    files = {os.path.basename(p)[3:-4]: p for p in glob.glob(os.path.join(src, "*.png"))}

    def eps(pre):
        found = set()
        for k in files:
            m = re.fullmatch(r"%s-(\d+)ep" % pre, k)
            if m:
                found.add(int(m.group(1)))
        return found
    pairs = sorted(eps("ldm") & eps("nerc"))
    if not pairs:
        raise SystemExit("no matched epochs between the two runs")

    vars_css, rules, radios, tabs, rows = [], [], [], [], []
    for k, ep in enumerate(pairs):
        for run in ("ldm", "nerc"):
            vars_css.append(f'  --{run}-{ep}: url("{durl(files[f"{run}-{ep}ep"])}");')
        radios.append(f'<input type="radio" name="ep" id="e{ep}"'
                      f'{" checked" if k == len(pairs) - 1 else ""}>')
        tabs.append(f'<label for="e{ep}">{ep} epochs</label>')
        for run in ("ldm", "nerc"):
            r = RFID[run].get(ep)
            p = psnr.get(f"{run}-{ep}ep", "—")
            rules += [
                f'#e{ep}:checked ~ .cmp .s-{run}{{background-image:var(--{run}-{ep})}}',
                f'#e{ep}:checked ~ .cmp .r-{run}::after{{content:"PSNR {p}'
                + (f'  \\00b7  rFID {r}"}}' if r else '  \\00b7  rFID n/a"}'),
            ]
        rules.append(f'#e{ep}:checked ~ .bar label[for="e{ep}"]'
                     '{background:var(--accent);color:#fff;border-color:var(--accent)}')
        rl, rn = RFID["ldm"].get(ep, "—"), RFID["nerc"].get(ep, "—")
        win = " class=\"n w\"" if rl != "—" and rn != "—" and float(rn) < float(rl) else ' class="n"'
        rows.append(f'<tr><td class="n">{ep}</td><td class="n">{rl}</td><td{win}>{rn}</td>'
                    f'<td class="n">{psnr.get(f"ldm-{ep}ep","—")}</td>'
                    f'<td class="n">{psnr.get(f"nerc-{ep}ep","—")}</td></tr>')

    refs = []
    for stem, name, note in [("original", "Original", "ground truth"),
                             ("init-sd-vae", "init · sd-vae",
                              f"shared starting weights · rFID {INIT_RFID} · "
                              f"PSNR {psnr.get('init-sd-vae','—')}")]:
        key = stem if stem in files else next(k for k in files if k.endswith(stem))
        refs.append(f'<figure class="ref"><img src="{durl(files[key])}" width="256" '
                    f'height="256" alt="{html.escape(name)}"><figcaption>'
                    f'<span class="n">{html.escape(name)}</span>'
                    f'<span class="d">{html.escape(note)}</span></figcaption></figure>')

    tpl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ab_template.html")
    page = (open(tpl).read()
            .replace("/*__VARS__*/", "\n".join(vars_css))
            .replace("/*__RULES__*/", "\n".join(rules))
            .replace("<!--RADIOS-->", "".join(radios))
            .replace("<!--TABS-->", "".join(tabs))
            .replace("<!--REFS-->", "".join(refs))
            .replace("<!--ROWS-->", "".join(rows))
            .replace("__TITLE__", html.escape(title))
            .replace("__SUBJECT__", html.escape(subject))
            .replace("__INIT_PSNR__", psnr.get("init-sd-vae", "—")))
    open(out, "w").write(page)
    print(f"wrote {out} ({os.path.getsize(out)/1e6:.2f} MB), matched epochs: {pairs}")


if __name__ == "__main__":
    main()
