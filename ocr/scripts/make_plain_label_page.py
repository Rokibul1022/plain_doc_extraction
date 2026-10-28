"""Build ONE self-contained HTML page for labelling a folder of plate photos in the browser (no server needed).

    python ocr/scripts/make_plain_label_page.py "new samples+3d" --out "new samples+3d/label_page.html"

Per plate: the photo and a contrast-enhanced copy (embossed text is easier to see), a box for the six digits (Latin or
Bengali numerals, any separators), a city picker and a class-letter picker (so no Bangla typing is needed), and an
"unreadable" box. Progress is kept in the browser. "Download CSV" writes  file,digits,city,letter  (UTF-8), the format
`ocr/scripts/read_plates.py --truth` reads. Nothing is pre-filled, so the labels are your own reading.
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ocr.synth_row import CITY_WORDS, LETTERS  # noqa: E402

EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MAX_W = 900


def _b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _enhanced(img: np.ndarray) -> np.ndarray:
    """Local contrast boost on lightness: shows raised characters that have no ink contrast."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Plate labelling</title>
<style>
:root{--bg:#f4f5f7;--card:#fff;--ink:#14181f;--mute:#667085;--line:#d9dde4;--accent:#0b6bcb;--ok:#1a8f4a;--warn:#b54708}
@media (prefers-color-scheme:dark){:root{--bg:#0f1319;--card:#171c24;--ink:#e9edf3;--mute:#9aa4b2;--line:#2b3340;--accent:#5aa9ff;--ok:#43c279;--warn:#f0a24a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,Segoe UI,Roboto,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);padding:.7rem 1rem;display:flex;gap:1rem;align-items:center;flex-wrap:wrap}
header h1{font-size:1.05rem;margin:0}.bar{flex:1;min-width:140px;height:8px;background:var(--line);border-radius:6px;overflow:hidden}.bar i{display:block;height:100%;width:0;background:var(--ok);transition:width .2s}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:8px;padding:.45rem .9rem;cursor:pointer}button.primary{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}
main{max-width:1100px;margin:0 auto;padding:1rem}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:1rem;margin-bottom:1rem}
.card.done{border-color:var(--ok)}.card h2{margin:0 0 .6rem;font-size:.95rem;color:var(--mute);display:flex;justify-content:space-between}
.imgs{display:grid;grid-template-columns:1fr 1fr;gap:.7rem}.imgs figure{margin:0}.imgs img{width:100%;height:auto;border-radius:8px;border:1px solid var(--line);cursor:zoom-in;background:#000}
.imgs figcaption{font-size:.78rem;color:var(--mute);margin-top:.2rem}
.fields{display:grid;grid-template-columns:200px 1fr 120px auto;gap:.7rem;margin-top:.8rem;align-items:end}
label{display:block;font-size:.8rem;color:var(--mute);margin-bottom:.2rem}input[type=text],select{width:100%;font:inherit;padding:.5rem .6rem;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink)}
input.digits{font-size:1.3rem;letter-spacing:.15em;font-variant-numeric:tabular-nums}.msg{font-size:.8rem;min-height:1.1em;margin-top:.2rem}.msg.bad{color:var(--warn)}.msg.good{color:var(--ok)}
.chk{display:flex;gap:.4rem;align-items:center;padding-bottom:.55rem;white-space:nowrap}
#zoom{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;z-index:20;cursor:zoom-out}#zoom img{max-width:96vw;max-height:94vh}
@media (max-width:760px){.imgs{grid-template-columns:1fr}.fields{grid-template-columns:1fr 1fr}}
</style></head><body>
<header><h1>Plate labelling</h1><div class="bar"><i id="bar"></i></div><span id="count"></span>
<button id="clear">Clear all</button><button class="primary" id="dl">Download CSV</button></header>
<main id="list"></main><div id="zoom"><img alt=""></div>
<script>
const DATA=__DATA__, CITIES=__CITIES__, LETTERS=__LETTERS__, KEY="plate_labels_v1";
const BN="০১২৩৪৫৬৭৮৯";
let S={};try{S=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){}};
const digitsOf=t=>[...t].map(c=>{const i=BN.indexOf(c);return i>=0?String(i):(c>="0"&&c<="9"?c:"")}).join("");
const el=(t,c,x)=>{const n=document.createElement(t);if(c)n.className=c;if(x!==undefined)n.textContent=x;return n};
function isDone(f){const s=S[f];return !!s&&(s.unreadable||digitsOf(s.digits||"").length===6)}
function refresh(){const n=DATA.filter(d=>isDone(d.file)).length;document.getElementById("count").textContent=n+" / "+DATA.length+" done";document.getElementById("bar").style.width=(100*n/DATA.length)+"%";
  document.querySelectorAll(".card").forEach(c=>c.classList.toggle("done",isDone(c.dataset.f)))}
function pick(label,opts,val,onchange,blank){const w=el("div");w.appendChild(el("label","",label));const s=el("select");
  const b=el("option","",blank);b.value="";s.appendChild(b);opts.forEach(o=>{const x=el("option","",o);x.value=o;s.appendChild(x)});s.value=val||"";s.onchange=()=>onchange(s.value);w.appendChild(s);return w}
DATA.forEach((d,i)=>{
  const f=d.file;S[f]=S[f]||{};const card=el("div","card");card.dataset.f=f;
  const h=el("h2");h.appendChild(el("span","",(i+1)+". "+f));card.appendChild(h);
  const g=el("div","imgs");[["photo",d.img],["contrast-enhanced (helps with embossed plates)",d.enh]].forEach(([cap,src])=>{
    const fig=el("figure");const im=el("img");im.src=src;im.alt=cap;im.onclick=()=>{const z=document.getElementById("zoom");z.firstElementChild.src=src;z.style.display="flex"};fig.appendChild(im);fig.appendChild(el("figcaption","",cap+" - click to zoom"));g.appendChild(fig)});
  card.appendChild(g);
  const fl=el("div","fields");
  const dw=el("div");dw.appendChild(el("label","","Number: 6 digits (e.g. 11-2789)"));
  const di=el("input","digits");di.type="text";di.inputMode="numeric";di.placeholder="112789";di.value=S[f].digits||"";const msg=el("div","msg");
  const check=()=>{const n=digitsOf(di.value);S[f].digits=di.value;
    if(S[f].unreadable){msg.textContent="marked unreadable";msg.className="msg"}
    else if(!n){msg.textContent="";msg.className="msg"}
    else if(n.length!==6){msg.textContent=n.length+" digits - need 6";msg.className="msg bad"}
    else{msg.textContent="ok: "+n.slice(0,2)+"-"+n.slice(2);msg.className="msg good"}
    save();refresh()};
  di.oninput=check;dw.appendChild(di);dw.appendChild(msg);fl.appendChild(dw);
  fl.appendChild(pick("City / top-row word (pick one)",CITIES,S[f].city,v=>{S[f].city=v;save()},"- not sure -"));
  fl.appendChild(pick("Class letter",[...LETTERS],S[f].letter,v=>{S[f].letter=v;save()},"-"));
  const cw=el("div","chk");const cb=el("input");cb.type="checkbox";cb.checked=!!S[f].unreadable;cb.onchange=()=>{S[f].unreadable=cb.checked;check()};cw.appendChild(cb);cw.appendChild(el("span","","unreadable"));fl.appendChild(cw);
  card.appendChild(fl);document.getElementById("list").appendChild(card);check();
});
document.getElementById("zoom").onclick=e=>{e.currentTarget.style.display="none"};
document.getElementById("clear").onclick=()=>{if(confirm("Clear every label?")){S={};save();location.reload()}};
document.getElementById("dl").onclick=()=>{
  const q=v=>/[",\n]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v;
  const rows=["file,digits,city,letter"];
  DATA.forEach(d=>{const s=S[d.file]||{};if(s.unreadable)return;const n=digitsOf(s.digits||"");if(n.length!==6)return;rows.push([d.file,n,s.city||"",s.letter||""].map(q).join(","))});
  const done=rows.length-1;if(done<DATA.length&&!confirm((DATA.length-done)+" plate(s) are not finished and will be left out. Download anyway?"))return;
  const a=document.createElement("a");a.href=URL.createObjectURL(new Blob(["﻿"+rows.join("\n")+"\n"],{type:"text/csv;charset=utf-8"}));a.download="labels_embossed10.csv";a.click()};
refresh();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default=None, help="default: <folder>/label_page.html")
    a = ap.parse_args()
    folder = Path(a.folder)
    out = Path(a.out) if a.out else folder / "label_page.html"
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXT)
    names = [p.name for p in files]
    if len(set(names)) != len(names):
        sys.exit("two images share a file name; rename one so the CSV can tell them apart")
    data = []
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        s = min(1.0, MAX_W / img.shape[1])
        if s < 1.0:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        elif img.shape[1] < 500:  # small crops: enlarge so the text is easy to see
            img = cv2.resize(img, None, fx=500 / img.shape[1], fy=500 / img.shape[1], interpolation=cv2.INTER_CUBIC)
        data.append({"file": p.name, "img": _b64(img), "enh": _b64(_enhanced(img))})
    html = (HTML.replace("__DATA__", json.dumps(data)).replace("__CITIES__", json.dumps(CITY_WORDS, ensure_ascii=False))
            .replace("__LETTERS__", json.dumps(LETTERS, ensure_ascii=False)))
    out.write_text(html, encoding="utf-8")
    print(f"{len(data)} plates -> {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
