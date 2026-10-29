"""Train the merged reader (PlateNet) and export it to ONNX. Needs torch (GPU recommended).

    # 1. from scratch on the four synthetic sets in `synthetic data/` (ink + embossed, digit row + top row)
    python ocr_merge/scripts/train_merged.py --steps 8000 --lr 1e-3 --save ocr_merge/weights/base_synthetic.pt
    # 2. fine-tune on the real labels (both looks of every labelled plate)
    python ocr_merge/scripts/train_merged.py --init ocr_merge/weights/base_synthetic.pt --real --steps 1500 --lr 3e-4 \
        --save ocr_merge/weights/plate_net.pt --onnx-out ocr_merge/plate_net.onnx [--holdout a.png,b.png | --holdout-csv test.csv]

Labels: labels.csv and labels_embossed10.csv (file,digits[,city,letter]); photos are looked up in the plate folders.
Hold-outs drop the named photos and every other photo of the same plate number, so the test stays honest.
"""

import argparse
import csv
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ocr.bangla_reader import VIEWS, region  # noqa: E402
from ocr.digits import DIGITS, normalize_digits  # noqa: E402
from ocr.synth_row import CITY_WORDS, LETTERS, to_top_region  # noqa: E402
from ocr_merge.model import N_CLASSES, PlateNet, export  # noqa: E402
from opencv.relief import relief_view  # noqa: E402

SYN = ROOT / "synthetic data"
LABELS = [ROOT / "labels.csv", ROOT / "labels_embossed10.csv"]
FOLDERS = [ROOT / "LPR_200 1st", ROOT / "LPR catagory", ROOT / "new samples+3d"]
EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
EVAL_N = 300  # the last plates of every synthetic set are kept out of training for the progress check


def shifted(c: np.ndarray, s: float) -> np.ndarray:
    h = c.shape[0]
    if s > 0:
        return c[int(s * h):]
    if s < 0:
        return cv2.copyMakeBorder(c, int(-s * h), 0, 0, 0, cv2.BORDER_REPLICATE)
    return c


def labelled_plates(cache: Path) -> dict:
    """{file: {digits, city, letter, crops}} for every labelled photo; the OpenCV scan is cached (it is the slow part)."""
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    from opencv.scan import scan

    paths = {}
    for folder in FOLDERS:
        for p in folder.rglob("*"):
            if p.suffix.lower() in EXT:
                paths.setdefault(p.name, p)
    plates = {}
    for lab in LABELS:
        for r in csv.DictReader(open(lab, encoding="utf-8-sig")):
            digits = normalize_digits(r.get("digits") or "")
            if len(digits) != 6 or r.get("status", "labelled") == "unreadable" or r["file"] not in paths or r["file"] in plates:
                continue
            s = scan(cv2.imread(str(paths[r["file"]])), relief="off", make_page=False)
            city, letter = (r.get("city") or "").strip(), (r.get("letter") or "").strip()
            plates[r["file"]] = {"digits": digits, "city": CITY_WORDS.index(city) if city in CITY_WORDS else -1,
                                 "letter": LETTERS.index(letter) if letter in LETTERS else -1, "crops": [s.rectified, s.restored]}
            print(f"  scanned {len(plates)}", end="\r", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(plates, open(cache, "wb"))
    return plates


def real_arrays(plates: dict, files: list[str]):
    """Training views of the real plates, in both looks (ink and relief_view): rows (x, y) and tops (x, city, letter)."""
    xr, yr, xt, ct, lt = [], [], [], [], []
    for f in files:
        p = plates[f]
        for c in p["crops"]:
            for look in (lambda v: v, relief_view):
                v = look(c)
                for top, trim in VIEWS:
                    xr.append(region(v, top, trim))
                    yr.append([DIGITS.index(ch) for ch in p["digits"]])
                for sh in (-0.03, 0.0, 0.03):
                    xt.append(to_top_region(look(shifted(c, sh))))
                    ct.append(p["city"])
                    lt.append(p["letter"])
    return np.stack(xr), np.array(yr, np.int64), np.stack(xt), np.array(ct, np.int64), np.array(lt, np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=96, help="digit-row images per step (the top row uses 2/3 of it)")
    ap.add_argument("--init", help="start from these weights (.pt)")
    ap.add_argument("--real", action="store_true", help="mix in the labelled real plates")
    ap.add_argument("--real-frac", type=float, default=0.35)
    ap.add_argument("--holdout", default="", help="comma separated photo names kept out of training (and their twins)")
    ap.add_argument("--holdout-csv", action="append", default=[], help="CSV whose `file` column is kept out of training")
    ap.add_argument("--cache", default=str(ROOT / "ocr_merge" / "weights" / "real_plates.pkl"))
    ap.add_argument("--save", required=True)
    ap.add_argument("--onnx-out")
    a = ap.parse_args()

    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("training device:", dev, flush=True)
    model = PlateNet()
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu"))
    model = model.to(dev)
    print("params", sum(p.numel() for p in model.parameters()), flush=True)

    def load(name, keys):
        d = np.load(SYN / f"{name}_20k.npz")
        return [torch.from_numpy(d[k]).to(dev) for k in keys]

    syn_row = [load("ink_row", "xy"), load("relief_row", "xy")]
    syn_top = [load("ink_top", "xcl"), load("relief_top", "xcl")]
    n_syn = syn_row[0][0].shape[0] - EVAL_N

    real = None
    if a.real:
        plates = labelled_plates(Path(a.cache))
        held = {x.strip() for x in a.holdout.split(",") if x.strip()}
        for hc in a.holdout_csv:
            held |= {r["file"] for r in csv.DictReader(open(hc, encoding="utf-8-sig"))}
        held_numbers = {plates[f]["digits"] for f in held if f in plates}
        train = [f for f in plates if f not in held and plates[f]["digits"] not in held_numbers]
        print(f"{len(plates)} labelled plates | {len(held)} held out | training on {len(train)}", flush=True)
        real = [torch.from_numpy(v).to(dev) for v in real_arrays(plates, train)]
        print(f"real views: {len(real[1])} digit-row, {len(real[3])} top-row", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    ctc = nn.CTCLoss(blank=N_CLASSES - 1, zero_infinity=True)
    g = torch.Generator(device=dev).manual_seed(7)

    def photometric(x: torch.Tensor) -> torch.Tensor:
        k = x.shape[0]
        x = x.float() / 255.0
        x = x * torch.empty(k, 1, 1, 1, device=dev).uniform_(0.7, 1.2) + torch.empty(k, 1, 1, 1, device=dev).uniform_(-0.08, 0.08)
        x = x + torch.randn_like(x) * torch.empty(k, 1, 1, 1, device=dev).uniform_(0.0, 0.03)
        return x.clamp(0, 1)

    def geometric(x: torch.Tensor) -> torch.Tensor:
        """Random affine (scale, slant, rotation, shift) + light blur: many distinct views of each real plate."""
        k = x.shape[0]
        u = lambda lo, hi: torch.empty(k, device=dev).uniform_(lo, hi)  # noqa: E731
        ang, sh = u(-0.07, 0.07), u(-0.18, 0.18)
        sx, sy, tx, ty = u(0.88, 1.12), u(0.88, 1.12), u(-0.06, 0.06), u(-0.10, 0.10)
        cos, sin = torch.cos(ang), torch.sin(ang)
        theta = torch.stack([torch.stack([sx * cos, -sin + sh, tx], 1), torch.stack([sin, sy * cos, ty], 1)], 1)
        x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), padding_mode="border", align_corners=False)
        blur = torch.rand(k, device=dev) < 0.4
        if blur.any():
            ker = torch.tensor([1.0, 2.0, 1.0], device=dev).view(1, 1, 1, 3) / 4.0
            x = x.clone()
            x[blur] = F.conv2d(F.pad(x[blur], (1, 1, 0, 0), mode="replicate"), ker)
        return x

    def pick(n):
        return torch.randint(0, n, (1,), generator=g, device=dev)

    def batch_rows(n):
        n_real = int(n * a.real_frac) if real else 0
        xs, ys = [], []
        for k, (sx, sy) in enumerate(syn_row):  # half ink, half embossed
            m = (n - n_real) // 2 + (k * ((n - n_real) % 2))
            idx = torch.randint(0, n_syn, (m,), generator=g, device=dev)
            xs.append(photometric(sx[idx].unsqueeze(1)))
            ys.append(sy[idx])
        if n_real:
            idx = torch.randint(0, len(real[1]), (n_real,), generator=g, device=dev)
            xs.append(geometric(photometric(real[0][idx].unsqueeze(1))))
            ys.append(real[1][idx])
        x = torch.cat(xs)
        return torch.roll(x, int(torch.randint(-3, 4, (1,), generator=g, device=dev)), dims=3), torch.cat(ys)

    def batch_tops(n):
        n_real = int(n * a.real_frac) if real else 0
        xs, cs, ls = [], [], []
        for k, (sx, sc, sl) in enumerate(syn_top):
            m = (n - n_real) // 2 + (k * ((n - n_real) % 2))
            idx = torch.randint(0, n_syn, (m,), generator=g, device=dev)
            xs.append(photometric(sx[idx].unsqueeze(1)))
            cs.append(sc[idx])
            ls.append(sl[idx])
        if n_real:
            idx = torch.randint(0, len(real[3]), (n_real,), generator=g, device=dev)
            xs.append(photometric(real[2][idx].unsqueeze(1)))
            cs.append(real[3][idx])
            ls.append(real[4][idx])
        return torch.cat(xs), torch.cat(cs), torch.cat(ls)

    @torch.no_grad()
    def evaluate():
        model.eval()
        res = []
        for (rx, ry), (tx, tc, tl) in zip(syn_row, syn_top):
            logp = model.read_row(rx[-EVAL_N:].unsqueeze(1).float() / 255.0)
            pred = logp.argmax(-1).cpu().numpy()
            exact = 0
            for seq, want in zip(pred, ry[-EVAL_N:].cpu().numpy()):
                s, prev = [], -1
                for k in seq:
                    if k != prev and k != N_CLASSES - 1:
                        s.append(k)
                    prev = k
                exact += s == list(want)
            pc, pl = model.read_top(tx[-EVAL_N:].unsqueeze(1).float() / 255.0)
            res += [exact / EVAL_N, (pc.argmax(1) == tc[-EVAL_N:]).float().mean().item(), (pl.argmax(1) == tl[-EVAL_N:]).float().mean().item()]
        model.train()
        return res

    model.train()
    t0, run = time.time(), 0.0
    for step in range(1, a.steps + 1):
        xr, yr = batch_rows(a.batch)
        xt, ct, lt = batch_tops(a.batch * 2 // 3)
        logp, pc, pl = model(xr, xt)
        t = logp.shape[1]
        loss = ctc(logp.permute(1, 0, 2).float(), yr, torch.full((len(yr),), t, dtype=torch.long), torch.full((len(yr),), 6, dtype=torch.long))
        loss = loss + F.cross_entropy(pc, ct, ignore_index=-1) + F.cross_entropy(pl, lt, ignore_index=-1)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 500 == 0 or step == a.steps:
            e = evaluate()
            print(f"step {step:5d} loss {run:.3f} | synthetic ink: digits {e[0]:.3f} city {e[1]:.3f} letter {e[2]:.3f}"
                  f" | embossed: digits {e[3]:.3f} city {e[4]:.3f} letter {e[5]:.3f} | {time.time() - t0:.0f}s", flush=True)
    Path(a.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, a.save)
    if a.onnx_out:
        export(model, a.onnx_out)
        print("exported", a.onnx_out, Path(a.onnx_out).stat().st_size // 1024, "KB", flush=True)


if __name__ == "__main__":
    main()
