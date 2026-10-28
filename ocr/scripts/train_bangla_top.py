"""Train the Bengali plate top-row reader (city word + class letter) and export it to ONNX.

Two softmax heads over a small CNN. Training-only (needs torch); inference uses onnxruntime.

Usage:
    python ocr/scripts/train_bangla_top.py --steps 3000 [--real top_labels.npz]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ocr.synth_row import CITY_WORDS, LETTERS, TOP_H, TOP_W, sample_top  # noqa: E402

ONNX_PATH = Path(__file__).resolve().parent.parent / "bangla_top.onnx"


class TopNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]

        self.cnn = nn.Sequential(*block(1, 32), nn.MaxPool2d(2), *block(32, 64), nn.MaxPool2d(2),
                                 *block(64, 128), nn.MaxPool2d(2), *block(128, 192), nn.MaxPool2d((2, 2)))
        feat = 192 * (TOP_H // 16) * (TOP_W // 16)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat, 256), nn.ReLU(inplace=True), nn.Dropout(0.2))
        self.city = nn.Linear(256, len(CITY_WORDS))
        self.letter = nn.Linear(256, len(LETTERS))

    def forward(self, x: torch.Tensor):
        x = (x - x.mean(dim=(2, 3), keepdim=True)) / (x.std(dim=(2, 3), keepdim=True) + 1e-3)
        h = self.fc(self.cnn(x).flatten(1))
        return self.city(h), self.letter(h)


def _gen_chunk(args):
    import cv2

    cv2.setNumThreads(1)
    seed, n, style = args
    rng = np.random.default_rng(seed)
    xs, cs, ls = [], [], []
    for _ in range(n):
        img, c, l = sample_top(rng, style)
        xs.append(img)
        cs.append(c)
        ls.append(l)
    return np.stack(xs), np.array(cs), np.array(ls)


def build_dataset(n: int, seed: int, procs: int | None = None, style: str = "ink"):
    import os
    from multiprocessing import Pool

    procs = procs or int(os.environ.get("GEN_PROCS", "10"))  # fewer processes leave a core for a training run on the GPU
    with Pool(procs) as pool:
        parts = pool.map(_gen_chunk, [(seed * 100003 + i, 500, style) for i in range(max(1, n // 500))])
    return tuple(np.concatenate([p[k] for p in parts]) for k in range(3))


def evaluate(model: nn.Module, n: int = 300, style: str = "ink") -> tuple[float, float]:
    rng = np.random.default_rng(777)
    model.eval()
    xs, cs, ls = zip(*(sample_top(rng, style) for _ in range(n)))
    dev = next(model.parameters()).device
    x = torch.from_numpy(np.stack(xs)).float().div(255).unsqueeze(1).to(dev)
    with torch.no_grad():
        pc, pl = model(x)
    model.train()
    return float((pc.argmax(1).cpu().numpy() == np.array(cs)).mean()), float((pl.argmax(1).cpu().numpy() == np.array(ls)).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--samples", type=int, default=40000)
    ap.add_argument("--dataset")
    ap.add_argument("--eval-n", type=int, default=300, help="synthetic plates used for each progress check (small = fast)")
    ap.add_argument("--style", choices=("ink", "relief"), default="ink", help="relief: embossed plates seen through opencv.relief.relief_view")
    ap.add_argument("--onnx-out")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--init")
    ap.add_argument("--real", help="npz: x (n,48,160) uint8, city (n,) and letter (n,) indices (-1 = unknown)")
    ap.add_argument("--real-frac", type=float, default=0.35)
    ap.add_argument("--save", default=str(ONNX_PATH.with_suffix(".pt")))
    a = ap.parse_args()
    if a.style == "relief" and not a.onnx_out:  # a separate model: the shipped ink model stays untouched
        a.onnx_out = str(ONNX_PATH.with_name("bangla_top_relief.onnx"))
        a.save = str(ONNX_PATH.with_name("bangla_top_relief.pt"))

    torch.manual_seed(0)
    torch.set_num_threads(8)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("training device:", dev, flush=True)
    model = TopNet()
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu"))
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    t0 = time.time()
    if a.dataset and Path(a.dataset).exists():
        d = np.load(a.dataset)
        sx, sc, sl = d["x"], d["c"], d["l"]
    else:
        sx, sc, sl = build_dataset(a.samples, seed=2, style=a.style)
        if a.dataset:
            np.savez(a.dataset, x=sx, c=sc, l=sl)
    print(f"synthetic dataset {sx.shape} ready in {time.time() - t0:.0f}s", flush=True)
    sxt, sct, slt = torch.from_numpy(sx).unsqueeze(1).to(dev), torch.from_numpy(sc).long().to(dev), torch.from_numpy(sl).long().to(dev)
    g = torch.Generator(device=dev).manual_seed(11)
    real = None
    if a.real:
        d = np.load(a.real)
        real = (torch.from_numpy(d["x"]).float().div(255).unsqueeze(1).to(dev), torch.from_numpy(d["city"]).long().to(dev), torch.from_numpy(d["letter"]).long().to(dev))
        print("real samples", len(real[1]))
    rng = np.random.default_rng(3)
    model.train()
    t0 = time.time()
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, len(sct), (a.batch,), generator=g, device=dev)
        x = sxt[idx].float() / 255.0
        x = (x * torch.empty(a.batch, 1, 1, 1, device=dev).uniform_(0.7, 1.2) + torch.empty(a.batch, 1, 1, 1, device=dev).uniform_(-0.08, 0.08)).clamp(0, 1)
        c, l = sct[idx], slt[idx]
        if real is not None:
            k = int(a.batch * a.real_frac)
            idx = torch.from_numpy(rng.integers(0, len(real[1]), k)).to(dev)
            rx = (real[0][idx] * torch.empty(k, 1, 1, 1, device=dev).uniform_(0.7, 1.2) + torch.empty(k, 1, 1, 1, device=dev).uniform_(-0.08, 0.08)).clamp(0, 1)
            x, c, l = torch.cat([x[: a.batch - k], rx]), torch.cat([c[: a.batch - k], real[1][idx]]), torch.cat([l[: a.batch - k], real[2][idx]])
        pc, pl = model(x)
        loss = F.cross_entropy(pc, c, ignore_index=-1) + F.cross_entropy(pl, l, ignore_index=-1)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if step % 250 == 0 or step == a.steps:
            ec, el = evaluate(model, n=a.eval_n, style=a.style)
            print(f"step {step:5d} loss {loss.item():.3f} synthetic city acc {ec:.3f} letter acc {el:.3f} {time.time() - t0:.0f}s", flush=True)
    model = model.to("cpu")
    torch.save(model.state_dict(), a.save)
    model.eval()
    out_path = Path(a.onnx_out) if a.onnx_out else ONNX_PATH
    torch.onnx.export(model, torch.zeros(1, 1, TOP_H, TOP_W), str(out_path), input_names=["x"],
                      output_names=["city", "letter"], dynamic_axes={"x": {0: "b"}, "city": {0: "b"}, "letter": {0: "b"}},
                      opset_version=17, dynamo=False)
    print("exported", out_path)


if __name__ == "__main__":
    main()
