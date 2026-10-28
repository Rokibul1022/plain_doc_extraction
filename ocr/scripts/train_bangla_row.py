"""Train the Bengali plate digit-row reader (CRNN + CTC) on synthetic plates, then export to ONNX.

Training-only: needs torch (CPU is fine). At run time the exported model is used through
onnxruntime, so the application itself does not depend on torch.

Usage:
    python ocr/scripts/train_bangla_row.py --steps 4000 [--real labels.npz] [--init model.pt]
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

from ocr.digits import DIGITS  # noqa: E402
from ocr.synth_row import IN_H, IN_W, sample  # noqa: E402

ONNX_PATH = Path(__file__).resolve().parent.parent / "bangla_row.onnx"
N_CLASSES = 11  # 10 digits + CTC blank (index 10)


class CRNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]

        self.cnn = nn.Sequential(
            *block(1, 16), nn.MaxPool2d(2, 2),
            *block(16, 32), nn.MaxPool2d(2, 2),
            *block(32, 64), *block(64, 64), nn.MaxPool2d((2, 1)),
            *block(64, 96), nn.MaxPool2d((2, 1)),
        )
        self.proj = nn.Linear(96 * (IN_H // 16), 160)
        self.rnn = nn.LSTM(160, 128, num_layers=2, bidirectional=True, dropout=0.2, batch_first=True)
        self.out = nn.Linear(256, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, 1, H, W) -> (B, T, C) log-probs
        x = (x - x.mean(dim=(2, 3), keepdim=True)) / (x.std(dim=(2, 3), keepdim=True) + 1e-3)
        f = self.cnn(x)  # (B, C, H/16, W/4)
        f = f.permute(0, 3, 1, 2).flatten(2)  # (B, W/4, C*H/16); flatten keeps the batch dimension dynamic in ONNX
        f, _ = self.rnn(F.relu(self.proj(f)))
        return F.log_softmax(self.out(f), dim=-1)


def _gen_chunk(args):
    """Worker: `n` synthetic plates with their own seed (module-level so Windows can pickle it)."""
    import cv2

    cv2.setNumThreads(1)
    seed, n, style = args
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for _ in range(n):
        img, label = sample(rng, style)
        xs.append(img)
        ys.append([DIGITS.index(c) for c in label])
    return np.stack(xs), np.array(ys, np.int64)


def build_dataset(n: int, seed: int, procs: int | None = None, style: str = "ink"):
    import os
    from multiprocessing import Pool

    procs = procs or max(2, min(int(os.environ.get("GEN_PROCS", "10")), (os.cpu_count() or 4) - 2))
    chunk = 500
    jobs = [(seed * 100003 + i, chunk, style) for i in range(max(1, n // chunk))]
    with Pool(procs) as pool:
        parts = pool.map(_gen_chunk, jobs)
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


class Synth(torch.utils.data.IterableDataset):  # (kept for quick experiments; training uses build_dataset)
    def __iter__(self):
        import cv2

        cv2.setNumThreads(1)  # one thread per worker: many workers x many OpenCV threads crawls
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng((info.seed if info else 0) % (2 ** 32))
        while True:
            img, label = sample(rng)
            yield torch.from_numpy(img).float().div(255).unsqueeze(0), torch.tensor([DIGITS.index(c) for c in label])


def decode(logp: torch.Tensor) -> list[str]:
    out = []
    for seq in logp.argmax(-1).cpu().numpy():
        s, prev = [], -1
        for k in seq:
            if k != prev and k != N_CLASSES - 1:
                s.append(DIGITS[k])
            prev = k
        out.append("".join(s))
    return out


def evaluate(model: nn.Module, n: int = 300, style: str = "ink") -> tuple[float, float]:
    rng = np.random.default_rng(12345)
    model.eval()
    imgs, labels = zip(*(sample(rng, style) for _ in range(n)))
    dev = next(model.parameters()).device
    x = torch.from_numpy(np.stack(imgs)).float().div(255).unsqueeze(1).to(dev)
    with torch.no_grad():
        pred = decode(model(x))
    exact = float(np.mean([p == l for p, l in zip(pred, labels)]))
    char = float(np.mean([sum(a == b for a, b in zip(p.ljust(6), l)) / 6 for p, l in zip(pred, labels)]))
    model.train()
    return exact, char


def export(model: nn.Module, path: Path | None = None) -> None:
    path = path or ONNX_PATH
    model = model.to("cpu")
    model.eval()
    dummy = torch.zeros(1, 1, IN_H, IN_W)
    torch.onnx.export(model, dummy, str(path), input_names=["x"], output_names=["logp"],
                      dynamic_axes={"x": {0: "b"}, "logp": {0: "b"}}, opset_version=17, dynamo=False)
    print("exported", path, path.stat().st_size // 1024, "KB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--samples", type=int, default=60000, help="synthetic plates to pre-generate")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--init")
    ap.add_argument("--export-only", action="store_true", help="load --init weights and just export the ONNX model")
    ap.add_argument("--real", help="npz with x (n,48,128) uint8 and y (n,) digit strings for fine-tuning")
    ap.add_argument("--real-frac", type=float, default=0.35)
    ap.add_argument("--real-aug", action="store_true", help="heavy geometric augmentation of the real plates")
    ap.add_argument("--onnx-out", help="export path (default: the shipped model)")
    ap.add_argument("--dataset", help="cache file for the synthetic dataset (npz); created if missing")
    ap.add_argument("--eval-n", type=int, default=300, help="synthetic plates used for each progress check (small = fast)")
    ap.add_argument("--style", choices=("ink", "relief", "relief_easy"), default="ink",
                    help="relief: train on embossed plates (no ink) seen through opencv.relief.relief_view")
    ap.add_argument("--save", default=str(Path(__file__).resolve().parent.parent / "bangla_row.pt"))
    a = ap.parse_args()
    if a.style.startswith("relief"):  # a separate model: the shipped ink reader stays untouched
        a.onnx_out = a.onnx_out or str(ONNX_PATH.with_name("bangla_row_relief.onnx"))
        if a.save.endswith("bangla_row.pt"):
            a.save = a.save.replace("bangla_row.pt", "bangla_row_relief.pt")

    torch.manual_seed(0)
    torch.set_num_threads(8)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("training device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "(no CUDA build/GPU)", flush=True)
    model = CRNN()
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu"))
    if a.export_only:
        export(model, Path(a.onnx_out) if a.onnx_out else None)
        return
    model = model.to(dev)
    print("params", sum(p.numel() for p in model.parameters()), flush=True)

    t0 = time.time()
    if a.dataset and Path(a.dataset).exists():
        d = np.load(a.dataset)
        sx, sy = d["x"], d["y"]
    else:
        sx, sy = build_dataset(a.samples, seed=1, style=a.style)
        if a.dataset:
            np.savez(a.dataset, x=sx, y=sy)
    print(f"synthetic dataset {sx.shape} ready in {time.time() - t0:.0f}s", flush=True)
    sxt = torch.from_numpy(sx).unsqueeze(1).to(dev)  # uint8 (N, 1, H, W) - kept on the GPU, no per-step copies
    syt = torch.from_numpy(sy).to(dev)

    real = None
    if a.real:
        d = np.load(a.real, allow_pickle=True)
        real = (torch.from_numpy(d["x"]).unsqueeze(1).to(dev), torch.tensor([[DIGITS.index(c) for c in s_] for s_ in d["y"]]).to(dev))
        print("real fine-tune samples", len(real[1]), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    ctc = nn.CTCLoss(blank=N_CLASSES - 1, zero_infinity=True)
    g = torch.Generator(device=dev).manual_seed(7)

    def geo_augment(x: torch.Tensor) -> torch.Tensor:
        """Random affine (scale, slant, rotation, shift) + blur on the GPU: many distinct views per real plate."""
        k = x.shape[0]
        ang = torch.empty(k, device=dev).uniform_(-0.07, 0.07)
        sh = torch.empty(k, device=dev).uniform_(-0.18, 0.18)
        sx = torch.empty(k, device=dev).uniform_(0.88, 1.12)
        sy = torch.empty(k, device=dev).uniform_(0.88, 1.12)
        tx = torch.empty(k, device=dev).uniform_(-0.06, 0.06)
        ty = torch.empty(k, device=dev).uniform_(-0.10, 0.10)
        cos, sin = torch.cos(ang), torch.sin(ang)
        theta = torch.stack([torch.stack([sx * cos, -sin + sh, tx], 1), torch.stack([sin, sy * cos, ty], 1)], 1)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
        blur = torch.rand(k, device=dev) < 0.4
        if blur.any():
            ker = torch.tensor([1.0, 2.0, 1.0], device=dev) / 4.0
            xb = F.conv2d(F.pad(x[blur], (1, 1, 0, 0), mode="replicate"), ker.view(1, 1, 1, 3))
            x = x.clone()
            x[blur] = xb
        return x

    def augment(x: torch.Tensor) -> torch.Tensor:
        k = x.shape[0]
        x = x.float() / 255.0
        x = x * torch.empty(k, 1, 1, 1, device=dev).uniform_(0.7, 1.2) + torch.empty(k, 1, 1, 1, device=dev).uniform_(-0.08, 0.08)
        x = x + torch.randn_like(x) * torch.empty(k, 1, 1, 1, device=dev).uniform_(0.0, 0.03)
        shift = int(torch.randint(-3, 4, (1,), generator=g, device=dev))
        return torch.roll(x, shift, dims=3).clamp(0, 1)

    model.train()
    t0, run = time.time(), 0.0
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, len(sy), (a.batch,), generator=g, device=dev)
        x, y = augment(sxt[idx]), syt[idx]
        if real is not None:
            k = int(a.batch * a.real_frac)
            ridx = torch.randint(0, len(real[1]), (k,), generator=g, device=dev)
            rx = augment(real[0][ridx])
            if a.real_aug:
                rx = geo_augment(rx)
            x = torch.cat([x[: a.batch - k], rx])
            y = torch.cat([y[: a.batch - k], real[1][ridx]])
        logp = model(x)  # (B, T, C)
        t = logp.shape[1]
        loss = ctc(logp.permute(1, 0, 2).float(), y, torch.full((x.shape[0],), t, dtype=torch.long),
                   torch.full((x.shape[0],), y.shape[1], dtype=torch.long))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 250 == 0 or step == a.steps:
            ex, ch = evaluate(model, n=a.eval_n, style=a.style)
            print(f"step {step:5d}  loss {run:.3f}  synthetic exact {ex:.3f}  char {ch:.3f}  {time.time() - t0:.0f}s", flush=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, a.save)
    export(model, Path(a.onnx_out) if a.onnx_out else None)


if __name__ == "__main__":
    main()
