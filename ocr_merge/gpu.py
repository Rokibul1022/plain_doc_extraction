"""Run the merged network on the GPU (PyTorch + CUDA) instead of ONNX Runtime on the CPU.

The ONNX Runtime installed here is the CPU build, so `plate_net.onnx` always runs on the CPU. This gives the same reader a
PyTorch session with the training weights (`ocr_merge/weights/plate_net.pt`, local only) on `cuda`; the reader, the voting and
the decision rules are untouched. Only the model call moves (YOLO and the OpenCV scan stay on the CPU). Needs torch.

    from ocr_merge.gpu import use_gpu
    print(use_gpu())               # cuda:0 (NVIDIA ...)
    from ocr_merge.pipeline import read_plate
"""

from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).with_name("weights") / "plate_net.pt"


class TorchSession:
    """Stands in for the ONNX session: `run(None, {"row": ..., "top": ...})` -> [logp, city, letter] as numpy arrays."""

    def __init__(self, weights: Path = WEIGHTS, device: str = "cuda") -> None:
        import torch

        from ocr_merge.model import PlateNet

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available to PyTorch")
        self.torch = torch
        self.device = torch.device(device)
        self.model = PlateNet()
        self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model.to(self.device).eval()

    def run(self, _outputs, feed: dict) -> list[np.ndarray]:
        t = self.torch
        with t.no_grad():
            out = self.model(t.from_numpy(feed["row"]).to(self.device), t.from_numpy(feed["top"]).to(self.device))
        return [o.cpu().numpy() for o in out]

    @property
    def name(self) -> str:
        return f"{self.device} ({self.torch.cuda.get_device_name(self.device)})" if self.device.type == "cuda" else str(self.device)


def attach(reader, weights: Path | None = None, device: str = "cuda") -> str | None:
    """Put `reader` on the GPU if PyTorch with CUDA and the weights are available; returns the device name, else None (CPU kept)."""
    try:
        import torch

        weights = Path(weights) if weights else WEIGHTS
        if not (torch.cuda.is_available() and weights.exists()):
            return None
        reader._sess = TorchSession(weights, device)
        reader._tried = True
        return reader._sess.name
    except Exception:
        return None


def use_gpu(weights: Path = WEIGHTS, device: str = "cuda") -> str:
    """Make the pipeline's merged reader run on the GPU (raises if that is not possible); returns the device name."""
    from ocr_merge.pipeline import reader

    r = reader()
    r._sess = TorchSession(Path(weights), device)
    r._tried = True
    return r._sess.name
