"""PlateNet: the four Bengali readers (ink/relief x digit row/top row) merged into one network. Training only (torch).

One CNN backbone is shared by both rows of the plate and by both looks of it (ink, or `relief_view` for embossed plates):

    row (B, 1, 48, 128) --+                      +-- BiLSTM -> CTC digits      logp   (B, 32, 11)
                          +-- shared CNN (3 x W/4)
    top (B, 1, 48, 160) --+                      +-- pool -> FC -> city word    city   (B, 19)
                                                               -> class letter  letter (B, 32)

Exported as one ONNX file with two inputs and three outputs, so a plate needs one inference call instead of four.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ocr.rowprep import IN_H
from ocr.synth_row import CITY_WORDS, LETTERS

N_CLASSES = 11  # 10 digits + CTC blank (index 10)


class PlateNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]

        self.cnn = nn.Sequential(
            *block(1, 32), nn.MaxPool2d(2, 2),
            *block(32, 64), nn.MaxPool2d(2, 2),
            *block(64, 128), *block(128, 128), nn.MaxPool2d((2, 1)),
            *block(128, 160), nn.MaxPool2d((2, 1)),
        )  # (B, 160, H/16, W/4)
        c = 160 * (IN_H // 16)
        self.proj = nn.Linear(c, 160)
        self.rnn = nn.LSTM(160, 128, num_layers=2, bidirectional=True, dropout=0.2, batch_first=True)
        self.out = nn.Linear(256, N_CLASSES)
        self.pool = nn.AdaptiveAvgPool2d((1, 8))
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(160 * 8, 256), nn.ReLU(inplace=True), nn.Dropout(0.2))
        self.city = nn.Linear(256, len(CITY_WORDS))
        self.letter = nn.Linear(256, len(LETTERS))

    @staticmethod
    def _norm(x: torch.Tensor) -> torch.Tensor:
        return (x - x.mean(dim=(2, 3), keepdim=True)) / (x.std(dim=(2, 3), keepdim=True) + 1e-3)

    def read_row(self, x: torch.Tensor) -> torch.Tensor:  # (B, 1, 48, 128) -> (B, T, 11) log-probs
        f = self.cnn(self._norm(x)).permute(0, 3, 1, 2).flatten(2)
        f, _ = self.rnn(F.relu(self.proj(f)))
        return F.log_softmax(self.out(f), dim=-1)

    def read_top(self, x: torch.Tensor):  # (B, 1, 48, 160) -> city logits, letter logits
        h = self.fc(self.pool(self.cnn(self._norm(x))).flatten(1))
        return self.city(h), self.letter(h)

    def forward(self, row: torch.Tensor, top: torch.Tensor):
        return (self.read_row(row), *self.read_top(top))


def export(model: PlateNet, path) -> None:
    from ocr.synth_row import TOP_H, TOP_W
    from ocr.rowprep import IN_W

    model = model.to("cpu").eval()
    torch.onnx.export(model, (torch.zeros(1, 1, IN_H, IN_W), torch.zeros(1, 1, TOP_H, TOP_W)), str(path),
                      input_names=["row", "top"], output_names=["logp", "city", "letter"],
                      dynamic_axes={"row": {0: "b"}, "top": {0: "t"}, "logp": {0: "b"}, "city": {0: "t"}, "letter": {0: "t"}},
                      opset_version=17, dynamo=False)
