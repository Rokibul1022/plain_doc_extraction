"""Video plate scanning: track each plate, pick its best frames, vote readings, fuse frames.

A single CCTV frame of a moving car is often unreadable, but the same plate appears in dozens of
frames. Reading several of its sharpest frames and voting beats any single-frame enhancement,
and registering + averaging the frames gives a cleaner image (less noise and compression error)
than any one of them.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

import cv2
import numpy as np

from ocr.detector import PlateDetection, PlateDetector, rectify
from ocr.ocr import PlateOCR, PlateReading, vote
from ocr import verdict_bn
from ocr.verdict import verdict_from
from opencv.quality import assess_quality
from ocr.scan import get_detector, get_ocr, read_plate_full

KEEP_PER_TRACK = 12
TOP_N_READ = 6


@dataclass
class Observation:
    frame_idx: int
    time_s: float
    detection: PlateDetection
    crop: np.ndarray
    score: float


@dataclass
class PlateTrack:
    track_id: int
    obs: list[Observation] = field(default_factory=list)
    last_box: tuple[float, float, float, float] = (0, 0, 0, 0)
    velocity: tuple[float, float] = (0.0, 0.0)
    misses: int = 0
    n_seen: int = 0

    def predicted_box(self) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.last_box
        vx, vy = self.velocity
        return (x1 + vx, y1 + vy, x2 + vx, y2 + vy)

    def update(self, o: Observation) -> None:
        x1, y1, x2, y2 = o.detection.box
        px1, py1, px2, py2 = self.last_box
        if self.n_seen:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
            self.velocity = (0.6 * self.velocity[0] + 0.4 * (cx - pcx), 0.6 * self.velocity[1] + 0.4 * (cy - pcy))
        self.last_box = (x1, y1, x2, y2)
        self.n_seen += 1
        self.misses = 0
        self.obs.append(o)
        if len(self.obs) > KEEP_PER_TRACK:  # keep only the best frames
            self.obs.sort(key=lambda z: -z.score)
            self.obs = self.obs[:KEEP_PER_TRACK]


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def associate(tracks: list[PlateTrack], dets: list[PlateDetection]) -> dict[int, int]:
    """Greedy det->track assignment by overlap with the track's predicted box."""
    pairs = []
    for ti, t in enumerate(tracks):
        pb = t.predicted_box()
        pw = max(pb[2] - pb[0], 1.0)
        for di, d in enumerate(dets):
            iou = _iou(pb, d.box)
            cx, cy = (d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2
            dist = np.hypot(cx - (pb[0] + pb[2]) / 2, cy - (pb[1] + pb[3]) / 2) / pw
            if iou > 0.15 or dist < 0.9:
                pairs.append((iou - 0.3 * dist, ti, di))
    pairs.sort(reverse=True)
    used_t, used_d, out = set(), set(), {}
    for _, ti, di in pairs:
        if ti not in used_t and di not in used_d:
            used_t.add(ti)
            used_d.add(di)
            out[di] = ti
    return out


def frame_score(crop: np.ndarray, width_px: float) -> float:
    q = assess_quality(crop, width_px)
    return q.score + 0.15 * min(width_px, 200.0)


def fuse_crops(crops: list[np.ndarray], weights: list[float], scale: float = 2.0, max_w: int = 480) -> np.ndarray:
    """Register crops to the strongest one (ECC affine) and combine them with quality weights."""
    order = int(np.argmax(weights))
    ref = crops[order]
    s = min(scale, max_w / max(ref.shape[1], 1))
    rw, rh = max(16, int(ref.shape[1] * s)), max(8, int(ref.shape[0] * s))
    ref_big = cv2.resize(ref, (rw, rh), interpolation=cv2.INTER_CUBIC)
    ref_g = cv2.cvtColor(ref_big, cv2.COLOR_BGR2GRAY).astype(np.float32)
    acc = ref_big.astype(np.float32) * weights[order]
    wsum = weights[order]
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
    for i, (c, w) in enumerate(zip(crops, weights)):
        if i == order:
            continue
        big = cv2.resize(c, (rw, rh), interpolation=cv2.INTER_CUBIC)
        g = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY).astype(np.float32)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            cc, warp = cv2.findTransformECC(ref_g, g, warp, cv2.MOTION_AFFINE, crit, None, 5)
        except cv2.error:
            continue
        shift = np.hypot(warp[0, 2] / rw, warp[1, 2] / rh)
        if cc < 0.55 or shift > 0.18 or abs(warp[0, 0] - 1) > 0.15 or abs(warp[1, 1] - 1) > 0.15:
            continue
        aligned = cv2.warpAffine(big, warp, (rw, rh), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_REPLICATE)
        acc += aligned.astype(np.float32) * w
        wsum += w
    return np.clip(acc / wsum, 0, 255).astype(np.uint8)


@dataclass
class PlateTrackResult:
    track_id: int
    first_frame: int
    last_frame: int
    n_seen: int
    best_frame: int
    best_time_s: float
    best_crop: np.ndarray
    fused: np.ndarray | None
    restored: np.ndarray
    readable: np.ndarray
    reading: PlateReading
    frame_readings: list[tuple[int, str, float]]
    verdict: str
    steps: list[str]
    document: np.ndarray | None = None
    scan_document: np.ndarray | None = None
    typed: np.ndarray | None = None

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "frames_seen": self.n_seen,
            "best_frame": self.best_frame,
            "best_time_s": round(self.best_time_s, 2),
            "reading": self.reading.to_dict(),
            "frame_readings": [(i, t, round(c, 3)) for i, t, c in self.frame_readings],
            "verdict": self.verdict,
            "steps": self.steps,
        }


def _combine_reliability(readings: list[PlateReading], final_text: str) -> float:
    """Reliability of a multi-frame read: independent-ish frames that agree reinforce each other.

    Heuristic (errors across frames of one plate are correlated, so each frame counts at 70 %)."""
    miss = 1.0
    for r in readings:
        if r.text == final_text and r.reliability:
            miss *= 1.0 - 0.7 * r.reliability
    return float(min(0.995, 1.0 - miss))


def finalize_track(track: PlateTrack, ocr: PlateOCR, do_fuse: bool = True, latin: bool = False) -> PlateTrackResult:
    obs = sorted(track.obs, key=lambda o: -o.score)[:TOP_N_READ]
    readings: list[PlateReading] = []
    frame_readings: list[tuple[int, str, float]] = []
    steps: list[str] = []
    for o in obs:
        reading = read_plate_full(o.crop, plate_width=o.detection.width_px, ocr=ocr, latin=latin).reading
        if reading.text:
            readings.append(reading)
            frame_readings.append((o.frame_idx, reading.text, reading.confidence))

    fused = None
    if do_fuse and len(obs) >= 2:
        fused = fuse_crops([o.crop for o in obs], [max(o.score, 1.0) ** 2 for o in obs])
        steps.append(f"fused {len(obs)} frames")
    base_crop = fused if fused is not None else obs[0].crop
    fr = read_plate_full(base_crop, plate_width=obs[0].detection.width_px, ocr=ocr, latin=latin)
    steps += fr.steps
    if fused is not None and fr.reading.text:
        readings.append(fr.reading)

    reading = vote(readings, ocr.region) if readings else PlateReading(model=ocr.model_name)
    if readings:
        scripts = [r.script for r in readings if r.text == reading.text]
        reading.script = scripts[0] if scripts else readings[0].script
        reading.reliability = _combine_reliability(readings, reading.text)
    if not reading.text:
        verdict = "unreadable"
    elif reading.script == "bengali":
        verdict = verdict_bn.verdict(reading.reliability or 0.0)
    else:
        verdict = verdict_from(reading.reliability or 0.0)
    return PlateTrackResult(
        track_id=track.track_id,
        first_frame=min(o.frame_idx for o in track.obs),
        last_frame=max(o.frame_idx for o in track.obs),
        n_seen=track.n_seen,
        best_frame=obs[0].frame_idx,
        best_time_s=obs[0].time_s,
        best_crop=obs[0].crop,
        fused=fused,
        restored=fr.restored,
        readable=fr.readable,
        reading=reading,
        frame_readings=frame_readings,
        verdict=verdict,
        steps=steps,
        document=fr.document,
        scan_document=fr.scan_document,
        typed=fr.typed,
    )


def _sane_fps(fps: float | None) -> float:
    """Container metadata sometimes reports 0 or absurd rates (e.g. 1000 fps in some WebM)."""
    return float(fps) if fps and 1.0 <= fps <= 120.0 else 30.0


def iter_frames(path: str, target_fps: float, max_seconds: float | None) -> Iterator[tuple[int, float, np.ndarray]]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = _sane_fps(cap.get(cv2.CAP_PROP_FPS))
    stride = max(1, int(round(fps / max(target_fps, 0.5))))
    idx = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % stride == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                t = idx / fps
                if max_seconds is not None and t > max_seconds:
                    break
                yield idx, t, frame
            idx += 1
    finally:
        cap.release()


@dataclass
class VideoScanResult:
    tracks: list[PlateTrackResult]
    frames_processed: int
    duration_s: float
    elapsed_s: float

    def to_dict(self) -> dict:
        return {
            "frames_processed": self.frames_processed,
            "duration_s": round(self.duration_s, 2),
            "elapsed_s": round(self.elapsed_s, 2),
            "plates": [t.to_dict() for t in self.tracks],
        }


def scan_video(
    path: str,
    *,
    detector: PlateDetector | None = None,
    ocr: PlateOCR | None = None,
    target_fps: float = 6.0,
    max_seconds: float | None = None,
    max_gap: int = 6,
    progress: Callable[[float], None] | None = None,
    latin: bool = False,
) -> VideoScanResult:
    det = detector or get_detector()
    engine = ocr or get_ocr()
    t_start = time.perf_counter()
    cap = cv2.VideoCapture(path)
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()

    tracks: list[PlateTrack] = []
    finished: list[PlateTrack] = []
    next_id = 1
    n_proc = 0
    last_t = 0.0

    for idx, t, frame in iter_frames(path, target_fps, max_seconds):
        n_proc += 1
        last_t = t
        use_tiles = len(tracks) == 0 or n_proc % 5 == 0
        dets = det.detect(frame, tile=use_tiles)
        mapping = associate(tracks, dets)
        for di, d in enumerate(dets):
            crop = rectify(frame, d.quad)
            obs = Observation(idx, t, d, crop, frame_score(crop, d.width_px))
            if di in mapping:
                tracks[mapping[di]].update(obs)
            else:
                tr = PlateTrack(next_id)
                next_id += 1
                tr.update(obs)
                tracks.append(tr)
        matched_tracks = set(mapping.values())
        for ti, tr in enumerate(tracks):
            if ti not in matched_tracks:
                tr.misses += 1
        alive = []
        for tr in tracks:
            (finished if tr.misses > max_gap else alive).append(tr)
        tracks = alive
        if progress and total:
            progress(min(0.9, idx / total))

    finished += tracks
    results = []
    for k, tr in enumerate(t for t in finished if t.n_seen >= 2 or (t.obs and t.obs[0].detection.confidence >= 0.6)):
        results.append(finalize_track(tr, engine, latin=latin))
        if progress:
            progress(0.9 + 0.1 * (k + 1) / max(1, len(finished)))
    results.sort(key=lambda r: (-r.reading.confidence, r.first_frame))
    return VideoScanResult(results, n_proc, last_t, time.perf_counter() - t_start)
