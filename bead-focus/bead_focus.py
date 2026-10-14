"""Align bead diffraction images, then compare 2-D and radial intensities.

Requires numpy and Pillow. No hardware movement is performed.
A correlation is a similarity score, NOT an axial distance measurement.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


class FocusError(ValueError):
    """Invalid input or an image that cannot be compared reliably."""


@dataclass(frozen=True)
class Config:
    roi_size: int = 129
    max_shift_px: int = 8
    inner_radius_px: float = 0.0
    threshold: float | None = None
    saturation_level: float | None = None

    def validate(self) -> None:
        if not isinstance(self.roi_size, int) or self.roi_size < 33 or self.roi_size % 2 == 0:
            raise FocusError("roi_size 必须是至少 33 的奇数，例如 129。")
        if not isinstance(self.max_shift_px, int) or self.max_shift_px < 0:
            raise FocusError("允许 XY 偏移必须是非负整数像素。")
        radius = self.roi_size // 2 - self.max_shift_px - 4
        if radius < 8:
            raise FocusError("允许 XY 偏移过大；请减小偏移范围或增加选区边长。")
        if not np.isfinite(self.inner_radius_px) or not 0 <= self.inner_radius_px <= radius - 8:
            raise FocusError("内圈排除半径过大；应至少保留 8 个径向采样点。")
        if self.threshold is not None and not -1 <= self.threshold <= 1:
            raise FocusError("相关阈值必须在 -1 到 1 之间。")
        if self.saturation_level is not None and (
            not np.isfinite(self.saturation_level) or self.saturation_level <= 0
        ):
            raise FocusError("saturation_level 必须是正的有限数值。")


def as_gray(array: np.ndarray) -> np.ndarray:
    """Preserve grayscale precision. RGB is converted to luminance in float64."""
    a = np.asarray(array)
    if a.ndim == 3 and a.shape[-1] in (3, 4):
        if a.shape[-1] == 4:
            alpha = a[..., 3]
            opaque = np.iinfo(a.dtype).max if np.issubdtype(a.dtype, np.integer) else 1.0
            if not np.all(alpha == opaque):
                raise FocusError("不支持带半透明像素的图片；请使用原始相机灰度图。")
        a = np.asarray(a[..., :3], dtype=np.float64) @ np.array([0.2126, 0.7152, 0.0722])
    elif a.ndim != 2:
        raise FocusError("输入应为二维灰度图或 RGB 图，不应为视频/Z 图像堆栈。")
    try:
        a = np.asarray(a, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise FocusError("图像必须由数值像素组成。") from exc
    if min(a.shape) < 3 or not np.all(np.isfinite(a)):
        raise FocusError("图片过小或包含 NaN/无穷值。")
    return a


def load_image(path: str | Path, frame: int | None = None) -> np.ndarray:
    """Read PNG/TIFF/BMP/JPEG without reducing a grayscale TIFF to 8 bit.

    For a multi-page TIFF, explicitly select a frame using --frame.
    A multi-frame image is rejected unless its frame is explicitly selected.
    """
    path = Path(path)
    if path.suffix.lower() == ".npy":
        if frame not in (None, 0):
            raise FocusError(".npy 只支持一张二维图片；frame 应为 0。")
        return as_gray(np.load(path, allow_pickle=False))
    with Image.open(path) as im:
        if getattr(im, "n_frames", 1) > 1 and frame is None:
            raise FocusError("这是多帧图片；请用 --frame 明确选择一帧，或导出单帧图。")
        frame = 0 if frame is None else frame
        if frame < 0 or frame >= getattr(im, "n_frames", 1):
            raise FocusError(f"图片不存在第 {frame} 帧。")
        im.seek(frame)
        if im.mode in ("P", "CMYK"):
            im = im.convert("RGB")
        return as_gray(np.asarray(im))


def crop_bead(
    image: np.ndarray, size: int, center: Sequence[float] | None = None
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    image = as_gray(image)
    h, w = image.shape
    cx, cy = ((w - 1) / 2, (h - 1) / 2) if center is None else center
    if not np.isfinite(cx) or not np.isfinite(cy):
        raise FocusError("珠中心必须是有限的 x、y 像素坐标。")
    x0, y0 = int(round(cx)) - size // 2, int(round(cy)) - size // 2
    if x0 < 0 or y0 < 0 or x0 + size > w or y0 + size > h:
        raise FocusError("单珠 ROI 超出图片边界。请移动中心或减小 ROI，不能补零后比较。")
    return image[y0:y0 + size, x0:x0 + size].copy(), (x0, y0, x0 + size, y0 + size)


def normalize(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    a = a - a.mean()
    std = float(np.sqrt(np.mean(a * a)))
    if std <= np.finfo(float).eps * max(1.0, float(np.max(np.abs(a)))) * 100:
        raise FocusError("选区没有可比较的衍射纹理，可能是纯背景、全饱和或空白图。")
    return a / std


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise FocusError("相关计算的两个输入形状必须相同。")
    a, b = normalize(a), normalize(b)
    return float(np.clip(np.mean(a * b), -1, 1))


def sample_bilinear(image: np.ndarray, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    """Sample original intensities without padding, wrapping or clipping positions."""
    h, w = image.shape
    if np.min(xx) < 0 or np.min(yy) < 0 or np.max(xx) > w-1 or np.max(yy) > h-1:
        raise FocusError("平移后的采样范围超出选区；请减小允许 XY 偏移或扩大选区。")
    xa, ya = np.floor(xx).astype(int), np.floor(yy).astype(int)
    xb, yb = np.minimum(xa+1, w-1), np.minimum(ya+1, h-1)
    fx, fy = xx-xa, yy-ya
    return (image[ya, xa]*(1-fx)*(1-fy) + image[ya, xb]*fx*(1-fy)
            + image[yb, xa]*(1-fx)*fy + image[yb, xb]*fx*fy)


def local_sums(image: np.ndarray, size: int) -> np.ndarray:
    integral = np.pad(image, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]


def align_xy(reference: np.ndarray, current: np.ndarray, reference_center,
             current_center, max_shift: int) -> dict:
    """Bounded linear NCC registration; displacement is current minus reference.

    Entered centers first define corresponding coordinate origins. Returned dx/dy
    are residual offsets in these coordinates, not whole-field bead motion.
    All candidates use the same inner reference area and valid original pixels.
    """
    margin = max_shift + 2
    template = reference[margin:-margin, margin:-margin]
    normalize(template)
    size = template.shape[0]
    base = np.array(current_center, dtype=float) - np.array(reference_center, dtype=float)
    yy, xx = np.indices((size+2*max_shift, size+2*max_shift), dtype=float)
    search = sample_bilinear(current, xx+margin-max_shift+base[0], yy+margin-max_shift+base[1])
    search = search-search.mean()
    t = template-template.mean()
    shape = (search.shape[0]+size-1, search.shape[1]+size-1)
    convolution = np.fft.irfft2(np.fft.rfft2(search, s=shape)*np.fft.rfft2(t[::-1, ::-1], s=shape), s=shape)
    numerator = convolution[size-1:search.shape[0], size-1:search.shape[1]]
    count = size*size
    total = local_sums(search, size)
    variance = np.maximum(local_sums(search*search, size)-total*total/count, 0)
    denominator = np.sqrt(np.sum(t*t)*variance)
    scores = np.full_like(numerator, -np.inf)
    np.divide(numerator, denominator, out=scores, where=denominator > np.finfo(float).eps)
    if not np.any(np.isfinite(scores)):
        raise FocusError("当前选区没有可用于自动对齐的衍射纹理。")
    iy, ix = np.unravel_index(np.argmax(scores), scores.shape)
    dx, dy = float(ix-max_shift), float(iy-max_shift)
    refinement = []
    for index, values in ((ix, scores[iy, :]), (iy, scores[:, ix])):
        delta = 0.0
        if 0 < index < len(values)-1:
            left, middle, right = values[index-1:index+2]
            curvature = left-2*middle+right
            if np.all(np.isfinite((left, middle, right))) and curvature < -1e-12:
                delta = float(np.clip(0.5*(left-right)/curvature, -.5, .5))
        refinement.append(delta)
    yy, xx = np.indices(template.shape, dtype=float)
    def matched_score(offset_x, offset_y):
        matched = sample_bilinear(current, xx+margin+base[0]+offset_x, yy+margin+base[1]+offset_y)
        return pearson(template, matched)
    ncc = matched_score(dx, dy)
    refined_dx, refined_dy = dx+refinement[0], dy+refinement[1]
    refined_ncc = matched_score(refined_dx, refined_dy)
    if refined_ncc > ncc+1e-12:
        dx, dy, ncc = refined_dx, refined_dy, refined_ncc
    at_limit = max_shift > 0 and (ix in (0, 2*max_shift) or iy in (0, 2*max_shift))
    return {"ncc": ncc, "shift_dx_px": dx, "shift_dy_px": dy,
            "alignment_at_limit": bool(at_limit),
            "aligned_center": (reference_center[0]+base[0]+dx, reference_center[1]+base[1]+dy)}


def radial_profile(
    image: np.ndarray, inner_radius: float = 0, center: Sequence[float] | None = None,
    outer_radius: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Average 720 angular samples at each radius, using the exact center.

    Fractional coordinates are honored through bilinear intensity sampling.
    Registration is handled separately; this function never rescales ring radii.
    """
    image = as_gray(image)
    h, w = image.shape
    cx, cy = ((w - 1) / 2, (h - 1) / 2) if center is None else center
    if not np.isfinite(cx) or not np.isfinite(cy) or not (0 <= cx < w and 0 <= cy < h):
        raise FocusError("径向轮廓的中心坐标不在图片内。")
    outer = min(h, w) // 2 - 2 if outer_radius is None else outer_radius
    if not np.isfinite(inner_radius) or inner_radius < 0 or not isinstance(outer, (int, np.integer)):
        raise FocusError("径向采样半径无效。")
    radii = np.arange(int(np.ceil(inner_radius)), outer + 1, dtype=float) + 0.5
    if len(radii) < 8 or radii[-1] > min(cx, cy, w-1-cx, h-1-cy):
        raise FocusError("中心位置或半径不合适，不能取得完整的衍射环轮廓。")
    angle = np.arange(720, dtype=float) * (2 * np.pi / 720)
    xx = cx + radii[:, None] * np.cos(angle)
    yy = cy + radii[:, None] * np.sin(angle)
    samples = sample_bilinear(image, xx, yy)
    return radii, samples.mean(axis=1)


class FocusReference:
    """Persist one reference ROI and compare subsequent bead images."""

    def __init__(self, roi: np.ndarray, config: Config = Config(), source: str = "", center=None):
        config.validate()
        self.roi = as_gray(roi).copy()
        if self.roi.shape != (config.roi_size, config.roi_size):
            raise FocusError("参考 ROI 大小与配置不一致。")
        normalize(self.roi)
        self.config, self.source = config, source
        self.center = tuple(map(float, center if center is not None else (config.roi_size//2, config.roi_size//2)))
        self.outer_radius = config.roi_size//2-config.max_shift_px-4
        self.radius, self.profile = radial_profile(self.roi, config.inner_radius_px, self.center, self.outer_radius)
        normalize(self.profile)
        self.roi.setflags(write=False)
        self.legacy_reference = False
        self.reference_id = hashlib.sha256(self.roi.tobytes()+np.array(self.center).tobytes()).hexdigest()[:16]

    @classmethod
    def from_image(cls, image: np.ndarray, center=None, config: Config = Config(), source=""):
        config.validate()
        roi, bounds = crop_bead(image, config.roi_size, center)
        h, w = image.shape[:2]
        cx, cy = ((w-1)/2, (h-1)/2) if center is None else center
        local_center = (cx-bounds[0], cy-bounds[1])
        return cls(roi, config, source, local_center)

    def save(self, path: str | Path) -> None:
        metadata = json.dumps({"version": 3, "config": asdict(self.config), "source": self.source, "center": self.center}, ensure_ascii=False)
        with Path(path).open("wb") as stream:
            np.savez_compressed(stream, roi=self.roi, metadata=np.array(metadata))

    @classmethod
    def load(cls, path: str | Path):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            version = metadata.get("version")
            if version not in (1, 2, 3):
                raise FocusError("不支持的参考文件版本。")
            config_data = metadata["config"]
            if version in (1, 2):
                config_data = {key: config_data[key] for key in ("roi_size", "max_shift_px", "inner_radius_px", "saturation_level") if key in config_data}
                config_data["threshold"] = None
            reference = cls(data["roi"], Config(**config_data), metadata.get("source", ""), metadata.get("center"))
            reference.legacy_reference = version in (1, 2)
            return reference

    def compare(self, image: np.ndarray, center=None) -> dict:
        """Align the current ROI, then calculate 2-D NCC and radial correlation."""
        c = self.config
        current, bounds = crop_bead(image, c.roi_size, center)
        h, w = image.shape[:2]
        cx, cy = ((w-1)/2, (h-1)/2) if center is None else center
        local_center = (cx-bounds[0], cy-bounds[1])
        alignment = align_xy(self.roi, current, self.center, local_center, c.max_shift_px)
        radius, cur_profile = radial_profile(current, c.inner_radius_px, alignment["aligned_center"], self.outer_radius)
        radial = pearson(self.profile, cur_profile)
        warnings = ["跨珠图样相似度不等于相同物理 Z；正式轨迹测量仍需为本组珠建立高度图像库。"]
        quality_issues = []
        if alignment["alignment_at_limit"]:
            quality_issues.append("最佳平移位于允许 XY 偏移的边界；请修正当前中心坐标或扩大允许范围后复核。")
        if c.saturation_level is not None:
            for name, roi in (("参考", self.roi), ("当前", current)):
                fraction = float(np.mean(roi >= c.saturation_level))
                if fraction > 0.01:
                    quality_issues.append(f"{name}选区至少 {fraction:.1%} 像素达到设定饱和值。")
        warnings.extend(quality_issues)
        return {
            "reference_id": self.reference_id,
            "ncc": alignment["ncc"],
            "shift_dx_px": alignment["shift_dx_px"],
            "shift_dy_px": alignment["shift_dy_px"],
            "alignment_at_limit": alignment["alignment_at_limit"],
            "aligned_current_center": [float(bounds[0]+alignment["aligned_center"][0]), float(bounds[1]+alignment["aligned_center"][1])],
            "radial_correlation": radial,
            "threshold": c.threshold,
            "matches_threshold": (radial >= c.threshold) if c.threshold is not None and not quality_issues else None,
            "quality_ok": not quality_issues,
            "roi_bounds": list(bounds),
            "warnings": warnings,
            "radius_px": radius.tolist(),
            "reference_profile": self.profile.tolist(),
            "current_profile": cur_profile.tolist(),
        }


CSV_FIELDS = ["image", "status", "reference_id", "ncc", "radial_correlation", "shift_dx_px", "shift_dy_px",
              "alignment_at_limit", "quality_ok", "threshold", "matches_threshold",
              "warnings", "error"]


def write_results_csv(path: str | Path, results: list[dict]) -> None:
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for item in results:
            row = {**item, "warnings": " | ".join(item.get("warnings", []))}
            writer.writerow(row)


def compare_files(reference: FocusReference, paths: Sequence[str | Path], center=None, frame=None) -> list[dict]:
    rows = []
    for path in paths:
        try:
            row = reference.compare(load_image(path, frame), center)
            row.update(image=str(path), status="ok")
        except (OSError, ValueError, KeyError) as exc:
            row = {"image": str(path), "status": "error", "error": str(exc)}
        rows.append(row)
    return rows


def scan_focus(reference: FocusReference, manifest: str | Path) -> dict:
    """Find the stage position whose image most resembles the reference.

    CSV columns: stage_z_nm,image_path,center_x,center_y (centers optional).
    Stage coordinates are supplied by the user, NOT inferred bead heights.
    Repeated positions are averaged. An interior peak can be interpolated.
    """
    manifest = Path(manifest)
    rows, errors = [], []
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"stage_z_nm", "image_path"}.issubset(reader.fieldnames or []):
            raise FocusError("扫描 CSV 必须包含 stage_z_nm、image_path 两列。")
        for index, item in enumerate(reader, 2):
            try:
                z = float(item["stage_z_nm"])
                if not np.isfinite(z):
                    raise FocusError("Z 位置必须是有限数值。")
                path = Path(item["image_path"])
                if not path.is_absolute():
                    path = manifest.parent / path
                x, y = item.get("center_x", ""), item.get("center_y", "")
                if bool(x) != bool(y):
                    raise FocusError("center_x 与 center_y 必须同时填写。")
                center = (float(x), float(y)) if x else None
                frame_text = item.get("frame", "")
                frame = int(frame_text) if frame_text else None
                result = reference.compare(load_image(path, frame), center)
                if not result["quality_ok"]:
                    raise FocusError("不能用于选焦：" + " | ".join(result["warnings"][1:]))
                rows.append({"stage_z_nm": z, "image": str(path),
                             **{key: result[key] for key in ("ncc", "radial_correlation", "shift_dx_px", "shift_dy_px")}})
            except (OSError, ValueError, KeyError) as exc:
                errors.append({"line": index, "error": str(exc)})
    if errors:
        raise FocusError("扫描存在无效行，请修正后再选焦：" + json.dumps(errors, ensure_ascii=False))
    positions = sorted({r["stage_z_nm"] for r in rows})
    if len(positions) < 3:
        raise FocusError("扫描至少需要三个不同 Z 位置，并覆盖相关峰两侧。")
    means = np.array([np.mean([r["radial_correlation"] for r in rows if r["stage_z_nm"] == z]) for z in positions])
    peak = int(np.argmax(means))
    warnings = ["返回的是输入扫描坐标中最匹配图样的位置，不是相对于另一颗珠的物理 Z 偏移。"]
    tied = bool(np.sum(np.isclose(means, means[peak], atol=1e-7, rtol=0)) > 1)
    edge = peak in (0, len(positions)-1)
    interpolated = None
    if edge:
        warnings.append("峰在扫描端点，请扩大 Z 范围，尚不能确认已找到焦位。")
    if tied:
        warnings.append("存在多个同高相关峰，不能给出唯一焦位。")
    if not edge and not tied:
        zz = np.array(positions[peak-1:peak+2]) - positions[peak]
        coefficients = np.polyfit(zz, means[peak-1:peak+2], 2)
        if coefficients[0] < -1e-15:
            vertex = -coefficients[1] / (2 * coefficients[0])
            if zz[0] <= vertex <= zz[-1]:
                interpolated = float(positions[peak] + vertex)
    return {
        "reference_id": reference.reference_id,
        "best_sampled_stage_z_nm": positions[peak],
        "interpolated_stage_z_nm": interpolated,
        "best_radial_correlation": float(means[peak]),
        "peak_at_edge": edge,
        "ambiguous_peak": tied,
        "warnings": warnings,
        "curve": [{"stage_z_nm": z, "radial_correlation": float(score)} for z, score in zip(positions, means)],
        "frames": rows,
    }


def write_scan_svg(path: str | Path, result: dict) -> None:
    """Write a small standalone correlation-versus-stage-position plot."""
    points = result["curve"]
    z = np.array([p["stage_z_nm"] for p in points])
    s = np.array([p["radial_correlation"] for p in points])
    xmin, xmax = float(z.min()), float(z.max())
    ymin, ymax = max(-1, float(s.min())-.05), min(1, float(s.max())+.05)
    ymax = max(ymax, ymin+.01)
    sx = lambda v: 80 + 570*(v-xmin)/(xmax-xmin)
    sy = lambda v: 330 - 270*(v-ymin)/(ymax-ymin)
    coords = " ".join(f"{sx(a):.2f},{sy(b):.2f}" for a,b in zip(z,s))
    circles = "".join(f'<circle cx="{sx(a):.2f}" cy="{sy(b):.2f}" r="4" fill="#2463eb"/>' for a,b in zip(z,s))
    text = f'''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="420" viewBox="0 0 720 420">
<rect width="720" height="420" fill="white"/><g font-family="Arial" fill="#1e293b">
<text x="80" y="30" font-size="18">Radial correlation vs. supplied stage Z</text>
<path d="M80 60V330H650" stroke="#334155" fill="none"/>
<polyline points="{coords}" fill="none" stroke="#2463eb" stroke-width="2"/>{circles}
<text x="80" y="355">{xmin:g}</text><text x="620" y="355">{xmax:g}</text>
<text x="15" y="65">{ymax:.3f}</text><text x="15" y="330">{ymin:.3f}</text>
<text x="300" y="383">Stage position (nm)</text>
<text x="90" y="407" font-size="12">Cross-bead matching does not establish equal physical bead Z.</text></g></svg>'''
    Path(path).write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="衍射图样参考匹配：输出相关值，不从单张图猜测 Z。")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="选一张参考图并保存参考文件")
    init.add_argument("image")
    init.add_argument("--output", required=True)
    init.add_argument("--center", nargs=2, type=float, metavar=("X", "Y"))
    init.add_argument("--roi-size", type=int, default=129)
    init.add_argument("--max-shift", type=int, default=8, help="选区内允许的残余 XY 偏移，单位 px")
    init.add_argument("--inner-radius", type=float, default=0)
    init.add_argument("--threshold", type=float)
    init.add_argument("--saturation-level", type=float)
    init.add_argument("--frame", type=int)
    compare = sub.add_parser("compare", help="比较一张或多张图片")
    compare.add_argument("reference")
    compare.add_argument("images", nargs="+")
    compare.add_argument("--center", nargs=2, type=float, metavar=("X", "Y"))
    compare.add_argument("--frame", type=int)
    compare.add_argument("--csv")
    scan = sub.add_parser("scan", help="用带已知位置的 Z 扫描寻找相关峰")
    scan.add_argument("reference")
    scan.add_argument("manifest")
    scan.add_argument("--output", required=True)
    scan.add_argument("--plot")
    args = parser.parse_args()
    try:
        if args.command == "init":
            config = Config(roi_size=args.roi_size, max_shift_px=args.max_shift, inner_radius_px=args.inner_radius,
                            threshold=args.threshold, saturation_level=args.saturation_level)
            ref = FocusReference.from_image(load_image(args.image, args.frame), args.center, config, str(args.image))
            ref.save(args.output)
            print(json.dumps({"reference_id": ref.reference_id, "saved": args.output}, ensure_ascii=False))
        elif args.command == "compare":
            rows = compare_files(FocusReference.load(args.reference), args.images, args.center, args.frame)
            if args.csv:
                write_results_csv(args.csv, rows)
            compact = [{k:v for k,v in row.items() if k not in ("radius_px", "reference_profile", "current_profile")} for row in rows]
            print(json.dumps(compact, ensure_ascii=False, indent=2, allow_nan=False))
            if any(r["status"] == "error" for r in rows):
                raise SystemExit(2)
        else:
            result = scan_focus(FocusReference.load(args.reference), args.manifest)
            Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            if args.plot:
                write_scan_svg(args.plot, result)
            print(json.dumps({k:v for k,v in result.items() if k not in ("frames", "curve")}, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"错误：{exc}\n")


if __name__ == "__main__":
    main()
