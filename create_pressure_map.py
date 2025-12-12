#!/usr/bin/env python3
"""
Generate a synthetic plantar pressure map from a foot OBJ using
 a height / distance-to-ground heuristic.

Usage:
    python create_pressure_map.py /path/to/session_dir [--wii-log /path/to/bb_log.txt] [--show]

Example:
    python create_pressure_map.py 20251204_153203__urntype_vn_capture__administrative_ski__d3c09ace3d846e2a4680d1fb8f18d208cc2ac6a7746f3b6beb004fb0ca3b714b --wii-log bb1_log.txt --show
"""
import argparse
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np

TEMPLATE_FOLDER = Path(__file__).parent / "pressure_templates"

LEFT_TEMPLATE_PATH = TEMPLATE_FOLDER / "left_template.png"
RIGHT_TEMPLATE_PATH = TEMPLATE_FOLDER / "right_template.png"
LEFT_TEMPLATE_VALUES = TEMPLATE_FOLDER / "left_template_values.npy"
RIGHT_TEMPLATE_VALUES = TEMPLATE_FOLDER / "right_template_values.npy"
LEFT_TOE_MASK = TEMPLATE_FOLDER / "left_0.png"
RIGHT_TOE_MASK = TEMPLATE_FOLDER / "right_0.png"
LEFT_FRONT_MASKS = [
    TEMPLATE_FOLDER / "left_front_0.png",
    TEMPLATE_FOLDER / "left_front_1.png",
]
RIGHT_FRONT_MASKS = [
    TEMPLATE_FOLDER / "right_front_0.png",
    TEMPLATE_FOLDER / "right_front_1.png",
]
LEFT_BACK_MASKS = [
    TEMPLATE_FOLDER / "left_back.png",
]
RIGHT_BACK_MASKS = [
    TEMPLATE_FOLDER / "right_back.png",
]
LEFT_FOOT_MASK = TEMPLATE_FOLDER / "left_mask.png"
RIGHT_FOOT_MASK = TEMPLATE_FOLDER / "right_mask.png"
BLUR_KERNELS = [5, 13, 25]


def load_obj_vertices(path: Path) -> np.ndarray:
    """Minimal OBJ loader that returns only vertex positions."""
    vertices = []
    with open(path, "r") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            x, y, z = map(float, parts[1:4])
            vertices.append((x, y, z))
    if not vertices:
        raise ValueError(f"No vertices found in OBJ: {path}")
    return np.asarray(vertices, dtype=np.float32)


def show_image(window_name: str, image: np.ndarray) -> None:
    """Display an image with OpenCV, handling headless environments gracefully."""
    try:
        cv2.imshow(window_name, image)
        cv2.waitKey(1)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
    except cv2.error:
        print(f"OpenCV GUI not available; skipped displaying '{window_name}'.")


def display_frame(window_name: str, image: np.ndarray) -> bool:
    """Display a frame non-blocking; returns False if window closed/quit."""
    try:
        cv2.imshow(window_name, image)
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            return False
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            return False
        return True
    except cv2.error:
        print(f"OpenCV GUI not available; skipping visualization.")
        return False


def compute_pressure(vertices: np.ndarray) -> np.ndarray:
    """Compute a pressure heuristic from vertex heights."""
    z = vertices[:, 2]
    z_min = np.min(z)
    z_max = np.max(z)
    dist_to_ground = z - z_min
    if z_max == z_min:
        return np.ones_like(z)
    dist_norm = dist_to_ground / (z_max - z_min)
    pressure = 1.0 - dist_norm
    gamma = 1.5
    return np.power(pressure, gamma)


def select_ground_slice(vertices: np.ndarray, max_height_above_ground: float = 0.02) -> np.ndarray:
    """Return only vertices that lie within `max_height_above_ground` above the lowest point."""
    z = vertices[:, 2]
    z_min = np.min(z)
    mask = (z - z_min) <= max_height_above_ground
    if not np.any(mask):
        raise ValueError(
            "No vertices found within the requested ground slice height; "
            "increase max_height_above_ground."
        )
    return vertices[mask]


def rasterize_pressure(vertices: np.ndarray, pressure: np.ndarray, resolution: int = 512) -> np.ndarray:
    """Project vertices onto XY plane, keeping aspect ratio, and rasterize to a grid."""
    x = vertices[:, 0]
    y = vertices[:, 1]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range, 1e-6)

    x = (x - x_min) / max_range
    y = (y - y_min) / max_range

    x_fill = x_range / max_range
    y_fill = y_range / max_range
    if x_fill < 1.0:
        x += 0.5 * (1.0 - x_fill)
    if y_fill < 1.0:
        y += 0.5 * (1.0 - y_fill)

    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)

    bins = resolution
    H_sum, xedges, yedges = np.histogram2d(
        x, y, bins=bins, range=[[0, 1], [0, 1]], weights=pressure
    )
    H_count, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
    with np.errstate(invalid="ignore", divide="ignore"):
        H = H_sum / H_count
    H[H_count == 0] = np.nan
    return H.T

def compute_pressure_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Compute (y_min, y_max, x_min, x_max) bounding box of the finite-pressure mask."""
    indices = np.column_stack(np.where(mask))
    if indices.size == 0:
        raise ValueError("Pressure grid mask is empty; cannot compute bounding box.")
    y_min = int(indices[:, 0].min())
    y_max = int(indices[:, 0].max())
    x_min = int(indices[:, 1].min())
    x_max = int(indices[:, 1].max())
    return y_min, y_max, x_min, x_max


def transform_template_to_bbox(
    template_path: Path,
    target_bbox: tuple[int, int, int, int],
    canvas_shape: tuple[int, int],
) -> np.ndarray:
    """Scale the entire template image so it fills the target bounding box."""
    if not template_path.exists():
        raise FileNotFoundError(f"Template image not found: {template_path}")

    template = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
    if template is None:
        raise ValueError(f"Failed to read template image: {template_path}")

    if template.shape[2] == 4:
        template_rgb = template[:, :, :3]
        template_alpha = template[:, :, 3].astype(np.float32) / 255.0
    else:
        template_rgb = template
        template_alpha = np.ones(template.shape[:2], dtype=np.float32)

    target_y_min, target_y_max, target_x_min, target_x_max = target_bbox
    target_h = target_y_max - target_y_min + 1
    target_w = target_x_max - target_x_min + 1
    if target_h <= 0 or target_w <= 0:
        raise ValueError("Invalid target bounding box dimensions.")

    scaled_rgb = cv2.resize(template_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    scaled_alpha = cv2.resize(template_alpha, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    canvas_h, canvas_w = canvas_shape
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    offset_y = target_y_min
    offset_x = target_x_min
    y_start = max(0, offset_y)
    x_start = max(0, offset_x)
    y_end = min(canvas_h, offset_y + scaled_rgb.shape[0])
    x_end = min(canvas_w, offset_x + scaled_rgb.shape[1])
    if y_start >= y_end or x_start >= x_end:
        raise ValueError("Scaled template does not overlap with canvas; check bounding boxes.")

    src_y_start = y_start - offset_y
    src_x_start = x_start - offset_x
    src_y_end = src_y_start + (y_end - y_start)
    src_x_end = src_x_start + (x_end - x_start)

    roi = canvas[y_start:y_end, x_start:x_end]
    src_rgb = scaled_rgb[src_y_start:src_y_end, src_x_start:src_x_end]
    src_alpha = scaled_alpha[src_y_start:src_y_end, src_x_start:src_x_end][..., None]
    blended = (src_alpha * src_rgb + (1.0 - src_alpha) * roi).astype(np.uint8)
    canvas[y_start:y_end, x_start:x_end] = blended

    return canvas


def project_scalar_to_grid(
    scalar_data: np.ndarray,
    target_bbox: tuple[int, int, int, int],
    canvas_shape: tuple[int, int],
) -> np.ndarray:
    """Project a scalar template map onto the pressure grid canvas."""
    target_y_min, target_y_max, target_x_min, target_x_max = target_bbox
    target_h = target_y_max - target_y_min + 1
    target_w = target_x_max - target_x_min + 1
    if target_h <= 0 or target_w <= 0:
        raise ValueError("Invalid target bounding box dimensions.")

    resized = cv2.resize(
        scalar_data.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR
    )
    canvas = np.zeros(canvas_shape, dtype=np.float32)

    offset_y = target_y_min
    offset_x = target_x_min
    y_start = max(0, offset_y)
    x_start = max(0, offset_x)
    y_end = min(canvas_shape[0], offset_y + resized.shape[0])
    x_end = min(canvas_shape[1], offset_x + resized.shape[1])
    if y_start >= y_end or x_start >= x_end:
        return canvas

    src_y_start = y_start - offset_y
    src_x_start = x_start - offset_x
    src_y_end = src_y_start + (y_end - y_start)
    src_x_end = src_x_start + (x_end - x_start)

    canvas[y_start:y_end, x_start:x_end] = resized[src_y_start:src_y_end, src_x_start:src_x_end]
    return canvas


def load_scalar_with_mask(foot_prefix: str, mask_path: Path) -> np.ndarray | None:
    if foot_prefix == "left":
        scalar_path = LEFT_TEMPLATE_VALUES
    else:
        scalar_path = RIGHT_TEMPLATE_VALUES

    if not scalar_path.exists() or not mask_path.exists():
        return None

    try:
        scalar_values = np.load(scalar_path).astype(np.float32)
    except Exception:
        return None

    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        return None
    mask = np.clip(mask_img.astype(np.float32) / 255.0, 0.0, 1.0)
    mask_resized = cv2.resize(
        mask, (scalar_values.shape[1], scalar_values.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    mask_blurred = cv2.GaussianBlur(mask_resized, (11, 11), 0)
    return scalar_values * mask_blurred


def adjust_template_with_blobs(
    template_canvas: np.ndarray,
    bbox: tuple[int, int, int, int],
    foot_prefix: str,
    show_comparison: bool,
    blob_scales: list[float],
) -> np.ndarray:
    if foot_prefix == "left":
        mask_paths = [LEFT_TOE_MASK] + LEFT_FRONT_MASKS + LEFT_BACK_MASKS
    else:
        mask_paths = [RIGHT_TOE_MASK] + RIGHT_FRONT_MASKS + RIGHT_BACK_MASKS
    if len(blob_scales) < len(mask_paths):
        blob_scales = blob_scales + [blob_scales[-1]] * (len(mask_paths) - len(blob_scales))
    adjusted = template_canvas.astype(np.float32).copy()

    for mask_path, blob_scale in zip(mask_paths, blob_scales):
        toe_scalar = load_scalar_with_mask(foot_prefix, mask_path)
        if toe_scalar is None:
            continue
        toe_canvas = project_scalar_to_grid(toe_scalar, bbox, template_canvas.shape[:2])
        max_val = np.max(toe_canvas)
        if max_val <= 0:
            continue
        normalized_toe = np.clip(toe_canvas / max_val, 0.0, 1.0)
        toe_canvas_norm = normalized_toe ** 1.2

        time_variation = 1.0 + 0.3 * np.sin(time.time())
        foot_variation = 1.0 + 0.2 * (np.random.rand() - 0.5)
        strength = blob_scale * (time_variation * foot_variation)

        overlay_color = np.array([50.0, 0.0, 200.0], dtype=np.float32)
        adjusted += toe_canvas_norm[..., None] * overlay_color * (strength / 100.0)

    adjusted = np.clip(adjusted, 0, 255)

    blurred = adjusted.copy()
    for _ in range(2):
        for k in BLUR_KERNELS:
            blurred = cv2.GaussianBlur(blurred, (k, k), 0)

    mask_path = LEFT_FOOT_MASK if foot_prefix == "left" else RIGHT_FOOT_MASK
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_img is not None:
        mask = np.clip(mask_img.astype(np.float32) / 255.0, 0.0, 1.0)
        mask_canvas = project_scalar_to_grid(mask, bbox, template_canvas.shape[:2])
        mask_canvas = mask_canvas[..., None]
        blurred = blurred * mask_canvas + 255.0 * (1.0 - mask_canvas)

    adjusted_final = np.clip(blurred, 0, 255).astype(np.uint8)

    if show_comparison:
        comparison = np.hstack((template_canvas, adjusted_final))
        show_image(f"{foot_prefix.capitalize()} Template (Before vs After)", comparison)

    return adjusted_final


def _align_scales(values: List[float], target_len: int) -> List[float]:
    if target_len <= 0:
        return []
    if not values:
        return [0.0] * target_len
    result: List[float] = []
    idx = 0
    while len(result) < target_len:
        result.append(values[idx] if idx < len(values) else values[-1])
        idx += 1
    return result[:target_len]


def compute_blob_scales_from_record(record: dict[str, float], foot_prefix: str) -> List[float]:
    TL = record.get("TL", 0.0)
    TR = record.get("TR", 0.0)
    BL = record.get("BL", 0.0)
    BR = record.get("BR", 0.0)
    total = record.get("Total", TL + TR + BL + BR)
    if total <= 0:
        total = max(TL + TR + BL + BR, 1e-6)

    if foot_prefix == "left":
        toe = (TL * 0.85 + BL * 0.1 + TR * 0.05) / total * 100.0
        front_vals = [
            (TL * 0.6 + BL * 0.3 + TR * 0.05 + BR * 0.05) / total * 100.0,
            (TL * 0.7 + BL * 0.15 + TR * 0.05 + BR * 0.05) / total * 100.0,
        ]
        back_val = (TL * 0.1 + BL * 0.85 + BR * 0.05) / total * 100.0
        front_masks = LEFT_FRONT_MASKS
        back_masks = LEFT_BACK_MASKS
    else:
        toe = (TR * 0.8 + BR * 0.1 + TL * 0.05) / total * 100.0
        front_vals = [
            (TR * 0.6 + BR * 0.3 + TL * 0.05 + BL * 0.05) / total * 100.0,
            (TR * 0.7 + BR * 0.15 + TL * 0.05 + BL * 0.05) / total * 100.0,
        ]
        back_val = (TR * 0.1 + BR * 0.85 + TL * 0.05) / total * 100.0
        front_masks = RIGHT_FRONT_MASKS
        back_masks = RIGHT_BACK_MASKS

    front_scales = _align_scales(front_vals, len(front_masks))
    back_scales = _align_scales([back_val], len(back_masks))
    return [toe] + front_scales + back_scales


def process_foot(
    obj_path: Path,
    out_prefix: str,
    template_path: Path,
    show_comparison: bool = False,
    blob_scales: list[float] | None = None,
) -> np.ndarray:
    """Process a single foot OBJ and return the transformed template canvas."""
    if not obj_path.exists():
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")

    vertices = load_obj_vertices(obj_path)
    ground_vertices = select_ground_slice(vertices, max_height_above_ground=0.02)
    pressure = compute_pressure(ground_vertices)
    grid = rasterize_pressure(ground_vertices, pressure, resolution=512)
    mask = np.isfinite(grid)
    bbox = compute_pressure_bbox(mask)

    template_canvas = transform_template_to_bbox(template_path, bbox, grid.shape)
    if blob_scales is None:
        blob_scales = [50.0] * (
            1 + len(LEFT_FRONT_MASKS) + len(LEFT_BACK_MASKS)
            if out_prefix == "left"
            else 1 + len(RIGHT_FRONT_MASKS) + len(RIGHT_BACK_MASKS)
        )
    template_canvas = adjust_template_with_blobs(
        template_canvas,
        bbox,
        out_prefix,
        show_comparison=show_comparison,
        blob_scales=blob_scales,
    )
    return template_canvas


def render_combined_frame(
    left_obj: Path,
    right_obj: Path,
    record: dict[str, float],
    blob_scale: float,
) -> np.ndarray:
    def fade_template(template: np.ndarray, primary: float, secondary: float) -> np.ndarray:
        if primary < 1.0 and secondary < 1.0:
            return np.full_like(template, 255)
        if primary < 5.0 and secondary < 5.0:
            fade_factor = (max(primary, secondary) - 1.0) / 4.0
            fade_factor = float(np.clip(fade_factor, 0.0, 1.0))
            if fade_factor <= 0.0:
                return np.full_like(template, 255)
            white = np.full_like(template, 255, dtype=np.uint8)
            blended = (
                template.astype(np.float32) * fade_factor
                + white.astype(np.float32) * (1.0 - fade_factor)
            )
            return np.clip(blended, 0, 255).astype(np.uint8)
        return template

    scale_factor = blob_scale / 100.0
    left_scales_base = compute_blob_scales_from_record(record, "left")
    right_scales_base = compute_blob_scales_from_record(record, "right")
    left_scales = [value * scale_factor for value in left_scales_base]
    right_scales = [value * scale_factor for value in right_scales_base]

    left_template = process_foot(
        left_obj,
        "left",
        LEFT_TEMPLATE_PATH,
        show_comparison=False,
        blob_scales=left_scales,
    )
    right_template = process_foot(
        right_obj,
        "right",
        RIGHT_TEMPLATE_PATH,
        show_comparison=False,
        blob_scales=right_scales,
    )

    left_template = fade_template(left_template, record.get("TL", 0.0), record.get("BL", 0.0))
    right_template = fade_template(right_template, record.get("TR", 0.0), record.get("BR", 0.0))

    return np.hstack((left_template, right_template))


def run_wii_loop(
    left_obj: Path,
    right_obj: Path,
    records: List[dict[str, float]],
    blob_scale: float,
) -> None:
    if not records:
        print("No [BB] records found in input.", file=sys.stderr)
        return

    window_name = "Left/Right Templates"
    index = 0
    try:
        while True:
            record = records[index]
            combined = render_combined_frame(left_obj, right_obj, record, blob_scale)
            if not display_frame(window_name, combined):
                break
            index = (index + 1) % len(records)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


def save_visualization_loop(
    left_obj: Path,
    right_obj: Path,
    records: List[dict[str, float]],
    blob_scale: float,
    output_path: Path,
    fps: int = 60,
) -> Path:
    if not records:
        raise ValueError("No records available to save visualization loop.")

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    try:
        for record in records:
            frame = render_combined_frame(left_obj, right_obj, record, blob_scale)
            if writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer for {output_path}")
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()

    return output_path

BB_PATTERN = re.compile(
    r"t=\s*([0-9.]+)s\s+"
    r"TL=\s*([0-9.]+)\s*kg\s+TR=\s*([0-9.]+)\s*kg\s+"
    r"BL=\s*([0-9.]+)\s*kg\s+BR=\s*([0-9.]+)\s*kg\s+Total=\s*([0-9.]+)\s*kg"
)
DEFAULT_RECORD = {"TL": 31.0, "TR": 8.0, "BL": 30.0, "BR": 4.0, "Total": 73.0}

def parse_records(stream: Iterable[str]) -> List[dict[str, float]]:
    records: List[dict[str, float]] = []
    for line in stream:
        match = BB_PATTERN.search(line)
        if not match:
            continue
        t_val, tl, tr, bl, br, total = match.groups()
        tl_f, tr_f, bl_f, br_f, total_f = map(float, (tl, tr, bl, br, total))
        records.append(
            {
                "t": float(t_val),
                "TL": tl_f,
                "TR": tr_f,
                "BL": bl_f,
                "BR": br_f,
                "Total": total_f,
            }
        )
    return records


def load_input(args: list[str]) -> List[dict[str, float]]:
    if args:
        text = Path(args[0]).read_text(encoding="utf-8")
        lines = text.splitlines()
    else:
        lines = sys.stdin.read().splitlines()
    return parse_records(lines)


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic plantar pressure maps and visualize foot templates."
    )
    parser.add_argument(
        "session_dir",
        type=Path,
        help="Directory containing left.obj/right.obj and optional bb_log.txt",
    )
    parser.add_argument(
        "--wii-log",
        dest="wii_log",
        type=Path,
        default=None,
        help="Optional Wii balance board log (defaults to session_dir/bb_log.txt)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Enable interactive visualization window",
    )
    parser.add_argument(
        "--loop-video",
        dest="loop_video",
        type=Path,
        default=None,
        help="Path to save one-loop visualization video (default: session_dir/loop_visualization.mp4)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_cli_args()

    session_dir = args.session_dir
    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")

    left_obj = session_dir / "left.obj"
    right_obj = session_dir / "right.obj"

    log_path = args.wii_log if args.wii_log is not None else session_dir / "bb_log.txt"
    if log_path.exists():
        records = load_input([str(log_path)])
        if not records:
            print(f"No records found in {log_path}.", file=sys.stderr)
            sys.exit(1)

        if True:
            loop_video_path = args.loop_video if args.loop_video is not None else session_dir / "loop_visualization.mp4"
            saved_path = save_visualization_loop(
                left_obj,
                right_obj,
                records,
                blob_scale=200.0,
                output_path=loop_video_path,
            )
            print(f"Saved loop visualization to {saved_path}")

        if args.show:
            run_wii_loop(left_obj, right_obj, records, blob_scale=200.0)
        else:
            print("Visualization flag not set; skipping interactive display.")
    else:
        print(f"Wii log {log_path} not found; using default static visualization.")
        combined = render_combined_frame(left_obj, right_obj, DEFAULT_RECORD, blob_scale=100.0)
        if args.show:
            show_image("Left/Right Templates", combined)
        else:
            output_image = session_dir / "static_template.png"
            cv2.imwrite(str(output_image), combined)
            print(f"Saved static template image to {output_image}")


if __name__ == "__main__":
    main()
