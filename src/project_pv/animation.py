"""Vector-to-pixel skeleton animation rendering helpers."""

from __future__ import annotations

import logging
import math
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw

from project_pv.skeleton_model import (
    BarImage,
    DEFAULT_SKELETON,
    SkeletonKeyFrame,
    SkeletonDefinition,
    Point,
    Segment,
    interpolate_keyframe_state,
    joint_points,
    keyframe_properties_at,
    keyframe_from_positions,
    positioned_joints,
    skeleton_segments,
    skeleton_from_record,
    skeleton_to_record,
)

VECTOR_FIELD_WIDTH = 360
VECTOR_FIELD_HEIGHT = 360
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MotionSpec:
    """Parameters that describe a vector animation before raster export."""

    width: int = 50
    height: int = 50
    frames: int = 32
    duration_ms: int = 60
    fill: str = "#1b8f87"
    outline: str = "#17202a"
    background: str = "#f7f4ea"
    show_joints: bool = True
    hidden_bar_images: tuple[str, ...] = ()
    keyframes: tuple[SkeletonKeyFrame, ...] = field(default_factory=tuple)
    skeleton: SkeletonDefinition = DEFAULT_SKELETON

    def validate(self) -> None:
        if self.width < 32 or self.height < 32:
            raise ValueError("animation size must be at least 32x32 pixels")
        if self.frames < 2:
            raise ValueError("animation requires at least 2 frames")
        if self.duration_ms < 10:
            raise ValueError("frame duration must be at least 10 ms")
        defined_bars = {bar.name for bar in self.skeleton.bars}
        for bar_name in self.hidden_bar_images:
            if bar_name not in defined_bars:
                raise ValueError(f"hidden body part references undefined bar {bar_name}")
        for keyframe in self.keyframes:
            if keyframe.frame < 0:
                raise ValueError("keyframe frame must be zero or greater")
            if keyframe.frame >= self.frames:
                raise ValueError("keyframe frame must be inside the animation frame range")
            if keyframe.scale <= 0:
                raise ValueError("keyframe scale must be greater than zero")


@dataclass(frozen=True)
class _CachedBarImage:
    image: Image.Image
    source_start: Point
    source_end: Point

    @property
    def source_length(self) -> float:
        return math.hypot(
            self.source_end[0] - self.source_start[0],
            self.source_end[1] - self.source_start[1],
        ) or 1.0

    @property
    def source_angle(self) -> float:
        return math.degrees(
            math.atan2(
                self.source_end[1] - self.source_start[1],
                self.source_end[0] - self.source_start[0],
            )
        )


_BarImageCache = dict[tuple[object, ...], _CachedBarImage | None]


def transform_points(points: Sequence[Point], center: Point, scale: float, angle_degrees: float) -> list[Point]:
    """Scale, rotate, and translate vector points."""

    angle = math.radians(angle_degrees)
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    cx, cy = center
    transformed = []
    for x, y in points:
        sx = x * scale
        sy = y * scale
        transformed.append(
            (
                cx + sx * cos_angle - sy * sin_angle,
                cy + sx * sin_angle + sy * cos_angle,
            )
        )
    return transformed


def transform_segments(segments: Sequence[Segment], center: Point, scale: float, angle_degrees: float) -> list[Segment]:
    """Scale, rotate, and translate bar segments."""

    transformed: list[Segment] = []
    for start, end in segments:
        [pixel_start, pixel_end] = transform_points((start, end), center, scale, angle_degrees)
        transformed.append((pixel_start, pixel_end))
    return transformed


def frame_state(spec: MotionSpec, frame_index: int) -> tuple[Point, float, float]:
    """Return center, scale, and angle for a frame."""

    return interpolate_keyframe_state(spec.keyframes, frame_index)[1:]


def fit_vector_points_to_pixel_field(points: Sequence[Point], pixel_width: int, pixel_height: int) -> tuple[list[Point], float]:
    """Fit vector-field points inside a pixel output without cropping the field."""

    fit_scale = min(pixel_width / VECTOR_FIELD_WIDTH, pixel_height / VECTOR_FIELD_HEIGHT)
    field_width = VECTOR_FIELD_WIDTH * fit_scale
    field_height = VECTOR_FIELD_HEIGHT * fit_scale
    offset_x = (pixel_width - field_width) / 2
    offset_y = (pixel_height - field_height) / 2
    return [(offset_x + x * fit_scale, offset_y + y * fit_scale) for x, y in points], fit_scale


def _bar_image_cache_key(bar_image: BarImage) -> tuple[object, ...] | None:
    path = Path(bar_image.image_path)
    if not path.exists():
        LOGGER.warning("Skipping missing bar image %s", path)
        return None
    try:
        stat = path.stat()
    except OSError:
        LOGGER.warning("Skipping unreadable bar image %s", path)
        return None
    return (
        str(path.resolve()),
        stat.st_mtime_ns,
        stat.st_size,
        bar_image.source_x,
        bar_image.source_y,
        bar_image.source_width,
        bar_image.source_height,
        bar_image.anchor_start_x,
        bar_image.anchor_start_y,
        bar_image.anchor_end_x,
        bar_image.anchor_end_y,
    )


def _load_cached_bar_image(bar_image: BarImage, cache: _BarImageCache) -> _CachedBarImage | None:
    key = _bar_image_cache_key(bar_image)
    if key is None:
        return None
    if key in cache:
        return cache[key]

    path = Path(bar_image.image_path)
    with Image.open(path) as source:
        attachment = source.convert("RGBA")
        if bar_image.source_box is not None:
            left, upper, right, lower = bar_image.source_box
            left = min(max(0, left), attachment.width)
            upper = min(max(0, upper), attachment.height)
            right = min(max(left, right), attachment.width)
            lower = min(max(upper, lower), attachment.height)
            if right <= left or lower <= upper:
                LOGGER.warning("Skipping empty source area for bar image %s", path)
                cache[key] = None
                return None
            attachment = attachment.crop((left, upper, right, lower))
        else:
            left = 0
            upper = 0
        if attachment.width <= 0 or attachment.height <= 0:
            cache[key] = None
            return None

        source_start = (
            bar_image.anchor_start_x - left,
            bar_image.anchor_start_y - upper,
        )
        source_end = (
            bar_image.anchor_end_x - left,
            bar_image.anchor_end_y - upper,
        )
        if source_start == source_end:
            source_start = (0, attachment.height / 2)
            source_end = (attachment.width, attachment.height / 2)

    cached = _CachedBarImage(attachment, source_start, source_end)
    cache[key] = cached
    return cached


def _paste_bar_image(canvas: Image.Image, bar_image: BarImage, start: Point, end: Point, cache: _BarImageCache | None = None) -> None:
    """Paste an image by using the source anchor vector as the body-part local axis."""

    source_cache = cache if cache is not None else {}
    cached = _load_cached_bar_image(bar_image, source_cache)
    if cached is None:
        return

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy) or 1.0
    angle = math.degrees(math.atan2(dy, dx))
    scale = length / cached.source_length
    rotation = math.radians(angle - cached.source_angle)
    cos_angle = math.cos(rotation)
    sin_angle = math.sin(rotation)
    start_x, start_y = start
    anchor_x, anchor_y = cached.source_start
    inverse_scale = 1.0 / scale
    coefficients = (
        cos_angle * inverse_scale,
        sin_angle * inverse_scale,
        anchor_x - ((cos_angle * start_x + sin_angle * start_y) * inverse_scale),
        -sin_angle * inverse_scale,
        cos_angle * inverse_scale,
        anchor_y - ((-sin_angle * start_x + cos_angle * start_y) * inverse_scale),
    )
    mapped = cached.image.transform(
        canvas.size,
        Image.Transform.AFFINE,
        coefficients,
        resample=Image.Resampling.BICUBIC,
    )
    canvas.alpha_composite(mapped)


def render_frame(
    spec: MotionSpec,
    frame_index: int,
    *,
    transparent_background: bool = False,
    bar_image_cache: _BarImageCache | None = None,
) -> Image.Image:
    """Rasterize one animation frame."""

    spec.validate()
    LOGGER.debug(
        "Rendering skeleton frame %s/%s size=%sx%s",
        frame_index + 1,
        spec.frames,
        spec.width,
        spec.height,
    )
    if transparent_background:
        image = Image.new("RGBA", (spec.width, spec.height), (0, 0, 0, 0))
    else:
        image = Image.new("RGB", (spec.width, spec.height), spec.background)
    draw = ImageDraw.Draw(image)
    pose, center, scale, angle = interpolate_keyframe_state(spec.keyframes, frame_index, spec.skeleton)
    bar_images = keyframe_properties_at(spec.keyframes, frame_index, spec.skeleton.bars)
    joints = positioned_joints(pose, spec.skeleton.joints)
    vector_segments = transform_segments(skeleton_segments(joints, spec.skeleton.bars), center, scale, angle)
    vector_joints = transform_points(joint_points(joints, spec.skeleton.bars), center, scale, angle)
    flattened_points = [point for segment in vector_segments for point in segment]
    pixel_points, fit_scale = fit_vector_points_to_pixel_field(flattened_points, spec.width, spec.height)
    pixel_joints, _ = fit_vector_points_to_pixel_field(vector_joints, spec.width, spec.height)
    pixel_segments = list(zip(pixel_points[0::2], pixel_points[1::2]))
    pixel_joint_map = {joint.name: point for joint, point in zip(joints, pixel_joints)}

    bar_width = max(1, round(5 * scale * fit_scale))
    joint_radius = max(1, round(5 * scale * fit_scale))
    for start, end in pixel_segments:
        draw.line((start, end), fill=spec.outline, width=bar_width)
    if bar_images:
        for bar in sorted(spec.skeleton.bars, key=lambda item: item.layer):
            bar_image = bar_images.get(bar.name)
            if not bar_image or bar.name in spec.hidden_bar_images:
                continue
            pixel_start = pixel_joint_map[bar.start_joint]
            pixel_end = pixel_joint_map[bar.end_joint]
            if image.mode != "RGBA":
                image = image.convert("RGBA")
                draw = ImageDraw.Draw(image)
            _paste_bar_image(image, bar_image, pixel_start, pixel_end, bar_image_cache)

    if spec.show_joints:
        for _joint, (x, y) in zip(joints, pixel_joints):
            draw.ellipse(
                (x - joint_radius, y - joint_radius, x + joint_radius, y + joint_radius),
                fill=spec.fill,
                outline=spec.outline,
                width=max(1, round(1.5 * fit_scale)),
            )
    return image


def motion_spec_to_record(spec: MotionSpec) -> dict[str, Any]:
    """Return a JSON-serializable animation record."""

    spec.validate()
    return {
        "version": 1,
        "parameters": {
            "width": spec.width,
            "height": spec.height,
            "frames": spec.frames,
            "duration_ms": spec.duration_ms,
            "fill": spec.fill,
            "outline": spec.outline,
            "background": spec.background,
            "show_joints": spec.show_joints,
            "hidden_bar_images": list(spec.hidden_bar_images),
        },
        "skeleton": skeleton_to_record(spec.skeleton),
        "keyframes": [
            {
                "frame": keyframe.frame,
                "origin_x": keyframe.origin_x,
                "origin_y": keyframe.origin_y,
                "scale": keyframe.scale,
                "rotation_degrees": keyframe.rotation_degrees,
                "bar_images": [
                    {
                        "bar_name": bar_image.bar_name,
                        "image_path": bar_image.image_path,
                        "source_x": bar_image.source_x,
                        "source_y": bar_image.source_y,
                        "source_width": bar_image.source_width,
                        "source_height": bar_image.source_height,
                        "anchor_start_x": bar_image.anchor_start_x,
                        "anchor_start_y": bar_image.anchor_start_y,
                        "anchor_end_x": bar_image.anchor_end_x,
                        "anchor_end_y": bar_image.anchor_end_y,
                    }
                    for bar_image in keyframe.bar_images
                ],
                "joint_positions": [
                    {"joint_name": position.joint_name, "x": position.x, "y": position.y}
                    for position in keyframe.joint_positions
                ],
            }
            for keyframe in spec.keyframes
        ],
    }


def motion_spec_from_record(record: dict[str, Any]) -> MotionSpec:
    """Build a motion spec from a JSON animation record."""

    parameters = record.get("parameters", {})
    skeleton = skeleton_from_record(record["skeleton"]) if "skeleton" in record else DEFAULT_SKELETON
    fallback_start = (
        float(parameters.get("start_x", 180.0)),
        float(parameters.get("start_y", 180.0)),
    )
    fallback_end = (
        float(parameters.get("end_x", fallback_start[0])),
        float(parameters.get("end_y", fallback_start[1])),
    )
    fallback_scale_start = float(parameters.get("scale_start", 1.0))
    fallback_scale_end = float(parameters.get("scale_end", fallback_scale_start))
    fallback_rotation = float(parameters.get("rotation_degrees", 0.0))
    keyframes = []
    for keyframe_record in record.get("keyframes", ()):
        frame = int(keyframe_record["frame"])
        frame_count = max(2, int(parameters.get("frames", MotionSpec.frames)))
        progress = frame / (frame_count - 1)
        positions = {
            position["joint_name"]: (float(position["x"]), float(position["y"]))
            for position in keyframe_record.get("joint_positions", ())
        }
        keyframes.append(
            keyframe_from_positions(
                frame,
                positions,
                origin_x=float(keyframe_record.get("origin_x", fallback_start[0] + (fallback_end[0] - fallback_start[0]) * progress)),
                origin_y=float(keyframe_record.get("origin_y", fallback_start[1] + (fallback_end[1] - fallback_start[1]) * progress)),
                scale=float(keyframe_record.get("scale", fallback_scale_start + (fallback_scale_end - fallback_scale_start) * progress)),
                rotation_degrees=float(keyframe_record.get("rotation_degrees", fallback_rotation * progress)),
                bar_images={
                    bar_image["bar_name"]: {
                        "image_path": bar_image["image_path"],
                        "source_x": bar_image.get("source_x", 0),
                        "source_y": bar_image.get("source_y", 0),
                        "source_width": bar_image.get("source_width", 0),
                        "source_height": bar_image.get("source_height", 0),
                        "anchor_start_x": bar_image.get("anchor_start_x", 0),
                        "anchor_start_y": bar_image.get("anchor_start_y", 0),
                        "anchor_end_x": bar_image.get("anchor_end_x", 0),
                        "anchor_end_y": bar_image.get("anchor_end_y", 0),
                    }
                    for bar_image in keyframe_record.get("bar_images", ())
                },
                joints=skeleton.joints,
                bars=skeleton.bars,
            )
        )

    spec = MotionSpec(
        width=int(parameters.get("width", MotionSpec.width)),
        height=int(parameters.get("height", MotionSpec.height)),
        frames=int(parameters.get("frames", MotionSpec.frames)),
        duration_ms=int(parameters.get("duration_ms", MotionSpec.duration_ms)),
        fill=str(parameters.get("fill", MotionSpec.fill)),
        outline=str(parameters.get("outline", MotionSpec.outline)),
        background=str(parameters.get("background", MotionSpec.background)),
        show_joints=bool(parameters.get("show_joints", True)),
        hidden_bar_images=tuple(parameters.get("hidden_bar_images", ())),
        keyframes=tuple(keyframes),
        skeleton=skeleton,
    )
    spec.validate()
    return spec


def save_record(spec: MotionSpec, output_path: str | Path) -> Path:
    """Save all animation parameters and keyframes to a JSON file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(motion_spec_to_record(spec), indent=2), encoding="utf-8")
    return path


def load_record(input_path: str | Path) -> MotionSpec:
    """Load all animation parameters and keyframes from a JSON file."""

    path = Path(input_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("animation record must be a JSON object")
    return motion_spec_from_record(record)


def build_frames(spec: MotionSpec, *, transparent_background: bool = False) -> list[Image.Image]:
    """Render all frames for the animation."""

    spec.validate()
    LOGGER.info("Building %s skeleton GIF frames", spec.frames)
    bar_image_cache: _BarImageCache = {}
    return [
        render_frame(spec, index, transparent_background=transparent_background, bar_image_cache=bar_image_cache)
        for index in range(spec.frames)
    ]


def _prepare_transparent_gif_frames(frames: Sequence[Image.Image]) -> list[Image.Image]:
    """Convert RGBA frames to paletted GIF frames with a stable transparent index."""

    prepared = []
    for frame in frames:
        rgba_frame = frame.convert("RGBA")
        paletted = rgba_frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=255)
        palette = paletted.getpalette() or []
        paletted.putpalette((palette + [0] * 768)[:768])
        transparent_mask = rgba_frame.getchannel("A").point(lambda alpha: 255 if alpha == 0 else 0)
        paletted.paste(255, transparent_mask)
        paletted.info["transparency"] = 255
        paletted.info["disposal"] = 2
        prepared.append(paletted)
    return prepared


def save_gif(spec: MotionSpec, output_path: str | Path) -> Path:
    """Render and save the animation as a GIF."""

    frames = _prepare_transparent_gif_frames(build_frames(spec, transparent_background=True))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Saving GIF to %s", path)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=spec.duration_ms,
        loop=0,
        transparency=255,
        background=255,
        disposal=2,
        optimize=False,
    )
    LOGGER.info("Saved GIF to %s", path)
    return path
