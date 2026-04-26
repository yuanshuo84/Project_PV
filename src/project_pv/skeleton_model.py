"""Named skeleton joints and anatomy rules for articulated figures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Iterable, Mapping

Point = tuple[float, float]
Segment = tuple[Point, Point]

BASIC_JOINT_RULE = (
    "A joint is a named anatomical point used as a bar endpoint; bars must "
    "connect only defined joints, and rendered joints come only from those "
    "defined endpoints."
)


@dataclass(frozen=True)
class Joint:
    """A named anatomical point in the neutral skeleton pose."""

    name: str
    x: float
    y: float
    radius: float = 5.0

    @property
    def point(self) -> Point:
        return (self.x, self.y)


@dataclass(frozen=True)
class Bar:
    """A rigid skeleton bar between two defined joints."""

    name: str
    start_joint: str
    end_joint: str
    layer: int = 0


@dataclass(frozen=True)
class JointPosition:
    """A keyframe parameter that sets one joint to an absolute position."""

    joint_name: str
    x: float
    y: float

    @property
    def point(self) -> Point:
        return (self.x, self.y)


@dataclass(frozen=True)
class BarImage:
    """A keyframe image attachment for one skeleton bar."""

    bar_name: str
    image_path: str
    source_x: int = 0
    source_y: int = 0
    source_width: int = 0
    source_height: int = 0
    anchor_start_x: int = 0
    anchor_start_y: int = 0
    anchor_end_x: int = 0
    anchor_end_y: int = 0

    @property
    def source_box(self) -> tuple[int, int, int, int] | None:
        if self.source_width <= 0 or self.source_height <= 0:
            return None
        return (
            self.source_x,
            self.source_y,
            self.source_x + self.source_width,
            self.source_y + self.source_height,
        )


@dataclass(frozen=True)
class SkeletonKeyFrame:
    """A frame-local set of joint positions and local coordinate transform."""

    frame: int
    joint_positions: tuple[JointPosition, ...]
    origin_x: float = 180.0
    origin_y: float = 180.0
    scale: float = 1.0
    rotation_degrees: float = 0.0
    bar_images: tuple[BarImage, ...] = ()


@dataclass(frozen=True)
class SkeletonDefinition:
    """Importable neutral skeleton definition."""

    joints: tuple[Joint, ...]
    bars: tuple[Bar, ...]
    rigid_bars: tuple[Bar, ...] = ()

    def rigid_hierarchy(self) -> tuple[Bar, ...]:
        return self.rigid_bars or self.bars


JOINTS: tuple[Joint, ...] = (
    Joint("head", 0, -76),
    Joint("neck", 0, -56),
    Joint("left_shoulder", -28, -42),
    Joint("right_shoulder", 28, -42),
    Joint("left_elbow", -50, -8),
    Joint("right_elbow", 50, -8),
    Joint("left_wrist", -64, 26),
    Joint("right_wrist", 64, 26),
    Joint("pelvis", 0, 28),
    Joint("left_hip", -20, 34),
    Joint("right_hip", 20, 34),
    Joint("left_knee", -24, 76),
    Joint("right_knee", 24, 76),
    Joint("left_ankle", -28, 116),
    Joint("right_ankle", 28, 116),
)

BARS: tuple[Bar, ...] = (
    Bar("head_to_neck", "head", "neck"),
    Bar("left_collar", "left_shoulder", "neck"),
    Bar("right_collar", "neck", "right_shoulder"),
    Bar("left_upper_arm", "left_shoulder", "left_elbow"),
    Bar("left_forearm", "left_elbow", "left_wrist"),
    Bar("right_upper_arm", "right_shoulder", "right_elbow"),
    Bar("right_forearm", "right_elbow", "right_wrist"),
    Bar("spine", "neck", "pelvis"),
    Bar("left_pelvis", "pelvis", "left_hip"),
    Bar("right_pelvis", "pelvis", "right_hip"),
    Bar("left_thigh", "left_hip", "left_knee"),
    Bar("left_shin", "left_knee", "left_ankle"),
    Bar("right_thigh", "right_hip", "right_knee"),
    Bar("right_shin", "right_knee", "right_ankle"),
)

RIGID_BAR_HIERARCHY: tuple[Bar, ...] = (
    Bar("head_to_neck", "neck", "head"),
    Bar("left_collar", "neck", "left_shoulder"),
    Bar("right_collar", "neck", "right_shoulder"),
    Bar("spine", "neck", "pelvis"),
    Bar("left_upper_arm", "left_shoulder", "left_elbow"),
    Bar("left_forearm", "left_elbow", "left_wrist"),
    Bar("right_upper_arm", "right_shoulder", "right_elbow"),
    Bar("right_forearm", "right_elbow", "right_wrist"),
    Bar("left_pelvis", "pelvis", "left_hip"),
    Bar("right_pelvis", "pelvis", "right_hip"),
    Bar("left_thigh", "left_hip", "left_knee"),
    Bar("left_shin", "left_knee", "left_ankle"),
    Bar("right_thigh", "right_hip", "right_knee"),
    Bar("right_shin", "right_knee", "right_ankle"),
)

DEFAULT_SKELETON = SkeletonDefinition(JOINTS, BARS, RIGID_BAR_HIERARCHY)


def skeleton_to_record(skeleton: SkeletonDefinition) -> dict[str, Any]:
    """Return a JSON-serializable skeleton definition."""

    validate_skeleton_model(skeleton.joints, skeleton.bars)
    _validate_bar_references(skeleton.joints, skeleton.rigid_hierarchy())
    return {
        "version": 1,
        "joints": [
            {"name": joint.name, "x": joint.x, "y": joint.y, "radius": joint.radius}
            for joint in skeleton.joints
        ],
        "bars": [
            {"name": bar.name, "start_joint": bar.start_joint, "end_joint": bar.end_joint, "layer": bar.layer}
            for bar in skeleton.bars
        ],
        "rigid_bars": [
            {"name": bar.name, "start_joint": bar.start_joint, "end_joint": bar.end_joint, "layer": bar.layer}
            for bar in skeleton.rigid_hierarchy()
        ],
    }


def skeleton_from_record(record: Mapping[str, Any]) -> SkeletonDefinition:
    """Build a skeleton definition from JSON-compatible data."""

    joints = tuple(
        Joint(
            str(item["name"]),
            float(item["x"]),
            float(item["y"]),
            float(item.get("radius", 5.0)),
        )
        for item in record.get("joints", ())
    )
    bars = tuple(
        Bar(str(item["name"]), str(item["start_joint"]), str(item["end_joint"]), int(item.get("layer", 0)))
        for item in record.get("bars", ())
    )
    rigid_bars = tuple(
        Bar(str(item["name"]), str(item["start_joint"]), str(item["end_joint"]), int(item.get("layer", 0)))
        for item in record.get("rigid_bars", ())
    )
    skeleton = SkeletonDefinition(joints, bars, rigid_bars)
    validate_skeleton_model(skeleton.joints, skeleton.bars)
    _validate_bar_references(skeleton.joints, skeleton.rigid_hierarchy())
    return skeleton


def save_skeleton(skeleton: SkeletonDefinition, output_path: str | Path) -> Path:
    """Save a skeleton definition JSON file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(skeleton_to_record(skeleton), indent=2), encoding="utf-8")
    return path


def load_skeleton(input_path: str | Path) -> SkeletonDefinition:
    """Load a skeleton definition JSON file."""

    path = Path(input_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("skeleton record must be a JSON object")
    return skeleton_from_record(record)


def joint_map(joints: Iterable[Joint] = JOINTS) -> dict[str, Joint]:
    """Return joints by name, rejecting duplicate definitions."""

    mapped: dict[str, Joint] = {}
    for joint in joints:
        if joint.name in mapped:
            raise ValueError(f"duplicate joint definition: {joint.name}")
        mapped[joint.name] = joint
    return mapped


def validate_skeleton_model(joints: Iterable[Joint] = JOINTS, bars: Iterable[Bar] = BARS) -> None:
    """Apply the basic joint rule to a skeleton definition."""

    joints = tuple(joints)
    bars = tuple(bars)
    mapped = joint_map(joints)
    used_joint_names: set[str] = set()
    for bar in bars:
        for joint_name in (bar.start_joint, bar.end_joint):
            if joint_name not in mapped:
                raise ValueError(f"bar {bar.name} references undefined joint {joint_name}")
            used_joint_names.add(joint_name)

    unused_joint_names = set(mapped) - used_joint_names
    if unused_joint_names:
        names = ", ".join(sorted(unused_joint_names))
        raise ValueError(f"defined joints must be used as bar endpoints: {names}")


def _validate_bar_references(joints: Iterable[Joint], bars: Iterable[Bar]) -> None:
    mapped = joint_map(joints)
    used_names: set[str] = set()
    for bar in bars:
        if bar.name in used_names:
            raise ValueError(f"duplicate bar definition: {bar.name}")
        used_names.add(bar.name)
        for joint_name in (bar.start_joint, bar.end_joint):
            if joint_name not in mapped:
                raise ValueError(f"bar {bar.name} references undefined joint {joint_name}")


def neutral_joint_positions(joints: Iterable[Joint] = JOINTS) -> dict[str, Point]:
    """Return the default absolute position of each defined joint."""

    return {name: joint.point for name, joint in joint_map(joints).items()}


def bar_lengths(joint_positions: Mapping[str, Point], bars: Iterable[Bar] = BARS) -> dict[str, float]:
    """Return bar lengths for a pose."""

    lengths = {}
    for bar in bars:
        start = joint_positions[bar.start_joint]
        end = joint_positions[bar.end_joint]
        lengths[bar.name] = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
    return lengths


def neutral_bar_lengths(joints: Iterable[Joint] = JOINTS, bars: Iterable[Bar] = BARS) -> dict[str, float]:
    """Return the rigid neutral length of each bar."""

    return bar_lengths(neutral_joint_positions(joints), bars)


def enforce_rigid_bar_lengths(
    joint_positions: Mapping[str, Point],
    bars: Iterable[Bar] = RIGID_BAR_HIERARCHY,
    target_lengths: Mapping[str, float] | None = None,
    joints: Iterable[Joint] = JOINTS,
) -> dict[str, Point]:
    """Project a pose so each child joint keeps its neutral bar length."""

    pose = dict(joint_positions)
    bars = tuple(bars)
    lengths = dict(target_lengths or neutral_bar_lengths(joints, bars))
    neutral_pose = neutral_joint_positions(joints)
    for bar in bars:
        parent = pose[bar.start_joint]
        child = pose[bar.end_joint]
        target_length = lengths[bar.name]
        dx = child[0] - parent[0]
        dy = child[1] - parent[1]
        current_length = (dx**2 + dy**2) ** 0.5
        if current_length == 0:
            neutral_parent = neutral_pose[bar.start_joint]
            neutral_child = neutral_pose[bar.end_joint]
            dx = neutral_child[0] - neutral_parent[0]
            dy = neutral_child[1] - neutral_parent[1]
            current_length = (dx**2 + dy**2) ** 0.5 or 1.0
        scale = target_length / current_length
        pose[bar.end_joint] = (parent[0] + dx * scale, parent[1] + dy * scale)
    return pose


def validate_joint_positions(joint_positions: Mapping[str, Point], joints: Iterable[Joint] = JOINTS) -> None:
    """Validate a set of joint position parameters against defined joints."""

    defined_names = set(joint_map(joints))
    for joint_name, point in joint_positions.items():
        if joint_name not in defined_names:
            raise ValueError(f"joint position references undefined joint {joint_name}")
        if len(point) != 2:
            raise ValueError(f"joint position for {joint_name} must have x and y values")


def validate_keyframe_properties(
    bar_images: Iterable[BarImage] = (),
    bars: Iterable[Bar] = BARS,
) -> None:
    """Validate non-interpolated keyframe display properties."""

    defined_bars = {bar.name for bar in bars}
    used_bars: set[str] = set()
    for bar_image in bar_images:
        if bar_image.bar_name not in defined_bars:
            raise ValueError(f"bar image references undefined bar {bar_image.bar_name}")
        if bar_image.bar_name in used_bars:
            raise ValueError(f"duplicate bar image for {bar_image.bar_name}")
        used_bars.add(bar_image.bar_name)
        if not bar_image.image_path:
            raise ValueError(f"bar image for {bar_image.bar_name} requires an image path")
        if bar_image.source_width < 0 or bar_image.source_height < 0:
            raise ValueError(f"bar image source size for {bar_image.bar_name} cannot be negative")


def keyframe_from_positions(
    frame: int,
    joint_positions: Mapping[str, Point],
    *,
    origin_x: float = 180.0,
    origin_y: float = 180.0,
    scale: float = 1.0,
    rotation_degrees: float = 0.0,
    bar_images: Mapping[str, str | Mapping[str, object]] | Iterable[BarImage] = (),
    joints: Iterable[Joint] = JOINTS,
    bars: Iterable[Bar] = BARS,
) -> SkeletonKeyFrame:
    """Create a stable keyframe from joint position parameters."""

    if frame < 0:
        raise ValueError("keyframe frame must be zero or greater")
    if scale <= 0:
        raise ValueError("keyframe scale must be greater than zero")
    validate_joint_positions(joint_positions, joints)
    if isinstance(bar_images, Mapping):
        attachments = []
        for bar_name, value in sorted(bar_images.items()):
            if isinstance(value, Mapping):
                image_path = str(value.get("image_path", ""))
                if not image_path:
                    continue
                attachments.append(
                    BarImage(
                        bar_name=bar_name,
                        image_path=image_path,
                        source_x=int(value.get("source_x", 0)),
                        source_y=int(value.get("source_y", 0)),
                        source_width=int(value.get("source_width", 0)),
                        source_height=int(value.get("source_height", 0)),
                        anchor_start_x=int(value.get("anchor_start_x", 0)),
                        anchor_start_y=int(value.get("anchor_start_y", 0)),
                        anchor_end_x=int(value.get("anchor_end_x", 0)),
                        anchor_end_y=int(value.get("anchor_end_y", 0)),
                    )
                )
            elif value:
                attachments.append(BarImage(bar_name, str(value)))
        image_attachments = tuple(attachments)
    else:
        image_attachments = tuple(sorted(bar_images, key=lambda bar_image: bar_image.bar_name))
    validate_keyframe_properties(image_attachments, bars)
    positions = tuple(
        JointPosition(joint_name, float(point[0]), float(point[1]))
        for joint_name, point in sorted(joint_positions.items())
    )
    return SkeletonKeyFrame(
        frame=frame,
        joint_positions=positions,
        origin_x=float(origin_x),
        origin_y=float(origin_y),
        scale=float(scale),
        rotation_degrees=float(rotation_degrees),
        bar_images=image_attachments,
    )


def keyframe_position_map(keyframe: SkeletonKeyFrame) -> dict[str, Point]:
    """Return a keyframe's joint positions keyed by joint name."""

    return {position.joint_name: position.point for position in keyframe.joint_positions}


def keyframe_bar_image_map(keyframe: SkeletonKeyFrame) -> dict[str, BarImage]:
    """Return a keyframe's bar image attachments keyed by bar name."""

    return {bar_image.bar_name: bar_image for bar_image in keyframe.bar_images}


def interpolate_keyframes(
    keyframes: Iterable[SkeletonKeyFrame],
    frame_index: int,
    skeleton: SkeletonDefinition = DEFAULT_SKELETON,
) -> dict[str, Point]:
    """Interpolate all joint positions for a frame from surrounding keyframes."""

    return interpolate_keyframe_state(keyframes, frame_index, skeleton)[0]


def interpolate_keyframe_state(
    keyframes: Iterable[SkeletonKeyFrame],
    frame_index: int,
    skeleton: SkeletonDefinition = DEFAULT_SKELETON,
) -> tuple[dict[str, Point], Point, float, float]:
    """Interpolate joint positions and local coordinate transform for a frame."""

    ordered = sorted(keyframes, key=lambda keyframe: keyframe.frame)
    if not ordered:
        return neutral_joint_positions(skeleton.joints), (180.0, 180.0), 1.0, 0.0

    for keyframe in ordered:
        if keyframe.frame < 0:
            raise ValueError("keyframe frame must be zero or greater")
        if keyframe.scale <= 0:
            raise ValueError("keyframe scale must be greater than zero")
        validate_joint_positions(keyframe_position_map(keyframe), skeleton.joints)
        validate_keyframe_properties(keyframe.bar_images, skeleton.bars)

    if frame_index <= ordered[0].frame:
        return _state_from_keyframe(ordered[0], skeleton)
    if frame_index >= ordered[-1].frame:
        return _state_from_keyframe(ordered[-1], skeleton)

    previous = ordered[0]
    following = ordered[-1]
    for index, keyframe in enumerate(ordered[:-1]):
        next_keyframe = ordered[index + 1]
        if keyframe.frame <= frame_index <= next_keyframe.frame:
            previous = keyframe
            following = next_keyframe
            break

    frame_span = max(1, following.frame - previous.frame)
    progress = (frame_index - previous.frame) / frame_span
    start_pose = _pose_from_keyframe(previous, skeleton)
    end_pose = _pose_from_keyframe(following, skeleton)
    pose = {
        joint_name: (
            start_pose[joint_name][0] + (end_pose[joint_name][0] - start_pose[joint_name][0]) * progress,
            start_pose[joint_name][1] + (end_pose[joint_name][1] - start_pose[joint_name][1]) * progress,
        )
        for joint_name in start_pose
    }
    center = (
        previous.origin_x + (following.origin_x - previous.origin_x) * progress,
        previous.origin_y + (following.origin_y - previous.origin_y) * progress,
    )
    scale = previous.scale + (following.scale - previous.scale) * progress
    rotation = previous.rotation_degrees + (following.rotation_degrees - previous.rotation_degrees) * progress
    return enforce_rigid_bar_lengths(pose, skeleton.rigid_hierarchy(), joints=skeleton.joints), center, scale, rotation


def positioned_joints(joint_positions: Mapping[str, Point], joints: Iterable[Joint] = JOINTS) -> tuple[Joint, ...]:
    """Return joints with selected positions overridden by parameters."""

    validate_joint_positions(joint_positions, joints)
    positioned = []
    for joint in joints:
        x, y = joint_positions.get(joint.name, joint.point)
        positioned.append(Joint(joint.name, x, y, joint.radius))
    return tuple(positioned)


def _pose_from_keyframe(keyframe: SkeletonKeyFrame, skeleton: SkeletonDefinition = DEFAULT_SKELETON) -> dict[str, Point]:
    pose = neutral_joint_positions(skeleton.joints)
    pose.update(keyframe_position_map(keyframe))
    return enforce_rigid_bar_lengths(pose, skeleton.rigid_hierarchy(), joints=skeleton.joints)


def _state_from_keyframe(
    keyframe: SkeletonKeyFrame,
    skeleton: SkeletonDefinition = DEFAULT_SKELETON,
) -> tuple[dict[str, Point], Point, float, float]:
    return _pose_from_keyframe(keyframe, skeleton), (keyframe.origin_x, keyframe.origin_y), keyframe.scale, keyframe.rotation_degrees


def keyframe_properties_at(
    keyframes: Iterable[SkeletonKeyFrame],
    frame_index: int,
    bars: Iterable[Bar] = BARS,
) -> dict[str, BarImage]:
    """Return non-interpolated display properties for the nearest previous keyframe."""

    ordered = sorted(keyframes, key=lambda keyframe: keyframe.frame)
    if not ordered:
        return {}

    selected = ordered[0]
    for keyframe in ordered:
        if keyframe.frame <= frame_index:
            selected = keyframe
        else:
            break
    validate_keyframe_properties(selected.bar_images, bars)
    return keyframe_bar_image_map(selected)


def skeleton_segments(joints: Iterable[Joint] = JOINTS, bars: Iterable[Bar] = BARS) -> list[Segment]:
    """Return normalized bar segments after validating the skeleton model."""

    joints = tuple(joints)
    bars = tuple(bars)
    validate_skeleton_model(joints, bars)
    mapped = joint_map(joints)
    return [(mapped[bar.start_joint].point, mapped[bar.end_joint].point) for bar in bars]


def joint_points(joints: Iterable[Joint] = JOINTS, bars: Iterable[Bar] = BARS) -> list[Point]:
    """Return defined joint points in model order after applying the joint rule."""

    joints = tuple(joints)
    bars = tuple(bars)
    validate_skeleton_model(joints, bars)
    return [joint.point for joint in joints]
