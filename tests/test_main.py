import os

from PIL import Image, ImageDraw

from project_pv.animation import (
    MotionSpec,
    _paste_bar_image,
    build_frames,
    fit_vector_points_to_pixel_field,
    load_record,
    motion_spec_from_record,
    motion_spec_to_record,
    render_frame,
    save_record,
    save_gif,
)
from project_pv.main import configure_tcl_paths, main
from project_pv.skeleton_model import (
    BARS,
    Bar,
    BarImage,
    Joint,
    SkeletonDefinition,
    bar_lengths,
    enforce_rigid_bar_lengths,
    keyframe_properties_at,
    interpolate_keyframe_state,
    interpolate_keyframes,
    joint_points,
    keyframe_from_positions,
    load_skeleton,
    save_skeleton,
    skeleton_segments,
    validate_skeleton_model,
)


def test_configure_tcl_paths_sets_bundled_python_tcl_paths(monkeypatch, tmp_path):
    tcl_root = tmp_path / "tcl"
    tcl_library = tcl_root / "tcl8.6"
    tk_library = tcl_root / "tk8.6"
    tcl_library.mkdir(parents=True)
    tk_library.mkdir(parents=True)
    (tcl_library / "init.tcl").write_text("", encoding="utf-8")
    (tk_library / "tk.tcl").write_text("", encoding="utf-8")

    monkeypatch.delenv("TCL_LIBRARY", raising=False)
    monkeypatch.delenv("TK_LIBRARY", raising=False)
    monkeypatch.setattr("project_pv.main.sys.base_prefix", str(tmp_path))

    configure_tcl_paths()

    assert "TCL_LIBRARY" in os.environ
    assert os.environ["TCL_LIBRARY"] == str(tcl_library)
    assert "TK_LIBRARY" in os.environ
    assert os.environ["TK_LIBRARY"] == str(tk_library)


def test_main_opens_ui(monkeypatch):
    opened = False

    def fake_run_app():
        nonlocal opened
        opened = True

    monkeypatch.setattr("project_pv.ui.run_app", fake_run_app)

    main()

    assert opened


def test_build_frames_uses_requested_frame_count_and_size():
    spec = MotionSpec(width=64, height=48, frames=4, duration_ms=20)

    frames = build_frames(spec)

    assert len(frames) == 4
    assert frames[0].size == (64, 48)


def test_motion_spec_defaults_to_50_pixel_output():
    spec = MotionSpec()

    assert spec.width == 50
    assert spec.height == 50
    assert spec.keyframes == ()


def test_skeleton_definition_json_round_trips(tmp_path):
    skeleton = SkeletonDefinition(
        joints=(Joint("root", 0, 0), Joint("tip", 108, 0)),
        bars=(Bar("root_to_tip", "root", "tip", 3),),
        rigid_bars=(Bar("root_to_tip", "root", "tip", 3),),
    )
    path = tmp_path / "skeleton.json"

    saved = save_skeleton(skeleton, path)
    loaded = load_skeleton(saved)

    assert loaded == skeleton
    assert loaded.bars[0].layer == 3


def test_motion_spec_uses_custom_skeleton_neutral_lengths():
    skeleton = SkeletonDefinition(
        joints=(Joint("neck", 0, 0), Joint("right_shoulder", 108, 0)),
        bars=(Bar("right_collar", "neck", "right_shoulder"),),
        rigid_bars=(Bar("right_collar", "neck", "right_shoulder"),),
    )
    keyframes = (
        keyframe_from_positions(
            0,
            {"right_shoulder": (108, 0)},
            joints=skeleton.joints,
            bars=skeleton.bars,
        ),
    )

    pose, _center, _scale, _rotation = interpolate_keyframe_state(keyframes, 0, skeleton)

    assert pose["right_shoulder"] == (108.0, 0.0)


def test_render_frame_draws_higher_body_part_layer_on_top(tmp_path):
    lower_path = tmp_path / "lower.png"
    upper_path = tmp_path / "upper.png"
    Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(lower_path)
    Image.new("RGBA", (8, 8), (0, 0, 255, 255)).save(upper_path)
    skeleton = SkeletonDefinition(
        joints=(Joint("root", 0, 0), Joint("tip", 40, 0)),
        bars=(
            Bar("lower_body", "root", "tip", 0),
            Bar("upper_body", "root", "tip", 10),
        ),
        rigid_bars=(Bar("lower_body", "root", "tip", 0),),
    )
    keyframes = (
        keyframe_from_positions(
            0,
            {},
            joints=skeleton.joints,
            bars=skeleton.bars,
            bar_images={
                "lower_body": {
                    "image_path": str(lower_path),
                    "anchor_start_x": 0,
                    "anchor_start_y": 4,
                    "anchor_end_x": 8,
                    "anchor_end_y": 4,
                },
                "upper_body": {
                    "image_path": str(upper_path),
                    "anchor_start_x": 0,
                    "anchor_start_y": 4,
                    "anchor_end_x": 8,
                    "anchor_end_y": 4,
                },
            },
        ),
    )
    spec = MotionSpec(
        width=96,
        height=96,
        frames=2,
        duration_ms=20,
        outline="#00000000",
        show_joints=False,
        keyframes=keyframes,
        skeleton=skeleton,
    )

    frame = render_frame(spec, 0, transparent_background=True)
    bbox = frame.getchannel("A").getbbox()
    assert bbox is not None
    sample = frame.getpixel(((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2))

    assert sample[:3] == (0, 0, 255)


def test_skeleton_joints_are_bar_endpoints_without_duplicates():
    segments = skeleton_segments()
    joints = joint_points()
    endpoints = {point for segment in segments for point in segment}

    assert set(joints) == endpoints
    assert len(joints) == len(endpoints)


def test_skeleton_model_rejects_bar_with_undefined_joint():
    joints = (Joint("root", 0, 0), Joint("hand", 10, 0))
    bars = (Bar("arm", "root", "missing_hand"),)

    try:
        validate_skeleton_model(joints, bars)
    except ValueError as exc:
        assert "undefined joint missing_hand" in str(exc)
    else:
        raise AssertionError("undefined bar joints should be rejected")


def test_keyframes_interpolate_joint_position_parameters():
    keyframes = (
        keyframe_from_positions(0, {"head": (0, -76)}),
        keyframe_from_positions(10, {"head": (20, -56)}),
    )

    pose = interpolate_keyframes(keyframes, 5)

    assert round(bar_lengths(pose)["head_to_neck"], 3) == 20.0


def test_keyframes_interpolate_local_coordinate_transform():
    keyframes = (
        keyframe_from_positions(0, {"head": (0, -76)}, origin_x=90, origin_y=120, scale=1.0, rotation_degrees=0),
        keyframe_from_positions(10, {"head": (20, -56)}, origin_x=270, origin_y=180, scale=2.0, rotation_degrees=90),
    )

    pose, center, scale, rotation = interpolate_keyframe_state(keyframes, 5)

    assert round(bar_lengths(pose)["head_to_neck"], 3) == 20.0
    assert center == (180, 150)
    assert scale == 1.5
    assert rotation == 45


def test_rigid_bar_projection_keeps_neutral_bar_lengths():
    pose = {
        "neck": (0, -56),
        "head": (0, -90),
        "left_shoulder": (-28, -42),
        "right_shoulder": (28, -42),
        "left_elbow": (-50, -8),
        "right_elbow": (90, -90),
        "left_wrist": (-64, 26),
        "right_wrist": (120, -130),
        "pelvis": (0, 28),
        "left_hip": (-20, 34),
        "right_hip": (20, 34),
        "left_knee": (-24, 76),
        "right_knee": (24, 76),
        "left_ankle": (-28, 116),
        "right_ankle": (28, 116),
    }

    rigid_pose = enforce_rigid_bar_lengths(pose)
    lengths = bar_lengths(rigid_pose)

    assert round(lengths["right_upper_arm"], 3) == 40.497
    assert round(lengths["right_forearm"], 3) == 36.77


def test_motion_spec_accepts_keyframes_inside_frame_range():
    keyframes = (
        keyframe_from_positions(0, {"head": (0, -76)}),
        keyframe_from_positions(3, {"head": (12, -64)}),
    )
    spec = MotionSpec(width=64, height=48, frames=4, duration_ms=20, keyframes=keyframes)

    frames = build_frames(spec)

    assert len(frames) == 4


def test_motion_spec_record_round_trips_parameters_and_keyframes():
    keyframes = (
        keyframe_from_positions(
            0,
            {"head": (0, -76)},
            origin_x=90,
            origin_y=180,
            scale=0.8,
            bar_images={
                "spine": {
                    "image_path": "spine.png",
                    "source_x": 1,
                    "source_y": 2,
                    "source_width": 3,
                    "source_height": 4,
                    "anchor_start_x": 1,
                    "anchor_start_y": 4,
                    "anchor_end_x": 4,
                    "anchor_end_y": 4,
                }
            },
        ),
        keyframe_from_positions(3, {"head": (12, -64)}, origin_x=270, origin_y=180, scale=1.2, rotation_degrees=30),
    )
    spec = MotionSpec(
        width=64,
        height=48,
        frames=4,
        duration_ms=20,
        fill="#abcdef",
        show_bars=False,
        hidden_bars=("left_collar",),
        hidden_bar_images=("spine",),
        keyframes=keyframes,
    )

    loaded = motion_spec_from_record(motion_spec_to_record(spec))

    assert loaded == spec
    assert loaded.keyframes[0].bar_images[0].source_box == (1, 2, 4, 6)
    assert loaded.keyframes[0].bar_images[0].anchor_end_x == 4
    assert loaded.show_bars is False
    assert loaded.hidden_bars == ("left_collar",)
    assert loaded.hidden_bar_images == ("spine",)


def test_keyframe_bar_images_use_nearest_previous_keyframe():
    keyframes = (
        keyframe_from_positions(0, {"head": (0, -76)}, bar_images={"left_collar": "left.png"}),
        keyframe_from_positions(10, {"head": (20, -56)}, bar_images={"spine": "spine.png"}),
    )

    bar_images = keyframe_properties_at(keyframes, 5)

    assert bar_images == {"left_collar": BarImage("left_collar", "left.png")}


def test_save_and_load_record_json(tmp_path):
    spec = MotionSpec(
        width=64,
        height=48,
        frames=4,
        duration_ms=20,
        keyframes=(keyframe_from_positions(2, {"left_wrist": (-70, 30)}),),
    )
    output_path = tmp_path / "animation.json"

    saved_path = save_record(spec, output_path)
    loaded = load_record(saved_path)

    assert loaded == spec


def test_render_frame_uses_global_joint_visibility_and_bar_images(tmp_path):
    image_path = tmp_path / "bar.png"
    source = Image.new("RGBA", (12, 8), (0, 0, 255, 255))
    for x in range(6, 12):
        for y in range(0, 8):
            source.putpixel((x, y), (255, 0, 0, 255))
    source.save(image_path)
    spec = MotionSpec(
        width=96,
        height=96,
        frames=2,
        duration_ms=20,
        keyframes=(
            keyframe_from_positions(
                0,
                {"head": (0, -76)},
                bar_images={
                    BARS[0].name: {
                        "image_path": str(image_path),
                        "source_x": 6,
                        "source_y": 0,
                        "source_width": 6,
                        "source_height": 8,
                        "anchor_start_x": 6,
                        "anchor_start_y": 4,
                        "anchor_end_x": 12,
                        "anchor_end_y": 4,
                    }
                },
            ),
            keyframe_from_positions(1, {"head": (0, -76)}),
        ),
    )
    no_joints_spec = MotionSpec(
        width=96,
        height=96,
        frames=2,
        duration_ms=20,
        show_joints=False,
        keyframes=spec.keyframes,
    )

    frames = build_frames(no_joints_spec, transparent_background=True)

    assert len(frames) == 2
    assert frames[0].getbbox() is not None


def test_render_frame_uses_global_bar_visibility():
    skeleton = SkeletonDefinition(
        joints=(Joint("start", 0, 0), Joint("end", 40, 0)),
        bars=(Bar("body", "start", "end"),),
        rigid_bars=(Bar("body", "start", "end"),),
    )
    visible = MotionSpec(
        width=96,
        height=96,
        frames=2,
        duration_ms=20,
        outline="#ff0000",
        show_joints=False,
        skeleton=skeleton,
    )
    hidden = MotionSpec(
        width=96,
        height=96,
        frames=2,
        duration_ms=20,
        outline="#ff0000",
        show_joints=False,
        show_bars=False,
        skeleton=skeleton,
    )

    visible_frame = render_frame(visible, 0, transparent_background=True)
    hidden_frame = render_frame(hidden, 0, transparent_background=True)

    assert visible_frame.getchannel("A").getbbox() is not None
    assert hidden_frame.getchannel("A").getbbox() is None


def test_bar_image_anchors_map_to_rotated_bar_ends(tmp_path):
    image_path = tmp_path / "anchored_bar.png"
    source = Image.new("RGBA", (12, 12), (0, 0, 0, 0))
    draw = ImageDraw.Draw(source)
    draw.line((2, 6, 10, 6), fill=(0, 0, 255, 255), width=3)
    draw.ellipse((0, 4, 4, 8), fill=(255, 0, 0, 255))
    draw.ellipse((8, 4, 12, 8), fill=(0, 255, 0, 255))
    source.save(image_path)

    canvas = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    _paste_bar_image(
        canvas,
        BarImage(
            bar_name=BARS[0].name,
            image_path=str(image_path),
            anchor_start_x=2,
            anchor_start_y=6,
            anchor_end_x=10,
            anchor_end_y=6,
        ),
        (32, 12),
        (32, 52),
    )

    alpha = canvas.getchannel("A")
    bbox = alpha.getbbox()
    assert bbox is not None
    assert bbox[0] <= 32 <= bbox[2]
    assert bbox[1] <= 12 <= bbox[3]
    assert bbox[1] <= 52 <= bbox[3]
    assert max(alpha.crop((30, 10, 35, 15)).getdata()) > 0
    assert max(alpha.crop((30, 50, 35, 55)).getdata()) > 0


def test_bar_image_uses_full_skeleton_pixel_fit(tmp_path):
    image_path = tmp_path / "neck_piece.png"
    source = Image.new("RGBA", (12, 12), (0, 0, 255, 255))
    source.save(image_path)
    spec = MotionSpec(
        width=96,
        height=96,
        frames=2,
        duration_ms=20,
        outline="#00000000",
        show_joints=False,
        keyframes=(
            keyframe_from_positions(
                0,
                {},
                bar_images={
                    BARS[0].name: {
                        "image_path": str(image_path),
                        "anchor_start_x": 6,
                        "anchor_start_y": 1,
                        "anchor_end_x": 6,
                        "anchor_end_y": 11,
                    }
                },
            ),
            keyframe_from_positions(1, {}),
        ),
    )

    frame = render_frame(spec, 0, transparent_background=True)
    bbox = frame.getchannel("A").getbbox()

    assert bbox is not None
    assert bbox[3] - bbox[1] < 30


def test_build_frames_caches_bar_image_cutouts(tmp_path, monkeypatch):
    image_path = tmp_path / "cached_piece.png"
    Image.new("RGBA", (12, 12), (0, 0, 255, 255)).save(image_path)
    spec = MotionSpec(
        width=96,
        height=96,
        frames=4,
        duration_ms=20,
        show_joints=False,
        keyframes=(
            keyframe_from_positions(
                0,
                {},
                bar_images={
                    BARS[0].name: {
                        "image_path": str(image_path),
                        "source_x": 0,
                        "source_y": 0,
                        "source_width": 12,
                        "source_height": 12,
                        "anchor_start_x": 6,
                        "anchor_start_y": 1,
                        "anchor_end_x": 6,
                        "anchor_end_y": 11,
                    }
                },
            ),
            keyframe_from_positions(3, {}),
        ),
    )
    original_open = Image.open
    open_calls = 0

    def counting_open(*args, **kwargs):
        nonlocal open_calls
        open_calls += 1
        return original_open(*args, **kwargs)

    monkeypatch.setattr("project_pv.animation.Image.open", counting_open)

    frames = build_frames(spec, transparent_background=True)

    assert len(frames) == 4
    assert open_calls == 1


def test_vector_field_fits_inside_non_square_pixel_output():
    points, scale = fit_vector_points_to_pixel_field([(0, 0), (360, 360)], 120, 720)

    assert scale == 120 / 360
    assert points == [(0, 300), (120, 420)]


def test_save_gif_writes_animated_gif(tmp_path):
    spec = MotionSpec(
        width=64,
        height=48,
        frames=3,
        duration_ms=20,
        keyframes=(
            keyframe_from_positions(0, {"head": (0, -76)}, origin_x=120, origin_y=180),
            keyframe_from_positions(2, {"head": (0, -76)}, origin_x=240, origin_y=180),
        ),
    )
    output_path = tmp_path / "animation.gif"

    saved_path = save_gif(spec, output_path)

    with Image.open(saved_path) as image:
        assert image.format == "GIF"
        assert image.n_frames == 3


def test_save_gif_uses_transparent_background(tmp_path):
    spec = MotionSpec(
        width=64,
        height=48,
        frames=3,
        duration_ms=20,
        background="#ff00ff",
        keyframes=(
            keyframe_from_positions(0, {"head": (0, -76)}, origin_x=120, origin_y=180),
            keyframe_from_positions(2, {"head": (0, -76)}, origin_x=240, origin_y=180),
        ),
    )
    output_path = tmp_path / "transparent.gif"

    saved_path = save_gif(spec, output_path)

    with Image.open(saved_path) as image:
        assert "transparency" in image.info
        assert image.convert("RGBA").getpixel((0, 0))[3] == 0


def test_save_gif_restores_background_between_frames(tmp_path):
    keyframes = (
        keyframe_from_positions(0, {"right_wrist": (64, 26)}),
        keyframe_from_positions(1, {"right_wrist": (96, -96)}),
        keyframe_from_positions(2, {"right_wrist": (64, 26)}),
    )
    spec = MotionSpec(
        width=96,
        height=96,
        frames=3,
        duration_ms=20,
        keyframes=keyframes,
    )
    output_path = tmp_path / "disposal.gif"

    saved_path = save_gif(spec, output_path)

    with Image.open(saved_path) as image:
        assert image.disposal_method == 2
        image.seek(1)
        assert image.disposal_method == 2
