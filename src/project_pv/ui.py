"""Tkinter UI for defining skeleton motion and exporting pixel GIFs."""

from __future__ import annotations

import logging
import math
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk
from PIL.Image import Resampling

from project_pv.animation import (
    MotionSpec,
    fit_vector_points_to_pixel_field,
    load_record,
    render_frame,
    save_record,
    save_gif,
    transform_points,
    transform_segments,
)
from project_pv.skeleton_model import (
    Bar,
    BarImage,
    DEFAULT_SKELETON,
    SkeletonDefinition,
    Joint,
    interpolate_keyframe_state,
    joint_points,
    keyframe_bar_image_map,
    keyframe_position_map,
    keyframe_from_positions,
    load_skeleton,
    neutral_joint_positions,
    positioned_joints,
    save_skeleton,
    skeleton_segments,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_BODY_PIXEL_MAP = Path(__file__).resolve().parents[2] / "release" / "DoorKeeper.png"
DEFAULT_ANIMATION_RECORD = Path(__file__).resolve().parents[2] / "release" / "waving_hand_skeleton.json"
DEFAULT_SKELETON_RECORD = Path(__file__).resolve().parents[2] / "release" / "skeleton.json"


class ProjectPVApp(tk.Tk):
    """Main desktop window for the vector-to-GIF workflow."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Project PV")
        self.minsize(980, 680)
        self._preview_photo: ImageTk.PhotoImage | None = None
        self._cutout_photo: ImageTk.PhotoImage | None = None
        self._pixel_map_photo: ImageTk.PhotoImage | None = None
        self._pixel_map_window: tk.Toplevel | None = None
        self._pixel_map_canvas: tk.Canvas | None = None
        self._pixel_cutout_canvas: tk.Canvas | None = None
        self._skeleton_window: tk.Toplevel | None = None
        self._skeleton_axis_canvas: tk.Canvas | None = None
        self._skeleton_editor_joints: dict[str, Joint] = {}
        self._skeleton_editor_bars: dict[str, Bar] = {}
        self._pixel_map_scale = 1.0
        self._pixel_cutout_view: tuple[int, int, int, int, float, float, float, float] | None = None
        self._pixel_selection_start: tuple[int, int] | None = None
        self._pixel_drag_anchor: str | None = None
        self._bar_image_cache: dict[tuple[object, ...], object] = {}
        self._preview_image_id: int | None = None
        self._after_id: str | None = None
        self._frame_index = 0
        self._skeleton: SkeletonDefinition = DEFAULT_SKELETON
        self._selected_joints: set[str] = set()
        self._vector_joint_locations: dict[str, tuple[float, float]] = {}
        self._record_path: Path | None = DEFAULT_ANIMATION_RECORD
        self._skeleton_path: Path | None = DEFAULT_SKELETON_RECORD

        self.width_var = tk.IntVar(value=50)
        self.height_var = tk.IntVar(value=50)
        self.frames_var = tk.IntVar(value=32)
        self.duration_var = tk.IntVar(value=60)
        self.fill_var = tk.StringVar(value="#1b8f87")
        self.outline_var = tk.StringVar(value="#17202a")
        self.background_var = tk.StringVar(value="#f7f4ea")
        self.show_joints_var = tk.BooleanVar(value=True)
        self.preview_zoom_var = tk.DoubleVar(value=2.0)
        self.preview_zoom_label_var = tk.StringVar(value="200%")
        self.played_frame_var = tk.IntVar(value=0)
        self.played_frame_label_var = tk.StringVar(value="0 / 31")
        first_joint = self._skeleton.joints[0]
        first_bar = self._skeleton.bars[0]
        self.selected_joint_var = tk.StringVar(value=first_joint.name)
        self.keyframe_var = tk.IntVar(value=0)
        self.joint_x_var = tk.DoubleVar(value=first_joint.x)
        self.joint_y_var = tk.DoubleVar(value=first_joint.y)
        self.offset_x_var = tk.DoubleVar(value=0)
        self.offset_y_var = tk.DoubleVar(value=0)
        self.local_x_var = tk.DoubleVar(value=180)
        self.local_y_var = tk.DoubleVar(value=180)
        self.local_scale_var = tk.DoubleVar(value=1.0)
        self.local_rotation_var = tk.DoubleVar(value=0)
        self.selected_bar_var = tk.StringVar(value=first_bar.name)
        self.bar_visible_var = tk.BooleanVar(value=True)
        self.bar_image_path_var = tk.StringVar(value=str(DEFAULT_BODY_PIXEL_MAP))
        self.bar_source_x_var = tk.IntVar(value=0)
        self.bar_source_y_var = tk.IntVar(value=0)
        self.bar_source_width_var = tk.IntVar(value=0)
        self.bar_source_height_var = tk.IntVar(value=0)
        self.bar_anchor_start_x_var = tk.IntVar(value=0)
        self.bar_anchor_start_y_var = tk.IntVar(value=0)
        self.bar_anchor_end_x_var = tk.IntVar(value=0)
        self.bar_anchor_end_y_var = tk.IntVar(value=0)
        self.apply_transform_var = tk.BooleanVar(value=False)
        self.apply_selected_joints_var = tk.BooleanVar(value=False)
        self.apply_all_joints_var = tk.BooleanVar(value=False)
        self.apply_selected_bar_var = tk.BooleanVar(value=True)
        self.apply_all_bars_var = tk.BooleanVar(value=False)
        self.skeleton_joint_name_var = tk.StringVar(value="")
        self.skeleton_joint_x_var = tk.DoubleVar(value=0)
        self.skeleton_joint_y_var = tk.DoubleVar(value=0)
        self.skeleton_joint_radius_var = tk.DoubleVar(value=5.0)
        self.skeleton_bar_name_var = tk.StringVar(value="")
        self.skeleton_bar_start_var = tk.StringVar(value="")
        self.skeleton_bar_end_var = tk.StringVar(value="")
        self.skeleton_bar_layer_var = tk.IntVar(value=0)
        self._keyframe_positions = {
            0: neutral_joint_positions(self._skeleton.joints),
            self.frames_var.get() - 1: neutral_joint_positions(self._skeleton.joints),
        }
        self._keyframe_transforms = {
            0: self._default_keyframe_transform(),
            self.frames_var.get() - 1: self._default_keyframe_transform(),
        }
        self._keyframe_bar_images = {
            0: {},
            self.frames_var.get() - 1: {},
        }
        self._hidden_bar_images: set[str] = set()
        self._updating_playhead = False
        self._selected_joints.add(self.selected_joint_var.get())

        self._build_layout()
        self._load_default_record()
        self._redraw()
        LOGGER.info("Project PV UI initialized")

    def _default_keyframe_transform(self) -> dict[str, float]:
        return {"origin_x": 180.0, "origin_y": 180.0, "scale": 1.0, "rotation_degrees": 0.0}

    def _load_default_record(self) -> None:
        if not DEFAULT_ANIMATION_RECORD.exists():
            return
        try:
            self._apply_spec(load_record(DEFAULT_ANIMATION_RECORD))
            self._record_path = DEFAULT_ANIMATION_RECORD
        except Exception:
            LOGGER.exception("Default animation record failed to load")

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=16)
        left.grid(row=0, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)

        right = ttk.Frame(self, padding=16)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)

        ttk.Label(left, text="Skeleton motion", font=("Segoe UI", 15, "bold")).grid(row=0, column=0, sticky="w")
        editor_tabs = ttk.Notebook(left)
        editor_tabs.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        left.rowconfigure(1, weight=1)

        output_tab = ttk.Frame(editor_tabs, padding=12)
        keyframe_tab = ttk.Frame(editor_tabs, padding=12)
        editor_tabs.add(output_tab, text="Output")
        editor_tabs.add(keyframe_tab, text="Keyframes")

        controls = ttk.Frame(output_tab)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)
        output_tab.columnconfigure(0, weight=1)

        row = -1
        for label, variable, lower, upper, step in (
            ("Pixel width", self.width_var, 32, 720, 1),
            ("Pixel height", self.height_var, 32, 720, 1),
            ("Frames", self.frames_var, 2, 120, 1),
            ("Frame ms", self.duration_var, 10, 500, 10),
        ):
            row += 1
            self._add_scale(controls, row, label, variable, lower, upper, step)

        color_frame = ttk.Frame(output_tab)
        color_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        color_frame.columnconfigure((1, 3, 5), weight=1)
        self._add_color_entry(color_frame, 0, "Fill", self.fill_var)
        self._add_color_entry(color_frame, 2, "Line", self.outline_var)
        self._add_color_entry(color_frame, 4, "BG", self.background_var)
        ttk.Checkbutton(color_frame, text="Show joints", variable=self.show_joints_var, command=self._redraw).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

        self.vector_canvas = tk.Canvas(keyframe_tab, width=360, height=240, bg="#fbfaf6", highlightthickness=1, highlightbackground="#cfc8b8")
        self.vector_canvas.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        self.vector_canvas.bind("<Button-1>", self._select_vector_joint)

        self.time_axis = tk.Canvas(keyframe_tab, width=360, height=36, bg="#fbfaf6", highlightthickness=1, highlightbackground="#cfc8b8")
        self.time_axis.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        self.time_axis.bind("<Button-1>", self._select_timeline_frame)

        pose_frame = ttk.Frame(keyframe_tab)
        pose_frame.grid(row=2, column=0, sticky="ew")
        pose_frame.columnconfigure(1, weight=1)
        pose_frame.columnconfigure(4, weight=1)
        keyframe_tab.columnconfigure(0, weight=1)
        keyframe_tab.rowconfigure(3, weight=1)
        ttk.Label(pose_frame, text="Keyframe").grid(row=0, column=0, sticky="w", pady=3)
        keyframe_spin = ttk.Spinbox(
            pose_frame,
            from_=0,
            to=max(0, self.frames_var.get() - 1),
            textvariable=self.keyframe_var,
            width=8,
            command=self._load_keyframe_joint,
        )
        keyframe_spin.grid(row=0, column=1, sticky="ew", padx=(8, 10), pady=3)
        keyframe_spin.bind("<Return>", lambda _event: self._load_keyframe_joint())
        keyframe_spin.bind("<FocusOut>", lambda _event: self._load_keyframe_joint())
        ttk.Button(pose_frame, text="Add keyframe", command=self._add_keyframe).grid(row=0, column=2, sticky="ew", pady=3)
        ttk.Button(pose_frame, text="Apply selected", command=self._apply_current_to_later_keyframes).grid(
            row=0, column=3, columnspan=2, sticky="ew", pady=3
        )

        ttk.Label(pose_frame, text="Local X").grid(row=1, column=0, sticky="w", pady=3)
        local_x_spin = ttk.Spinbox(pose_frame, from_=0, to=360, increment=1, textvariable=self.local_x_var, width=8, command=self._set_keyframe_transform)
        local_x_spin.grid(row=1, column=1, sticky="ew", padx=(8, 10), pady=3)
        ttk.Label(pose_frame, text="Local Y").grid(row=1, column=2, sticky="w", pady=3)
        local_y_spin = ttk.Spinbox(pose_frame, from_=0, to=360, increment=1, textvariable=self.local_y_var, width=8, command=self._set_keyframe_transform)
        local_y_spin.grid(row=1, column=3, sticky="ew", padx=(8, 10), pady=3)
        ttk.Button(pose_frame, text="Set transform", command=self._set_keyframe_transform).grid(row=1, column=4, sticky="ew", pady=3)

        ttk.Label(pose_frame, text="Scale").grid(row=2, column=0, sticky="w", pady=3)
        local_scale_spin = ttk.Spinbox(pose_frame, from_=0.2, to=2.5, increment=0.1, textvariable=self.local_scale_var, width=8, command=self._set_keyframe_transform)
        local_scale_spin.grid(row=2, column=1, sticky="ew", padx=(8, 10), pady=3)
        ttk.Label(pose_frame, text="Rotation").grid(row=2, column=2, sticky="w", pady=3)
        local_rotation_spin = ttk.Spinbox(pose_frame, from_=-720, to=720, increment=15, textvariable=self.local_rotation_var, width=8, command=self._set_keyframe_transform)
        local_rotation_spin.grid(row=2, column=3, sticky="ew", padx=(8, 10), pady=3)
        for spin in (local_x_spin, local_y_spin, local_scale_spin, local_rotation_spin):
            spin.bind("<Return>", lambda _event: self._set_keyframe_transform())
            spin.bind("<FocusOut>", lambda _event: self._set_keyframe_transform())
        ttk.Checkbutton(pose_frame, text="Apply transform", variable=self.apply_transform_var).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=3
        )

        ttk.Label(pose_frame, text="Joint").grid(row=4, column=0, sticky="w", pady=3)
        joint_box = ttk.Combobox(
            pose_frame,
            textvariable=self.selected_joint_var,
            values=tuple(joint.name for joint in self._skeleton.joints),
            state="readonly",
        )
        self.joint_box = joint_box
        joint_box.grid(row=4, column=1, columnspan=4, sticky="ew", padx=(8, 0), pady=3)
        joint_box.bind("<<ComboboxSelected>>", lambda _event: self._select_joint_from_box())

        ttk.Label(pose_frame, text="Joint X").grid(row=5, column=0, sticky="w", pady=3)
        x_spin = ttk.Spinbox(pose_frame, from_=-180, to=180, increment=1, textvariable=self.joint_x_var, width=8, command=self._set_keyframe_joint)
        x_spin.grid(row=5, column=1, sticky="ew", padx=(8, 10), pady=3)
        ttk.Label(pose_frame, text="Joint Y").grid(row=5, column=2, sticky="w", pady=3)
        y_spin = ttk.Spinbox(pose_frame, from_=-180, to=180, increment=1, textvariable=self.joint_y_var, width=8, command=self._set_keyframe_joint)
        y_spin.grid(row=5, column=3, sticky="ew", padx=(8, 10), pady=3)
        ttk.Button(pose_frame, text="Set", command=self._set_keyframe_joint).grid(row=5, column=4, sticky="ew", pady=3)
        for spin in (x_spin, y_spin):
            spin.bind("<Return>", lambda _event: self._set_keyframe_joint())
            spin.bind("<FocusOut>", lambda _event: self._set_keyframe_joint())

        ttk.Checkbutton(pose_frame, text="Apply selected joints", variable=self.apply_selected_joints_var).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=3
        )
        ttk.Checkbutton(pose_frame, text="Apply all joints", variable=self.apply_all_joints_var).grid(
            row=6, column=2, columnspan=2, sticky="w", pady=3
        )
        ttk.Button(pose_frame, text="Use skeleton default", command=self._set_keyframe_to_skeleton_default).grid(
            row=6, column=4, sticky="ew", pady=3
        )

        ttk.Label(pose_frame, text="Offset X").grid(row=7, column=0, sticky="w", pady=3)
        offset_x_spin = ttk.Spinbox(pose_frame, from_=-180, to=180, increment=1, textvariable=self.offset_x_var, width=8)
        offset_x_spin.grid(row=7, column=1, sticky="ew", padx=(8, 10), pady=3)
        ttk.Label(pose_frame, text="Offset Y").grid(row=7, column=2, sticky="w", pady=3)
        offset_y_spin = ttk.Spinbox(pose_frame, from_=-180, to=180, increment=1, textvariable=self.offset_y_var, width=8)
        offset_y_spin.grid(row=7, column=3, sticky="ew", padx=(8, 10), pady=3)
        ttk.Button(pose_frame, text="Apply offset", command=self._offset_selected_joints).grid(row=7, column=4, sticky="ew", pady=3)
        for spin in (offset_x_spin, offset_y_spin):
            spin.bind("<Return>", lambda _event: self._offset_selected_joints())

        ttk.Label(pose_frame, text="Bar").grid(row=8, column=0, sticky="w", pady=3)
        bar_box = ttk.Combobox(
            pose_frame,
            textvariable=self.selected_bar_var,
            values=tuple(bar.name for bar in self._skeleton.bars),
            state="readonly",
        )
        self.bar_box = bar_box
        bar_box.grid(row=8, column=1, columnspan=4, sticky="ew", padx=(8, 0), pady=3)
        bar_box.bind("<<ComboboxSelected>>", lambda _event: self._load_selected_bar())

        ttk.Checkbutton(pose_frame, text="Show body part", variable=self.bar_visible_var, command=self._set_selected_bar_visibility).grid(
            row=9, column=0, columnspan=2, sticky="w", pady=3
        )
        ttk.Checkbutton(pose_frame, text="Apply selected body", variable=self.apply_selected_bar_var).grid(
            row=9, column=2, columnspan=2, sticky="w", pady=3
        )
        ttk.Checkbutton(pose_frame, text="Apply all bodies", variable=self.apply_all_bars_var).grid(
            row=9, column=4, sticky="w", pady=3
        )

        ttk.Label(pose_frame, text="Bar image").grid(row=10, column=0, sticky="w", pady=3)
        bar_image_entry = ttk.Entry(pose_frame, textvariable=self.bar_image_path_var)
        bar_image_entry.grid(row=10, column=1, columnspan=2, sticky="ew", padx=(8, 10), pady=3)
        bar_image_entry.bind("<Return>", lambda _event: self._set_bar_image())
        bar_image_entry.bind("<FocusOut>", lambda _event: self._set_bar_image())
        ttk.Button(pose_frame, text="Browse", command=self._browse_bar_image).grid(row=10, column=3, sticky="ew", padx=(0, 8), pady=3)
        ttk.Button(pose_frame, text="Clear", command=self._clear_bar_image).grid(row=10, column=4, sticky="ew", pady=3)

        for column, (label, variable) in enumerate(
            (
                ("Src X", self.bar_source_x_var),
                ("Src Y", self.bar_source_y_var),
                ("Src W", self.bar_source_width_var),
                ("Src H", self.bar_source_height_var),
            )
        ):
            ttk.Label(pose_frame, text=label).grid(row=11, column=column, sticky="w", pady=3)
            source_spin = ttk.Spinbox(pose_frame, from_=0, to=10000, increment=1, textvariable=variable, width=7, command=self._set_bar_image)
            source_spin.grid(row=12, column=column, sticky="ew", padx=(0 if column == 0 else 8, 0), pady=3)
            source_spin.bind("<Return>", lambda _event: self._set_bar_image())
            source_spin.bind("<FocusOut>", lambda _event: self._set_bar_image())

        for column, (label, variable) in enumerate(
            (
                ("A1 X", self.bar_anchor_start_x_var),
                ("A1 Y", self.bar_anchor_start_y_var),
                ("A2 X", self.bar_anchor_end_x_var),
                ("A2 Y", self.bar_anchor_end_y_var),
            )
        ):
            ttk.Label(pose_frame, text=label).grid(row=13, column=column, sticky="w", pady=3)
            anchor_spin = ttk.Spinbox(pose_frame, from_=0, to=10000, increment=1, textvariable=variable, width=7, command=self._set_bar_image)
            anchor_spin.grid(row=14, column=column, sticky="ew", padx=(0 if column == 0 else 8, 0), pady=3)
            anchor_spin.bind("<Return>", lambda _event: self._set_bar_image())
            anchor_spin.bind("<FocusOut>", lambda _event: self._set_bar_image())

        ttk.Button(pose_frame, text="Pixel map", command=self._open_pixel_map_window).grid(
            row=15, column=0, columnspan=5, sticky="ew", pady=(8, 3)
        )

        self.joint_table = ttk.Treeview(keyframe_tab, columns=("x", "y"), show="tree headings", height=8, selectmode="extended")
        self.joint_table.heading("#0", text="Joint")
        self.joint_table.heading("x", text="X")
        self.joint_table.heading("y", text="Y")
        self.joint_table.column("#0", width=140, stretch=True)
        self.joint_table.column("x", width=70, anchor="e")
        self.joint_table.column("y", width=70, anchor="e")
        self.joint_table.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        self.joint_table.bind("<<TreeviewSelect>>", self._select_joint_from_table)

        ttk.Label(right, text="Pixel GIF conversion", font=("Segoe UI", 15, "bold")).grid(row=0, column=0, sticky="w")
        zoom_controls = ttk.Frame(right)
        zoom_controls.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        zoom_controls.columnconfigure(2, weight=1)
        ttk.Button(zoom_controls, text="-", width=3, command=lambda: self._change_preview_zoom(-0.25)).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(zoom_controls, text="+", width=3, command=lambda: self._change_preview_zoom(0.25)).grid(row=0, column=1, padx=(0, 10))
        zoom_scale = ttk.Scale(
            zoom_controls,
            from_=0.25,
            to=4.0,
            variable=self.preview_zoom_var,
            command=self._set_preview_zoom,
        )
        zoom_scale.grid(row=0, column=2, sticky="ew", padx=(0, 10))
        ttk.Label(zoom_controls, textvariable=self.preview_zoom_label_var, width=5).grid(row=0, column=3, padx=(0, 8))
        ttk.Button(zoom_controls, text="100%", width=6, command=self._reset_preview_zoom).grid(row=0, column=4)

        preview_shell = ttk.Frame(right, borderwidth=1, relief="solid")
        preview_shell.grid(row=2, column=0, sticky="nsew", pady=(12, 14))
        preview_shell.columnconfigure(0, weight=1)
        preview_shell.rowconfigure(0, weight=1)
        self.pixel_preview = tk.Canvas(preview_shell, bg="#fbfaf6", highlightthickness=0)
        self.pixel_preview.grid(row=0, column=0, sticky="nsew")
        x_scroll = ttk.Scrollbar(preview_shell, orient="horizontal", command=self.pixel_preview.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        y_scroll = ttk.Scrollbar(preview_shell, orient="vertical", command=self.pixel_preview.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        self.pixel_preview.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)

        playhead = ttk.Frame(right)
        playhead.grid(row=3, column=0, sticky="ew", pady=(0, 12))
        playhead.columnconfigure(1, weight=1)
        ttk.Label(playhead, text="Frame").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.played_frame_scale = ttk.Scale(
            playhead,
            from_=0,
            to=max(1, self.frames_var.get() - 1),
            variable=self.played_frame_var,
            command=self._set_played_frame,
        )
        self.played_frame_scale.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        ttk.Label(playhead, textvariable=self.played_frame_label_var, width=9).grid(row=0, column=2, sticky="e")

        buttons = ttk.Frame(right)
        buttons.grid(row=4, column=0, sticky="ew")
        buttons.columnconfigure(2, weight=1)
        ttk.Button(buttons, text="Play", command=self.play).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(buttons, text="Stop", command=self.stop).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(buttons, text="Skeleton", command=self.open_skeleton_definition_window).grid(row=0, column=3, padx=(0, 8))
        ttk.Button(buttons, text="Import skeleton", command=self.import_skeleton).grid(row=0, column=4, padx=(0, 8))
        ttk.Button(buttons, text="Export skeleton", command=self.export_skeleton).grid(row=0, column=5, padx=(0, 8))
        ttk.Button(buttons, text="Refresh skeleton", command=self.refresh_skeleton).grid(row=1, column=3, padx=(0, 8), pady=(8, 0))
        ttk.Button(buttons, text="Import JSON", command=self.import_record).grid(row=1, column=4, padx=(0, 8), pady=(8, 0))
        ttk.Button(buttons, text="Export JSON", command=self.export_record).grid(row=1, column=5, padx=(0, 8), pady=(8, 0))
        ttk.Button(buttons, text="Export GIF", command=self.export_gif).grid(row=2, column=5, pady=(8, 0))

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(right, textvariable=self.status_var).grid(row=5, column=0, sticky="w", pady=(12, 0))

    def _add_scale(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.Variable,
        lower: float,
        upper: float,
        step: float,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        scale = ttk.Scale(parent, from_=lower, to=upper, variable=variable, command=lambda _value: self._redraw())
        scale.grid(row=row, column=1, sticky="ew", pady=3, padx=(10, 8))
        spin = ttk.Spinbox(parent, from_=lower, to=upper, increment=step, textvariable=variable, width=8, command=self._redraw)
        spin.grid(row=row, column=2, sticky="e", pady=3)
        spin.bind("<Return>", lambda _event: self._redraw())
        spin.bind("<FocusOut>", lambda _event: self._redraw())

    def _add_color_entry(self, parent: ttk.Frame, column: int, label: str, variable: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=0, column=column, sticky="w", padx=(0, 4))
        entry = ttk.Entry(parent, textvariable=variable, width=9)
        entry.grid(row=0, column=column + 1, sticky="ew", padx=(0, 10))
        entry.bind("<Return>", lambda _event: self._redraw())
        entry.bind("<FocusOut>", lambda _event: self._redraw())

    def _change_preview_zoom(self, delta: float) -> None:
        zoom = min(4.0, max(0.25, self.preview_zoom_var.get() + delta))
        self.preview_zoom_var.set(zoom)
        self._redraw()

    def _reset_preview_zoom(self) -> None:
        self.preview_zoom_var.set(1.0)
        self._redraw()

    def _set_preview_zoom(self, _value: str) -> None:
        self._redraw()

    def _sync_played_frame_control(self, spec: MotionSpec) -> None:
        max_frame = max(0, spec.frames - 1)
        self._frame_index = min(max(0, self._frame_index), max_frame)
        if hasattr(self, "played_frame_scale"):
            self.played_frame_scale.configure(to=max(1, max_frame))
        self._updating_playhead = True
        try:
            self.played_frame_var.set(self._frame_index)
            self.played_frame_label_var.set(f"{self._frame_index} / {max_frame}")
        finally:
            self._updating_playhead = False

    def _set_played_frame(self, value: str) -> None:
        if self._updating_playhead:
            return
        try:
            spec = self._spec()
            frame = min(max(0, round(float(value))), max(0, spec.frames - 1))
            self._frame_index = frame
            self.played_frame_var.set(frame)
            self.played_frame_label_var.set(f"{frame} / {max(0, spec.frames - 1)}")
            self._redraw()
        except Exception as exc:
            LOGGER.exception("Played frame update failed")
            self.status_var.set(str(exc))

    def _show_keyframe_in_pixel_preview(self, frame: int) -> None:
        self._frame_index = min(max(0, frame), max(0, int(self.frames_var.get()) - 1))

    def _keyframe_frame(self) -> int:
        return min(max(0, int(self.keyframe_var.get())), max(0, int(self.frames_var.get()) - 1))

    def _select_joint_from_box(self) -> None:
        self._selected_joints = {self.selected_joint_var.get()}
        self._load_keyframe_joint()

    def _select_joint_from_table(self, _event: tk.Event) -> None:
        selected = set(self.joint_table.selection())
        if not selected:
            return
        self._selected_joints = selected
        first_joint = next(iter(selected))
        self.selected_joint_var.set(first_joint)
        positions, _transform, _bar_images = self._preview_keyframe_state(self._keyframe_frame())
        x, y = positions[first_joint]
        self.joint_x_var.set(x)
        self.joint_y_var.set(y)
        self._redraw()

    def _select_vector_joint(self, event: tk.Event) -> None:
        if not self._vector_joint_locations:
            return

        nearest_joint = min(
            self._vector_joint_locations,
            key=lambda name: (self._vector_joint_locations[name][0] - event.x) ** 2
            + (self._vector_joint_locations[name][1] - event.y) ** 2,
        )
        x, y = self._vector_joint_locations[nearest_joint]
        if math.hypot(x - event.x, y - event.y) > 12:
            return

        multi_select = bool(event.state & 0x0001) or bool(event.state & 0x0004)
        if multi_select:
            if nearest_joint in self._selected_joints and len(self._selected_joints) > 1:
                self._selected_joints.remove(nearest_joint)
            else:
                self._selected_joints.add(nearest_joint)
        else:
            self._selected_joints = {nearest_joint}
        self.selected_joint_var.set(nearest_joint)
        self._load_keyframe_joint()

    def _select_timeline_frame(self, event: tk.Event) -> None:
        frames = max(2, int(self.frames_var.get()))
        width = max(1, int(self.time_axis["width"]) - 20)
        progress = min(1.0, max(0.0, (event.x - 10) / width))
        self.keyframe_var.set(round(progress * (frames - 1)))
        self._load_keyframe_joint()

    def _keyframes_from_editor(self) -> tuple:
        return tuple(
            keyframe_from_positions(
                frame,
                positions,
                origin_x=self._keyframe_transforms[frame]["origin_x"],
                origin_y=self._keyframe_transforms[frame]["origin_y"],
                scale=self._keyframe_transforms[frame]["scale"],
                rotation_degrees=self._keyframe_transforms[frame]["rotation_degrees"],
                bar_images={
                    bar_name: {
                        "image_path": attachment.image_path,
                        "source_x": attachment.source_x,
                        "source_y": attachment.source_y,
                        "source_width": attachment.source_width,
                        "source_height": attachment.source_height,
                        "anchor_start_x": attachment.anchor_start_x,
                        "anchor_start_y": attachment.anchor_start_y,
                        "anchor_end_x": attachment.anchor_end_x,
                        "anchor_end_y": attachment.anchor_end_y,
                    }
                    for bar_name, attachment in self._keyframe_bar_images.get(frame, {}).items()
                },
                joints=self._skeleton.joints,
                bars=self._skeleton.bars,
            )
            for frame, positions in sorted(self._keyframe_positions.items())
            if frame in self._keyframe_transforms
        )

    def _preview_keyframe_state(
        self, frame: int
    ) -> tuple[dict[str, tuple[float, float]], dict[str, float], dict[str, BarImage]]:
        if frame in self._keyframe_positions and frame in self._keyframe_transforms:
            return (
                dict(self._keyframe_positions[frame]),
                dict(self._keyframe_transforms[frame]),
                dict(self._keyframe_bar_images.get(frame, {})),
            )
        pose, center, scale, rotation = interpolate_keyframe_state(self._keyframes_from_editor(), frame, self._skeleton)
        previous_frames = [stored_frame for stored_frame in self._keyframe_positions if stored_frame <= frame]
        property_frame = max(previous_frames) if previous_frames else min(self._keyframe_positions)
        return (
            dict(pose),
            {
                "origin_x": center[0],
                "origin_y": center[1],
                "scale": scale,
                "rotation_degrees": rotation,
            },
            dict(self._keyframe_bar_images.get(property_frame, {})),
        )

    def _add_keyframe(self) -> None:
        frame = self._keyframe_frame()
        positions, transform, bar_images = self._preview_keyframe_state(frame)
        self._keyframe_positions[frame] = positions
        self._keyframe_transforms[frame] = transform
        self._keyframe_bar_images[frame] = bar_images
        self._load_keyframe_joint()
        self.status_var.set(f"Added keyframe {frame}")

    def _apply_current_to_later_keyframes(self) -> None:
        frame = self._keyframe_frame()
        positions, transform, bar_images = self._preview_keyframe_state(frame)
        later_frames = sorted(stored_frame for stored_frame in self._keyframe_positions if stored_frame > frame)
        if not later_frames:
            self.status_var.set(f"No keyframes after frame {frame}")
            return
        apply_transform = bool(self.apply_transform_var.get())
        apply_all_joints = bool(self.apply_all_joints_var.get())
        apply_selected_joints = bool(self.apply_selected_joints_var.get())
        apply_all_bars = bool(self.apply_all_bars_var.get())
        apply_selected_bar = bool(self.apply_selected_bar_var.get())
        if not any((apply_transform, apply_all_joints, apply_selected_joints, apply_all_bars, apply_selected_bar)):
            self.status_var.set("Select parameters before applying to later keyframes")
            return
        selected_joint_names = set(self._selected_joints) or {self.selected_joint_var.get()}
        selected_bar_name = self.selected_bar_var.get()
        for stored_frame in later_frames:
            if apply_transform:
                self._keyframe_transforms[stored_frame] = dict(transform)
            if apply_all_joints:
                self._keyframe_positions[stored_frame] = dict(positions)
            elif apply_selected_joints:
                target_positions = dict(self._keyframe_positions[stored_frame])
                for joint_name in selected_joint_names:
                    target_positions[joint_name] = positions[joint_name]
                self._keyframe_positions[stored_frame] = target_positions
            if apply_all_bars:
                self._keyframe_bar_images[stored_frame] = dict(bar_images)
            elif apply_selected_bar:
                target_bar_images = dict(self._keyframe_bar_images.get(stored_frame, {}))
                if selected_bar_name in bar_images:
                    target_bar_images[selected_bar_name] = bar_images[selected_bar_name]
                else:
                    target_bar_images.pop(selected_bar_name, None)
                self._keyframe_bar_images[stored_frame] = target_bar_images
        self._load_keyframe_joint()
        groups = []
        if apply_transform:
            groups.append("transform")
        if apply_all_joints:
            groups.append("all joints")
        elif apply_selected_joints:
            groups.append("selected joints")
        if apply_all_bars:
            groups.append("all bodies")
        elif apply_selected_bar:
            groups.append("selected body")
        self.status_var.set(f"Applied {', '.join(groups)} from frame {frame} to {len(later_frames)} later keyframes")

    def _load_keyframe_joint(self) -> None:
        frame = self._keyframe_frame()
        self.keyframe_var.set(frame)
        self._show_keyframe_in_pixel_preview(frame)
        positions, transform, bar_images = self._preview_keyframe_state(frame)
        self._load_keyframe_transform(transform)
        x, y = positions[self.selected_joint_var.get()]
        self.joint_x_var.set(x)
        self.joint_y_var.set(y)
        self._refresh_joint_table(positions)
        self._load_bar_image_path(bar_images)
        self._redraw()

    def _load_keyframe_transform(self, transform: dict[str, float]) -> None:
        self.local_x_var.set(transform["origin_x"])
        self.local_y_var.set(transform["origin_y"])
        self.local_scale_var.set(transform["scale"])
        self.local_rotation_var.set(transform["rotation_degrees"])

    def _set_keyframe_transform(self) -> None:
        frame = self._keyframe_frame()
        self.keyframe_var.set(frame)
        self._show_keyframe_in_pixel_preview(frame)
        if frame not in self._keyframe_transforms:
            self._load_keyframe_joint()
            self.status_var.set(f"Add keyframe {frame} before editing transform")
            return
        self._keyframe_transforms[frame] = {
            "origin_x": float(self.local_x_var.get()),
            "origin_y": float(self.local_y_var.get()),
            "scale": float(self.local_scale_var.get()),
            "rotation_degrees": float(self.local_rotation_var.get()),
        }
        self._redraw()

    def _refresh_joint_table(self, positions: dict[str, tuple[float, float]]) -> None:
        for row_id in self.joint_table.get_children():
            if row_id not in positions:
                self.joint_table.delete(row_id)
        for joint in self._skeleton.joints:
            x, y = positions[joint.name]
            values = (f"{x:.1f}", f"{y:.1f}")
            if self.joint_table.exists(joint.name):
                self.joint_table.item(joint.name, text=joint.name, values=values)
            else:
                self.joint_table.insert("", "end", iid=joint.name, text=joint.name, values=values)
        self.joint_table.selection_set(tuple(self._selected_joints))

    def _load_selected_bar(self) -> None:
        self.bar_visible_var.set(self.selected_bar_var.get() not in self._hidden_bar_images)
        self._load_bar_image_path()

    def _set_selected_bar_visibility(self) -> None:
        bar_name = self.selected_bar_var.get()
        if self.bar_visible_var.get():
            self._hidden_bar_images.discard(bar_name)
        else:
            self._hidden_bar_images.add(bar_name)
        self._redraw()

    def _load_bar_image_path(self, bar_images: dict[str, BarImage] | None = None) -> None:
        if bar_images is None:
            _positions, _transform, bar_images = self._preview_keyframe_state(self._keyframe_frame())
        self.bar_visible_var.set(self.selected_bar_var.get() not in self._hidden_bar_images)
        attachment = bar_images.get(self.selected_bar_var.get())
        self.bar_image_path_var.set(attachment.image_path if attachment else str(DEFAULT_BODY_PIXEL_MAP))
        self.bar_source_x_var.set(attachment.source_x if attachment else 0)
        self.bar_source_y_var.set(attachment.source_y if attachment else 0)
        self.bar_source_width_var.set(attachment.source_width if attachment else 0)
        self.bar_source_height_var.set(attachment.source_height if attachment else 0)
        self.bar_anchor_start_x_var.set(attachment.anchor_start_x if attachment else 0)
        self.bar_anchor_start_y_var.set(attachment.anchor_start_y if attachment else 0)
        self.bar_anchor_end_x_var.set(attachment.anchor_end_x if attachment else 0)
        self.bar_anchor_end_y_var.set(attachment.anchor_end_y if attachment else 0)
        self._draw_pixel_map_window()

    def _bar_source_box_for_image(self, image: Image.Image) -> tuple[int, int, int, int]:
        source_x = min(max(0, int(self.bar_source_x_var.get())), image.width)
        source_y = min(max(0, int(self.bar_source_y_var.get())), image.height)
        source_width = int(self.bar_source_width_var.get())
        source_height = int(self.bar_source_height_var.get())
        if source_width <= 0 or source_height <= 0:
            return 0, 0, image.width, image.height
        right = min(max(source_x, source_x + source_width), image.width)
        lower = min(max(source_y, source_y + source_height), image.height)
        return source_x, source_y, right, lower

    def _current_pixel_map_path(self) -> Path:
        return Path(self.bar_image_path_var.get().strip() or DEFAULT_BODY_PIXEL_MAP)

    def _open_pixel_map_window(self) -> None:
        if self._pixel_map_window is not None and self._pixel_map_window.winfo_exists():
            self._pixel_map_window.lift()
            self._draw_pixel_map_window()
            return

        window = tk.Toplevel(self)
        window.title("Pixel map cutout")
        window.minsize(640, 560)
        window.columnconfigure(0, weight=1)
        window.rowconfigure(1, weight=1)
        window.protocol("WM_DELETE_WINDOW", self._close_pixel_map_window)
        self._pixel_map_window = window

        ttk.Label(window, textvariable=self.selected_bar_var, font=("Segoe UI", 11, "bold")).grid(
            row=0, column=0, sticky="w", padx=12, pady=(12, 6)
        )
        self._pixel_map_canvas = tk.Canvas(window, width=600, height=400, bg="#fbfaf6", highlightthickness=1, highlightbackground="#cfc8b8")
        self._pixel_map_canvas.grid(row=1, column=0, sticky="nsew", padx=12)
        self._pixel_map_canvas.bind("<ButtonPress-1>", self._start_pixel_map_selection)
        self._pixel_map_canvas.bind("<B1-Motion>", self._drag_pixel_map_selection)
        self._pixel_map_canvas.bind("<ButtonRelease-1>", self._finish_pixel_map_selection)

        self._pixel_cutout_canvas = tk.Canvas(window, width=600, height=140, bg="#fbfaf6", highlightthickness=1, highlightbackground="#cfc8b8")
        self._pixel_cutout_canvas.grid(row=2, column=0, sticky="ew", padx=12, pady=12)
        self._pixel_cutout_canvas.bind("<ButtonPress-1>", self._start_pixel_cutout_anchor_drag)
        self._pixel_cutout_canvas.bind("<B1-Motion>", self._drag_pixel_cutout_anchor)
        self._pixel_cutout_canvas.bind("<ButtonRelease-1>", self._finish_pixel_cutout_anchor_drag)
        window.update_idletasks()
        self._draw_pixel_map_window()
        window.after(50, self._draw_pixel_map_window)

    def _close_pixel_map_window(self) -> None:
        if self._pixel_map_window is not None:
            self._pixel_map_window.destroy()
        self._pixel_map_window = None
        self._pixel_map_canvas = None
        self._pixel_cutout_canvas = None
        self._pixel_map_photo = None
        self._cutout_photo = None
        self._pixel_cutout_view = None
        self._pixel_selection_start = None
        self._pixel_drag_anchor = None

    def _image_point_from_pixel_canvas(self, event: tk.Event) -> tuple[int, int] | None:
        path = self._current_pixel_map_path()
        if not path.exists() or self._pixel_map_canvas is None:
            return None
        with Image.open(path) as source:
            width, height = source.size
        canvas_x = self._pixel_map_canvas.canvasx(event.x)
        canvas_y = self._pixel_map_canvas.canvasy(event.y)
        x = min(max(0, round(canvas_x / self._pixel_map_scale)), width)
        y = min(max(0, round(canvas_y / self._pixel_map_scale)), height)
        return x, y

    def _start_pixel_map_selection(self, event: tk.Event) -> None:
        point = self._image_point_from_pixel_canvas(event)
        if point is None:
            return
        self._pixel_selection_start = point

    def _drag_pixel_map_selection(self, event: tk.Event) -> None:
        if self._pixel_selection_start is None:
            return
        current = self._image_point_from_pixel_canvas(event)
        if current is None:
            return
        self._set_source_rect_from_points(self._pixel_selection_start, current)
        self._draw_pixel_map_window()

    def _finish_pixel_map_selection(self, event: tk.Event) -> None:
        if self._pixel_selection_start is None:
            return
        current = self._image_point_from_pixel_canvas(event)
        if current is not None:
            self._set_source_rect_from_points(self._pixel_selection_start, current)
            self._set_bar_image()
        self._pixel_selection_start = None

    def _image_point_from_cutout_canvas(self, event: tk.Event) -> tuple[int, int] | None:
        if self._pixel_cutout_canvas is None or self._pixel_cutout_view is None:
            return None
        left, upper, right, lower, x, y, scale_x, scale_y = self._pixel_cutout_view
        canvas_x = self._pixel_cutout_canvas.canvasx(event.x)
        canvas_y = self._pixel_cutout_canvas.canvasy(event.y)
        image_x = left + ((canvas_x - x) / scale_x)
        image_y = upper + ((canvas_y - y) / scale_y)
        return (
            min(max(left, round(image_x)), right),
            min(max(upper, round(image_y)), lower),
        )

    def _start_pixel_cutout_anchor_drag(self, event: tk.Event) -> None:
        point = self._image_point_from_cutout_canvas(event)
        if point is None:
            return
        self._pixel_drag_anchor = self._nearest_bar_anchor(point)
        self._set_dragged_bar_anchor(point)

    def _drag_pixel_cutout_anchor(self, event: tk.Event) -> None:
        if self._pixel_drag_anchor is None:
            return
        point = self._image_point_from_cutout_canvas(event)
        if point is not None:
            self._set_dragged_bar_anchor(point)

    def _finish_pixel_cutout_anchor_drag(self, event: tk.Event) -> None:
        if self._pixel_drag_anchor is None:
            return
        point = self._image_point_from_cutout_canvas(event)
        if point is not None:
            self._set_dragged_bar_anchor(point)
        self._pixel_drag_anchor = None

    def _nearest_bar_anchor(self, point: tuple[int, int]) -> str:
        start = (self.bar_anchor_start_x_var.get(), self.bar_anchor_start_y_var.get())
        end = (self.bar_anchor_end_x_var.get(), self.bar_anchor_end_y_var.get())
        start_distance = math.hypot(point[0] - start[0], point[1] - start[1])
        end_distance = math.hypot(point[0] - end[0], point[1] - end[1])
        if math.isclose(start_distance, end_distance):
            midpoint_x = self.bar_source_x_var.get() + (self.bar_source_width_var.get() / 2)
            return "start" if point[0] <= midpoint_x else "end"
        return "start" if start_distance <= end_distance else "end"

    def _set_dragged_bar_anchor(self, point: tuple[int, int]) -> None:
        if self._pixel_drag_anchor == "start":
            self.bar_anchor_start_x_var.set(point[0])
            self.bar_anchor_start_y_var.set(point[1])
        elif self._pixel_drag_anchor == "end":
            self.bar_anchor_end_x_var.set(point[0])
            self.bar_anchor_end_y_var.set(point[1])
        self._set_bar_image()

    def _set_source_rect_from_points(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        left = min(start[0], end[0])
        upper = min(start[1], end[1])
        right = max(start[0], end[0])
        lower = max(start[1], end[1])
        self.bar_source_x_var.set(left)
        self.bar_source_y_var.set(upper)
        self.bar_source_width_var.set(max(0, right - left))
        self.bar_source_height_var.set(max(0, lower - upper))

    def _draw_pixel_map_window(self) -> None:
        if self._pixel_map_window is None or not self._pixel_map_window.winfo_exists() or self._pixel_map_canvas is None:
            return
        canvas = self._pixel_map_canvas
        canvas.delete("all")
        path = self._current_pixel_map_path()
        if not path.exists():
            canvas.create_text(300, 200, text="Missing bitmap", fill="#9f3a38")
            self._draw_bar_cutout_preview()
            return

        with Image.open(path) as source:
            image = source.convert("RGBA")
        max_width = int(canvas.winfo_width())
        max_height = int(canvas.winfo_height())
        if max_width <= 1:
            max_width = int(canvas["width"])
        if max_height <= 1:
            max_height = int(canvas["height"])
        self._pixel_map_scale = min(max_width / image.width, max_height / image.height, 1.0)
        preview_size = (
            max(1, round(image.width * self._pixel_map_scale)),
            max(1, round(image.height * self._pixel_map_scale)),
        )
        preview = image.resize(preview_size, Resampling.NEAREST)
        self._pixel_map_photo = ImageTk.PhotoImage(preview)
        canvas.create_image(0, 0, anchor="nw", image=self._pixel_map_photo)
        canvas.configure(scrollregion=(0, 0, preview_size[0], preview_size[1]))

        left, upper, right, lower = self._bar_source_box_for_image(image)
        if right > left and lower > upper:
            x0 = left * self._pixel_map_scale
            y0 = upper * self._pixel_map_scale
            x1 = right * self._pixel_map_scale
            y1 = lower * self._pixel_map_scale
            canvas.create_rectangle(x0, y0, x1, y1, outline="#f39c12", width=2)
        self._draw_bar_cutout_preview()

    def _draw_bar_cutout_preview(self) -> None:
        if self._pixel_cutout_canvas is None:
            return
        canvas = self._pixel_cutout_canvas
        canvas.delete("all")
        self._pixel_cutout_view = None
        canvas_width = int(canvas.winfo_width())
        canvas_height = int(canvas.winfo_height())
        if canvas_width <= 1:
            canvas_width = int(canvas["width"])
        if canvas_height <= 1:
            canvas_height = int(canvas["height"])
        canvas.create_rectangle(0, 0, canvas_width, canvas_height, fill="#fbfaf6", outline="")

        path = self._current_pixel_map_path()
        if not path.exists():
            canvas.create_text(canvas_width / 2, canvas_height / 2, text="Missing bitmap", fill="#9f3a38")
            return

        try:
            with Image.open(path) as source:
                image = source.convert("RGBA")
                left, upper, right, lower = self._bar_source_box_for_image(image)
                if right <= left or lower <= upper:
                    canvas.create_text(canvas_width / 2, canvas_height / 2, text="Empty cutout", fill="#9f3a38")
                    return
                cutout = image.crop((left, upper, right, lower))
        except Exception as exc:
            LOGGER.exception("Cutout preview failed")
            canvas.create_text(canvas_width / 2, canvas_height / 2, text=str(exc), fill="#9f3a38", width=canvas_width - 12)
            return

        fit_scale = min((canvas_width - 12) / cutout.width, (canvas_height - 12) / cutout.height)
        preview_size = (
            max(1, round(cutout.width * fit_scale)),
            max(1, round(cutout.height * fit_scale)),
        )
        cutout = cutout.resize(preview_size, Resampling.NEAREST)
        self._cutout_photo = ImageTk.PhotoImage(cutout)
        x = (canvas_width - preview_size[0]) / 2
        y = (canvas_height - preview_size[1]) / 2
        canvas.create_image(x, y, anchor="nw", image=self._cutout_photo)
        canvas.create_rectangle(x, y, x + preview_size[0], y + preview_size[1], outline="#5d6d7e")

        cutout_scale_x = preview_size[0] / max(1, right - left)
        cutout_scale_y = preview_size[1] / max(1, lower - upper)
        self._pixel_cutout_view = (left, upper, right, lower, x, y, cutout_scale_x, cutout_scale_y)
        start_anchor = (
            x + (self.bar_anchor_start_x_var.get() - left) * cutout_scale_x,
            y + (self.bar_anchor_start_y_var.get() - upper) * cutout_scale_y,
        )
        end_anchor = (
            x + (self.bar_anchor_end_x_var.get() - left) * cutout_scale_x,
            y + (self.bar_anchor_end_y_var.get() - upper) * cutout_scale_y,
        )
        canvas.create_line(start_anchor[0], start_anchor[1], end_anchor[0], end_anchor[1], fill="#f39c12", width=2)
        self._draw_cutout_anchor(canvas, start_anchor, "#2f6f4e")
        self._draw_cutout_anchor(canvas, end_anchor, "#9f3a38")

    def _draw_cutout_anchor(self, canvas: tk.Canvas, point: tuple[float, float], color: str) -> None:
        anchor_x, anchor_y = point
        canvas.create_line(anchor_x - 7, anchor_y, anchor_x + 7, anchor_y, fill=color, width=2)
        canvas.create_line(anchor_x, anchor_y - 7, anchor_x, anchor_y + 7, fill=color, width=2)
        canvas.create_oval(anchor_x - 3, anchor_y - 3, anchor_x + 3, anchor_y + 3, fill=color, outline="#17202a")

    def _set_bar_image(self) -> None:
        self._draw_pixel_map_window()
        frame = self._keyframe_frame()
        self._show_keyframe_in_pixel_preview(frame)
        if frame not in self._keyframe_bar_images:
            self._load_keyframe_joint()
            self.status_var.set(f"Add keyframe {frame} before attaching bar images")
            return
        image_path = self.bar_image_path_var.get().strip()
        bar_name = self.selected_bar_var.get()
        if image_path:
            self._keyframe_bar_images[frame][bar_name] = BarImage(
                bar_name=bar_name,
                image_path=image_path,
                source_x=int(self.bar_source_x_var.get()),
                source_y=int(self.bar_source_y_var.get()),
                source_width=int(self.bar_source_width_var.get()),
                source_height=int(self.bar_source_height_var.get()),
                anchor_start_x=int(self.bar_anchor_start_x_var.get()),
                anchor_start_y=int(self.bar_anchor_start_y_var.get()),
                anchor_end_x=int(self.bar_anchor_end_x_var.get()),
                anchor_end_y=int(self.bar_anchor_end_y_var.get()),
            )
        else:
            self._keyframe_bar_images[frame].pop(bar_name, None)
        self._draw_pixel_map_window()
        self._redraw()

    def _browse_bar_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Attach bar image",
            initialdir=DEFAULT_BODY_PIXEL_MAP.parent,
            initialfile=DEFAULT_BODY_PIXEL_MAP.name,
            filetypes=(("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"), ("All files", "*.*")),
        )
        if not path:
            return
        self.bar_image_path_var.set(path)
        self._set_bar_image()

    def _clear_bar_image(self) -> None:
        frame = self._keyframe_frame()
        self._show_keyframe_in_pixel_preview(frame)
        if frame in self._keyframe_bar_images:
            self._keyframe_bar_images[frame].pop(self.selected_bar_var.get(), None)
        self.bar_image_path_var.set(str(DEFAULT_BODY_PIXEL_MAP))
        self.bar_source_x_var.set(0)
        self.bar_source_y_var.set(0)
        self.bar_source_width_var.set(0)
        self.bar_source_height_var.set(0)
        self.bar_anchor_start_x_var.set(0)
        self.bar_anchor_start_y_var.set(0)
        self.bar_anchor_end_x_var.set(0)
        self.bar_anchor_end_y_var.set(0)
        self._draw_pixel_map_window()
        self._redraw()

    def _set_keyframe_joint(self) -> None:
        frame = self._keyframe_frame()
        self.keyframe_var.set(frame)
        self._show_keyframe_in_pixel_preview(frame)
        if frame not in self._keyframe_positions:
            self._load_keyframe_joint()
            self.status_var.set(f"Add keyframe {frame} before editing joints")
            return
        positions = self._keyframe_positions[frame]
        positions[self.selected_joint_var.get()] = (float(self.joint_x_var.get()), float(self.joint_y_var.get()))
        self._refresh_joint_table(positions)
        self._redraw()

    def _set_keyframe_to_skeleton_default(self) -> None:
        frame = self._keyframe_frame()
        self.keyframe_var.set(frame)
        self._show_keyframe_in_pixel_preview(frame)
        self._keyframe_positions[frame] = neutral_joint_positions(self._skeleton.joints)
        self._keyframe_transforms.setdefault(frame, self._default_keyframe_transform())
        self._keyframe_bar_images.setdefault(frame, {})
        positions = self._keyframe_positions[frame]
        selected_joint = self.selected_joint_var.get()
        if selected_joint in positions:
            x, y = positions[selected_joint]
            self.joint_x_var.set(x)
            self.joint_y_var.set(y)
        self._refresh_joint_table(positions)
        self.status_var.set(f"Applied skeleton default joints to keyframe {frame}")
        self._redraw()

    def _offset_selected_joints(self) -> None:
        frame = self._keyframe_frame()
        self.keyframe_var.set(frame)
        self._show_keyframe_in_pixel_preview(frame)
        if frame not in self._keyframe_positions:
            self._load_keyframe_joint()
            self.status_var.set(f"Add keyframe {frame} before offsetting joints")
            return
        positions = self._keyframe_positions[frame]
        dx = float(self.offset_x_var.get())
        dy = float(self.offset_y_var.get())
        selected = self._selected_joints or {self.selected_joint_var.get()}
        for joint_name in selected:
            x, y = positions[joint_name]
            positions[joint_name] = (x + dx, y + dy)
        if self.selected_joint_var.get() in positions:
            x, y = positions[self.selected_joint_var.get()]
            self.joint_x_var.set(x)
            self.joint_y_var.set(y)
        self._refresh_joint_table(positions)
        self._redraw()

    def _sync_keyframes_to_frame_count(self) -> None:
        final_frame = max(0, int(self.frames_var.get()) - 1)
        synced: dict[int, dict[str, tuple[float, float]]] = {}
        synced_transforms: dict[int, dict[str, float]] = {}
        synced_bar_images: dict[int, dict[str, BarImage]] = {}
        for frame, positions in self._keyframe_positions.items():
            synced[min(frame, final_frame)] = positions
        for frame, transform in self._keyframe_transforms.items():
            synced_transforms[min(frame, final_frame)] = transform
        for frame, bar_images in self._keyframe_bar_images.items():
            synced_bar_images[min(frame, final_frame)] = dict(bar_images)
        synced.setdefault(0, neutral_joint_positions(self._skeleton.joints))
        synced.setdefault(final_frame, neutral_joint_positions(self._skeleton.joints))
        synced_transforms.setdefault(0, self._default_keyframe_transform())
        synced_transforms.setdefault(final_frame, self._default_keyframe_transform())
        synced_bar_images.setdefault(0, {})
        synced_bar_images.setdefault(final_frame, {})
        self._keyframe_positions = synced
        self._keyframe_transforms = synced_transforms
        self._keyframe_bar_images = synced_bar_images

    def _spec(self) -> MotionSpec:
        self._sync_keyframes_to_frame_count()
        final_frame = max(0, int(self.frames_var.get()) - 1)
        keyframes = tuple(keyframe for keyframe in self._keyframes_from_editor() if keyframe.frame <= final_frame)
        return MotionSpec(
            width=int(self.width_var.get()),
            height=int(self.height_var.get()),
            frames=int(self.frames_var.get()),
            duration_ms=int(self.duration_var.get()),
            fill=self.fill_var.get(),
            outline=self.outline_var.get(),
            background=self.background_var.get(),
            show_joints=bool(self.show_joints_var.get()),
            hidden_bar_images=tuple(sorted(self._hidden_bar_images)),
            keyframes=keyframes,
            skeleton=self._skeleton,
        )

    def _redraw(self) -> None:
        try:
            spec = self._spec()
            spec.validate()
            self._sync_played_frame_control(spec)
            self._draw_vector_preview(spec)
            self._draw_time_axis(spec)
            image = render_frame(spec, self._frame_index % spec.frames, bar_image_cache=self._bar_image_cache)
            zoom = self.preview_zoom_var.get()
            preview_size = (
                max(1, round(image.width * zoom)),
                max(1, round(image.height * zoom)),
            )
            image = image.resize(preview_size, Resampling.NEAREST)
            self._preview_photo = ImageTk.PhotoImage(image)
            self._draw_pixel_preview(image.width, image.height)
            self.preview_zoom_label_var.set(f"{zoom:.0%}")
            self.status_var.set(f"{spec.width}x{spec.height}, {spec.frames} frames, GIF-ready")
        except Exception as exc:
            LOGGER.exception("Preview redraw failed")
            self.status_var.set(str(exc))

    def _draw_pixel_preview(self, width: int, height: int) -> None:
        if self._preview_image_id is None:
            self._preview_image_id = self.pixel_preview.create_image(0, 0, anchor="nw", image=self._preview_photo)
        else:
            self.pixel_preview.itemconfigure(self._preview_image_id, image=self._preview_photo)
        self.pixel_preview.configure(scrollregion=(0, 0, width, height))

    def _draw_vector_preview(self, spec: MotionSpec) -> None:
        canvas = self.vector_canvas
        canvas.delete("all")
        self._vector_joint_locations = {}
        width = int(canvas["width"])
        height = int(canvas["height"])

        frame = self._keyframe_frame()
        positions, transform, bar_images = self._preview_keyframe_state(frame)
        center = (transform["origin_x"], transform["origin_y"])
        scale = transform["scale"]
        rotation = transform["rotation_degrees"]
        joints = positioned_joints(positions, self._skeleton.joints)
        vector_segments = transform_segments(skeleton_segments(joints, self._skeleton.bars), center, scale, rotation)
        vector_joints = transform_points(joint_points(joints, self._skeleton.bars), center, scale, rotation)
        flattened_points = [point for segment in vector_segments for point in segment]
        pixel_points, _ = fit_vector_points_to_pixel_field(flattened_points, width, height)
        joint_locations, _ = fit_vector_points_to_pixel_field(vector_joints, width, height)
        segments = list(zip(pixel_points[0::2], pixel_points[1::2]))
        pixel_joint_map = {joint.name: point for joint, point in zip(joints, joint_locations)}

        axis_points = transform_points(((0, 0), (40, 0), (0, 40)), center, scale, rotation)
        [origin, x_axis, y_axis], _ = fit_vector_points_to_pixel_field(axis_points, width, height)
        canvas.create_line(origin, x_axis, fill="#9f3a38", width=2, arrow=tk.LAST)
        canvas.create_line(origin, y_axis, fill="#2f6f4e", width=2, arrow=tk.LAST)
        canvas.create_oval(origin[0] - 4, origin[1] - 4, origin[0] + 4, origin[1] + 4, fill="#17202a", outline="")

        for segment_start, segment_end in segments:
            canvas.create_line(segment_start, segment_end, fill=spec.outline, width=4, capstyle=tk.ROUND)
        if bar_images:
            for bar in self._skeleton.bars:
                attachment = bar_images.get(bar.name)
                if not attachment or bar.name in spec.hidden_bar_images:
                    continue
                bar_start = pixel_joint_map[bar.start_joint]
                bar_end = pixel_joint_map[bar.end_joint]
                canvas.create_line(bar_start, bar_end, fill="#d68910", width=8, capstyle=tk.ROUND)
        for joint, (x, y) in zip(self._skeleton.joints, joint_locations):
            self._vector_joint_locations[joint.name] = (x, y)
            if spec.show_joints:
                selected = joint.name in self._selected_joints
                radius = 7 if selected else 5
                outline = "#f39c12" if selected else spec.outline
                width = 3 if selected else 2
                canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=spec.fill, outline=outline, width=width)

    def _draw_time_axis(self, spec: MotionSpec) -> None:
        canvas = self.time_axis
        canvas.delete("all")
        width = int(canvas["width"])
        height = int(canvas["height"])
        x0 = 10
        x1 = width - 10
        y = height // 2
        canvas.create_line(x0, y, x1, y, fill="#8a8173", width=2)

        frame_span = max(1, spec.frames - 1)
        current_x = x0 + (x1 - x0) * (self._keyframe_frame() / frame_span)
        canvas.create_line(current_x, 6, current_x, height - 6, fill="#2f6f4e", width=2)
        for frame in sorted(self._keyframe_positions):
            if frame > frame_span:
                continue
            x = x0 + (x1 - x0) * (frame / frame_span)
            fill = "#f39c12" if frame == self._keyframe_frame() else "#1b8f87"
            canvas.create_polygon(x, y - 7, x + 7, y, x, y + 7, x - 7, y, fill=fill, outline="#17202a")
        canvas.create_text(x0, height - 8, text="0", anchor="w", fill="#4d463d")
        canvas.create_text(x1, height - 8, text=str(spec.frames - 1), anchor="e", fill="#4d463d")

    def play(self) -> None:
        LOGGER.info("Starting preview playback")
        self.stop()
        self._animate()

    def stop(self) -> None:
        if self._after_id:
            LOGGER.info("Stopping preview playback")
            self.after_cancel(self._after_id)
            self._after_id = None

    def _animate(self) -> None:
        try:
            spec = self._spec()
            spec.validate()
            self._frame_index = (self._frame_index + 1) % spec.frames
            self._redraw()
            self._after_id = self.after(spec.duration_ms, self._animate)
        except Exception as exc:
            LOGGER.exception("Preview animation failed")
            self.status_var.set(str(exc))
            self._after_id = None

    def _set_skeleton(self, skeleton: SkeletonDefinition) -> None:
        old_neutral = neutral_joint_positions(self._skeleton.joints)
        self._skeleton = skeleton
        joint_names = tuple(joint.name for joint in skeleton.joints)
        bar_names = tuple(bar.name for bar in skeleton.bars)
        if not joint_names or not bar_names:
            raise ValueError("skeleton requires at least one joint and one bar")
        if hasattr(self, "joint_box"):
            self.joint_box.configure(values=joint_names)
        if hasattr(self, "bar_box"):
            self.bar_box.configure(values=bar_names)
        if self.selected_joint_var.get() not in joint_names:
            self.selected_joint_var.set(joint_names[0])
            self._selected_joints = {joint_names[0]}
        else:
            self._selected_joints = {name for name in self._selected_joints if name in joint_names} or {self.selected_joint_var.get()}
        if self.selected_bar_var.get() not in bar_names:
            self.selected_bar_var.set(bar_names[0])
        self._hidden_bar_images = {name for name in self._hidden_bar_images if name in bar_names}

        neutral = neutral_joint_positions(skeleton.joints)
        migrated_positions = {}
        for frame, positions in self._keyframe_positions.items():
            frame_positions = {}
            for name, neutral_point in neutral.items():
                if name not in positions:
                    frame_positions[name] = neutral_point
                    continue
                point = positions[name]
                previous_neutral = old_neutral.get(name)
                if previous_neutral is None:
                    frame_positions[name] = point
                    continue
                dx = neutral_point[0] - previous_neutral[0]
                dy = neutral_point[1] - previous_neutral[1]
                frame_positions[name] = (point[0] + dx, point[1] + dy)
            migrated_positions[frame] = frame_positions
        self._keyframe_positions = migrated_positions
        self._keyframe_bar_images = {
            frame: {name: attachment for name, attachment in bar_images.items() if name in bar_names}
            for frame, bar_images in self._keyframe_bar_images.items()
        }

    def open_skeleton_definition_window(self) -> None:
        if self._skeleton_window is not None and self._skeleton_window.winfo_exists():
            self._skeleton_window.lift()
            self._load_skeleton_editor_state()
            return

        window = tk.Toplevel(self)
        window.title("Skeleton definition")
        window.minsize(760, 520)
        window.columnconfigure((0, 1), weight=1)
        window.rowconfigure(1, weight=1)
        window.protocol("WM_DELETE_WINDOW", self._close_skeleton_definition_window)
        self._skeleton_window = window

        ttk.Label(window, text="Neutral joints", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w", padx=12, pady=(12, 6))
        ttk.Label(window, text="Bars", font=("Segoe UI", 11, "bold")).grid(row=0, column=1, sticky="w", padx=12, pady=(12, 6))

        joint_frame = ttk.Frame(window)
        joint_frame.grid(row=1, column=0, sticky="nsew", padx=(12, 6))
        joint_frame.columnconfigure(0, weight=1)
        joint_frame.rowconfigure(0, weight=1)
        self.skeleton_joint_table = ttk.Treeview(joint_frame, columns=("x", "y", "radius"), show="tree headings", height=10)
        self.skeleton_joint_table.heading("#0", text="Joint")
        self.skeleton_joint_table.heading("x", text="X")
        self.skeleton_joint_table.heading("y", text="Y")
        self.skeleton_joint_table.heading("radius", text="Radius")
        self.skeleton_joint_table.column("#0", width=140, stretch=True)
        self.skeleton_joint_table.column("x", width=70, anchor="e")
        self.skeleton_joint_table.column("y", width=70, anchor="e")
        self.skeleton_joint_table.column("radius", width=70, anchor="e")
        self.skeleton_joint_table.grid(row=0, column=0, sticky="nsew")
        self.skeleton_joint_table.bind("<<TreeviewSelect>>", self._load_selected_skeleton_joint)
        ttk.Scrollbar(joint_frame, orient="vertical", command=self.skeleton_joint_table.yview).grid(row=0, column=1, sticky="ns")
        self.skeleton_joint_table.configure(yscrollcommand=joint_frame.children["!scrollbar"].set)

        joint_edit = ttk.Frame(window)
        joint_edit.grid(row=2, column=0, sticky="ew", padx=(12, 6), pady=8)
        joint_edit.columnconfigure((1, 3), weight=1)
        ttk.Label(joint_edit, text="Name").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(joint_edit, textvariable=self.skeleton_joint_name_var, width=16).grid(row=0, column=1, sticky="ew", padx=(6, 8), pady=3)
        ttk.Label(joint_edit, text="X").grid(row=0, column=2, sticky="w", pady=3)
        ttk.Spinbox(joint_edit, from_=-1000, to=1000, increment=1, textvariable=self.skeleton_joint_x_var, width=8).grid(row=0, column=3, sticky="ew", padx=(6, 0), pady=3)
        ttk.Label(joint_edit, text="Y").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Spinbox(joint_edit, from_=-1000, to=1000, increment=1, textvariable=self.skeleton_joint_y_var, width=8).grid(row=1, column=1, sticky="ew", padx=(6, 8), pady=3)
        ttk.Label(joint_edit, text="Radius").grid(row=1, column=2, sticky="w", pady=3)
        ttk.Spinbox(joint_edit, from_=1, to=100, increment=1, textvariable=self.skeleton_joint_radius_var, width=8).grid(row=1, column=3, sticky="ew", padx=(6, 0), pady=3)
        ttk.Button(joint_edit, text="Set joint", command=self._set_skeleton_editor_joint).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(joint_edit, text="Delete joint", command=self._delete_skeleton_editor_joint).grid(row=2, column=2, columnspan=2, sticky="ew", padx=(8, 0), pady=(6, 0))

        bar_frame = ttk.Frame(window)
        bar_frame.grid(row=1, column=1, sticky="nsew", padx=(6, 12))
        bar_frame.columnconfigure(0, weight=1)
        bar_frame.rowconfigure(0, weight=1)
        self.skeleton_bar_table = ttk.Treeview(bar_frame, columns=("start", "end", "layer"), show="tree headings", height=10)
        self.skeleton_bar_table.heading("#0", text="Bar")
        self.skeleton_bar_table.heading("start", text="Start")
        self.skeleton_bar_table.heading("end", text="End")
        self.skeleton_bar_table.heading("layer", text="Layer")
        self.skeleton_bar_table.column("#0", width=150, stretch=True)
        self.skeleton_bar_table.column("start", width=120, stretch=True)
        self.skeleton_bar_table.column("end", width=120, stretch=True)
        self.skeleton_bar_table.column("layer", width=60, anchor="e")
        self.skeleton_bar_table.grid(row=0, column=0, sticky="nsew")
        self.skeleton_bar_table.bind("<<TreeviewSelect>>", self._load_selected_skeleton_bar)
        ttk.Scrollbar(bar_frame, orient="vertical", command=self.skeleton_bar_table.yview).grid(row=0, column=1, sticky="ns")
        self.skeleton_bar_table.configure(yscrollcommand=bar_frame.children["!scrollbar"].set)

        bar_edit = ttk.Frame(window)
        bar_edit.grid(row=2, column=1, sticky="ew", padx=(6, 12), pady=8)
        bar_edit.columnconfigure(1, weight=1)
        ttk.Label(bar_edit, text="Name").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(bar_edit, textvariable=self.skeleton_bar_name_var).grid(row=0, column=1, sticky="ew", padx=(6, 0), pady=3)
        ttk.Label(bar_edit, text="Start").grid(row=1, column=0, sticky="w", pady=3)
        self.skeleton_bar_start_box = ttk.Combobox(bar_edit, textvariable=self.skeleton_bar_start_var, state="readonly")
        self.skeleton_bar_start_box.grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=3)
        ttk.Label(bar_edit, text="End").grid(row=2, column=0, sticky="w", pady=3)
        self.skeleton_bar_end_box = ttk.Combobox(bar_edit, textvariable=self.skeleton_bar_end_var, state="readonly")
        self.skeleton_bar_end_box.grid(row=2, column=1, sticky="ew", padx=(6, 0), pady=3)
        ttk.Label(bar_edit, text="Layer").grid(row=3, column=0, sticky="w", pady=3)
        ttk.Spinbox(bar_edit, from_=-1000, to=1000, increment=1, textvariable=self.skeleton_bar_layer_var, width=8).grid(
            row=3, column=1, sticky="ew", padx=(6, 0), pady=3
        )
        ttk.Button(bar_edit, text="Set bar", command=self._set_skeleton_editor_bar).grid(row=4, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(bar_edit, text="Delete bar", command=self._delete_skeleton_editor_bar).grid(row=4, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))

        actions = ttk.Frame(window)
        actions.grid(row=4, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 12))
        actions.columnconfigure(0, weight=1)
        ttk.Button(actions, text="Apply skeleton", command=self._apply_skeleton_definition_window).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(actions, text="Close", command=self._close_skeleton_definition_window).grid(row=0, column=2)

        self._skeleton_axis_canvas = tk.Canvas(window, height=180, bg="#fbfaf6", highlightthickness=1, highlightbackground="#cfc8b8")
        self._skeleton_axis_canvas.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 12))
        self._load_skeleton_editor_state()

    def _close_skeleton_definition_window(self) -> None:
        if self._skeleton_window is not None:
            self._skeleton_window.destroy()
        self._skeleton_window = None
        self._skeleton_axis_canvas = None

    def _load_skeleton_editor_state(self) -> None:
        self._skeleton_editor_joints = {joint.name: joint for joint in self._skeleton.joints}
        self._skeleton_editor_bars = {bar.name: bar for bar in self._skeleton.bars}
        self._refresh_skeleton_editor_tables()

    def _refresh_skeleton_editor_tables(self) -> None:
        if self._skeleton_window is None or not self._skeleton_window.winfo_exists():
            return
        for row_id in self.skeleton_joint_table.get_children():
            self.skeleton_joint_table.delete(row_id)
        for joint in self._skeleton_editor_joints.values():
            self.skeleton_joint_table.insert("", "end", iid=joint.name, text=joint.name, values=(f"{joint.x:.1f}", f"{joint.y:.1f}", f"{joint.radius:.1f}"))
        for row_id in self.skeleton_bar_table.get_children():
            self.skeleton_bar_table.delete(row_id)
        for bar in self._skeleton_editor_bars.values():
            self.skeleton_bar_table.insert("", "end", iid=bar.name, text=bar.name, values=(bar.start_joint, bar.end_joint, bar.layer))
        joint_names = tuple(self._skeleton_editor_joints)
        self.skeleton_bar_start_box.configure(values=joint_names)
        self.skeleton_bar_end_box.configure(values=joint_names)
        self._draw_skeleton_axis_preview()

    def _draw_skeleton_axis_preview(self) -> None:
        canvas = self._skeleton_axis_canvas
        if canvas is None:
            return
        canvas.delete("all")
        width = max(1, int(canvas.winfo_width()))
        height = max(1, int(canvas.winfo_height()))
        if width <= 1:
            width = int(canvas["width"]) if "width" in canvas.keys() else 720
        if height <= 1:
            height = int(canvas["height"])

        joints = tuple(self._skeleton_editor_joints.values())
        bars = tuple(self._skeleton_editor_bars.values())
        points = [joint.point for joint in joints] + [(0, 0), (80, 0), (0, 80), (-80, 0), (0, -80)]
        if not points:
            return
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(1.0, max_x - min_x)
        span_y = max(1.0, max_y - min_y)
        padding = 18
        scale = min((width - padding * 2) / span_x, (height - padding * 2) / span_y)

        def map_point(point: tuple[float, float]) -> tuple[float, float]:
            return (
                padding + (point[0] - min_x) * scale,
                padding + (point[1] - min_y) * scale,
            )

        origin = map_point((0, 0))
        x_axis = map_point((80, 0))
        y_axis = map_point((0, 80))
        canvas.create_line(origin, x_axis, fill="#9f3a38", width=2, arrow=tk.LAST)
        canvas.create_line(origin, y_axis, fill="#2f6f4e", width=2, arrow=tk.LAST)
        canvas.create_text(x_axis[0] + 8, x_axis[1], text="X", anchor="w", fill="#9f3a38")
        canvas.create_text(y_axis[0], y_axis[1] + 8, text="Y", anchor="n", fill="#2f6f4e")
        canvas.create_oval(origin[0] - 3, origin[1] - 3, origin[0] + 3, origin[1] + 3, fill="#17202a", outline="")

        mapped = {joint.name: map_point(joint.point) for joint in joints}
        for bar in sorted(bars, key=lambda item: item.layer):
            if bar.start_joint in mapped and bar.end_joint in mapped:
                start = mapped[bar.start_joint]
                end = mapped[bar.end_joint]
                layer_color = "#8a8173" if bar.layer < 0 else "#5d6d7e" if bar.layer == 0 else "#d68910"
                canvas.create_line(start, end, fill=layer_color, width=3, capstyle=tk.ROUND)
        selected_joint = self.skeleton_joint_table.selection()[0] if self.skeleton_joint_table.selection() else ""
        for joint in joints:
            x, y = mapped[joint.name]
            selected = joint.name == selected_joint
            radius = 5 if selected else 3
            fill = "#f39c12" if selected else "#1b8f87"
            canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=fill, outline="#17202a")
            canvas.create_text(x + 6, y, text=joint.name, anchor="w", fill="#4d463d")

    def _load_selected_skeleton_joint(self, _event: tk.Event) -> None:
        selected = self.skeleton_joint_table.selection()
        if not selected:
            return
        joint = self._skeleton_editor_joints[selected[0]]
        self.skeleton_joint_name_var.set(joint.name)
        self.skeleton_joint_x_var.set(joint.x)
        self.skeleton_joint_y_var.set(joint.y)
        self.skeleton_joint_radius_var.set(joint.radius)
        self._draw_skeleton_axis_preview()

    def _load_selected_skeleton_bar(self, _event: tk.Event) -> None:
        selected = self.skeleton_bar_table.selection()
        if not selected:
            return
        bar = self._skeleton_editor_bars[selected[0]]
        self.skeleton_bar_name_var.set(bar.name)
        self.skeleton_bar_start_var.set(bar.start_joint)
        self.skeleton_bar_end_var.set(bar.end_joint)
        self.skeleton_bar_layer_var.set(bar.layer)
        self._draw_skeleton_axis_preview()

    def _set_skeleton_editor_joint(self) -> None:
        name = self.skeleton_joint_name_var.get().strip()
        if not name:
            self.status_var.set("Skeleton joint requires a name")
            return
        selected = self.skeleton_joint_table.selection()
        old_name = selected[0] if selected and selected[0] != name else None
        joint = Joint(name, float(self.skeleton_joint_x_var.get()), float(self.skeleton_joint_y_var.get()), float(self.skeleton_joint_radius_var.get()))
        if old_name:
            self._skeleton_editor_joints.pop(old_name, None)
            self._skeleton_editor_bars = {
                bar_name: Bar(
                    bar.name,
                    name if bar.start_joint == old_name else bar.start_joint,
                    name if bar.end_joint == old_name else bar.end_joint,
                    bar.layer,
                )
                for bar_name, bar in self._skeleton_editor_bars.items()
            }
        self._skeleton_editor_joints[name] = joint
        self._refresh_skeleton_editor_tables()
        self.skeleton_joint_table.selection_set(name)

    def _delete_skeleton_editor_joint(self) -> None:
        name = self.skeleton_joint_name_var.get().strip()
        if not name:
            return
        self._skeleton_editor_joints.pop(name, None)
        self._skeleton_editor_bars = {
            bar_name: bar
            for bar_name, bar in self._skeleton_editor_bars.items()
            if bar.start_joint != name and bar.end_joint != name
        }
        self._refresh_skeleton_editor_tables()

    def _set_skeleton_editor_bar(self) -> None:
        name = self.skeleton_bar_name_var.get().strip()
        start = self.skeleton_bar_start_var.get().strip()
        end = self.skeleton_bar_end_var.get().strip()
        if not name or not start or not end:
            self.status_var.set("Skeleton bar requires name, start, and end")
            return
        self._skeleton_editor_bars[name] = Bar(name, start, end, int(self.skeleton_bar_layer_var.get()))
        self._refresh_skeleton_editor_tables()
        self.skeleton_bar_table.selection_set(name)

    def _delete_skeleton_editor_bar(self) -> None:
        name = self.skeleton_bar_name_var.get().strip()
        if not name:
            return
        self._skeleton_editor_bars.pop(name, None)
        self._refresh_skeleton_editor_tables()

    def _apply_skeleton_definition_window(self) -> None:
        try:
            joints = tuple(self._skeleton_editor_joints.values())
            bars = tuple(self._skeleton_editor_bars.values())
            existing_rigid = {bar.name: bar for bar in self._skeleton.rigid_hierarchy()}
            rigid_bars = []
            for bar in bars:
                rigid = existing_rigid.get(bar.name)
                if rigid and {rigid.start_joint, rigid.end_joint} == {bar.start_joint, bar.end_joint}:
                    rigid_bars.append(rigid)
                else:
                    rigid_bars.append(bar)
            self._set_skeleton(SkeletonDefinition(joints, bars, tuple(rigid_bars)))
            self._load_keyframe_joint()
            self.status_var.set("Applied skeleton definition")
        except Exception as exc:
            LOGGER.exception("Skeleton definition apply failed")
            messagebox.showerror("Skeleton definition failed", str(exc))

    def import_skeleton(self) -> None:
        try:
            path = filedialog.askopenfilename(
                title="Import skeleton definition",
                initialdir=DEFAULT_SKELETON_RECORD.parent,
                initialfile=DEFAULT_SKELETON_RECORD.name,
                filetypes=(("Skeleton JSON", "*.json"), ("All files", "*.*")),
            )
            if not path:
                return
            self._skeleton_path = Path(path)
            self._set_skeleton(load_skeleton(path))
            self._load_keyframe_joint()
            self.status_var.set(f"Imported skeleton {path}")
        except Exception as exc:
            LOGGER.exception("Skeleton import failed")
            messagebox.showerror("Skeleton import failed", str(exc))

    def export_skeleton(self) -> None:
        try:
            path = filedialog.asksaveasfilename(
                title="Export skeleton definition",
                defaultextension=".json",
                initialdir=DEFAULT_SKELETON_RECORD.parent,
                initialfile=DEFAULT_SKELETON_RECORD.name,
                filetypes=(("Skeleton JSON", "*.json"),),
            )
            if not path:
                return
            saved = save_skeleton(self._skeleton, path)
            self._skeleton_path = saved
            self.status_var.set(f"Exported skeleton {saved}")
        except Exception as exc:
            LOGGER.exception("Skeleton export failed")
            messagebox.showerror("Skeleton export failed", str(exc))

    def refresh_skeleton(self) -> None:
        if self._skeleton_path is None:
            self.status_var.set("No skeleton file selected")
            return
        try:
            self._set_skeleton(load_skeleton(self._skeleton_path))
            self._load_keyframe_joint()
            self.status_var.set(f"Refreshed skeleton {self._skeleton_path}")
        except Exception as exc:
            LOGGER.exception("Skeleton refresh failed")
            messagebox.showerror("Skeleton refresh failed", str(exc))

    def _apply_spec(self, spec: MotionSpec) -> None:
        self._set_skeleton(spec.skeleton)
        self.width_var.set(spec.width)
        self.height_var.set(spec.height)
        self.frames_var.set(spec.frames)
        self.duration_var.set(spec.duration_ms)
        self.fill_var.set(spec.fill)
        self.outline_var.set(spec.outline)
        self.background_var.set(spec.background)
        self.show_joints_var.set(spec.show_joints)
        defined_bar_names = {bar.name for bar in self._skeleton.bars}
        self._hidden_bar_images = {name for name in spec.hidden_bar_images if name in defined_bar_names}
        self._keyframe_positions = {
            keyframe.frame: keyframe_position_map(keyframe)
            for keyframe in spec.keyframes
        }
        self._keyframe_transforms = {
            keyframe.frame: {
                "origin_x": keyframe.origin_x,
                "origin_y": keyframe.origin_y,
                "scale": keyframe.scale,
                "rotation_degrees": keyframe.rotation_degrees,
            }
            for keyframe in spec.keyframes
        }
        self._keyframe_bar_images = {
            keyframe.frame: keyframe_bar_image_map(keyframe)
            for keyframe in spec.keyframes
        }
        self._sync_keyframes_to_frame_count()
        self.keyframe_var.set(0)
        self._frame_index = 0
        self._load_keyframe_joint()

    def import_record(self) -> None:
        try:
            path = filedialog.askopenfilename(
                title="Import animation record",
                initialdir=DEFAULT_ANIMATION_RECORD.parent,
                initialfile=DEFAULT_ANIMATION_RECORD.name,
                filetypes=(("JSON record", "*.json"), ("All files", "*.*")),
            )
            if not path:
                LOGGER.info("Record import cancelled")
                return
            spec = load_record(path)
            self._record_path = Path(path)
            self._apply_spec(spec)
            self.status_var.set(f"Imported {path}")
        except Exception as exc:
            LOGGER.exception("Record import failed")
            messagebox.showerror("Import failed", str(exc))

    def export_record(self) -> None:
        try:
            spec = self._spec()
            default_path = Path.cwd() / "release" / "project_pv_animation.json"
            path = filedialog.asksaveasfilename(
                title="Export animation record",
                defaultextension=".json",
                initialfile=default_path.name,
                initialdir=default_path.parent,
                filetypes=(("JSON record", "*.json"),),
            )
            if not path:
                LOGGER.info("Record export cancelled")
                return
            saved = save_record(spec, path)
            self._record_path = saved
            self.status_var.set(f"Exported {saved}")
        except Exception as exc:
            LOGGER.exception("Record export failed")
            messagebox.showerror("Export failed", str(exc))

    def export_gif(self) -> None:
        try:
            spec = self._spec()
            if self._record_path is None:
                default_path = Path.cwd() / "release" / "project_pv_animation.gif"
            else:
                default_path = self._record_path.with_suffix(".gif")
            path = filedialog.asksaveasfilename(
                title="Export GIF",
                defaultextension=".gif",
                initialfile=default_path.name,
                initialdir=default_path.parent,
                filetypes=(("GIF image", "*.gif"),),
            )
            if not path:
                LOGGER.info("GIF export cancelled")
                return
            LOGGER.info("Exporting GIF to %s", path)
            saved = save_gif(spec, path)
            self.status_var.set(f"Exported {saved}")
        except Exception as exc:
            LOGGER.exception("GIF export failed")
            messagebox.showerror("Export failed", str(exc))


def run_app() -> None:
    """Start the Project PV UI."""

    LOGGER.info("Starting Tk main loop")
    app = ProjectPVApp()
    app.mainloop()
    LOGGER.info("Tk main loop closed")
