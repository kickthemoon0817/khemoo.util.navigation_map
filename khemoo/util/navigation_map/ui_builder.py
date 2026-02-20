from __future__ import annotations

import os
from typing import Callable

import omni.ui as ui
import omni.usd
from isaacsim.gui.components.style import get_style
from isaacsim.gui.components.ui_utils import (
    btn_builder,
    cb_builder,
    dropdown_builder,
    float_builder,
    multi_btn_builder,
    progress_bar_builder,
    str_builder,
    xyz_builder,
)
from pxr import Gf, Usd, UsdGeom

# Output type choices for the dropdown
OUTPUT_TYPE_IMAGE: str = "Aerial Photo"
OUTPUT_TYPE_NAV_MAP: str = "Navigation Map"
OUTPUT_TYPE_BOTH: str = "Both"
OUTPUT_TYPES: tuple[str, ...] = (OUTPUT_TYPE_IMAGE, OUTPUT_TYPE_NAV_MAP, OUTPUT_TYPE_BOTH)

# Status severity levels
STATUS_READY: int = 0
STATUS_INFO: int = 1
STATUS_WARNING: int = 2
STATUS_ERROR: int = 3
STATUS_SUCCESS: int = 4

# ARGB color codes for status bar messages
_STATUS_COLORS: dict[int, int] = {
    STATUS_READY: 0xFF999999,    # gray
    STATUS_INFO: 0xFFCCCCCC,     # light gray
    STATUS_WARNING: 0xFFFFAA00,  # amber
    STATUS_ERROR: 0xFFFF4444,    # red
    STATUS_SUCCESS: 0xFF44BB44,  # green
}


class NavigationMapUIBuilder:
    """
    Builds the unified Omni.UI panel for the Navigation Map Generator extension.

    Provides a shared Area Definition section (origin, bounds, cell size,
    positioning helpers) used by both the orthographic capture and the
    occupancy map generation workflows.  Separate action buttons let the
    user trigger each operation independently.

    This class owns all UI widget models and layout logic.  It does not
    perform any capture or generation itself — those are delegated to
    callbacks provided by the extension.
    """

    def __init__(self) -> None:
        self._models: dict[str, ui.AbstractValueModel] = {}
        self._prev_origin: tuple[float, float] = (0.0, 0.0)
        self._lower_bound: tuple[float, float] = (-1.0, -1.0)
        self._upper_bound: tuple[float, float] = (1.0, 1.0)
        self._wait_bound_update: bool = False
        self._bound_update_case: int = 0
        self._exclude_prim_paths: list[str] = []
        self._exclude_list_container: ui.VStack | None = None
        self._om: object | None = None
        self._status_label: ui.Label | None = None
        self._progress_model: ui.AbstractValueModel | None = None
        self._output_type_index: int = 2
        self._generate_btn: ui.Button | None = None

    @property
    def models(self) -> dict[str, ui.AbstractValueModel]:
        """Direct access to the UI value models keyed by field name."""
        return self._models

    @property
    def selected_output_type(self) -> str:
        """The currently selected output type from the dropdown."""
        return OUTPUT_TYPES[self._output_type_index]

    def build(
        self,
        frame: ui.Frame,
        omap_interface: object | None,
        on_generate: Callable[[], None],
        on_open_folder: Callable[[], None],
    ) -> None:
        """
        Construct the full UI inside the given frame.

        Args:
            frame: The parent UI frame to build widgets into.
            omap_interface: The OccupancyMap singleton for viewport visualization.
            on_generate: Callback for the single "Generate" button.
            on_open_folder: Callback to open the output directory in a file browser.
        """
        self._om = omap_interface
        with frame:
            with ui.VStack(spacing=5, height=0, style=get_style()):
                self._build_area_section()
                self._build_settings_section()
                self._build_exclusion_section()
                self._build_output_section()

                ui.Spacer(height=10)
                self._generate_btn = btn_builder(
                    label="",
                    text="GENERATE",
                    tooltip="Run the selected output workflow",
                    on_clicked_fn=on_generate,
                )
                ui.Spacer(height=5)
                self._progress_model = progress_bar_builder(
                    label="Progress",
                    default_val=0,
                    tooltip="Overall generation progress",
                )
                ui.Spacer(height=5)
                self._status_label = ui.Label(
                    "Ready",
                    word_wrap=True,
                    height=0,
                    style={"color": _STATUS_COLORS[STATUS_READY]},
                )
                ui.Spacer(height=5)
                btn_builder(
                    label="",
                    text="Open Output Folder",
                    tooltip="Open the output directory in your file browser",
                    on_clicked_fn=on_open_folder,
                )

    # ------------------------------------------------------------------
    # Public getters — Area Definition
    # ------------------------------------------------------------------

    def get_origin(self) -> tuple[float, float, float]:
        """
        Read the origin XYZ from the UI.

        Returns:
            Tuple of (x, y, z).
        """
        return (
            self._models["origin"][0].get_value_as_float(),
            self._models["origin"][1].get_value_as_float(),
            self._models["origin"][2].get_value_as_float(),
        )

    def get_lower_bound(self) -> tuple[float, float, float]:
        """
        Read the lower bound XYZ from the UI.

        Returns:
            Tuple of (x, y, z).
        """
        return (
            self._lower_bound[0],
            self._lower_bound[1],
            self._models["lower_bound"][2].get_value_as_float(),
        )

    def get_upper_bound(self) -> tuple[float, float, float]:
        """
        Read the upper bound XYZ from the UI.

        Returns:
            Tuple of (x, y, z).
        """
        return (
            self._upper_bound[0],
            self._upper_bound[1],
            self._models["upper_bound"][2].get_value_as_float(),
        )

    def get_boundary_values(self) -> tuple[float, float, float, float]:
        """
        Derive ortho-capture boundary from origin + bounds.

        Returns:
            Tuple of (x_min, x_max, y_min, y_max) in world coordinates.
        """
        ox, oy, _ = self.get_origin()
        lb = self.get_lower_bound()
        ub = self.get_upper_bound()
        return (ox + lb[0], ox + ub[0], oy + lb[1], oy + ub[1])

    def get_cell_size(self) -> float:
        """Read the cell size from the UI."""
        return self._models["cell_size"].get_value_as_float()

    def get_camera_height(self) -> float:
        """Read the camera Z height from the UI."""
        return self._models["z_height"].get_value_as_float()

    def get_meters_per_pixel(self) -> float:
        """Read the meters-per-pixel resolution from the UI."""
        return self._models["meters_per_pixel"].get_value_as_float()

    def get_camera_path(self) -> str:
        """Read the USD camera prim path from the UI."""
        return self._models["camera_path"].get_value_as_string()

    def get_output_directory(self) -> str:
        """Read the output directory path from the UI."""
        return self._models["output_dir"].get_value_as_string()

    def get_use_physx_geometry(self) -> bool:
        """Read the PhysX geometry checkbox state from the UI."""
        return self._models["physx_geom"].get_value_as_bool()

    def get_exclude_prim_paths(self) -> tuple[str, ...]:
        """
        Read the prim path exclusion list.

        Returns:
            Tuple of prim path strings to exclude from occupancy map generation.
        """
        return tuple(self._exclude_prim_paths)

    def get_max_traversable_slope_degrees(self) -> float:
        """Read the max traversable slope angle from the UI (degrees)."""
        return self._models["max_slope"].get_value_as_float()

    # ------------------------------------------------------------------
    # Public setters — UI state control
    # ------------------------------------------------------------------

    def set_status(self, message: str, severity: int = STATUS_INFO) -> None:
        """
        Update the status label text and color.

        Args:
            message: Status message to display.
            severity: One of STATUS_READY, STATUS_INFO, STATUS_WARNING,
                STATUS_ERROR, or STATUS_SUCCESS.
        """
        if self._status_label is None:
            return
        self._status_label.text = message
        color = _STATUS_COLORS.get(severity, _STATUS_COLORS[STATUS_INFO])
        self._status_label.style = {"color": color}

    def set_progress(self, fraction: float) -> None:
        """
        Update the progress bar value.

        Args:
            fraction: Progress value between 0.0 and 1.0.
        """
        if self._progress_model is None:
            return
        self._progress_model.set_value(max(0.0, min(1.0, fraction)))

    def set_generate_enabled(self, enabled: bool) -> None:
        """
        Enable or disable the Generate button.

        Args:
            enabled: Whether the button should be clickable.
        """
        if self._generate_btn is not None:
            self._generate_btn.enabled = enabled


    # ------------------------------------------------------------------
    # UI section builders
    # ------------------------------------------------------------------

    def _build_area_section(self) -> None:
        """Build the Step 1 — Define Map Area section."""
        with ui.CollapsableFrame(
            title="Step 1 \u2014 Define Map Area", style=get_style(), collapsed=False,
        ):
            with ui.VStack(spacing=2, height=0):
                self._models["origin"] = xyz_builder(
                    label="Origin",
                    tooltip="Center point of the map area in world coordinates",
                    on_value_changed_fn=[
                        self._on_area_value_changed,
                        self._on_area_value_changed,
                        self._on_area_value_changed,
                    ],
                )
                self._models["lower_bound"] = xyz_builder(
                    label="Area Min",
                    tooltip="Lower-left corner of the map area, relative to Origin",
                    default_val=[self._lower_bound[0], self._lower_bound[1], 0.0],
                    on_value_changed_fn=[
                        self._on_area_value_changed,
                        self._on_area_value_changed,
                        self._on_area_value_changed,
                    ],
                )
                self._models["upper_bound"] = xyz_builder(
                    label="Area Max",
                    tooltip="Upper-right corner of the map area, relative to Origin",
                    default_val=[self._upper_bound[0], self._upper_bound[1], 0.0],
                    on_value_changed_fn=[
                        self._on_area_value_changed,
                        self._on_area_value_changed,
                        self._on_area_value_changed,
                    ],
                )
                self._models["center_bound"] = multi_btn_builder(
                    "Quick Select",
                    text=["Fit to Selection", "Set Area from Selection"],
                    tooltip=[
                        "Use viewport selection to position the map area",
                        "Move the origin to the center of selected objects",
                        "Set area boundaries from the bounding box of selected objects",
                    ],
                    on_clicked_fn=[self._on_center_selection, self._on_bound_selection],
                )
                self._models["cell_size"] = float_builder(
                    label="Map Detail",
                    default_val=0.05,
                    min=0.001,
                    step=0.001,
                    format="%.3f",
                    tooltip=(
                        "Grid cell size in meters \u2014 smaller values produce a finer map "
                        "but take longer to generate. Default 0.05 m works well for "
                        "indoor environments."
                    ),
                )
                self._models["cell_size"].add_value_changed_fn(self._on_cell_size_changed)

    def _build_settings_section(self) -> None:
        """Build the Step 2 — Settings section with common and advanced options."""
        with ui.CollapsableFrame(
            title="Step 2 \u2014 Settings", style=get_style(), collapsed=True,
        ):
            with ui.VStack(spacing=2, height=0):
                self._models["z_height"] = float_builder(
                    "Camera Height",
                    default_val=50.0,
                    tooltip=(
                        "Height of the overhead camera in meters. Set higher than "
                        "the tallest object in your scene."
                    ),
                )
                self._models["meters_per_pixel"] = float_builder(
                    "Resolution (m/px)",
                    default_val=0.05,
                    tooltip=(
                        "Meters per pixel in the aerial photo. Smaller values give "
                        "sharper images but produce larger files and take longer."
                    ),
                )
                self._models["max_slope"] = float_builder(
                    "Slope Limit (\u00b0)",
                    default_val=0.0,
                    tooltip=(
                        "Maximum ground slope in degrees that a robot can traverse. "
                        "Steeper surfaces are marked as obstacles. Set to 0 to disable."
                    ),
                    min=0.0,
                    max=90.0,
                    step=1.0,
                    format="%.1f",
                )
                with ui.CollapsableFrame(
                    title="Advanced", style=get_style(), collapsed=True,
                ):
                    with ui.VStack(spacing=2, height=0):
                        self._models["camera_path"] = str_builder(
                            "Camera Path",
                            default_val="/World/OrthoCamera",
                            tooltip="USD prim path for the overhead camera (internal)",
                        )
                        self._models["physx_geom"] = cb_builder(
                            "Use Collision Geometry",
                            tooltip=(
                                "When enabled, uses the physics collision shapes instead "
                                "of visual meshes. Leave enabled unless you see missing "
                                "obstacles in the output."
                            ),
                            on_clicked_fn=None,
                            default_val=True,
                        )

    def _build_exclusion_section(self) -> None:
        """Build the Ignore Objects section."""
        with ui.CollapsableFrame(
            title="Ignore Objects", style=get_style(), collapsed=True,
        ):
            with ui.VStack(spacing=4, height=0):
                ui.Label(
                    "Objects listed here (and their children) will be hidden "
                    "during map generation.",
                    word_wrap=True,
                    height=0,
                )
                with ui.ScrollingFrame(height=120):
                    self._exclude_list_container = ui.VStack(spacing=1, height=0)
                self._rebuild_exclusion_list_ui()
                multi_btn_builder(
                    "Actions",
                    count=3,
                    text=["Add from Selection", "Remove Checked", "Clear All"],
                    tooltip=[
                        "Manage the ignore list",
                        "Add the currently selected viewport objects",
                        "Remove checked items from the list",
                        "Remove all items from the list",
                    ],
                    on_clicked_fn=[
                        self._on_add_exclusion_from_selection,
                        self._on_remove_selected_exclusion,
                        self._on_clear_exclusion_list,
                    ],
                )

    def _build_output_section(self) -> None:
        """Build the Step 3 — Output section with output type and directory."""
        with ui.CollapsableFrame(
            title="Step 3 \u2014 Output", style=get_style(), collapsed=False,
        ):
            with ui.VStack(spacing=2, height=0):
                dropdown_builder(
                    label="Output Type",
                    default_val=self._output_type_index,
                    items=list(OUTPUT_TYPES),
                    tooltip="Choose what to generate",
                    on_clicked_fn=self._on_output_type_changed,
                )
                self._models["output_dir"] = str_builder(
                    "Output Directory",
                    default_val=os.path.expanduser("~/navigation_maps"),
                    tooltip="Folder where generated files will be saved",
                    use_folder_picker=True,
                )

    # ------------------------------------------------------------------
    # Exclusion list callbacks
    # ------------------------------------------------------------------

    def _rebuild_exclusion_list_ui(self) -> None:
        """Clear and re-populate the exclusion list VStack with current paths."""
        if self._exclude_list_container is None:
            return
        self._exclude_list_container.clear()
        with self._exclude_list_container:
            if not self._exclude_prim_paths:
                ui.Label("  (empty)", height=20, style={"color": 0xFF888888})
            else:
                for idx, path in enumerate(self._exclude_prim_paths):
                    with ui.HStack(height=20, spacing=4):
                        cb = ui.CheckBox(width=16, name=f"excl_cb_{idx}")
                        cb.model.set_value(False)
                        ui.Label(path, word_wrap=False)

    def _on_add_exclusion_from_selection(self) -> None:
        """Add currently selected stage prims to the exclusion list."""
        selected: list[str] = list(
            omni.usd.get_context().get_selection().get_selected_prim_paths()
        )
        if not selected:
            return
        changed = False
        for prim_path in selected:
            if prim_path not in self._exclude_prim_paths:
                self._exclude_prim_paths.append(prim_path)
                changed = True
        if changed:
            self._rebuild_exclusion_list_ui()

    def _on_remove_selected_exclusion(self) -> None:
        """Remove checked items from the exclusion list."""
        if self._exclude_list_container is None:
            return
        indices_to_remove: list[int] = []
        for idx, child in enumerate(self._exclude_list_container.get_children()):
            hstack_children = child.get_children()
            if hstack_children:
                checkbox = hstack_children[0]
                if hasattr(checkbox, "model") and checkbox.model.get_value_as_bool():
                    indices_to_remove.append(idx)
        for idx in reversed(indices_to_remove):
            if idx < len(self._exclude_prim_paths):
                self._exclude_prim_paths.pop(idx)
        self._rebuild_exclusion_list_ui()

    def _on_clear_exclusion_list(self) -> None:
        """Clear the entire exclusion list."""
        self._exclude_prim_paths.clear()
        self._rebuild_exclusion_list_ui()

    def _on_output_type_changed(self, index: int) -> None:
        """Handle output type dropdown selection change."""
        self._output_type_index = index

    # ------------------------------------------------------------------
    # Positioning callbacks (ported from omap UI)
    # ------------------------------------------------------------------

    def _on_area_value_changed(self, _value: float) -> None:
        """Sync internal bound tracking when any area field changes."""
        lb_x = self._models["lower_bound"][0].get_value_as_float()
        lb_y = self._models["lower_bound"][1].get_value_as_float()
        ub_x = self._models["upper_bound"][0].get_value_as_float()
        ub_y = self._models["upper_bound"][1].get_value_as_float()

        if lb_x >= ub_x or lb_y >= ub_y:
            return

        if self._wait_bound_update:
            if self._bound_update_case == 0:
                self._lower_bound = (lb_x, self._lower_bound[1])
            elif self._bound_update_case == 1:
                self._lower_bound = (self._lower_bound[0], lb_y)
            elif self._bound_update_case == 2:
                self._upper_bound = (ub_x, self._upper_bound[1])
            elif self._bound_update_case == 3:
                self._upper_bound = (self._upper_bound[0], ub_y)
        else:
            self._lower_bound = (lb_x, lb_y)
            self._upper_bound = (ub_x, ub_y)

        self._update_viewport_visualization()

    def _on_cell_size_changed(self, _value: float) -> None:
        """Sync cell size to the omap interface for viewport grid rendering."""
        if self._om is not None:
            self._om.set_cell_size(self._models["cell_size"].get_value_as_float())

    def _update_viewport_visualization(self) -> None:
        """
        Push the current origin / bounds to the omap singleton so the
        bounding-box, grid and coordinate axes are drawn in the viewport.
        """
        if self._om is None:
            return
        origin = self.get_origin()
        lower = self.get_lower_bound()
        upper = self.get_upper_bound()
        self._om.set_transform(origin, lower, upper)
        self._om.update()

    def _on_center_selection(self) -> None:
        """Center the origin on the selected prims and adjust bounds to match."""
        origin = self._calculate_bounds(origin_calc=True, stationary_bounds=True)
        self._models["origin"][0].set_value(origin[0])
        self._models["origin"][1].set_value(origin[1])

        result = self._calculate_bounds(origin_calc=False, stationary_bounds=True)
        self._lower_bound, self._upper_bound = result[0], result[1]
        self._set_bound_values_in_ui()

    def _on_bound_selection(self) -> None:
        """Set bounds from the bounding box of selected prims."""
        result = self._calculate_bounds(origin_calc=False, stationary_bounds=False)
        self._lower_bound, self._upper_bound = result[0], result[1]
        self._set_bound_values_in_ui()

    def _set_bound_values_in_ui(self) -> None:
        """Push internal bound values into the UI widgets with change-guard."""
        self._wait_bound_update = True
        self._bound_update_case = 0
        self._models["lower_bound"][0].set_value(self._lower_bound[0])
        self._bound_update_case += 1
        self._models["lower_bound"][1].set_value(self._lower_bound[1])
        self._bound_update_case += 1
        self._models["upper_bound"][0].set_value(self._upper_bound[0])
        self._bound_update_case += 1
        self._models["upper_bound"][1].set_value(self._upper_bound[1])
        self._wait_bound_update = False

    def _calculate_bounds(
        self, origin_calc: bool, stationary_bounds: bool,
    ) -> tuple[float, float] | tuple[tuple[float, float], tuple[float, float]]:
        """
        Compute origin or bounds from the current prim selection.

        Args:
            origin_calc: If True, return the midpoint as the new origin.
            stationary_bounds: If True, adjust bounds relative to origin shift.

        Returns:
            Origin as (x, y) when origin_calc is True, otherwise a tuple
            of (lower_bound, upper_bound) each as (x, y).
        """
        origin_xy: tuple[float, float] = (
            self._models["origin"][0].get_value_as_float(),
            self._models["origin"][1].get_value_as_float(),
        )

        if not origin_calc and stationary_bounds:
            lower: tuple[float, float] = (
                self._lower_bound[0] + self._prev_origin[0] - origin_xy[0],
                self._lower_bound[1] + self._prev_origin[1] - origin_xy[1],
            )
            upper: tuple[float, float] = (
                self._upper_bound[0] + self._prev_origin[0] - origin_xy[0],
                self._upper_bound[1] + self._prev_origin[1] - origin_xy[1],
            )
            return lower, upper

        selected_paths = omni.usd.get_context().get_selection().get_selected_prim_paths()
        stage = omni.usd.get_context().get_stage()
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        bbox_cache.Clear()
        total_bbox = Gf.BBox3d()

        if len(selected_paths) > 0:
            for prim_path in selected_paths:
                prim = stage.GetPrimAtPath(prim_path)
                bounds = bbox_cache.ComputeWorldBound(prim)
                total_bbox = Gf.BBox3d.Combine(total_bbox, Gf.BBox3d(bounds.ComputeAlignedRange()))
            box_range = total_bbox.GetBox()

            if origin_calc:
                mid = box_range.GetMidpoint()
                self._prev_origin = origin_xy
                return (mid[0], mid[1])

            min_pt = box_range.GetMin()
            max_pt = box_range.GetMax()
            lower = (min_pt[0] - origin_xy[0], min_pt[1] - origin_xy[1])
            upper = (max_pt[0] - origin_xy[0], max_pt[1] - origin_xy[1])
            return lower, upper

        if origin_calc:
            return (0.0, 0.0)
        return (0.0, 0.0), (0.0, 0.0)
