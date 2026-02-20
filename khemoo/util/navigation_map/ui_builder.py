from __future__ import annotations

import os
from typing import Callable

import omni.ui as ui
import omni.usd
from isaacsim.gui.components.style import get_style
from isaacsim.gui.components.ui_utils import (
    btn_builder,
    cb_builder,
    float_builder,
    multi_btn_builder,
    str_builder,
    xyz_builder,
)
from pxr import Gf, Usd, UsdGeom


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
        self._prev_origin: list[float] = [0.0, 0.0]
        self._lower_bound: list[float] = [-1.0, -1.0]
        self._upper_bound: list[float] = [1.0, 1.0]
        self._wait_bound_update: bool = False
        self._bound_update_case: int = 0
        self._exclude_prim_paths: list[str] = []
        self._exclude_list_container: ui.VStack | None = None
        self._om: object | None = None

    @property
    def models(self) -> dict[str, ui.AbstractValueModel]:
        """Direct access to the UI value models keyed by field name."""
        return self._models

    def build(
        self,
        frame: ui.Frame,
        omap_interface: object | None,
        on_create_camera: Callable[[], None],
        on_capture_ortho: Callable[[], None],
        on_generate_omap: Callable[[], None],
    ) -> None:
        """
        Construct the full UI inside the given frame.

        Args:
            frame: The parent UI frame to build widgets into.
            omap_interface: The OccupancyMap singleton for viewport visualization.
            on_create_camera: Callback for the "Create Camera" button.
            on_capture_ortho: Callback for the "Capture Orthographic Map" button.
            on_generate_omap: Callback for the "Generate Occupancy Map" button.
        """
        self._om = omap_interface
        with frame:
            with ui.VStack(spacing=5, height=0, style=get_style()):
                self._build_area_section()
                self._build_ortho_section()
                self._build_omap_section()
                self._build_exclusion_section()
                self._build_output_section()

                ui.Spacer(height=10)
                btn_builder(
                    label="Create Camera",
                    text="CREATE CAMERA",
                    tooltip="Create orthographic camera based on area and camera settings",
                    on_clicked_fn=on_create_camera,
                )
                ui.Spacer(height=5)
                btn_builder(
                    label="Ortho Capture",
                    text="CAPTURE ORTHOGRAPHIC MAP",
                    tooltip="Capture a top-down orthographic image of the defined area",
                    on_clicked_fn=on_capture_ortho,
                )
                ui.Spacer(height=5)
                btn_builder(
                    label="Omap Generate",
                    text="GENERATE OCCUPANCY MAP",
                    tooltip="Generate a 2D occupancy map using PhysX raycasting",
                    on_clicked_fn=on_generate_omap,
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
    # UI section builders
    # ------------------------------------------------------------------

    def _build_area_section(self) -> None:
        """Build the Area Definition collapsable frame with origin, bounds, and positioning."""
        with ui.CollapsableFrame(title="Area Definition", style=get_style(), collapsed=False):
            with ui.VStack(spacing=2, height=0):
                self._models["origin"] = xyz_builder(
                    label="Origin",
                    on_value_changed_fn=[
                        self._on_area_value_changed,
                        self._on_area_value_changed,
                        self._on_area_value_changed,
                    ],
                )
                self._models["lower_bound"] = xyz_builder(
                    label="Lower Bound",
                    default_val=[self._lower_bound[0], self._lower_bound[1], 0.0],
                    on_value_changed_fn=[
                        self._on_area_value_changed,
                        self._on_area_value_changed,
                        self._on_area_value_changed,
                    ],
                )
                self._models["upper_bound"] = xyz_builder(
                    label="Upper Bound",
                    default_val=[self._upper_bound[0], self._upper_bound[1], 0.0],
                    on_value_changed_fn=[
                        self._on_area_value_changed,
                        self._on_area_value_changed,
                        self._on_area_value_changed,
                    ],
                )
                self._models["center_bound"] = multi_btn_builder(
                    "Positioning",
                    text=["Center to Selection", "Bound Selection"],
                    on_clicked_fn=[self._on_center_selection, self._on_bound_selection],
                )
                self._models["cell_size"] = float_builder(
                    label="Cell Size",
                    default_val=0.05,
                    min=0.001,
                    step=0.001,
                    format="%.3f",
                    tooltip="Cell size in stage units for occupancy map resolution",
                )
                self._models["cell_size"].add_value_changed_fn(self._on_cell_size_changed)

    def _build_ortho_section(self) -> None:
        """Build the Orthographic Settings collapsable frame."""
        with ui.CollapsableFrame(title="Orthographic Settings", style=get_style(), collapsed=False):
            with ui.VStack(spacing=2, height=0):
                self._models["z_height"] = float_builder(
                    "Z Height", default_val=50.0, tooltip="Camera height above the scene",
                )
                self._models["meters_per_pixel"] = float_builder(
                    "Meters per Pixel", default_val=0.05,
                    tooltip="Meters per pixel for calculating image resolution",
                )
                self._models["camera_path"] = str_builder(
                    "Camera Path", default_val="/World/OrthoCamera",
                    tooltip="USD path for the orthographic camera",
                )

    def _build_omap_section(self) -> None:
        """Build the Occupancy Map Settings collapsable frame."""
        with ui.CollapsableFrame(title="Occupancy Map Settings", style=get_style(), collapsed=False):
            with ui.VStack(spacing=2, height=0):
                self._models["physx_geom"] = cb_builder(
                    "Use PhysX Collision Geometry",
                    tooltip=(
                        "If True, current collision approximations are used. "
                        "If False, original USD meshes are used with RigidBody removal."
                    ),
                    on_clicked_fn=None,
                    default_val=True,
                )
                self._models["max_slope"] = float_builder(
                    "Max Traversable Slope (°)",
                    default_val=0.0,
                    tooltip=(
                        "Slope angle threshold in degrees for post-processing. "
                        "Occupied cells whose ground slope is below this angle are "
                        "reclassified as free space. Set to 0 to disable."
                    ),
                    min=0.0,
                    max=90.0,
                    step=1.0,
                    format="%.1f",
                )

    def _build_exclusion_section(self) -> None:
        """Build the Prim Exclusion List collapsable frame for omap generation."""
        with ui.CollapsableFrame(
            title="Prim Exclusion List", style=get_style(), collapsed=False,
        ):
            with ui.VStack(spacing=4, height=0):
                ui.Label(
                    "Prims below (and their descendants) are hidden during omap generation.",
                    word_wrap=True,
                    height=0,
                )
                with ui.ScrollingFrame(height=120):
                    self._exclude_list_container = ui.VStack(spacing=1, height=0)
                self._rebuild_exclusion_list_ui()
                multi_btn_builder(
                    "Exclusion",
                    text=["Add from Selection", "Remove Selected", "Clear All"],
                    on_clicked_fn=[
                        self._on_add_exclusion_from_selection,
                        self._on_remove_selected_exclusion,
                        self._on_clear_exclusion_list,
                    ],
                )

    def _build_output_section(self) -> None:
        """Build the Output Settings collapsable frame."""
        with ui.CollapsableFrame(title="Output Settings", style=get_style(), collapsed=False):
            with ui.VStack(spacing=2, height=0):
                self._models["output_dir"] = str_builder(
                    "Output Directory",
                    default_val=os.path.expanduser("~/navigation_maps"),
                    tooltip="Directory to save captured images and YAML files",
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
                self._lower_bound[0] = lb_x
            elif self._bound_update_case == 1:
                self._lower_bound[1] = lb_y
            elif self._bound_update_case == 2:
                self._upper_bound[0] = ub_x
            elif self._bound_update_case == 3:
                self._upper_bound[1] = ub_y
        else:
            self._lower_bound[0] = lb_x
            self._lower_bound[1] = lb_y
            self._upper_bound[0] = ub_x
            self._upper_bound[1] = ub_y

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
    ) -> list[float] | tuple[list[float], list[float]]:
        """
        Compute origin or bounds from the current prim selection.

        Args:
            origin_calc: If True, return the midpoint as the new origin.
            stationary_bounds: If True, adjust bounds relative to origin shift.

        Returns:
            Origin as [x, y] when origin_calc is True, otherwise a tuple
            of (lower_bound, upper_bound) each as [x, y].
        """
        origin_xy = [
            self._models["origin"][0].get_value_as_float(),
            self._models["origin"][1].get_value_as_float(),
        ]

        if not origin_calc and stationary_bounds:
            lower = [
                self._lower_bound[0] + self._prev_origin[0] - origin_xy[0],
                self._lower_bound[1] + self._prev_origin[1] - origin_xy[1],
            ]
            upper = [
                self._upper_bound[0] + self._prev_origin[0] - origin_xy[0],
                self._upper_bound[1] + self._prev_origin[1] - origin_xy[1],
            ]
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
                self._prev_origin = list(origin_xy)
                return [mid[0], mid[1]]

            min_pt = box_range.GetMin()
            max_pt = box_range.GetMax()
            lower = [min_pt[0] - origin_xy[0], min_pt[1] - origin_xy[1]]
            upper = [max_pt[0] - origin_xy[0], max_pt[1] - origin_xy[1]]
            return lower, upper

        if origin_calc:
            return [0.0, 0.0]
        return [0.0, 0.0], [0.0, 0.0]
