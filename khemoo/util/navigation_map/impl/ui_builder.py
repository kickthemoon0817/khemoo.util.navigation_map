from __future__ import annotations

import os
from typing import Callable, Optional

import omni.ui as ui
from isaacsim.gui.components.style import get_style
from isaacsim.gui.components.ui_utils import btn_builder, float_builder, str_builder


class NavigationMapUIBuilder:
    """
    Builds the Omni.UI panel for the Navigation Map Generator extension.

    This class owns all UI widget models and layout logic. It does not
    perform any capture or camera operations itself — those are delegated
    to callbacks provided by the extension.
    """

    def __init__(self) -> None:
        self._models: dict[str, ui.AbstractValueModel] = {}

    @property
    def models(self) -> dict[str, ui.AbstractValueModel]:
        """Direct access to the UI value models keyed by field name."""
        return self._models

    def build(
        self,
        frame: ui.Frame,
        on_create_camera: Callable[[], None],
        on_capture: Callable[[], None],
    ) -> None:
        """
        Construct the full UI inside the given frame.

        Args:
            frame: The parent UI frame to build widgets into.
            on_create_camera: Callback invoked when the user clicks "Create Camera".
            on_capture: Callback invoked when the user clicks "Capture".
        """
        with frame:
            with ui.VStack(spacing=5, height=0, style=get_style()):
                self._build_boundary_section()
                self._build_camera_section()
                self._build_output_section()

                ui.Spacer(height=10)
                btn_builder(
                    label="Create Camera",
                    text="CREATE CAMERA",
                    tooltip="Create orthographic camera based on settings",
                    on_clicked_fn=on_create_camera,
                )
                ui.Spacer(height=5)
                btn_builder(
                    label="Capture Image",
                    text="CAPTURE",
                    tooltip="Capture navigation map from the orthographic camera",
                    on_clicked_fn=on_capture,
                )

    def get_boundary_values(self) -> tuple[float, float, float, float]:
        """
        Read the current boundary coordinate values from the UI.

        Returns:
            Tuple of (x_min, x_max, y_min, y_max).
        """
        return (
            self._models["x_min"].get_value_as_float(),
            self._models["x_max"].get_value_as_float(),
            self._models["y_min"].get_value_as_float(),
            self._models["y_max"].get_value_as_float(),
        )

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

    def _build_boundary_section(self) -> None:
        """Build the Boundary Coordinates collapsable frame."""
        with ui.CollapsableFrame(title="Boundary Coordinates", style=get_style(), collapsed=False):
            with ui.VStack(spacing=2, height=0):
                self._models["x_min"] = float_builder("X Min", default_val=-10.0, tooltip="Minimum X coordinate")
                self._models["x_max"] = float_builder("X Max", default_val=10.0, tooltip="Maximum X coordinate")
                self._models["y_min"] = float_builder("Y Min", default_val=-10.0, tooltip="Minimum Y coordinate")
                self._models["y_max"] = float_builder("Y Max", default_val=10.0, tooltip="Maximum Y coordinate")

    def _build_camera_section(self) -> None:
        """Build the Camera Settings collapsable frame."""
        with ui.CollapsableFrame(title="Camera Settings", style=get_style(), collapsed=False):
            with ui.VStack(spacing=2, height=0):
                self._models["z_height"] = float_builder(
                    "Z Height", default_val=50.0, tooltip="Camera height above the scene"
                )
                self._models["meters_per_pixel"] = float_builder(
                    "Meters per Pixel", default_val=0.01,
                    tooltip="Meters per pixel for calculating image resolution",
                )
                self._models["camera_path"] = str_builder(
                    "Camera Path", default_val="/World/OrthoCamera",
                    tooltip="USD path for the orthographic camera",
                )

    def _build_output_section(self) -> None:
        """Build the Output Settings collapsable frame."""
        with ui.CollapsableFrame(title="Output Settings", style=get_style(), collapsed=False):
            with ui.VStack(spacing=2, height=0):
                self._models["output_dir"] = str_builder(
                    "Output Directory",
                    default_val=os.path.expanduser("~/navigation_maps"),
                    tooltip="Directory to save captured navigation maps",
                    use_folder_picker=True,
                )

