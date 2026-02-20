from __future__ import annotations

import asyncio
import gc
import os
import subprocess
import sys
import weakref
from typing import Any, Optional

import carb
import omni.ext
import omni.kit.app
import omni.ui as ui
from isaacsim.asset.gen.omap.bindings import _omap
from isaacsim.gui.components.element_wrappers import ScrollingWindow
from isaacsim.gui.components.menu import make_menu_item_description
from omni.kit.menu.utils import MenuItemDescription, add_menu_items, remove_menu_items

from .impl.omap_capture import OmapCapture
from .impl.omap_config import OmapConfig
from .impl.ortho_capture import OrthoMapCapture
from .impl.ortho_config import BoundaryRegion, OrthoMapConfig
from .ui_builder import (
    NavigationMapUIBuilder,
    OUTPUT_TYPE_BOTH,
    OUTPUT_TYPE_IMAGE,
    OUTPUT_TYPE_NAV_MAP,
    STATUS_ERROR,
    STATUS_INFO,
    STATUS_SUCCESS,
    STATUS_WARNING,
)

EXTENSION_TITLE = "Navigation Map Generator"


class NavigationMapExtension(omni.ext.IExt):
    """
    Extension entry-point that wires the OrthoMapCapture and OmapCapture
    engines into a unified Omni.UI panel accessible from the Tools menu.

    This class is intentionally thin — it delegates capture logic to
    OrthoMapCapture / OmapCapture and UI construction to
    NavigationMapUIBuilder.
    """

    def __init__(self) -> None:
        super().__init__()
        self._ext_id: Optional[str] = None
        self._window: Optional[ScrollingWindow] = None
        self._menu_items: list[MenuItemDescription] = []
        self._capture_engine: Optional[OrthoMapCapture] = None
        self._omap_engine: Optional[OmapCapture] = None
        self._om: Any | None = None
        self._ui_builder: NavigationMapUIBuilder = NavigationMapUIBuilder()

    def on_startup(self, ext_id: str) -> None:
        """
        Called by Kit when the extension is loaded.

        Args:
            ext_id: The unique extension identifier assigned by Kit.
        """
        self._ext_id = ext_id
        self._capture_engine = OrthoMapCapture()
        self._omap_engine = OmapCapture()
        self._om = _omap.acquire_omap_interface()

        self._window = ScrollingWindow(
            title=EXTENSION_TITLE, width=450, height=600, visible=False,
            dockPreference=ui.DockPreference.LEFT_BOTTOM,
        )
        self._window.set_visibility_changed_fn(self._on_window_visibility_changed)

        menu_entry = [
            make_menu_item_description(
                ext_id, EXTENSION_TITLE, lambda a=weakref.proxy(self): a._toggle_window()
            )
        ]
        self._menu_items = [MenuItemDescription("Utilities", sub_menu=menu_entry)]
        add_menu_items(self._menu_items, "Tools")

        carb.log_info(f"{EXTENSION_TITLE} ({ext_id}) loaded.")

    def on_shutdown(self) -> None:
        """Called by Kit when the extension is unloaded. Releases all resources."""
        carb.log_info(f"{EXTENSION_TITLE} shutting down.")

        if self._capture_engine is not None:
            self._capture_engine.destroy()
            self._capture_engine = None

        if self._omap_engine is not None:
            self._omap_engine.destroy()
            self._omap_engine = None

        if self._om is not None:
            _omap.release_omap_interface(self._om)
            self._om = None

        if self._menu_items:
            remove_menu_items(self._menu_items, "Tools")
            self._menu_items = []

        self._window = None
        self._ext_id = None
        gc.collect()

    def _on_window_visibility_changed(self, visible: bool) -> None:
        """Rebuild the UI each time the window becomes visible."""
        if visible and self._window is not None:
            self._ui_builder.build(
                frame=self._window.frame,
                omap_interface=self._om,
                on_generate=self._on_generate,
                on_open_folder=self._on_open_folder,
            )

    def _toggle_window(self) -> None:
        """Toggle the extension window visibility from the menu."""
        if self._window is not None:
            self._window.visible = not self._window.visible

    # ------------------------------------------------------------------
    # Generate workflow
    # ------------------------------------------------------------------

    def _on_generate(self) -> None:
        """Validate inputs and dispatch to the correct async workflow."""
        error = self._validate_inputs()
        if error is not None:
            self._ui_builder.set_status(error, STATUS_ERROR)
            return
        output_type = self._ui_builder.selected_output_type
        self._ui_builder.set_generate_enabled(False)
        self._ui_builder.set_progress(0.0)
        self._ui_builder.set_status("Starting...", STATUS_INFO)
        if output_type == OUTPUT_TYPE_IMAGE:
            asyncio.ensure_future(self._generate_image_async())
        elif output_type == OUTPUT_TYPE_NAV_MAP:
            asyncio.ensure_future(self._generate_navmap_async())
        else:
            asyncio.ensure_future(self._generate_both_async())

    def _validate_inputs(self) -> str | None:
        """
        Check that the UI inputs are valid for generation.

        Returns:
            An error message string if validation fails, or None if valid.
        """
        x_min, x_max, y_min, y_max = self._ui_builder.get_boundary_values()
        if x_min >= x_max or y_min >= y_max:
            return "Area Min must be less than Area Max in both X and Y."
        cell_size = self._ui_builder.get_cell_size()
        if cell_size <= 0:
            return "Map Detail (cell size) must be greater than zero."
        output_dir = self._ui_builder.get_output_directory()
        if not output_dir or not output_dir.strip():
            return "Output Directory must not be empty."
        return None

    def _build_ortho_config(self) -> OrthoMapConfig:
        """
        Construct an OrthoMapConfig from the current UI values.

        Returns:
            Immutable config ready for OrthoMapCapture.
        """
        x_min, x_max, y_min, y_max = self._ui_builder.get_boundary_values()
        boundary = BoundaryRegion(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        meters_per_pixel = self._ui_builder.get_meters_per_pixel()
        tile_grid = OrthoMapConfig.compute_tile_grid(boundary, meters_per_pixel)
        return OrthoMapConfig(
            boundary=boundary,
            camera_height_meters=self._ui_builder.get_camera_height(),
            meters_per_pixel=meters_per_pixel,
            camera_prim_path=self._ui_builder.get_camera_path(),
            output_directory=self._ui_builder.get_output_directory(),
            tile_grid=tile_grid,
        )

    def _build_omap_config(self) -> OmapConfig:
        """
        Construct an OmapConfig from the current UI values.

        Returns:
            Immutable config ready for OmapCapture.
        """
        return OmapConfig(
            origin=self._ui_builder.get_origin(),
            lower_bound=self._ui_builder.get_lower_bound(),
            upper_bound=self._ui_builder.get_upper_bound(),
            cell_size=self._ui_builder.get_cell_size(),
            use_physx_geometry=self._ui_builder.get_use_physx_geometry(),
            output_directory=self._ui_builder.get_output_directory(),
            exclude_prim_paths=self._ui_builder.get_exclude_prim_paths(),
            max_traversable_slope_degrees=self._ui_builder.get_max_traversable_slope_degrees(),
        )

    async def _generate_image_async(self) -> None:
        """Create camera, capture with progress, update UI on completion."""
        try:
            config = self._build_ortho_config()
            self._ui_builder.set_status("Creating camera...", STATUS_INFO)
            self._capture_engine.create_camera(config)
            self._ui_builder.set_status("Capturing aerial photo...", STATUS_INFO)
            self._deactivate_guide_visualization()
            await omni.kit.app.get_app().next_update_async()
            filepath = await self._capture_engine.capture_async(
                progress_fn=self._on_capture_progress,
            )
            self._restore_guide_visualization()
            if filepath:
                self._ui_builder.set_status(f"Saved: {filepath}", STATUS_SUCCESS)
            else:
                self._ui_builder.set_status("Capture failed — check console.", STATUS_ERROR)
        except Exception as exc:
            carb.log_error(f"Image capture failed: {exc}")
            self._ui_builder.set_status(f"Error: {exc}", STATUS_ERROR)
            self._restore_guide_visualization()
        finally:
            self._ui_builder.set_progress(1.0)
            self._ui_builder.set_generate_enabled(True)

    async def _generate_navmap_async(self) -> None:
        """Generate the occupancy map and update UI on completion."""
        try:
            config = self._build_omap_config()
            self._ui_builder.set_status("Generating navigation map...", STATUS_INFO)
            self._ui_builder.set_progress(0.1)
            filepath = await self._omap_engine.generate_async(config)
            if filepath:
                self._ui_builder.set_status(f"Saved: {filepath}", STATUS_SUCCESS)
            else:
                self._ui_builder.set_status("Generation failed — check console.", STATUS_ERROR)
        except Exception as exc:
            carb.log_error(f"Nav map generation failed: {exc}")
            self._ui_builder.set_status(f"Error: {exc}", STATUS_ERROR)
        finally:
            self._ui_builder.set_progress(1.0)
            self._ui_builder.set_generate_enabled(True)

    async def _generate_both_async(self) -> None:
        """Generate aerial photo then navigation map, with combined progress."""
        try:
            config_ortho = self._build_ortho_config()
            self._ui_builder.set_status("Creating camera...", STATUS_INFO)
            self._capture_engine.create_camera(config_ortho)
            self._ui_builder.set_status("Capturing aerial photo...", STATUS_INFO)
            self._deactivate_guide_visualization()
            await omni.kit.app.get_app().next_update_async()
            img_path = await self._capture_engine.capture_async(
                progress_fn=self._on_capture_progress_half,
            )
            self._restore_guide_visualization()
            if not img_path:
                self._ui_builder.set_status("Image capture failed — check console.", STATUS_ERROR)
                return
            config_omap = self._build_omap_config()
            self._ui_builder.set_status("Generating navigation map...", STATUS_INFO)
            self._ui_builder.set_progress(0.6)
            omap_path = await self._omap_engine.generate_async(config_omap)
            if omap_path:
                self._ui_builder.set_status(
                    f"Done! Image: {img_path}  |  Map: {omap_path}", STATUS_SUCCESS,
                )
            else:
                self._ui_builder.set_status(
                    f"Image saved ({img_path}) but map generation failed.", STATUS_WARNING,
                )
        except Exception as exc:
            carb.log_error(f"Combined generation failed: {exc}")
            self._ui_builder.set_status(f"Error: {exc}", STATUS_ERROR)
            self._restore_guide_visualization()
        finally:
            self._ui_builder.set_progress(1.0)
            self._ui_builder.set_generate_enabled(True)

    def _on_capture_progress(self, current_tile: int, total_tiles: int) -> None:
        """Progress callback for image-only mode (0.0 → 1.0)."""
        fraction = current_tile / max(1, total_tiles)
        self._ui_builder.set_progress(fraction)
        self._ui_builder.set_status(
            f"Capturing tile {current_tile}/{total_tiles}...", STATUS_INFO,
        )

    def _on_capture_progress_half(self, current_tile: int, total_tiles: int) -> None:
        """Progress callback for Both mode — image occupies first half (0.0 → 0.5)."""
        fraction = 0.5 * current_tile / max(1, total_tiles)
        self._ui_builder.set_progress(fraction)
        self._ui_builder.set_status(
            f"Capturing tile {current_tile}/{total_tiles}...", STATUS_INFO,
        )

    # ------------------------------------------------------------------
    # Guide visualization helpers
    # ------------------------------------------------------------------

    def _deactivate_guide_visualization(self) -> None:
        """Collapse the omap guide transform to zero so nothing is drawn."""
        if self._om is None:
            return
        self._om.set_transform((0, 0, 0), (0, 0, 0), (0, 0, 0))
        self._om.update()

    def _restore_guide_visualization(self) -> None:
        """Restore the omap guide to reflect the current UI bounds."""
        if self._om is None:
            return
        origin = self._ui_builder.get_origin()
        lower = self._ui_builder.get_lower_bound()
        upper = self._ui_builder.get_upper_bound()
        self._om.set_transform(origin, lower, upper)
        self._om.update()

    def _on_open_folder(self) -> None:
        """Open the output directory in the system file browser."""
        output_dir = self._ui_builder.get_output_directory()
        if not output_dir:
            self._ui_builder.set_status("No output directory set.", STATUS_WARNING)
            return
        path = os.path.expanduser(output_dir)
        os.makedirs(path, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

