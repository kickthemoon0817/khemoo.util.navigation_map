from __future__ import annotations

import asyncio
import gc
import weakref
from typing import Optional

import carb
import omni.ext
import omni.ui as ui
from isaacsim.gui.components.element_wrappers import ScrollingWindow
from isaacsim.gui.components.menu import make_menu_item_description
from omni.kit.menu.utils import MenuItemDescription, add_menu_items, remove_menu_items

from .impl.ortho_capture import OrthoMapCapture
from .impl.ortho_config import BoundaryRegion, OrthoMapConfig
from .ui_builder import NavigationMapUIBuilder

EXTENSION_TITLE = "Navigation Map Generator"


class NavigationMapExtension(omni.ext.IExt):
    """
    Extension entry-point that wires the OrthoMapCapture engine
    into an Omni.UI panel accessible from the Tools menu.

    This class is intentionally thin — it delegates capture logic to
    OrthoMapCapture and UI construction to NavigationMapUIBuilder.
    """

    def __init__(self) -> None:
        super().__init__()
        self._ext_id: Optional[str] = None
        self._window: Optional[ScrollingWindow] = None
        self._menu_items: list[MenuItemDescription] = []
        self._capture_engine: Optional[OrthoMapCapture] = None
        self._ui_builder: NavigationMapUIBuilder = NavigationMapUIBuilder()

    def on_startup(self, ext_id: str) -> None:
        """
        Called by Kit when the extension is loaded.

        Args:
            ext_id: The unique extension identifier assigned by Kit.
        """
        self._ext_id = ext_id
        self._capture_engine = OrthoMapCapture()

        self._window = ScrollingWindow(
            title=EXTENSION_TITLE, width=450, height=500, visible=False,
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
                on_create_camera=self._on_create_camera,
                on_capture=self._on_capture,
            )

    def _toggle_window(self) -> None:
        """Toggle the extension window visibility from the menu."""
        if self._window is not None:
            self._window.visible = not self._window.visible

    def _on_create_camera(self) -> None:
        """Read UI values, build an OrthoMapConfig, and create the camera."""
        x_min, x_max, y_min, y_max = self._ui_builder.get_boundary_values()
        boundary = BoundaryRegion(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        meters_per_pixel = self._ui_builder.get_meters_per_pixel()

        tile_grid = OrthoMapConfig.compute_tile_grid(boundary, meters_per_pixel)
        config = OrthoMapConfig(
            boundary=boundary,
            camera_height_meters=self._ui_builder.get_camera_height(),
            meters_per_pixel=meters_per_pixel,
            camera_prim_path=self._ui_builder.get_camera_path(),
            output_directory=self._ui_builder.get_output_directory(),
            tile_grid=tile_grid,
        )
        self._capture_engine.create_camera(config)

    def _on_capture(self) -> None:
        """Kick off the async tiled capture."""
        if not self._capture_engine.is_ready:
            carb.log_warn("No camera created. Please create a camera first.")
            return
        asyncio.ensure_future(self._capture_engine.capture_async())

