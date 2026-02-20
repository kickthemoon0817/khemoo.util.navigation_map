from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import carb
import numpy as np
import omni.kit.app
import omni.kit.commands
import omni.replicator.core as rep
import omni.usd
from PIL import Image
from pxr import Gf, UsdGeom

from .ortho_config import OrthoMapConfig

# USD orthographic camera aperture attributes are stored in centimeters,
# while tile dimensions are in meters.  This factor converts meters → cm.
APERTURE_METERS_TO_USD_CM: float = 10.0

# Camera clipping plane defaults.  NEAR_CLIP_DISTANCE is the minimum
# distance from the camera plane to the near clip; FAR_CLIP_BUFFER is
# added beyond twice the camera height to ensure tall geometry is captured.
NEAR_CLIP_DISTANCE: float = 0.1
FAR_CLIP_BUFFER: float = 100.0

# Number of frames to wait after teleporting the camera before reading the
# annotator buffer.  The Omniverse render pipeline is multi-stage (USD notice
# propagation → scene delegate update → Hydra render → output buffer write).
# A single step_async() can return a frame rendered from the *previous* camera
# position, causing misaligned or "rotated" tile sections in the stitched image.
# Two settle frames are the minimum: one propagates the USD xform attribute,
# the second ensures the render output buffer reflects the new viewpoint.
SETTLE_FRAMES_AFTER_TELEPORT: int = 2


class OrthoMapCapture:
    """
    Creates an orthographic camera and captures tiled top-down images of a scene region.

    This class manages the full lifecycle of orthographic map capture:
    creating the USD camera prim, setting up the Replicator render product,
    performing tiled rendering for large resolutions, and stitching the
    result into a single image saved to disk.

    The class is designed to be usable both from the UI extension and
    programmatically from scripts.
    """

    def __init__(self) -> None:
        self._config: Optional[OrthoMapConfig] = None
        self._camera_prim: Optional[UsdGeom.Camera] = None
        self._render_product: Optional[rep.create.render_product] = None
        self._rgb_annotator: Optional[rep.AnnotatorRegistry] = None

    @property
    def config(self) -> Optional[OrthoMapConfig]:
        """The current capture configuration, or None if not yet created."""
        return self._config

    @property
    def is_ready(self) -> bool:
        """Whether a camera and render product are set up and ready to capture."""
        return (
            self._config is not None
            and self._render_product is not None
            and self._rgb_annotator is not None
        )

    def create_camera(self, config: OrthoMapConfig) -> None:
        """
        Create an orthographic camera prim and render product from the given config.

        Tears down any existing camera/render resources before creating new ones.
        The camera is positioned at the first tile center, ready for tiled capture.

        Args:
            config: Immutable configuration describing the capture region and resolution.
        """
        stage = omni.usd.get_context().get_stage()
        if not stage:
            raise RuntimeError("No USD stage available")

        self._teardown_render_resources()
        self._config = config

        # Remove existing prim at the target path
        existing_prim = stage.GetPrimAtPath(config.camera_prim_path)
        if existing_prim.IsValid():
            omni.kit.commands.execute("DeletePrims", paths=[config.camera_prim_path])

        # Define the orthographic camera prim
        grid = config.tile_grid
        self._camera_prim = UsdGeom.Camera.Define(stage, config.camera_prim_path)

        initial_x = config.boundary.x_min + grid.tile_width_meters / 2.0
        initial_y = config.boundary.y_min + grid.tile_height_meters / 2.0

        xform_api = UsdGeom.XformCommonAPI(self._camera_prim)
        xform_api.SetTranslate(Gf.Vec3d(initial_x, initial_y, config.camera_height_meters))
        xform_api.SetRotate(Gf.Vec3f(0, 0, 0), UsdGeom.XformCommonAPI.RotationOrderXYZ)

        self._camera_prim.GetProjectionAttr().Set("orthographic")
        self._camera_prim.GetHorizontalApertureAttr().Set(
            grid.tile_width_meters * APERTURE_METERS_TO_USD_CM,
        )
        self._camera_prim.GetVerticalApertureAttr().Set(
            grid.tile_height_meters * APERTURE_METERS_TO_USD_CM,
        )
        self._camera_prim.GetClippingRangeAttr().Set(
            Gf.Vec2f(NEAR_CLIP_DISTANCE, config.camera_height_meters * 2 + FAR_CLIP_BUFFER)
        )

        # Create render product sized to a single tile
        self._render_product = rep.create.render_product(
            config.camera_prim_path, (grid.tile_width_pixels, grid.tile_height_pixels)
        )
        self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self._rgb_annotator.attach([self._render_product])

        carb.log_info(
            f"Created orthographic camera at {config.camera_prim_path} | "
            f"Resolution: {grid.total_width_pixels}x{grid.total_height_pixels} | "
            f"Tiles: {grid.num_tiles_x}x{grid.num_tiles_y}"
        )

    async def capture_async(self) -> Optional[str]:
        """
        Capture the full orthographic image by rendering all tiles and stitching them.

        Moves the camera across the tile grid, captures each tile via Replicator,
        and assembles the final image. The Y-axis is flipped so that y_min maps
        to the bottom of the output image.

        Returns:
            The file path of the saved image, or None if capture failed.
        """
        if not self.is_ready:
            carb.log_warn("Cannot capture: camera not created. Call create_camera first.")
            return None

        config = self._config
        grid = config.tile_grid
        os.makedirs(config.output_directory, exist_ok=True)

        final_image = np.zeros(
            (grid.total_height_pixels, grid.total_width_pixels, 3), dtype=np.uint8
        )
        xform_api = UsdGeom.XformCommonAPI(self._camera_prim)

        carb.log_info(
            f"Starting tiled capture: {grid.num_tiles_x}x{grid.num_tiles_y} "
            f"= {grid.total_tiles} tiles"
        )

        tile_count = 0
        for ty in range(grid.num_tiles_y):
            for tx in range(grid.num_tiles_x):
                tile_count += 1

                px_start = tx * grid.tile_width_pixels
                py_start = ty * grid.tile_height_pixels

                # Tile center in world coordinates
                tile_center_x = (
                    config.boundary.x_min
                    + (px_start + grid.tile_width_pixels / 2.0) * config.meters_per_pixel
                )
                tile_center_y = (
                    config.boundary.y_min
                    + (py_start + grid.tile_height_pixels / 2.0) * config.meters_per_pixel
                )

                xform_api.SetTranslate(
                    Gf.Vec3d(tile_center_x, tile_center_y, config.camera_height_meters)
                )

                # Wait for the render pipeline to fully propagate the new camera
                # position before capturing.  Without this, the annotator may
                # return a frame rendered from the previous tile's viewpoint.
                for _ in range(SETTLE_FRAMES_AFTER_TELEPORT):
                    await omni.kit.app.get_app().next_update_async()
                await rep.orchestrator.step_async()

                rgb_data = self._rgb_annotator.get_data()
                if rgb_data is None:
                    carb.log_error(f"Failed to get RGB data for tile ({tx}, {ty})")
                    continue

                tile_data = rgb_data[: grid.tile_height_pixels, : grid.tile_width_pixels, :3]

                # Place tile with Y flipped: ty=0 (y_min) -> bottom of image (high row index)
                img_y_start = grid.total_height_pixels - (py_start + grid.tile_height_pixels)
                img_y_end = grid.total_height_pixels - py_start
                final_image[img_y_start:img_y_end, px_start : px_start + grid.tile_width_pixels] = tile_data

                if grid.uses_tiling:
                    carb.log_info(f"  Captured tile {tile_count}/{grid.total_tiles}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nav_map_{timestamp}.png"
        filepath = os.path.join(config.output_directory, filename)

        img = Image.fromarray(final_image)
        img.save(filepath)
        carb.log_info(
            f"Navigation map saved: {filepath} "
            f"({grid.total_width_pixels}x{grid.total_height_pixels})"
        )
        return filepath

    def destroy(self) -> None:
        """Release all render resources and clear state."""
        self._teardown_render_resources()
        self._camera_prim = None
        self._config = None

    def _teardown_render_resources(self) -> None:
        """Detach annotator and destroy render product if they exist."""
        if self._rgb_annotator is not None:
            try:
                self._rgb_annotator.detach()
            except (RuntimeError, AttributeError) as exc:
                carb.log_warn(f"Failed to detach RGB annotator during teardown: {exc}")
            self._rgb_annotator = None

        if self._render_product is not None:
            try:
                self._render_product.destroy()
            except (RuntimeError, AttributeError) as exc:
                carb.log_warn(f"Failed to destroy render product during teardown: {exc}")
            self._render_product = None

