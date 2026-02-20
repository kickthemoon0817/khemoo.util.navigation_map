from __future__ import annotations

import math
import os
from datetime import datetime
from typing import Optional

import carb
import numpy as np
import omni.kit.app
import omni.kit.usd.layers
import omni.physx
import omni.timeline
import omni.usd
from isaacsim.asset.gen.omap.bindings import _omap
from isaacsim.core.utils.stage import get_stage_units
from omni.physx import get_physx_scene_query_interface
from omni.physx.scripts import utils as physx_utils
from PIL import Image
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from .omap_config import OmapConfig


class OmapCapture:
    """
    Generates 2D occupancy maps using the PhysX-based raycast engine.

    Wraps the native ``_omap.Generator`` binding and handles the full
    workflow: rigid-body cleanup in a session layer, PhysX simulation
    step, generation, image export, and ROS YAML metadata output.
    """

    def __init__(self) -> None:
        self._generator: Optional[_omap.Generator] = None
        self._config: Optional[OmapConfig] = None

    @property
    def config(self) -> Optional[OmapConfig]:
        """The current generation configuration, or None if not set."""
        return self._config

    async def generate_async(self, config: OmapConfig) -> Optional[str]:
        """
        Generate a 2D occupancy map and save the result as PNG + ROS YAML.

        Handles rigid-body removal in an anonymous session layer when
        ``use_physx_geometry`` is False, runs the PhysX simulation for
        one frame, then generates the occupancy grid.

        Args:
            config: Immutable configuration for the generation run.

        Returns:
            File path of the saved PNG image, or None on failure.
        """
        self._config = config
        context = omni.usd.get_context()
        stage = context.get_stage()
        if stage is None:
            carb.log_error("No USD stage available for omap generation.")
            return None

        timeline = omni.timeline.get_timeline_interface()
        app = omni.kit.app.get_app()

        if not config.use_physx_geometry:
            filepath = await self._generate_with_mesh_collision(
                stage, timeline, app, config,
            )
        else:
            filepath = await self._generate_with_physx_collision(
                timeline, app, config,
            )

        return filepath

    def destroy(self) -> None:
        """Release the generator and clear state."""
        self._generator = None
        self._config = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _generate_with_physx_collision(
        self,
        timeline: omni.timeline.ITimeline,
        app: omni.kit.app.IApp,
        config: OmapConfig,
    ) -> Optional[str]:
        """
        Generate using existing PhysX collision geometry.

        When ``exclude_prim_paths`` is non-empty, hides the excluded prims
        in an anonymous session layer before running the simulation.

        Args:
            timeline: The timeline interface for play/stop control.
            app: The Kit application interface for frame stepping.
            config: The generation configuration.

        Returns:
            File path of the saved PNG, or None on failure.
        """
        stage = omni.usd.get_context().get_stage()
        layer: Optional[Sdf.Layer] = None

        if config.exclude_prim_paths and stage is not None:
            layer = Sdf.Layer.CreateAnonymous("anon_omap_exclusion")
            stage.GetSessionLayer().subLayerPaths.append(layer.identifier)
            self._hide_excluded_prims(stage, layer, config.exclude_prim_paths)
            await app.next_update_async()

        timeline.play()
        for _ in range(5):
            await app.next_update_async()

        context = omni.usd.get_context()
        physx_interface = omni.physx.get_physx_interface()
        self._generator = _omap.Generator(physx_interface, context.get_stage_id())
        self._generator.update_settings(config.cell_size, 4, 5, 6)
        self._generator.set_transform(
            config.origin, config.lower_bound, config.upper_bound,
        )

        await app.next_update_async()
        self._generator.generate2d()
        await app.next_update_async()

        slope_free_mask: Optional[np.ndarray] = None
        if config.max_traversable_slope_degrees > 0.0 and self._generator is not None:
            slope_free_mask = self._compute_slope_free_mask(config)

        timeline.stop()

        if layer is not None and stage is not None:
            stage.GetSessionLayer().subLayerPaths.remove(layer.identifier)

        return self._save_results(config, slope_free_mask=slope_free_mask)

    async def _generate_with_mesh_collision(
        self,
        stage: Usd.Stage,
        timeline: omni.timeline.ITimeline,
        app: omni.kit.app.IApp,
        config: OmapConfig,
    ) -> Optional[str]:
        """
        Generate by stripping RigidBodyAPI and applying CollisionAPI to visible meshes.

        Creates an anonymous session layer so original stage data is untouched.
        Removes RigidBodyAPI from all prims that have both CollisionAPI and
        RigidBodyAPI, then applies CollisionAPI to visible geometry prims.

        Args:
            stage: The current USD stage.
            timeline: The timeline interface for play/stop control.
            app: The Kit application interface for frame stepping.
            config: The generation configuration.

        Returns:
            File path of the saved PNG, or None on failure.
        """
        layer = Sdf.Layer.CreateAnonymous("anon_omap_session")
        session = stage.GetSessionLayer()
        session.subLayerPaths.append(layer.identifier)

        with Usd.EditContext(stage, layer):
            if config.exclude_prim_paths:
                self._hide_excluded_prims(stage, layer, config.exclude_prim_paths)

            with Sdf.ChangeBlock():
                for prim in stage.Traverse():
                    if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        physx_utils.removePhysics(prim)

            await app.next_update_async()

            with Sdf.ChangeBlock():
                for prim in stage.Traverse():
                    imageable = UsdGeom.Imageable(prim)
                    if imageable:
                        visibility = imageable.ComputeVisibility(Usd.TimeCode.Default())
                        if visibility == UsdGeom.Tokens.invisible:
                            continue

                    if prim.IsA(UsdGeom.Mesh):
                        points_attr = UsdGeom.Mesh(prim).GetPointsAttr().Get()
                        if points_attr is None or len(points_attr) == 0:
                            continue

                    if prim.HasAPI(UsdPhysics.CollisionAPI):
                        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                            approx = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
                            if approx == "none":
                                continue
                        if prim.IsA(UsdGeom.Gprim):
                            if prim.IsInstanceable():
                                UsdPhysics.CollisionAPI.Apply(prim)
                                UsdPhysics.MeshCollisionAPI.Apply(prim)
                            else:
                                try:
                                    physx_utils.setCollider(prim, "none")
                                except Exception:
                                    continue
                    elif prim.IsA(UsdGeom.Xformable) and prim.IsInstanceable():
                        UsdPhysics.CollisionAPI.Apply(prim)
                        UsdPhysics.MeshCollisionAPI.Apply(prim)
                    elif prim.IsA(UsdGeom.Gprim):
                        UsdPhysics.CollisionAPI.Apply(prim)
                        UsdPhysics.MeshCollisionAPI.Apply(prim)

        timeline.play()
        for _ in range(5):
            await app.next_update_async()

        context = omni.usd.get_context()
        physx_interface = omni.physx.get_physx_interface()
        self._generator = _omap.Generator(physx_interface, context.get_stage_id())
        self._generator.update_settings(config.cell_size, 4, 5, 6)
        self._generator.set_transform(
            config.origin, config.lower_bound, config.upper_bound,
        )

        await app.next_update_async()
        self._generator.generate2d()
        await app.next_update_async()

        slope_free_mask: Optional[np.ndarray] = None
        if config.max_traversable_slope_degrees > 0.0 and self._generator is not None:
            slope_free_mask = self._compute_slope_free_mask(config)

        timeline.stop()

        session.subLayerPaths.remove(layer.identifier)
        return self._save_results(config, slope_free_mask=slope_free_mask)

    def _save_results(
        self,
        config: OmapConfig,
        slope_free_mask: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """
        Save the generated occupancy map as a PNG image and a ROS YAML file.

        The image is rotated 180° to match the native Isaac Sim omap
        extension default orientation.  The ROS YAML origin is computed
        from the generator's actual grid bounds (``get_min_bound``) with a
        half-cell offset, matching the native ``compute_coordinates``
        formula at 180° rotation.

        When *slope_free_mask* is provided, occupied cells marked ``True``
        in the mask are reclassified as free space before saving.

        Args:
            config: The configuration used for this generation run.
            slope_free_mask: Boolean array of length ``width * height``.
                ``True`` entries override occupied cells to free.

        Returns:
            File path of the saved PNG, or None if the buffer is empty.
        """
        if self._generator is None:
            return None

        dims = self._generator.get_dimensions()
        if dims[0] == 0 or dims[1] == 0:
            carb.log_warn("Occupancy map buffer is empty — no collision geometry in bounds?")
            return None

        buffer = self._generator.get_buffer()
        occupied_color = (0, 0, 0, 255)
        freespace_color = (255, 255, 255, 255)
        unknown_color = (127, 127, 127, 255)
        image_data = np.full((dims[1], dims[0], 4), unknown_color, dtype=np.uint8)

        flat_buffer = np.array(buffer, dtype=np.float32)
        if slope_free_mask is not None:
            flat_buffer[slope_free_mask] = 0.0

        occupied_mask = flat_buffer == 1.0
        free_mask = flat_buffer == 0.0
        image_flat = image_data.reshape(-1, 4)
        image_flat[occupied_mask] = occupied_color
        image_flat[free_mask] = freespace_color

        os.makedirs(config.output_directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        png_filename = f"omap_{timestamp}.png"
        png_filepath = os.path.join(config.output_directory, png_filename)

        img = Image.fromarray(image_data, mode="RGBA")
        img = img.rotate(-180, expand=True)
        img.save(png_filepath)

        # ROS YAML — origin from generator actual bounds, matching native
        # omap extension default (180° rotation).  At 180° the ROS origin
        # corresponds to compute_coordinates "top_right" which becomes
        # "bottom_left" after the corner swap: (min_b + half_cell).
        scale_to_meters = 1.0 / get_stage_units()
        min_b = self._generator.get_min_bound()
        half_w = config.cell_size * 0.5
        origin_x = min_b[0] + half_w
        origin_y = min_b[1] + half_w

        yaml_filename = f"omap_{timestamp}.yaml"
        yaml_filepath = os.path.join(config.output_directory, yaml_filename)
        yaml_content = (
            f"image: {png_filename}\n"
            f"resolution: {float(config.cell_size / scale_to_meters)}\n"
            f"origin: [{float(origin_x / scale_to_meters)}, "
            f"{float(origin_y / scale_to_meters)}, 0.0000]\n"
            f"negate: 0\n"
            f"occupied_thresh: 0.65\n"
            f"free_thresh: 0.196\n"
        )
        with open(yaml_filepath, "w") as f:
            f.write(yaml_content)

        carb.log_info(
            f"Occupancy map saved: {png_filepath} ({dims[0]}x{dims[1]} cells) | "
            f"YAML: {yaml_filepath}"
        )
        return png_filepath

    def _compute_slope_free_mask(self, config: OmapConfig) -> np.ndarray:
        """
        Identify occupied cells caused only by steep terrain, not real obstacles.

        For every occupied cell a two-stage check is performed:

        1. **Slope check** — a downward ray from above the map ceiling.  If
           the slope angle of the first surface hit (relative to world Z-up) is
           ≥ ``max_traversable_slope_degrees``, the cell stays occupied.
        2. **Thickness check** — an upward ray from below the map floor.  The
           vertical distance between the top-hit and bottom-hit gives the
           object thickness at that XY position.  If the thickness exceeds
           ``cell_size * 5`` the cell is treated as a solid obstacle (wall,
           box, etc.) and stays occupied.

        Only cells that pass *both* checks are reclassified as free space.

        Must be called **while the PhysX timeline is still playing** so that
        the scene query interface has an active scene.

        Args:
            config: The generation configuration (provides slope threshold).

        Returns:
            Boolean numpy array of length ``width * height``.  ``True``
            entries mark occupied cells that should become free.
        """
        dims = self._generator.get_dimensions()
        buffer = np.array(self._generator.get_buffer(), dtype=np.float32)
        min_bound = self._generator.get_min_bound()
        lower_bound_z: float = config.origin[2] + config.lower_bound[2]
        upper_bound_z: float = config.origin[2] + config.upper_bound[2]
        cell_size: float = config.cell_size
        threshold_rad: float = math.radians(config.max_traversable_slope_degrees)

        total_cells: int = dims[0] * dims[1]
        slope_free: np.ndarray = np.zeros(total_cells, dtype=bool)

        scene_query = get_physx_scene_query_interface()

        down_ray_start_z: float = upper_bound_z + 10.0
        down_ray_distance: float = down_ray_start_z - lower_bound_z + 10.0
        up_ray_start_z: float = lower_bound_z - 10.0
        up_ray_distance: float = upper_bound_z - up_ray_start_z + 10.0
        thickness_threshold: float = cell_size * 5.0

        occupied_indices = np.where(buffer == 1.0)[0]
        reclassified_count: int = 0
        for idx in occupied_indices:
            x_cell: int = int(idx % dims[0])
            y_cell: int = int(idx // dims[0])

            cell_x: float = min_bound[0] + (x_cell + 0.5) * cell_size
            cell_y: float = min_bound[1] + (y_cell + 0.5) * cell_size

            # Stage 1: downward ray — slope check.
            down_origin = carb.Float3(cell_x, cell_y, down_ray_start_z)
            down_dir = carb.Float3(0.0, 0.0, -1.0)
            down_hit = scene_query.raycast_closest(
                down_origin, down_dir, down_ray_distance,
            )
            if not down_hit["hit"]:
                continue

            normal = down_hit["normal"]
            dot_with_up: float = float(normal[2])
            slope_angle: float = math.acos(max(-1.0, min(1.0, dot_with_up)))
            if slope_angle >= threshold_rad:
                continue

            # Stage 2: upward ray — thickness check.
            top_z: float = float(down_hit["position"][2])
            up_origin = carb.Float3(cell_x, cell_y, up_ray_start_z)
            up_dir = carb.Float3(0.0, 0.0, 1.0)
            up_hit = scene_query.raycast_closest(
                up_origin, up_dir, up_ray_distance,
            )
            if not up_hit["hit"]:
                continue

            bottom_z: float = float(up_hit["position"][2])
            thickness: float = top_z - bottom_z
            if thickness > thickness_threshold:
                continue

            slope_free[idx] = True
            reclassified_count += 1

        carb.log_info(
            f"Slope filter: {reclassified_count}/{len(occupied_indices)} occupied cells "
            f"reclassified as free (threshold={config.max_traversable_slope_degrees}°)"
        )
        return slope_free

    @staticmethod
    def _hide_excluded_prims(
        stage: Usd.Stage,
        layer: Sdf.Layer,
        exclude_prim_paths: tuple[str, ...],
    ) -> None:
        """
        Make excluded prims invisible in the given session layer.

        Each path in *exclude_prim_paths* is treated as a prefix — both the
        prim itself and all descendants under it are hidden so that the
        PhysX raycast ignores them entirely.

        Args:
            stage: The current USD stage.
            layer: The anonymous session layer to write visibility overrides into.
            exclude_prim_paths: Prim path prefixes to exclude.
        """
        with Usd.EditContext(stage, layer):
            with Sdf.ChangeBlock():
                for prim in stage.Traverse():
                    prim_path: str = prim.GetPath().pathString
                    if any(prim_path.startswith(ep) for ep in exclude_prim_paths):
                        imageable = UsdGeom.Imageable(prim)
                        if imageable:
                            imageable.MakeInvisible()

