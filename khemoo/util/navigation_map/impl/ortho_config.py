from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundaryRegion:
    """
    Axis-aligned bounding box in world coordinates (meters).

    Defines the rectangular region of the scene to capture.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def width_meters(self) -> float:
        """Horizontal extent of the region in meters."""
        return abs(self.x_max - self.x_min)

    @property
    def height_meters(self) -> float:
        """Vertical extent of the region in meters."""
        return abs(self.y_max - self.y_min)


@dataclass(frozen=True)
class TileGrid:
    """
    Computed tiling layout for rendering large orthographic captures.

    When the total resolution exceeds MAX_TILE_SIZE, the capture is split
    into a grid of tiles that are rendered individually and stitched together.
    """

    num_tiles_x: int
    num_tiles_y: int
    tile_width_pixels: int
    tile_height_pixels: int
    tile_width_meters: float
    tile_height_meters: float
    total_width_pixels: int
    total_height_pixels: int

    @property
    def total_tiles(self) -> int:
        """Total number of tiles in the grid."""
        return self.num_tiles_x * self.num_tiles_y

    @property
    def uses_tiling(self) -> bool:
        """Whether the capture requires multiple tiles."""
        return self.total_tiles > 1


@dataclass(frozen=True)
class OrthoMapConfig:
    """
    Complete configuration for an orthographic map capture.

    Combines the spatial boundary, camera height, resolution, and
    the computed tile grid into a single immutable configuration object.
    """

    boundary: BoundaryRegion
    camera_height_meters: float
    meters_per_pixel: float
    camera_prim_path: str
    output_directory: str
    tile_grid: TileGrid

    @staticmethod
    def compute_tile_grid(
        boundary: BoundaryRegion,
        meters_per_pixel: float,
        max_tile_size: int = 2048,
    ) -> TileGrid:
        """
        Compute the tile grid layout from boundary and resolution settings.

        Args:
            boundary: The world-space region to capture.
            meters_per_pixel: Spatial resolution of the output image.
            max_tile_size: Maximum pixel dimension per tile to avoid VRAM blowup.

        Returns:
            A TileGrid describing how to partition the capture.
        """
        total_width_pixels = max(64, int(boundary.width_meters / meters_per_pixel))
        total_height_pixels = max(64, int(boundary.height_meters / meters_per_pixel))

        num_tiles_x = max(1, (total_width_pixels + max_tile_size - 1) // max_tile_size)
        num_tiles_y = max(1, (total_height_pixels + max_tile_size - 1) // max_tile_size)

        tile_width_pixels = (total_width_pixels + num_tiles_x - 1) // num_tiles_x
        tile_height_pixels = (total_height_pixels + num_tiles_y - 1) // num_tiles_y

        # Adjust totals to be exact multiples of tile size
        total_width_pixels = tile_width_pixels * num_tiles_x
        total_height_pixels = tile_height_pixels * num_tiles_y

        tile_width_meters = tile_width_pixels * meters_per_pixel
        tile_height_meters = tile_height_pixels * meters_per_pixel

        return TileGrid(
            num_tiles_x=num_tiles_x,
            num_tiles_y=num_tiles_y,
            tile_width_pixels=tile_width_pixels,
            tile_height_pixels=tile_height_pixels,
            tile_width_meters=tile_width_meters,
            tile_height_meters=tile_height_meters,
            total_width_pixels=total_width_pixels,
            total_height_pixels=total_height_pixels,
        )

