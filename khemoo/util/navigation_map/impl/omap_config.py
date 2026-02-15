from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OmapConfig:
    """
    Immutable configuration for occupancy map generation.

    Encapsulates the spatial region, cell resolution, and generation
    options required by the OmapCapture engine.
    """

    origin: tuple[float, float, float]
    lower_bound: tuple[float, float, float]
    upper_bound: tuple[float, float, float]
    cell_size: float
    use_physx_geometry: bool
    output_directory: str
    exclude_prim_paths: tuple[str, ...] = ()
    max_traversable_slope_degrees: float = 0.0

    @property
    def width_cells(self) -> int:
        """Estimated number of cells along the X axis."""
        return max(1, int((self.upper_bound[0] - self.lower_bound[0]) / self.cell_size))

    @property
    def height_cells(self) -> int:
        """Estimated number of cells along the Y axis."""
        return max(1, int((self.upper_bound[1] - self.lower_bound[1]) / self.cell_size))

