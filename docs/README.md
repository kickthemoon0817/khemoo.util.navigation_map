# Navigation Map Generator

> `khemoo.util.navigation_map` — Isaac Sim extension for generating navigation maps using orthographic capture and occupancy grid extraction.

## Overview

Navigation Map Generator creates high-resolution top-down orthographic images of Isaac Sim scenes. These images serve as navigation maps for autonomous robots, providing bird's-eye-view representations of the environment.

The extension supports:

- **Orthographic Capture** — Creates a USD orthographic camera and renders a top-down image of a defined scene region.
- **Tiled Rendering** — Automatically splits large captures into tiles (max 2048px per tile) to avoid VRAM limitations, then stitches them into a single output image.
- **Occupancy Map** *(planned)* — Future integration for binary occupancy grid extraction from scene geometry.

## Installation

Enable `khemoo.util.navigation_map` in your Isaac Sim `.kit` file or the Extension Manager.

```toml
[dependencies]
"khemoo.util.navigation_map" = {}
```

The extension directory must be registered as an extension search path. If placed under `extsUser/`, Isaac Sim discovers it automatically.

## Usage

### GUI

1. Open **Tools → Utilities → Navigation Map Generator**
2. Configure the **Boundary Coordinates** (X Min, X Max, Y Min, Y Max) to define the capture region in world meters
3. Set **Camera Settings**:
   - **Z Height** — Camera altitude above the scene (default: 50.0 m)
   - **Meters per Pixel** — Spatial resolution (default: 0.01 m/px)
   - **Camera Path** — USD prim path for the camera (default: `/World/OrthoCamera`)
4. Set **Output Directory** for saved images (default: `~/navigation_maps`)
5. Click **CREATE CAMERA** to place the orthographic camera
6. Click **CAPTURE** to render and save the navigation map

Output files are saved as `nav_map_YYYYMMDD_HHMMSS.png`.

### Programmatic API

The capture engine is usable without the UI:

```python
from khemoo.util.navigation_map import BoundaryRegion, OrthoMapConfig, OrthoMapCapture

boundary = BoundaryRegion(x_min=-20.0, x_max=20.0, y_min=-20.0, y_max=20.0)
tile_grid = OrthoMapConfig.compute_tile_grid(boundary, meters_per_pixel=0.01)

config = OrthoMapConfig(
    boundary=boundary,
    camera_height_meters=50.0,
    meters_per_pixel=0.01,
    camera_prim_path="/World/OrthoCamera",
    output_directory="/tmp/nav_maps",
    tile_grid=tile_grid,
)

capture = OrthoMapCapture()
capture.create_camera(config)
filepath = await capture.capture_async()
capture.destroy()
```

## Architecture

```
khemoo.util.navigation_map/
├── config/
│   └── extension.toml          # Extension manifest and dependencies
├── docs/
│   ├── README.md               # This file
│   └── VERSION.md              # Version history
└── khemoo/util/navigation_map/
    ├── __init__.py              # Public API exports
    ├── extension.py             # NavigationMapExtension — Kit entry point
    ├── ui_builder.py            # NavigationMapUIBuilder — UI widget construction
    └── impl/
        ├── __init__.py
        ├── ortho_config.py      # Frozen dataclasses: BoundaryRegion, TileGrid, OrthoMapConfig
        └── ortho_capture.py     # OrthoMapCapture — camera creation + tiled rendering
```

### Module Responsibilities

| Module | Class | Role |
|---|---|---|
| `ortho_config` | `BoundaryRegion` | Axis-aligned bounding box in world meters |
| `ortho_config` | `TileGrid` | Computed tiling layout (tile count, dimensions, resolution) |
| `ortho_config` | `OrthoMapConfig` | Complete immutable capture configuration |
| `ortho_capture` | `OrthoMapCapture` | Camera lifecycle, tiled rendering, image stitching |
| `ui_builder` | `NavigationMapUIBuilder` | Omni.UI panel with boundary/camera/output controls |
| `extension` | `NavigationMapExtension` | Thin wiring between capture engine and UI |

### Design Principles

- **Immutability** — All configuration objects are `@dataclass(frozen=True)`
- **Separation of Concerns** — Capture engine is standalone and usable without UI
- **Type Safety** — Full type hints on all signatures, attributes, and return types
- **No Micro-Methods** — Logic reads top-to-bottom; functions are cohesive, not fragmented

## Dependencies

| Package | Purpose |
|---|---|
| `isaacsim.gui.components` | ScrollingWindow, menu helpers, UI builders |
| `omni.kit.menu.utils` | Tools menu integration |
| `omni.replicator.core` | Render products and RGB annotators |
| `omni.ui` | Widget framework |
| `omni.usd` | USD stage access |
| `PIL` (Pillow) | Image assembly and saving |
| `numpy` | Pixel buffer manipulation |

## Roadmap

- [x] Orthographic top-down capture with tiled rendering
- [ ] Occupancy grid extraction from scene collision geometry
- [ ] Combined navigation map export (ortho + occupancy layers)

## License

Internal use — see repository for details.

