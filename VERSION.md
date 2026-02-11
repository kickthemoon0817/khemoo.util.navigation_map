# Version History

## v0.1.0 — 2026-02-09

Initial release. Migrated orthographic capture from `isaacsim.util.ortho_capture` and redesigned following AGENTS.md coding standards.

### Features

- **Orthographic top-down capture** with configurable boundary region, camera height, and resolution
- **Tiled rendering** — Automatically splits captures exceeding 2048px into tile grids to avoid VRAM limitations, then stitches tiles into a single output image
- **GUI panel** — Accessible via Tools → Utilities → Navigation Map Generator with boundary coordinates, camera settings, and output directory controls
- **Programmatic API** — `OrthoMapCapture` class usable without UI for scripted workflows
- **Immutable configuration** — All config objects (`BoundaryRegion`, `TileGrid`, `OrthoMapConfig`) are frozen dataclasses

### Architecture

- `ortho_config.py` — Frozen dataclasses for capture configuration
- `ortho_capture.py` — Standalone capture engine with camera lifecycle and tiled rendering
- `ui_builder.py` — Separated UI construction with callback delegation
- `extension.py` — Thin wiring layer between engine and UI

### Migration Notes

Renamed from `isaacsim.util.ortho_capture`. The monolithic extension class was decomposed into four modules with clear separation of concerns. Output filenames changed from `ortho_capture_*.png` to `nav_map_*.png`.

