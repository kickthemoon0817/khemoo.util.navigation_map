"""
Basic diagnostic test for occupancy map generation.

Creates a simple scene with cubes and verifies the occupancy map
generator produces correct results. Compares both the Generator API
and the OmapCapture wrapper.

Run with:
    ./python.sh extsUser/khemoo.util.navigation_map/tests/test_omap_basic.py --no-window
"""
from __future__ import annotations

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--no-window", action="store_true", default=False)
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

headless = args.headless or args.no_window
simulation_app = SimulationApp({"headless": headless})

import asyncio
import os

import numpy as np
import omni.kit.app
import omni.physx
import omni.timeline
import omni.usd
from isaacsim.core.utils.extensions import enable_extension
from PIL import Image
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

enable_extension("khemoo.util.navigation_map")
simulation_app.update()

from isaacsim.asset.gen.omap.bindings import _omap
from khemoo.util.navigation_map.impl.omap_capture import OmapCapture
from khemoo.util.navigation_map.impl.omap_config import OmapConfig

LOG = "/tmp/omap_test_log2.txt"
CELL_SIZE: float = 0.05


def log(msg: str) -> None:
    with open(LOG, "a") as f:
        f.write(msg + "\n")


async def run_tests() -> None:
    log("=== Test started ===")
    app = omni.kit.app.get_app()
    context = omni.usd.get_context()
    timeline = omni.timeline.get_timeline_interface()

    # ── Phase 1: Raw Generator test ──
    log("\n--- Phase 1: Raw _omap.Generator test ---")
    await context.new_stage_async()
    stage = context.get_stage()

    # Create 3 cubes with collision
    for path, pos in [
        ("/Cube1", (1.0, 0.0, 0.0)),
        ("/Cube2", (1.0, 2.0, 0.0)),
        ("/Cube3", (-1.5, -1.5, 0.0)),
    ]:
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        UsdGeom.Xformable(cube).AddTranslateOp().Set(pos)
        UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(path))

    UsdPhysics.Scene.Define(stage, Sdf.Path("/World/physicsScene"))
    ps = PhysxSchema.PhysxSceneAPI.Get(stage, "/World/physicsScene")
    ps.CreateEnableCCDAttr(True)
    ps.CreateEnableGPUDynamicsAttr(False)
    ps.CreateBroadphaseTypeAttr("MBP")
    ps.CreateSolverTypeAttr("TGS")
    await app.next_update_async()

    timeline.play()
    for _ in range(5):
        await app.next_update_async()

    physx_iface = omni.physx.get_physx_interface()
    gen = _omap.Generator(physx_iface, context.get_stage_id())
    gen.update_settings(CELL_SIZE, 4, 5, 6)
    gen.set_transform((0, 0, 0), (-3, -3, -1), (3, 3, 1))
    await app.next_update_async()
    gen.generate2d()

    buf = np.array(gen.get_buffer(), dtype=np.float32)
    dims = gen.get_dimensions()
    occ = int(np.sum(buf == 1.0))
    free = int(np.sum(buf == 0.0))
    unk = int(np.sum(buf == 0.5))
    log(f"  dims=({dims[0]},{dims[1]}) occupied={occ} free={free} unknown={unk}")
    log(f"  buffer_len={len(buf)} total_cells={dims[0]*dims[1]}")

    min_b = gen.get_min_bound()
    max_b = gen.get_max_bound()
    log(f"  min_bound=({min_b[0]:.3f},{min_b[1]:.3f},{min_b[2]:.3f})")
    log(f"  max_bound=({max_b[0]:.3f},{max_b[1]:.3f},{max_b[2]:.3f})")

    timeline.stop()
    await app.next_update_async()

    if occ > 0:
        log("  PASS: Raw generator detected occupied cells")
    else:
        log("  FAIL: Raw generator detected NO occupied cells")

    # ── Phase 2: OmapCapture wrapper test ──
    log("\n--- Phase 2: OmapCapture wrapper test ---")
    output_dir = os.path.expanduser("~/navigation_maps_test/basic")
    os.makedirs(output_dir, exist_ok=True)

    capture = OmapCapture()
    cfg = OmapConfig(
        origin=(0.0, 0.0, 0.0),
        lower_bound=(-3.0, -3.0, -1.0),
        upper_bound=(3.0, 3.0, 1.0),
        cell_size=CELL_SIZE,
        use_physx_geometry=True,
        output_directory=output_dir,
        exclude_prim_paths=(),
        max_traversable_slope_degrees=0.0,
    )
    path = await capture.generate_async(cfg)
    log(f"  Result path: {path}")

    if path and os.path.exists(path):
        img = np.array(Image.open(path))
        occ_px = int(np.sum(np.all(img[:, :, :3] == 0, axis=2)))
        free_px = int(np.sum(np.all(img[:, :, :3] == 255, axis=2)))
        grey_px = int(np.sum(np.all(img[:, :, :3] == 127, axis=2)))
        log(f"  img_shape={img.shape} occ_px={occ_px} free_px={free_px} grey_px={grey_px}")
        if occ_px > 0:
            log("  PASS: OmapCapture detected occupied cells")
        else:
            log("  FAIL: OmapCapture detected NO occupied cells")
    else:
        log("  FAIL: No output file generated")

    capture.destroy()

    # ── Phase 3: Viewport visualizer interface test ──
    log("\n--- Phase 3: Viewport visualizer interface test ---")
    om = _omap.acquire_omap_interface()
    try:
        om.set_transform(
            (0.0, 0.0, 0.0),
            (-3.0, -3.0, -1.0),
            (3.0, 3.0, 1.0),
        )
        om.set_cell_size(0.05)
        om.update()
        log("  PASS: set_transform + set_cell_size + update succeeded")
    except Exception as exc:
        log(f"  FAIL: Viewport interface error: {exc}")
    finally:
        _omap.release_omap_interface(om)

    log("\n=== Tests completed ===")


# Clear log
with open(LOG, "w") as f:
    f.write("")

task = asyncio.ensure_future(run_tests())
while not task.done():
    simulation_app.update()
if task.exception():
    log(f"EXCEPTION: {task.exception()}")
    import traceback
    log(traceback.format_exc())
    simulation_app.close()
    sys.exit(1)
simulation_app.close()

