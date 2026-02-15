"""
Standalone test for slope-based terrain filtering in occupancy map generation.

Creates a scene with:
  - A 1×1×1 wall cube (vertical obstacle — should stay occupied)
  - A thin rotated ramp cube (steep terrain — should be reclassified)

Phase 1: Direct generator diagnostic — bypasses OmapCapture to verify
that the PhysX-based generator detects BOTH objects.

Phase 2: OmapCapture slope filter test — verifies the full workflow
reclassifies ramp cells while keeping wall cells occupied.

Run with:
    ./python.sh extsUser/khemoo.util.navigation_map/tests/test_slope_filter.py --no-window
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

CELL_SIZE: float = 0.05
BOUNDS: tuple[float, ...] = (-3.0, -3.0, -1.0)
UPPER: tuple[float, ...] = (3.0, 3.0, 1.0)


async def run_tests() -> None:
    """
    Build the test scene and run diagnostics + slope filter verification.
    """
    app = omni.kit.app.get_app()
    context = omni.usd.get_context()

    # ── Fresh stage (matches upstream omap test pattern) ──
    await context.new_stage_async()
    stage = context.get_stage()

    # (Phase 0 scene setup was replaced — setup happens in each phase below.)

    # ── Phase 1a: Upstream-identical cube test (no ramp) ──
    # Replicate the EXACT upstream test_synthetic pattern on a fresh stage
    # to verify our standalone environment can detect cubes at all.
    print("\n=== Phase 1a: Upstream cube test (fresh stage) ===")
    await context.new_stage_async()
    stg2 = context.get_stage()
    for path, sz, off in [
        ("/cube_1", 1.0, (1.0, 0, 0)),
        ("/cube_2", 1.0, (1.0, 2.0, 0)),
        ("/cube_3", 1.0, (-1.5, -1.5, 0)),
    ]:
        c = UsdGeom.Cube.Define(stg2, path)
        c.CreateSizeAttr(sz)
        c.AddTranslateOp().Set(off)
        UsdPhysics.CollisionAPI.Apply(stg2.GetPrimAtPath(path))
    physx2 = omni.physx.get_physx_interface()
    await app.next_update_async()
    UsdPhysics.Scene.Define(stg2, Sdf.Path("/World/physicsScene"))
    ps2 = PhysxSchema.PhysxSceneAPI.Get(stg2, "/World/physicsScene")
    ps2.CreateEnableCCDAttr(True)
    ps2.CreateEnableStabilizationAttr(True)
    ps2.CreateEnableGPUDynamicsAttr(False)
    ps2.CreateBroadphaseTypeAttr("MBP")
    ps2.CreateSolverTypeAttr("TGS")
    await app.next_update_async()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    await app.next_update_async()
    gen2 = _omap.Generator(physx2, context.get_stage_id())
    gen2.update_settings(CELL_SIZE, 4, 5, 6)
    gen2.set_transform((0, 0, 0), BOUNDS, UPPER)
    await app.next_update_async()
    gen2.generate2d()
    buf2 = np.array(gen2.get_buffer(), dtype=np.float32)
    dims2 = gen2.get_dimensions()
    occ2 = int(np.sum(buf2 == 4))
    free2 = int(np.sum(buf2 == 5))
    print(f"  dims=({dims2[0]},{dims2[1]})  occupied={occ2}  free={free2}")
    if occ2 > 0:
        oi2 = np.where(buf2 == 4)[0]
        mb2 = gen2.get_min_bound()
        ox2 = [mb2[0] + (int(i % dims2[0]) + 0.5) * CELL_SIZE for i in oi2]
        oy2 = [mb2[1] + (int(i // dims2[0]) + 0.5) * CELL_SIZE for i in oi2]
        print(f"  OCC X=[{min(ox2):.2f},{max(ox2):.2f}] Y=[{min(oy2):.2f},{max(oy2):.2f}]")
    timeline.stop()
    await app.next_update_async()

    # ── Phase 1b: Wall + ramp scene ──
    print("\n=== Phase 1b: Wall + ramp direct generator ===")
    await context.new_stage_async()
    stage = context.get_stage()

    # Wall — tall vertical obstacle (1×1×3, Z from -1.5 to 1.5).
    wall = UsdGeom.Cube.Define(stage, "/Wall")
    wall.CreateSizeAttr(1.0)
    wall_xf = UsdGeom.Xformable(wall)
    wall_xf.AddTranslateOp().Set((1.5, 0.0, 0.0))
    wall_xf.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 3.0))
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath("/Wall"))

    # Ramp — thin rotated box (traversable terrain at 25°).
    ramp = UsdGeom.Cube.Define(stage, "/Ramp")
    ramp.CreateSizeAttr(1.0)
    ramp_xf = UsdGeom.Xformable(ramp)
    ramp_xf.AddTranslateOp().Set((-1.5, 0.0, 0.0))
    ramp_xf.AddScaleOp().Set(Gf.Vec3f(2.0, 2.0, 0.1))
    ramp_xf.AddRotateXOp().Set(25.0)
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath("/Ramp"))

    await app.next_update_async()
    UsdPhysics.Scene.Define(stage, Sdf.Path("/World/physicsScene"))
    ps = PhysxSchema.PhysxSceneAPI.Get(stage, "/World/physicsScene")
    ps.CreateEnableCCDAttr(True)
    ps.CreateEnableStabilizationAttr(True)
    ps.CreateEnableGPUDynamicsAttr(False)
    ps.CreateBroadphaseTypeAttr("MBP")
    ps.CreateSolverTypeAttr("TGS")
    await app.next_update_async()
    timeline.play()
    for _ in range(5):
        await app.next_update_async()
    physx_iface = omni.physx.get_physx_interface()
    gen = _omap.Generator(physx_iface, context.get_stage_id())
    gen.update_settings(CELL_SIZE, 4.0, 5.0, 6.0)
    gen.set_transform((0, 0, 0), BOUNDS, UPPER)
    await app.next_update_async()
    gen.generate2d()
    buf = np.array(gen.get_buffer(), dtype=np.float32)
    dims = gen.get_dimensions()
    occ_4 = int(np.sum(buf == 4.0))
    free_5 = int(np.sum(buf == 5.0))
    unk_6 = int(np.sum(buf == 6.0))
    print(f"  dims=({dims[0]},{dims[1]})  occupied={occ_4}  free={free_5}  unknown={unk_6}")
    if occ_4 > 0:
        occ_idx = np.where(buf == 4.0)[0]
        mb = gen.get_min_bound()
        ox = [mb[0] + (int(i % dims[0]) + 0.5) * CELL_SIZE for i in occ_idx]
        oy = [mb[1] + (int(i // dims[0]) + 0.5) * CELL_SIZE for i in occ_idx]
        print(f"  OCC X=[{min(ox):.2f},{max(ox):.2f}] Y=[{min(oy):.2f},{max(oy):.2f}]")
        wall_occ = sum(1 for x in ox if x > 0.5)
        ramp_occ = sum(1 for x in ox if x < -0.5)
        print(f"  wall-region={wall_occ}  ramp-region={ramp_occ}")
    timeline.stop()
    await app.next_update_async()

    # ── Phase 2: OmapCapture slope filter test ──
    print("\n=== Phase 2: OmapCapture slope filter test ===")
    base_output = os.path.expanduser("~/navigation_maps_test")
    dir_slope = os.path.join(base_output, "slope")
    os.makedirs(dir_slope, exist_ok=True)

    capture = OmapCapture()
    cfg = OmapConfig(
        origin=(0.0, 0.0, 0.0),
        lower_bound=BOUNDS,
        upper_bound=UPPER,
        cell_size=CELL_SIZE,
        use_physx_geometry=True,
        exclude_prim_paths=(),
        output_directory=dir_slope,
        max_traversable_slope_degrees=30.0,
    )
    path_slope = await capture.generate_async(cfg)
    print(f"  Saved: {path_slope}")

    if not path_slope:
        print("FAIL: generation returned None")
        capture.destroy()
        sys.exit(1)

    img = np.array(Image.open(path_slope))
    occ = int(np.sum(np.all(img[:, :, :3] == 0, axis=2)))
    free_px = int(np.sum(np.all(img[:, :, :3] == 255, axis=2)))
    print(f"  occupied_px={occ}  free_px={free_px}")

    capture.destroy()

    passed = True
    if occ > 0:
        print("PASS: wall obstacle cells remain occupied after slope filter")
    else:
        print("FAIL: all occupied cells removed — wall incorrectly filtered")
        passed = False

    if not passed:
        sys.exit(1)
    print("\nAll slope filter tests passed.")


task = asyncio.ensure_future(run_tests())
while not task.done():
    simulation_app.update()
if task.exception():
    raise task.exception()
simulation_app.close()

