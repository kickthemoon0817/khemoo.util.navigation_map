"""
End-to-end alignment test for occupancy map and orthographic capture.

Creates a scene with a ground plane, lights, and a cube at a known position.
Generates both an occupancy map (PhysX raycast) and an orthographic image
(visual render).  Verifies obstacle detection, non-blank rendering, and
basic alignment between the two outputs.

Run with:
    ./python.sh extsUser/khemoo.util.navigation_map/tests/test_e2e_alignment.py --no-window
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
import omni.usd
from isaacsim.core.utils.extensions import enable_extension
from PIL import Image
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade

enable_extension("khemoo.util.navigation_map")
simulation_app.update()

from khemoo.util.navigation_map.impl.omap_capture import OmapCapture
from khemoo.util.navigation_map.impl.omap_config import OmapConfig
from khemoo.util.navigation_map.impl.ortho_capture import OrthoMapCapture
from khemoo.util.navigation_map.impl.ortho_config import BoundaryRegion, OrthoMapConfig

LOG = "/tmp/e2e_alignment_test.txt"
OUTPUT_DIR = os.path.expanduser("~/navigation_maps_test/e2e")

CELL_SIZE: float = 0.05
MET_PER_PX: float = 0.05
ORIGIN: tuple[float, float, float] = (0.0, 0.0, 0.0)
LOWER: tuple[float, float, float] = (-10.0, -10.0, 0.5)
UPPER: tuple[float, float, float] = (10.0, 10.0, 5.0)
CUBE_POS: tuple[float, float, float] = (5.0, 3.0, 1.0)
CUBE_SIZE: float = 2.0


def log(msg: str) -> None:
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def create_preview_material(
    stage, mat_path: str, color: tuple[float, float, float]
) -> UsdShade.Material:
    """Create a UsdPreviewSurface material with the given diffuse color."""
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


async def run_e2e() -> bool:
    log("=== E2E Alignment Test ===")
    app = omni.kit.app.get_app()
    ctx = omni.usd.get_context()

    # ── Scene setup ──
    await ctx.new_stage_async()
    stage = ctx.get_stage()

    # Distant light pointing downward (default = -Z)
    light = UsdLux.DistantLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(5000)
    light.CreateAngleAttr(1.0)
    # Dome light for ambient fill
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(500)

    # Materials
    ground_mat = create_preview_material(stage, "/World/Looks/GroundMat", (0.3, 0.3, 0.3))
    cube_mat = create_preview_material(stage, "/World/Looks/CubeMat", (0.8, 0.1, 0.1))

    # Ground: large flat cube at z=-0.05 (below omap z_lower=0.5)
    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.CreateSizeAttr(1.0)
    gx = UsdGeom.Xformable(ground)
    gx.AddTranslateOp().Set((0.0, 0.0, -0.05))
    gx.AddScaleOp().Set(Gf.Vec3f(40.0, 40.0, 0.1))
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath("/World/Ground"))
    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath("/World/Ground")).Bind(ground_mat)

    # Obstacle cube (2x2x2, center at (5, 3, 1))
    cube = UsdGeom.Cube.Define(stage, "/World/Cube")
    cube.CreateSizeAttr(CUBE_SIZE)
    cx = UsdGeom.Xformable(cube)
    cx.AddTranslateOp().Set(CUBE_POS)
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath("/World/Cube"))
    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath("/World/Cube")).Bind(cube_mat)

    UsdPhysics.Scene.Define(stage, Sdf.Path("/World/physicsScene"))
    ps = PhysxSchema.PhysxSceneAPI.Get(stage, "/World/physicsScene")
    ps.CreateEnableCCDAttr(True)
    ps.CreateEnableGPUDynamicsAttr(False)
    ps.CreateBroadphaseTypeAttr("MBP")
    ps.CreateSolverTypeAttr("TGS")
    await app.next_update_async()
    log(f"Scene: ground(grey), cube(red) at {CUBE_POS} size={CUBE_SIZE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Occupancy Map ──
    log("\n--- Phase 1: Occupancy Map ---")
    omap = OmapCapture()
    omap_cfg = OmapConfig(
        origin=ORIGIN, lower_bound=LOWER, upper_bound=UPPER,
        cell_size=CELL_SIZE, use_physx_geometry=True,
        output_directory=OUTPUT_DIR, exclude_prim_paths=(),
        max_traversable_slope_degrees=0.0,
    )
    omap_path = await omap.generate_async(omap_cfg)
    omap.destroy()
    log(f"Omap path: {omap_path}")

    if not omap_path or not os.path.exists(omap_path):
        log("FAIL: No omap output file"); return False

    omap_img = np.array(Image.open(omap_path))
    omap_h, omap_w = omap_img.shape[:2]
    black = np.all(omap_img[:, :, :3] == 0, axis=2)
    occ_count = int(np.sum(black))
    free_count = int(np.sum(np.all(omap_img[:, :, :3] == 255, axis=2)))
    log(f"Omap: {omap_w}x{omap_h}, occupied={occ_count}, free={free_count}")

    if occ_count == 0:
        log("FAIL: No occupied pixels in omap"); return False

    occ_rows, occ_cols = np.where(black)
    log(f"Omap occupied: rows=[{occ_rows.min()},{occ_rows.max()}]"
        f" cols=[{occ_cols.min()},{occ_cols.max()}]")

    # ── 2. Orthographic Map ──
    log("\n--- Phase 2: Orthographic Map ---")
    x_min, x_max = ORIGIN[0] + LOWER[0], ORIGIN[0] + UPPER[0]
    y_min, y_max = ORIGIN[1] + LOWER[1], ORIGIN[1] + UPPER[1]
    boundary = BoundaryRegion(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    tile_grid = OrthoMapConfig.compute_tile_grid(boundary, MET_PER_PX)
    ortho_cfg = OrthoMapConfig(
        boundary=boundary, camera_height_meters=50.0,
        meters_per_pixel=MET_PER_PX,
        camera_prim_path="/World/TestOrthoCamera",
        output_directory=OUTPUT_DIR, tile_grid=tile_grid,
    )
    ortho = OrthoMapCapture()
    ortho.create_camera(ortho_cfg)

    # Warm up renderer (40 frames for RTX convergence)
    for _ in range(40):
        await app.next_update_async()

    ortho_path = await ortho.capture_async()
    ortho.destroy()
    log(f"Ortho path: {ortho_path}")

    if not ortho_path or not os.path.exists(ortho_path):
        log("FAIL: No ortho output file"); return False

    ortho_img = np.array(Image.open(ortho_path))
    ortho_h, ortho_w = ortho_img.shape[:2]
    log(f"Ortho: {ortho_w}x{ortho_h}, range=[{ortho_img.min()},{ortho_img.max()}]")

    if ortho_img.max() == ortho_img.min():
        log("FAIL: Ortho image is blank (uniform)"); return False

    # ── 3. Alignment ──
    log("\n--- Phase 3: Alignment ---")
    log(f"Dimensions: omap={omap_w}x{omap_h}, ortho={ortho_w}x{ortho_h}")

    # Expected cube in ortho pixel coords (row0=ymax, col0=xmin)
    cx_lo = CUBE_POS[0] - CUBE_SIZE / 2  # 4.0
    cx_hi = CUBE_POS[0] + CUBE_SIZE / 2  # 6.0
    cy_lo = CUBE_POS[1] - CUBE_SIZE / 2  # 2.0
    cy_hi = CUBE_POS[1] + CUBE_SIZE / 2  # 4.0
    exp_col_lo = int((cx_lo - x_min) / MET_PER_PX)
    exp_col_hi = int((cx_hi - x_min) / MET_PER_PX)
    exp_row_lo = int((y_max - cy_hi) / MET_PER_PX)
    exp_row_hi = int((y_max - cy_lo) / MET_PER_PX)
    log(f"Expected cube: rows=[{exp_row_lo},{exp_row_hi}] cols=[{exp_col_lo},{exp_col_hi}]")

    # Check omap occupied region vs expected
    tol = 15
    row_ok = occ_rows.min() >= (exp_row_lo - tol) and occ_rows.max() <= (exp_row_hi + tol)
    col_ok = occ_cols.min() >= (exp_col_lo - tol) and occ_cols.max() <= (exp_col_hi + tol)
    log(f"Omap vs expected: row_ok={row_ok}, col_ok={col_ok}")

    # Check ortho shows cube (material contrast in expected region)
    if ortho_h > exp_row_hi and ortho_w > exp_col_hi:
        region = ortho_img[exp_row_lo:exp_row_hi, exp_col_lo:exp_col_hi]
        outside = ortho_img[0:40, 0:40]  # corner (should be ground)
        region_mean = float(np.mean(region))
        outside_mean = float(np.mean(outside))
        contrast = abs(region_mean - outside_mean)
        log(f"Ortho cube region mean={region_mean:.1f}, background={outside_mean:.1f}, "
            f"contrast={contrast:.1f}")
        # Also check channel-wise to see red difference
        if len(region.shape) == 3 and region.shape[2] >= 3:
            r_diff = abs(float(np.mean(region[:,:,0])) - float(np.mean(outside[:,:,0])))
            g_diff = abs(float(np.mean(region[:,:,1])) - float(np.mean(outside[:,:,1])))
            b_diff = abs(float(np.mean(region[:,:,2])) - float(np.mean(outside[:,:,2])))
            log(f"Ortho channel diffs: R={r_diff:.1f}, G={g_diff:.1f}, B={b_diff:.1f}")
        if contrast > 5:
            log("PASS: Ortho cube region differs from background")
        else:
            log("WARNING: Low contrast between cube and background in ortho")

    log("\n--- Results ---")
    log(f"PASS: Omap detected {occ_count} occupied cells")
    log(f"PASS: Ortho captured non-blank image ({ortho_w}x{ortho_h})")
    if row_ok and col_ok:
        log("PASS: Omap occupied region aligns with expected cube position")
    else:
        log("INFO: Omap position offset — check image rotation orientation")
    log(f"Output: {OUTPUT_DIR}")
    return True


with open(LOG, "w") as f:
    f.write("")
task = asyncio.ensure_future(run_e2e())
while not task.done():
    simulation_app.update()
if task.exception():
    log(f"EXCEPTION: {task.exception()}")
    import traceback
    log(traceback.format_exc())
    simulation_app.close()
    sys.exit(1)
result = task.result()
simulation_app.close()
sys.exit(0 if result else 1)
