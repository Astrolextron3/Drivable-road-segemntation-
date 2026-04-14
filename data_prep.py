
"""
Data Preparation Pipeline for Drivable Area Segmentation.
Uses nuScenes v1.0-mini HD map to generate image-mask pairs.

Pipeline (per image):
  Stage 1: Geometric Projection  — 3D map polygons → 2D camera image
  Stage 2: Morphological Cleanup — Fill holes, remove noise
  Stage 3: GrabCut Refinement    — Use projection as seed, refine with image colors
  Stage 4: Horizon Guard         — Erase sky/building regions from mask
  Stage 5: Edge Snapping         — Cut mask at physical curbs using Canny edges
  Stage 6: Largest Component     — Keep only the main road blob
  Stage 7: Quality Validation    — Flag/skip masks with abnormal coverage

Usage (from Anaconda Prompt):
    python data_prep.py
"""

import os
import numpy as np
from PIL import Image
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points

# ═══════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════
DATAROOT      = r"C:\Users\Arindam S Katoch\Desktop\v1.0-mini"
OUTPUT_DIR    = r"C:\Users\Arindam S Katoch\Desktop\v1.0-mini\all py codes\processed_wow"
IMG_H, IMG_W  = 256, 512
MAX_SAMPLES   = 300
CAM_KEYS      = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
]
SEARCH_RADIUS = 60        # meters around ego to search for drivable area

# Refinement parameters
HORIZON_FRAC   = 0.40     # mask out top 40% (sky/buildings)
GRABCUT_ITERS  = 3        # GrabCut iterations (more = slower but better)
MIN_ROAD_FRAC  = 0.03     # minimum road coverage to accept a mask
MAX_ROAD_FRAC  = 0.75     # maximum road coverage (reject if too much = bad mask)
EDGE_DILATE    = 2        # dilation for edge-snapping zone
MORPH_CLOSE_K  = 11       # kernel for closing holes
MORPH_OPEN_K   = 5        # kernel for removing noise


# ═══════════════════════════════════════════════════════════
#  NUSCENES SETUP
# ═══════════════════════════════════════════════════════════

def init_nuscenes():
    """Initialize nuScenes database and map APIs."""
    nusc = NuScenes(version="v1.0-mini", dataroot=DATAROOT, verbose=True)
    map_names = [
        "singapore-onenorth", "singapore-hollandvillage",
        "singapore-queenstown", "boston-seaport",
    ]
    nusc_maps = {name: NuScenesMap(dataroot=DATAROOT, map_name=name)
                 for name in map_names}
    return nusc, nusc_maps


def get_scene_map(nusc, nusc_maps, scene_token):
    """Get the NuScenesMap for a given scene."""
    log = nusc.get("log", nusc.get("scene", scene_token)["log_token"])
    return nusc_maps[log["location"]]


# ═══════════════════════════════════════════════════════════
#  STAGE 1: GEOMETRIC PROJECTION (3D Map → 2D Image)
# ═══════════════════════════════════════════════════════════

def world_to_camera(pts_3xN, ego_rot, ego_trans, cam_rot, cam_trans):
    """Transform 3D world points → ego frame → camera frame."""
    pts_ego = ego_rot.inverse.rotation_matrix @ (pts_3xN - ego_trans.reshape(3, 1))
    pts_cam = cam_rot.inverse.rotation_matrix @ (pts_ego - cam_trans.reshape(3, 1))
    return pts_cam


def stage1_geometric_projection(nusc, nusc_map, sample, cam_key, img_h, img_w):
    """
    Project drivable area polygons from HD map onto camera image.
    Uses fillPoly (respects concave road shapes).
    Returns a rough binary mask (uint8, 0/255).
    """
    cam_token = sample["data"][cam_key]
    cam_data  = nusc.get("sample_data", cam_token)
    ego_pose  = nusc.get("ego_pose", cam_data["ego_pose_token"])
    calib     = nusc.get("calibrated_sensor",
                         cam_data["calibrated_sensor_token"])

    ego_rot   = Quaternion(ego_pose["rotation"])
    ego_trans = np.array(ego_pose["translation"])
    cam_rot   = Quaternion(calib["rotation"])
    cam_trans = np.array(calib["translation"])
    intrinsic = np.array(calib["camera_intrinsic"])

    orig_w, orig_h = cam_data["width"], cam_data["height"]

    # Scale intrinsic to target resolution
    K = intrinsic.copy()
    K[0] *= img_w / orig_w
    K[1] *= img_h / orig_h

    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    # Query drivable area polygons near ego vehicle
    ex, ey = ego_trans[0], ego_trans[1]
    r = SEARCH_RADIUS
    patch = (ex - r, ey - r, ex + r, ey + r)
    records = nusc_map.get_records_in_patch(
        patch, ["drivable_area"], mode="intersect"
    )

    for token in records["drivable_area"]:
        da = nusc_map.get("drivable_area", token)
        for poly_token in da["polygon_tokens"]:
            poly  = nusc_map.get("polygon", poly_token)
            nodes = [nusc_map.get("node", n)
                     for n in poly["exterior_node_tokens"]]
            if len(nodes) < 3:
                continue

            pts_world = np.array([[n["x"], n["y"], 0.0] for n in nodes]).T
            pts_cam = world_to_camera(
                pts_world, ego_rot, ego_trans, cam_rot, cam_trans
            )

            # Only keep points in front of camera
            front = pts_cam[2] > 0.5
            if front.sum() < 3:
                continue

            pts_2d = view_points(pts_cam[:, front], K, normalize=True)
            xs = pts_2d[0].astype(np.int32)
            ys = pts_2d[1].astype(np.int32)

            valid = (xs >= 0) & (xs < img_w) & (ys >= 0) & (ys < img_h)
            if valid.sum() < 3:
                continue

            # fillPoly respects concave shapes (not fillConvexPoly!)
            pts_img = np.stack([xs[valid], ys[valid]], axis=1)
            cv2.fillPoly(mask, [pts_img], 255)

    # Fallback: if projection yielded almost nothing, use a road trapezoid
    road_ratio = mask.sum() / (255 * img_h * img_w)
    if road_ratio < 0.02:
        trap = np.zeros((img_h, img_w), dtype=np.uint8)
        pts = np.array([[
            (int(img_w * 0.05), img_h - 1),
            (int(img_w * 0.95), img_h - 1),
            (int(img_w * 0.65), int(img_h * 0.55)),
            (int(img_w * 0.35), int(img_h * 0.55)),
        ]], dtype=np.int32)
        cv2.fillPoly(trap, pts, 255)
        mask = trap

    return mask


# ═══════════════════════════════════════════════════════════
#  STAGE 2: MORPHOLOGICAL CLEANUP
# ═══════════════════════════════════════════════════════════

def stage2_morphological_cleanup(mask):
    """
    Fill small holes inside the road (lane markings, shadows)
    and remove small floating noise blobs.
    """
    # Close: fill holes
    k_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_CLOSE_K, MORPH_CLOSE_K)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    # Open: remove noise
    k_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_OPEN_K, MORPH_OPEN_K)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    return mask


# ═══════════════════════════════════════════════════════════
#  STAGE 3: GRABCUT REFINEMENT
# ═══════════════════════════════════════════════════════════

def stage3_grabcut_refinement(img_bgr_resized, rough_mask):
    """
    Use GrabCut with the geometric projection as initialization.
    GrabCut learns a color model for road vs non-road from the image
    and refines the boundary to follow actual color transitions.

    This is the key step that turns a rough projected mask into one
    that hugs the real road edges.
    """
    h, w = rough_mask.shape

    # Build GrabCut marker mask:
    #   GC_BGD (0)     = definitely background
    #   GC_FGD (1)     = definitely foreground (road)
    #   GC_PR_BGD (2)  = probably background
    #   GC_PR_FGD (3)  = probably foreground
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

    # Definite background: top 35% of image (sky/buildings)
    gc_mask[:int(h * 0.35), :] = cv2.GC_BGD

    # Definite foreground: eroded core of the projected mask
    # (erode to be conservative — only the "safe" interior is certain road)
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    core_road = cv2.erode(rough_mask, k_erode, iterations=2)
    gc_mask[core_road > 0] = cv2.GC_FGD

    # Probable foreground: the rest of the projected mask (edges uncertain)
    gc_mask[(rough_mask > 0) & (core_road == 0)] = cv2.GC_PR_FGD

    # Skip GrabCut if there's no definite foreground seed
    if (gc_mask == cv2.GC_FGD).sum() < 100:
        return rough_mask

    # Run GrabCut
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(img_bgr_resized, gc_mask, None,
                    bgd_model, fgd_model,
                    GRABCUT_ITERS, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        # GrabCut can fail on edge cases; fall back to rough mask
        return rough_mask

    # Extract result: foreground = GC_FGD or GC_PR_FGD
    refined = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    return refined


# ═══════════════════════════════════════════════════════════
#  STAGE 4: HORIZON GUARD
# ═══════════════════════════════════════════════════════════

def stage4_horizon_guard(mask):
    """Erase anything in the top portion of the image (sky/buildings)."""
    h = mask.shape[0]
    mask[:int(h * HORIZON_FRAC), :] = 0
    return mask


# ═══════════════════════════════════════════════════════════
#  STAGE 5: EDGE SNAPPING
# ═══════════════════════════════════════════════════════════

def stage5_edge_snapping(mask, img_bgr_resized):
    """
    Use Canny edge detection on the actual photo to find physical
    boundaries (curbs, barriers, grass edges). Cut the mask at these
    edges so it doesn't bleed onto non-road surfaces.
    """
    gray = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2GRAY)

    # Bilateral filter preserves edges while smoothing noise
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    edges = cv2.Canny(gray, 50, 150)

    # Dilate edges to create a "no-go zone" for the mask
    edge_zone = cv2.dilate(
        edges, np.ones((3, 3), np.uint8), iterations=EDGE_DILATE
    )
    mask[edge_zone > 0] = 0

    # Re-close to repair small breaks caused by edge cutting
    k_repair = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_repair)

    return mask


# ═══════════════════════════════════════════════════════════
#  STAGE 6: LARGEST CONNECTED COMPONENT
# ═══════════════════════════════════════════════════════════

def stage6_largest_component(mask):
    """Keep only the largest connected region (the actual road)."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8) * 255
    return mask


# ═══════════════════════════════════════════════════════════
#  STAGE 7: QUALITY VALIDATION
# ═══════════════════════════════════════════════════════════

def stage7_quality_check(mask, img_h, img_w):
    """
    Validate mask quality. Returns (is_valid, road_fraction).
    Rejects masks that are too empty or too full (likely bad projection).
    """
    road_frac = mask.sum() / (255 * img_h * img_w)
    is_valid = MIN_ROAD_FRAC <= road_frac <= MAX_ROAD_FRAC
    return is_valid, road_frac


# ═══════════════════════════════════════════════════════════
#  FULL PIPELINE
# ═══════════════════════════════════════════════════════════

def generate_clean_mask(nusc, nusc_map, sample, cam_key,
                        img_bgr_resized, img_h, img_w):
    """
    Run the full 7-stage mask generation pipeline.

    Args:
        nusc, nusc_map, sample, cam_key: nuScenes data references
        img_bgr_resized: The camera image resized to (img_h, img_w) in BGR
        img_h, img_w: Target mask dimensions

    Returns:
        (mask, is_valid, road_frac): Final mask + quality info
    """
    # Stage 1: Raw geometric projection
    mask = stage1_geometric_projection(
        nusc, nusc_map, sample, cam_key, img_h, img_w
    )

    # Stage 2: Morphological cleanup
    mask = stage2_morphological_cleanup(mask)

    # Stage 3: GrabCut refinement (image-aware boundary correction)
    mask = stage3_grabcut_refinement(img_bgr_resized, mask)

    # Stage 4: Horizon guard
    mask = stage4_horizon_guard(mask)

    # Stage 5: Edge snapping (cut at curbs/barriers)
    mask = stage5_edge_snapping(mask, img_bgr_resized)

    # Stage 6: Keep largest component only
    mask = stage6_largest_component(mask)

    # Stage 7: Quality validation
    is_valid, road_frac = stage7_quality_check(mask, img_h, img_w)

    return mask, is_valid, road_frac


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    """Process all nuScenes samples through the full pipeline."""
    nusc, nusc_maps = init_nuscenes()
    os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/masks",  exist_ok=True)

    saved = 0
    skipped = 0

    for scene in tqdm(nusc.scene, desc="Scenes"):
        if saved >= MAX_SAMPLES:
            break

        nusc_map = get_scene_map(nusc, nusc_maps, scene["token"])
        sample_token = scene["first_sample_token"]

        while sample_token and saved < MAX_SAMPLES:
            sample = nusc.get("sample", sample_token)

            for cam_key in CAM_KEYS:
                if saved >= MAX_SAMPLES:
                    break

                cam_data = nusc.get("sample_data", sample["data"][cam_key])
                img_path = os.path.join(DATAROOT, cam_data["filename"])
                img_bgr  = cv2.imread(img_path)
                if img_bgr is None:
                    continue

                # Resize image to target dimensions
                img_bgr_resized = cv2.resize(img_bgr, (IMG_W, IMG_H))

                # Run full 7-stage pipeline
                mask, is_valid, road_frac = generate_clean_mask(
                    nusc, nusc_map, sample, cam_key,
                    img_bgr_resized, IMG_H, IMG_W,
                )

                if not is_valid:
                    skipped += 1
                    continue

                # Save image (RGB) and mask
                img_rgb = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)

                fname = f"{saved:05d}"
                img_pil.save(f"{OUTPUT_DIR}/images/{fname}.jpg")
                cv2.imwrite(f"{OUTPUT_DIR}/masks/{fname}.png", mask)
                saved += 1

            sample_token = sample["next"]

    print(f"\n{'='*50}")
    print(f"✅ Pipeline complete!")
    print(f"   Saved   : {saved} image-mask pairs")
    print(f"   Skipped : {skipped} (failed quality check)")
    print(f"   Output  : {OUTPUT_DIR}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()