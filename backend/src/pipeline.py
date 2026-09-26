"""Pixel-Ops trial-round reconstruction pipeline (orchestrator).

Stages:
  1. features : SIFT + ratio-test + RANSAC-F verified match graph
  2. sfm      : incremental SfM (E-init, PnP registration, DLT triangulation,
                alternating refinement) -> sparse point cloud + camera poses
  3. depth    : calibrated stereo rectification + SGBM on best pairs
  4. fusion   : build cleaned point cloud; dense-depth fusion is optional (.ply)
  5. mesh     : Poisson surface + vertex colours (.ply / .obj)
"""
import argparse
import json
import os
import sys
import time

from config import PipelineConfig
from features import FeatureStore, match_descriptors
from sfm import IncrementalSfM, build_tracks
from depth import (pick_stereo_pairs, rectify_geometry, rectify_images,
                   check_geometry, sgbm_depth, depth_to_world, save_depth_pngs)
from fusion import fuse
from meshing import reconstruct_mesh


def log(msg):
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-mesh", action="store_true")
    ap.add_argument("--fuse-depth", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cfg = PipelineConfig(image_dir=args.images, output_dir=args.out)
    cfg.fuse_depth = bool(args.fuse_depth)
    os.makedirs(args.out, exist_ok=True)
    depth_dir = os.path.join(args.out, "depth")
    os.makedirs(depth_dir, exist_ok=True)
    t0 = time.time()
    summary = {"stages": {}}

    names = sorted(f for f in os.listdir(args.images)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not names:
        sys.exit("no images found in " + args.images)
    if args.quick:
        names = names[:6]
        log("[quick] validation mode: using first 6 images only")

    fs = FeatureStore(cfg)
    for n in names:
        nf = fs.add_image(n, os.path.join(args.images, n))
        log(f"[features] {n}: {nf} keypoints")
    fs.match_all_pairs(log)
    summary["stages"]["features"] = {
        "images": len(names),
        "verified_pairs": len(fs.matches),
        "time_s": round(time.time() - t0, 1)
    }

    t1 = time.time()
    tracks = build_tracks(fs.matches, len(names))
    log(f"[sfm] {len(tracks)} tracks")
    sfm = IncrementalSfM(fs, cfg, log)
    if not sfm.run(args.images, tracks):
        sys.exit("SfM failed - not enough geometry")
    summary["stages"]["sfm"] = {
        "registered": len(sfm.cams),
        "sparse_points": int(len(sfm.points)),
        "time_s": round(time.time() - t1, 1)
    }

    t2 = time.time()
    candidates = pick_stereo_pairs(fs, sfm, tracks, cfg.depth_pairs * 40)
    depth_results = []
    for (i, j, med_full, p95_full, _) in candidates:
        if len(depth_results) >= cfg.depth_pairs:
            break
        Ri, ti = sfm.cams[i]
        Rj, tj = sfm.cams[j]
        w_full, h_full = fs.sizes[i]
        geom = rectify_geometry(fs.Ks[i], fs.Ks[j], Ri, ti, Rj, tj,
                                cfg.depth_scale, w_full, h_full)
        if geom is None:
            continue
        shared = [sfm.points[r] for r, tr in enumerate(sfm.pt_tracks)
                  if i in tr and j in tr][:400]
        if len(shared) < 12:
            continue
        import numpy as np
        shared = np.array(shared)
        ok, med_dy, med_d, p95_d = check_geometry(
            geom, shared, Ri, ti, Rj, tj, cfg.sgbm_num_disp)
        if not ok:
            continue
        rect = rectify_images(args.images, fs.names[i], fs.names[j], geom, cfg.depth_scale)
        res = sgbm_depth(rect, cfg)
        res["world_points"] = depth_to_world(res, cfg.depth_subsample)
        k = len(depth_results)
        prefix = os.path.join(depth_dir, f"depth_{k:02d}_{os.path.splitext(fs.names[i])[0]}")
        save_depth_pngs(res["depth"], res["valid"], prefix)
        depth_results.append((res, f"{fs.names[i]}+{fs.names[j]}"))

    summary["stages"]["depth"] = {
        "pairs": len(depth_results),
        "time_s": round(time.time() - t2, 1)
    }

    depth_for_fusion = depth_results if cfg.fuse_depth else []
    pcd, extent = fuse(sfm.points, sfm.colors, depth_for_fusion, cfg, log)

    import open3d as o3d
    o3d.io.write_point_cloud(os.path.join(args.out, "pointcloud.ply"), pcd)
    summary["stages"]["fusion"] = {
        "points": len(pcd.points),
        "dense_depth_fused": bool(cfg.fuse_depth),
        "time_s": 0.0
    }

    if not args.skip_mesh:
        t4 = time.time()
        mesh = reconstruct_mesh(pcd, sfm.cams, fs, args.images, cfg, log)
        o3d.io.write_triangle_mesh(os.path.join(args.out, "mesh.ply"), mesh)
        o3d.io.write_triangle_mesh(os.path.join(args.out, "mesh.obj"), mesh)
        summary["stages"]["mesh"] = {
            "vertices": len(mesh.vertices),
            "triangles": len(mesh.triangles),
            "time_s": round(time.time() - t4, 1)
        }

    summary["total_time_s"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    cams = {
        fs.names[i]: {"K": fs.Ks[i].tolist(), "R": R.tolist(), "t": t.tolist()}
        for i, (R, t) in sfm.cams.items()
    }
    with open(os.path.join(args.out, "cameras.json"), "w") as f:
        json.dump(cams, f)
    log("[done] " + json.dumps(summary))


if __name__ == "__main__":
    main()
