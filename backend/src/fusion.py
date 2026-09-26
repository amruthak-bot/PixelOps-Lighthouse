"""Build a clean cloud from sparse SfM points plus optional stereo back-projections."""
import numpy as np
import open3d as o3d


def fuse(sparse_pts, sparse_colors, depth_results, cfg, log=print):
    parts, cparts = [], []
    if sparse_pts is not None and len(sparse_pts):
        parts.append(np.asarray(sparse_pts, np.float64))
        cparts.append(np.asarray(sparse_colors, np.float64) / 255.0)
    for res, name in depth_results:
        pw = res["world_points"]
        if len(pw) == 0:
            continue
        parts.append(pw)
        cparts.append(np.tile([0.75, 0.78, 0.82], (len(pw), 1)))
        log(f"[fusion] +{len(pw)} pts from {name}")
    pts = np.vstack(parts)
    cols = np.vstack(cparts)

    extent = np.linalg.norm(pts.max(0) - pts.min(0))
    voxel = cfg.voxel_size_ratio * extent
    log(f"[fusion] raw {len(pts)} pts, extent {extent:.2f}, voxel {voxel:.4f}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    pcd = pcd.voxel_down_sample(voxel)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=cfg.sor_nb_neighbors,
                                            std_ratio=cfg.sor_std_ratio)
    log(f"[fusion] cleaned cloud: {len(pcd.points)} pts")
    return pcd, extent
