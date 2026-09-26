"""Surface mesh from the cleaned point cloud + per-vertex colours from the images.

Steps: orient normals towards the cameras -> screened Poisson reconstruction
-> drop low-density vertices -> colour every vertex by projecting it into the
registered views (angle-weighted average of the observing pixels).
"""
import os
import cv2
import numpy as np
import open3d as o3d


def reconstruct_mesh(pcd, cams, fs, img_dir, cfg, log=print):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=cfg.voxel_size_ratio * 20, max_nn=30))
    centres = np.array([-R.T @ t for R, t in cams.values()])
    pcd.orient_normals_towards_camera_location(centres.mean(0))
    log("[mesh] running Poisson reconstruction ...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=cfg.poisson_depth)
    d = np.asarray(densities)
    keep = d > np.quantile(d, cfg.density_quantile)
    mesh.remove_vertices_by_mask(~keep)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    log(f"[mesh] {len(mesh.vertices)} verts / {len(mesh.triangles)} tris")
    colour_vertices(mesh, pcd, cams, fs, img_dir, log)
    mesh.compute_vertex_normals()
    return mesh


def colour_vertices(mesh, pcd, cams, fs, img_dir, log=print, max_dim=1200):
    verts = np.asarray(mesh.vertices)
    vcols = np.zeros((len(verts), 3))
    vw = np.zeros(len(verts))
    n_done = 0
    for i, (R, t) in cams.items():
        path = os.path.join(img_dir, fs.names[i])
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h0, w0 = bgr.shape[:2]
        s = min(1.0, max_dim / max(h0, w0))
        if s < 1.0:
            bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        del bgr
        K = fs.Ks[i].copy() * s
        K[2, 2] = 1.0
        centre = -R.T @ t
        h, w = img.shape[:2]
        Xc = (R @ verts.T + t.reshape(3, 1)).T
        front = Xc[:, 2] > 0
        z = Xc[:, 2].clip(min=1e-9)
        uv = (K @ (Xc / z[:, None]).T).T[:, :2]
        inside = front & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        idx = np.flatnonzero(inside)
        if len(idx) == 0:
            continue
        view_dir = centre - verts[idx]
        view_dir /= np.linalg.norm(view_dir, axis=1, keepdims=True) + 1e-9
        nrm = np.asarray(mesh.vertex_normals)[idx]
        wgt = np.clip((view_dir * nrm).sum(1), 0, 1) ** 2 + 1e-3
        px = np.clip(uv[idx].astype(int), [0, 0], [w - 1, h - 1])
        col = img[px[:, 1], px[:, 0]]
        vcols[idx] += col * wgt[:, None]
        vw[idx] += wgt
        n_done += 1
        del img
    good = vw > 0
    vcols[good] /= vw[good, None]
    vcols[~good] = [0.7, 0.72, 0.75]
    mesh.vertex_colors = o3d.utility.Vector3dVector(vcols)
    log(f"[mesh] coloured {good.sum()}/{len(verts)} vertices from {n_done} views")
