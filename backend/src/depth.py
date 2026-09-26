"""Dense depth maps from calibrated stereo pairs.

For the best covisible image pairs (ranked by triangulation quality) we:
  1. Rectify the pair from the recovered SfM geometry,
  2. Run semi-global block matching (SGBM),
  3. Convert disparity into per-pixel depth in the reference camera frame.
"""
import os
import cv2
import numpy as np


def relative_pose(Ri, ti, Rj, tj):
    R_ji = Rj @ Ri.T
    t_ji = tj - R_ji @ ti
    return R_ji, t_ji


def predict_disp_stats(fs, sfm, i, j, pt_ids, max_pts=200):
    Ri, ti = sfm.cams[i]
    Rj, tj = sfm.cams[j]
    R_ji, t_ji = relative_pose(Ri, ti, Rj, tj)
    base = np.linalg.norm(t_ji)
    if base < 1e-9:
        return None, None
    f = fs.Ks[i][0, 0]
    disps = []
    for r in pt_ids[:max_pts]:
        z = (Ri @ sfm.points[r] + ti)[2]
        if z > 1e-6:
            disps.append(f * base / z)
    if len(disps) < 25:
        return None, None
    return float(np.median(disps)), float(np.percentile(disps, 95))


def pick_stereo_pairs(fs, sfm, tracks, n_pairs):
    reg = set(sfm.cams)
    vis = {}
    for r, tr in enumerate(sfm.pt_tracks):
        for v in tr:
            if v in reg:
                vis.setdefault(v, set()).add(r)
    reg_list = sorted(reg)
    scored = []
    for a in range(len(reg_list)):
        for b in range(a + 1, len(reg_list)):
            i, j = reg_list[a], reg_list[b]
            common = list(vis.get(i, set()) & vis.get(j, set()))
            cnt = len(common)
            if cnt < 25:
                continue
            med_full, p95_full = predict_disp_stats(fs, sfm, i, j, common)
            if med_full is None:
                continue
            Ri, ti = sfm.cams[i]
            Rj, tj = sfm.cams[j]
            _, t_ji = relative_pose(Ri, ti, Rj, tj)
            base = np.linalg.norm(t_ji)
            lateral = abs(t_ji[0]) / base
            if abs(t_ji[2]) / base > 0.92:
                continue
            if not (8.0 < med_full < 4000.0):
                continue
            scale = min(0.25, 230.0 / p95_full)
            if scale < 1.0 / 16:
                continue
            scored.append((cnt * (0.2 + lateral) * scale, i, j, med_full, p95_full, scale))
    scored.sort(reverse=True)
    return [(i, j, mf, p95, s) for _, i, j, mf, p95, s in scored[:n_pairs]]


def rectify_geometry(Ki, Kj, Ri, ti, Rj, tj, scale, w_full, h_full):
    Kis = Ki * scale
    Kjs = Kj * scale
    Kis[2, 2] = Kjs[2, 2] = 1.0
    w, h = int(w_full * scale), int(h_full * scale)
    R_ji = Rj @ Ri.T
    t_ji = tj - R_ji @ ti
    b_i = R_ji.T @ t_ji
    B = float(np.linalg.norm(b_i))
    if B < 1e-9:
        return None
    x = -b_i / B
    z0 = np.array([0.0, 0.0, 1.0])
    if abs(float(x @ z0)) > 0.9:
        return None
    y = np.cross(z0, x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    R1 = np.stack([x, y, z])
    R2 = R1 @ R_ji.T
    t_check = R2 @ t_ji
    if not (abs(t_check[0] + B) < 1e-6 * B and abs(t_check[1]) < 1e-6 * B and abs(t_check[2]) < 1e-6 * B):
        return None
    f = float((Kis[0, 0] + Kjs[0, 0]) / 2.0)
    cx, cy = w / 2.0, h / 2.0
    P1 = np.array([[f, 0, cx, 0], [0, f, cy, 0], [0, 0, 1, 0]])
    P2 = np.array([[f, 0, cx, -f * B], [0, f, cy, 0], [0, 0, 1, 0]])
    R_ri = R1 @ Ri
    t_ri = R1 @ ti
    R_wr = R_ri.T
    t_wr = -R_wr @ t_ri
    return dict(R1=R1, R2=R2, P1=P1, P2=P2, f=f, cx=cx, cy=cy, B=B,
                R_wr=R_wr, t_wr=t_wr, w=w, h=h, Kis=Kis, Kjs=Kjs)


def rectify_images(img_dir, name_i, name_j, geom, scale):
    gray_i = cv2.imread(os.path.join(img_dir, name_i), cv2.IMREAD_GRAYSCALE)
    gray_j = cv2.imread(os.path.join(img_dir, name_j), cv2.IMREAD_GRAYSCALE)
    if scale != 1.0:
        gray_i = cv2.resize(gray_i, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray_j = cv2.resize(gray_j, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    w, h = geom["w"], geom["h"]
    R1, R2, P1, P2 = geom["R1"], geom["R2"], geom["P1"], geom["P2"]
    Kis, Kjs = geom["Kis"], geom["Kjs"]
    map1x, map1y = cv2.initUndistortRectifyMap(Kis, np.zeros(5), R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(Kjs, np.zeros(5), R2, P2, (w, h), cv2.CV_32FC1)
    r_i = cv2.remap(gray_i, map1x, map1y, cv2.INTER_LINEAR)
    r_j = cv2.remap(gray_j, map2x, map2y, cv2.INTER_LINEAR)
    return dict(r_i=r_i, r_j=r_j, geom=geom)


def check_geometry(geom, pts_world, Ri, ti, Rj, tj, num_disp, max_dy_px=3.0, min_pts=12):
    if len(pts_world) < min_pts:
        return False, 0.0, 0.0, 0.0
    R1, R2, P1, P2 = geom["R1"], geom["R2"], geom["P1"], geom["P2"]
    dys, disps = [], []
    for X in pts_world:
        x1r = P1 @ np.append(R1 @ (Ri @ X + ti), 1.0)
        x2r = P2 @ np.append(R2 @ (Rj @ X + tj), 1.0)
        if x1r[2] <= 0 or x2r[2] <= 0:
            continue
        x1r, x2r = x1r / x1r[2], x2r / x2r[2]
        dys.append(abs(x1r[1] - x2r[1]))
        disps.append(x1r[0] - x2r[0])
    if len(dys) < min_pts:
        return False, 0.0, 0.0, 0.0
    dys, disps = np.array(dys), np.array(disps)
    med_dy = float(np.median(dys))
    med_d = float(np.median(disps))
    p5_d = float(np.percentile(disps, 5))
    p95_d = float(np.percentile(disps, 95))
    ok = med_dy < max_dy_px and p5_d > 0.5 and p95_d < num_disp - 8 and med_d > 4.0
    return bool(ok), med_dy, med_d, p95_d


def sgbm_depth(rect, cfg):
    r_i, r_j, geom = rect["r_i"], rect["r_j"], rect["geom"]
    num_disp = cfg.sgbm_num_disp
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=num_disp, blockSize=cfg.sgbm_block,
        P1=8 * cfg.sgbm_block ** 2, P2=32 * cfg.sgbm_block ** 2,
        disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100,
        speckleRange=32, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = sgbm.compute(r_i, r_j).astype(np.float32) / 16.0
    valid = (disp > 0.5) & (disp < num_disp - 1)
    f, cx, cy, B = geom["f"], geom["cx"], geom["cy"], geom["B"]
    h, w = disp.shape
    ys, xs = np.nonzero(valid)
    d = disp[ys, xs]
    Z = f * B / d
    X = (xs.astype(np.float32) - cx) * Z / f
    Y = (ys.astype(np.float32) - cy) * Z / f
    points_rect = np.zeros((h, w, 3), np.float32)
    points_rect[ys, xs, 0] = X
    points_rect[ys, xs, 1] = Y
    points_rect[ys, xs, 2] = Z
    depth = np.zeros_like(disp)
    depth[valid] = Z
    depth[~np.isfinite(depth)] = 0
    depth[depth < 0] = 0
    return dict(depth=depth, valid=valid, points_rect=points_rect,
                R_wr=geom["R_wr"], t_wr=geom["t_wr"], rect_gray=r_i)


def depth_to_world(res, subsample):
    depth, valid = res["depth"], res["valid"]
    ys, xs = np.nonzero(valid[::subsample, ::subsample])
    ys, xs = ys * subsample, xs * subsample
    pr = res["points_rect"][ys, xs]
    return (res["R_wr"] @ pr.T).T + res["t_wr"]


def save_depth_pngs(depth, valid, out_prefix):
    d = depth.copy()
    vmax = np.percentile(d[valid], 99.5) if valid.any() else 1.0
    scale = 65535.0 / max(vmax, 1e-9)
    png16 = np.zeros_like(d, dtype=np.uint16)
    png16[valid] = np.clip(d[valid] * scale, 0, 65535).astype(np.uint16)
    cv2.imwrite(out_prefix + "_depth16.png", png16)
    vis = np.zeros((*d.shape, 3), np.uint8)
    norm = np.clip(d / max(vmax, 1e-9), 0, 1)
    vis[valid] = cv2.applyColorMap((norm[valid] * 255).astype(np.uint8), cv2.COLORMAP_INFERNO).reshape(-1, 3)
    cv2.imwrite(out_prefix + "_vis.png", vis)
    return float(vmax)
