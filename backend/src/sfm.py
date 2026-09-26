"""Incremental Structure-from-Motion for Pixel-Ops."""
import os
import cv2
import numpy as np


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        p = self.p.setdefault(x, x)
        if p != x:
            self.p[x] = self.find(p)
        return self.p[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def proj_matrix(K, R, t):
    return K @ np.hstack([R, t.reshape(3, 1)])


def triangulate_multiview(obs, Ps):
    A = []
    for (x, y), P in zip(obs, Ps):
        A += [x * P[2] - P[0], y * P[2] - P[1]]
    _, _, Vt = np.linalg.svd(np.stack(A))
    X = Vt[-1]
    return None if abs(X[3]) < 1e-9 else X[:3] / X[3]


def reproj_errors(Xs, uv, K, R, t):
    Xc = (R @ Xs.T + t.reshape(3, 1)).T
    z = Xc[:, 2].clip(min=1e-9)
    proj = (K @ (Xc / z[:, None]).T).T[:, :2]
    return np.linalg.norm(proj - uv, axis=1)


def build_tracks(matches, n_images):
    uf = UnionFind()
    nodes = set()
    for (i, j), m in matches.items():
        for a, b in m:
            a, b = (i, int(a)), (j, int(b))
            uf.union(a, b)
            nodes.update((a, b))
    groups = {}
    for node in nodes:
        groups.setdefault(uf.find(node), {})[node[0]] = node[1]
    return [g for g in groups.values() if len(g) >= 2]


class IncrementalSfM:
    def __init__(self, fs, cfg, log=print):
        self.fs, self.cfg, self.log = fs, cfg, log
        self.cams = {}
        self.points = None
        self.colors = None
        self.pt_tracks = []

    def _essential_init(self, i, j):
        m = self.fs.matches[(i, j)]
        K1, K2 = self.fs.Ks[i], self.fs.Ks[j]
        p1 = np.hstack([self.fs.kps[i][m[:, 0]], np.ones((len(m), 1))])
        p2 = np.hstack([self.fs.kps[j][m[:, 1]], np.ones((len(m), 1))])
        n1 = (np.linalg.inv(K1) @ p1.T).T[:, :2]
        n2 = (np.linalg.inv(K2) @ p2.T).T[:, :2]
        E, mask = cv2.findEssentialMat(n1, n2, np.eye(3), method=cv2.RANSAC,
                                       prob=0.999, threshold=0.01)
        if E is None or mask is None:
            return None
        inl = mask.ravel().astype(bool)
        if inl.sum() < self.cfg.min_inliers_pair:
            return None
        _, R, t, _ = cv2.recoverPose(E, n1[inl], n2[inl], np.eye(3))
        return R, t.ravel(), m[inl]

    def initialise(self, tracks):
        for i, j in sorted(self.fs.matches, key=lambda k: -len(self.fs.matches[k]))[:25]:
            res = self._essential_init(i, j)
            if res is None:
                continue
            R, t, m = res
            P1 = proj_matrix(self.fs.Ks[i], np.eye(3), np.zeros(3))
            P2 = proj_matrix(self.fs.Ks[j], R, t)
            Xh = cv2.triangulatePoints(P1, P2,
                                       self.fs.kps[i][m[:, 0]].T,
                                       self.fs.kps[j][m[:, 1]].T)
            X = (Xh[:3] / Xh[3]).T
            ok = [(R @ x + t)[2] > 0 and x[2] > 0 for x in X]
            if sum(ok) < 0.6 * len(X):
                continue
            self.cams[i] = (np.eye(3), np.zeros(3))
            self.cams[j] = (R, t)
            self._triangulate_new_tracks(tracks, [j])
            self.log(f"[sfm] initialised pair {i},{j}")
            return True
        return False

    def _track_index(self):
        return {tuple(sorted(tr.items())): r for r, tr in enumerate(self.pt_tracks)}

    def _triangulate_new_tracks(self, tracks, new_imgs):
        existing = self._track_index()
        new_pts, new_trs = [], []
        reg = set(self.cams)
        for tr in tracks:
            if tuple(sorted(tr.items())) in existing:
                continue
            views = [v for v in tr if v in reg]
            if len(views) < 2 or not any(v in new_imgs for v in views):
                continue
            centers = {v: -self.cams[v][0].T @ self.cams[v][1] for v in views}
            best = None
            for a in range(len(views)):
                for b in range(a + 1, len(views)):
                    d = np.linalg.norm(centers[views[a]] - centers[views[b]])
                    if best is None or d > best[0]:
                        best = (d, views[a], views[b])
            _, vi, vj = best
            obs = np.array([self.fs.kps[vi][tr[vi]], self.fs.kps[vj][tr[vj]]])
            Ps = [proj_matrix(self.fs.Ks[v], *self.cams[v]) for v in (vi, vj)]
            X = triangulate_multiview(obs, Ps)
            if X is not None and all((self.cams[v][0] @ X + self.cams[v][1])[2] > 0 for v in (vi, vj)):
                new_pts.append(X)
                new_trs.append(tr)
        if new_pts:
            arr = np.array(new_pts)
            self.points = arr if self.points is None else np.vstack([self.points, arr])
            self.pt_tracks.extend(new_trs)
        return len(new_pts)

    def _pnp_register(self, img_idx):
        uv, X = [], []
        for r, tr in enumerate(self.pt_tracks):
            if img_idx in tr:
                uv.append(self.fs.kps[img_idx][tr[img_idx]])
                X.append(self.points[r])
        if len(X) < self.cfg.pnp_min_2d3d:
            return False
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            np.array(X, np.float64), np.array(uv, np.float32),
            self.fs.Ks[img_idx], None,
            reprojectionError=self.cfg.pnp_reproj_thresh,
            confidence=0.999, iterationsCount=2000)
        if not ok or inl is None or len(inl) < self.cfg.pnp_min_inliers:
            return False
        R, _ = cv2.Rodrigues(rvec)
        self.cams[img_idx] = (R, tvec.ravel())
        return True

    def grow(self, tracks):
        progress = True
        while progress:
            progress = False
            for i in range(len(self.fs.names)):
                if i in self.cams:
                    continue
                if self._pnp_register(i):
                    self._triangulate_new_tracks(tracks, [i])
                    progress = True

    def refine(self):
        for _ in range(self.cfg.refine_iters):
            for i in list(self.cams):
                if i == min(self.cams):
                    continue
                self._pnp_register(i)
            keep_pts, keep_trs = [], []
            for tr in self.pt_tracks:
                views = [v for v in tr if v in self.cams]
                if len(views) < 2:
                    continue
                obs = np.array([self.fs.kps[v][tr[v]] for v in views])
                Ps = [proj_matrix(self.fs.Ks[v], *self.cams[v]) for v in views]
                X = triangulate_multiview(obs, Ps)
                if X is None:
                    continue
                if not all((self.cams[v][0] @ X + self.cams[v][1])[2] > 0 for v in views):
                    continue
                errs = [reproj_errors(X[None], np.array([self.fs.kps[v][tr[v]]]),
                                      self.fs.Ks[v], *self.cams[v])[0] for v in views]
                if np.median(errs) < 4.0:
                    keep_pts.append(X)
                    keep_trs.append(tr)
            self.points = np.array(keep_pts) if keep_pts else None
            self.pt_tracks = keep_trs
        self._compute_colors()

    def _compute_colors(self):
        if self.points is None:
            return
        self.colors = np.zeros((len(self.points), 3), np.uint8)
        acc = np.zeros((len(self.points), 3), np.float32)
        counts = np.zeros(len(self.points))
        per_img = {}
        for r, tr in enumerate(self.pt_tracks):
            for v, k in tr.items():
                if v in self.cams:
                    per_img.setdefault(v, []).append((r, k))
        for v, rk in per_img.items():
            bgr = cv2.imread(os.path.join(self._img_dir, self.fs.names[v]))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            for r, k in rk:
                x, y = map(lambda z: int(round(z)), self.fs.kps[v][k])
                x, y = min(max(x, 0), w - 1), min(max(y, 0), h - 1)
                acc[r] += rgb[y, x]
                counts[r] += 1
        seen = counts > 0
        self.colors[seen] = (acc[seen] / counts[seen, None]).astype(np.uint8)
        self.colors[~seen] = 128

    def run(self, img_dir, tracks):
        self._img_dir = img_dir
        if not self.initialise(tracks):
            return False
        self.grow(tracks)
        self.refine()
        return self.points is not None and len(self.cams) >= 2
