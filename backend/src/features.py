"""Feature extraction and pairwise geometric matching.

Stage 1 of the pipeline: detect SIFT keypoints in every image, match descriptors
between image pairs (Lowe ratio test), and keep only pairs that pass a RANSAC
fundamental-matrix check. Everything downstream (SfM, depth) builds on this
verified match graph.
"""
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS


def read_exif_focal_px(path):
    """Focal length in pixels from EXIF 35mm-equivalent focal length.

    f_px = f_35mm * image_width / 36  (full-frame reference width 36mm).
    Returns None when the tag is missing.
    """
    try:
        img = Image.open(path)
        w, _ = img.size
        exif = img._getexif() or {}
        for tag_id, val in exif.items():
            if TAGS.get(tag_id) == "FocalLengthIn35mmFilm":
                return float(val) * w / 36.0
    except Exception:
        pass
    return None


def load_gray(path, max_dim):
    """Load image as grayscale, downscaling so the long side <= max_dim."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"cannot read {path}")
    h, w = img.shape
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img, scale, (w, h)


def make_intrinsics(orig_size, focal_ratio, f_px=None):
    """Pinhole intrinsics with square pixels, principal point at image centre."""
    w, h = orig_size
    f = f_px if f_px else focal_ratio * max(w, h)
    K = np.array([[f, 0, w / 2.0],
                  [0, f, h / 2.0],
                  [0, 0, 1.0]])
    return K


def extract_sift(gray, n_features):
    sift = cv2.SIFT_create(nfeatures=n_features)
    kps, descs = sift.detectAndCompute(gray, None)
    if descs is None or len(kps) == 0:
        return np.zeros((0, 2), np.float32), None
    pts = np.array([kp.pt for kp in kps], np.float32)
    return pts, descs


def match_descriptors(d1, d2, ratio):
    """FLANN kNN matching + Lowe ratio test. Returns (idx1, idx2) pairs."""
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return np.zeros((0, 2), np.int64)
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    knn = flann.knnMatch(d1, d2, k=2)
    good = []
    for m_n in knn:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good.append((m.queryIdx, m.trainIdx))
    return np.array(good, np.int64).reshape(-1, 2)


def verify_pair(pts1, pts2, K, ransac_thresh):
    """RANSAC fundamental matrix check. Returns (inlier_matches_idx, F, n_inliers)."""
    if len(pts1) < 8:
        return None, None, 0
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC,
                                     ransacReprojThreshold=ransac_thresh, confidence=0.999)
    if F is None or mask is None:
        return None, None, 0
    inl = mask.ravel().astype(bool)
    return np.flatnonzero(inl), F, int(inl.sum())


class FeatureStore:
    """Holds per-image keypoints/descriptors and the verified match graph."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.names = []
        self.kps = []
        self.descs = []
        self.scales = []
        self.sizes = []
        self.Ks = []
        self.matches = {}

    def add_image(self, name, path):
        gray, scale, (w, h) = load_gray(path, self.cfg.max_image_dim)
        pts, descs = extract_sift(gray, self.cfg.sift_features)
        pts = pts / scale
        self.names.append(name)
        self.kps.append(pts)
        self.descs.append(descs)
        self.scales.append(scale)
        self.sizes.append((w, h))
        f_px = read_exif_focal_px(path)
        self.Ks.append(make_intrinsics((w, h), self.cfg.focal_ratio, f_px))
        return len(pts)

    def match_all_pairs(self, log=print):
        n = len(self.names)
        pairs = set()
        W = self.cfg.pair_window
        for i in range(n):
            for k in range(1, W + 1):
                if i + k < n:
                    pairs.add((i, i + k))
        for s in self.cfg.pair_strides:
            for i in range(0, n, 3):
                if i + s < n:
                    pairs.add((i, i + s))
        pairs = sorted(pairs)
        if len(pairs) > self.cfg.max_pairs:
            step = len(pairs) / self.cfg.max_pairs
            pairs = [pairs[int(k * step)] for k in range(self.cfg.max_pairs)]
            log(f"[features] capping pairwise matching to {len(pairs)} pairs")
        kept, total_inl = 0, 0
        for t, (i, j) in enumerate(pairs):
            m = match_descriptors(self.descs[i], self.descs[j], self.cfg.ratio_test)
            if len(m) == 0:
                continue
            inl_idx, F, n_inl = verify_pair(self.kps[i][m[:, 0]], self.kps[j][m[:, 1]],
                                            self.Ks[i], self.cfg.ransac_thresh_px)
            if n_inl >= self.cfg.min_inliers_pair:
                self.matches[(i, j)] = m[inl_idx]
                kept += 1
                total_inl += n_inl
            if (t + 1) % 200 == 0:
                log(f"[features] matched {t + 1}/{len(pairs)} pairs, kept {kept}")
        log(f"[features] kept {kept}/{len(pairs)} pairs ({total_inl} inlier matches)")
        return kept
