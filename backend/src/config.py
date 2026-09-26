"""Shared configuration for the Pixel-Ops trial-round reconstruction pipeline."""
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    # --- input / output -----------------------------------------------------
    image_dir: str = "dataset/images"
    output_dir: str = "outputs"

    # --- features -----------------------------------------------------------
    max_image_dim: int = 1600
    sift_features: int = 8000
    ratio_test: float = 0.75
    ransac_thresh_px: float = 3.0
    min_inliers_pair: int = 40
    max_pairs: int = 2500
    pair_window: int = 6
    pair_strides: tuple = (20, 40)

    # --- intrinsics ---------------------------------------------------------
    focal_ratio: float = 1.2

    # --- incremental SfM ----------------------------------------------------
    pnp_min_2d3d: int = 30
    pnp_min_inliers: int = 25
    pnp_reproj_thresh: float = 6.0
    refine_iters: int = 3
    min_track_len_for_ba: int = 2

    # --- depth maps ---------------------------------------------------------
    depth_pairs: int = 6
    fuse_depth: bool = False
    depth_scale: float = 0.25
    sgbm_block: int = 5
    sgbm_num_disp: int = 256
    depth_subsample: int = 3

    # --- fusion -------------------------------------------------------------
    voxel_size_ratio: float = 0.001
    sor_nb_neighbors: int = 30
    sor_std_ratio: float = 2.0

    # --- meshing ------------------------------------------------------------
    poisson_depth: int = 9
    density_quantile: float = 0.05

    random_seed: int = 7
