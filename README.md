# PixelOps Lighthouse

Merged Pixel-Ops UAV photogrammetry project containing:

- Vercel-ready static frontend in `site/`
- Python/OpenCV/Open3D reconstruction backend in `backend/src/`
- Vercel build configuration
- Submitted-run metrics from the verified Lighthouse reconstruction

## Submitted run

- 12 input images
- 27 verified image pairs
- 10 / 12 registered cameras
- 1,352 sparse SfM points
- 5 stereo depth maps
- 1,161 final filtered points
- 5,493 mesh vertices
- 10,829 mesh triangles
- 45.2 s recorded runtime
- Dense depth fusion disabled in the submitted robust mode

## Pipeline

`Images -> SIFT -> FLANN -> Lowe -> RANSAC -> Essential Matrix / PnP -> Triangulation / SfM -> SGBM depth -> filtering -> Poisson mesh`

## Vercel

Vercel reads `vercel.json`, runs:

```bash
npm run build
```

and publishes the generated `dist/` directory.

The heavy OpenCV/Open3D reconstruction backend is kept as source code for local/external compute rather than being executed inside the static Vercel frontend.
