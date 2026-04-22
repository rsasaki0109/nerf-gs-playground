# GS Mapper Launch Kit

External SLAM outputs to browser-viewable 3D Gaussian Splats.

GS Mapper turns photos, robotics logs, and MASt3R-SLAM / VGGT-SLAM 2.0 / Pi3 / LoGeR artifacts into trainable 3DGS outputs with WebGL, Spark, and WebGPU viewers.

## Links

- [Project page](https://rsasaki0109.github.io/gs-mapper/) - First-stop page with the GS Mapper pitch and viewer entry points.
- [Live splat viewer](https://rsasaki0109.github.io/gs-mapper/splat.html) - WebGL scene picker with eight bundled comparison splats.
- [Spark mobile / VR viewer](https://rsasaki0109.github.io/gs-mapper/splat_spark.html) - Spark viewer for mobile and WebXR-capable devices.
- [WebGPU viewer](https://rsasaki0109.github.io/gs-mapper/splat_webgpu.html) - GPU-sort viewer for Chrome, Edge, and WebGPU-enabled browsers.
- [GitHub repository](https://github.com/rsasaki0109/gs-mapper) - Source, README benchmarks, external SLAM import docs, and tests.

## Where To Post

### X / Twitter

- URL: https://x.com/intent/tweet
- Audience: 3DGS, SLAM, robotics, and WebGPU builders who want a fast demo link.
- Angle: Lead with the live viewer and keep it visual. Use the short social post.
- Copy block: `short-social`

### Hacker News Show HN

- URL: https://news.ycombinator.com/submit
- Audience: Graphics, mapping, robotics, and developer-tool readers who inspect repos.
- Angle: Frame it as a small open-source bridge from external SLAM artifacts to browser demos.
- Copy block: `community-post`

### LinkedIn

- URL: https://www.linkedin.com/feed/
- Audience: Robotics, autonomy, geospatial, and simulation engineers.
- Angle: Emphasize the practical handoff from research SLAM outputs to reviewable artifacts.
- Copy block: `technical-social`

### Reddit communities

- URL: https://www.reddit.com/search/?q=3D%20Gaussian%20Splatting%20SLAM&type=communities
- Audience: Subreddits around Gaussian Splatting, photogrammetry, 3D scanning, and robotics.
- Angle: Pick one relevant community, check its rules, and post the community copy with screenshots.
- Copy block: `community-post`

### GitHub awesome lists

- URL: https://github.com/search?q=awesome+3d+gaussian+splatting&type=repositories
- Audience: Maintainers of curated 3DGS, SLAM, NeRF, robotics, and WebGPU resource lists.
- Angle: Open a small PR with the awesome-list entry and link directly to the live demo.
- Copy block: `awesome-list`

### Japanese robotics channels

- URL: https://rsasaki0109.github.io/gs-mapper/
- Audience: Japanese robotics, mapping, autonomy, and computer-vision builders.
- Angle: Use the Japanese announcement and point people to the live splat viewer first.
- Copy block: `japanese`


## Copy Blocks

### Short social post (233/280 chars)

```text
GS Mapper turns photos, robotics logs, and external SLAM outputs (MASt3R-SLAM, VGGT-SLAM 2.0, Pi3, LoGeR) into browser-viewable 3D Gaussian Splats.

Live demos: https://rsasaki0109.github.io/gs-mapper/splat.html
#3DGS #SLAM #Robotics
```

### Technical social post

```text
I released GS Mapper: a glue layer from DUSt3R / MASt3R pose-free preprocessing and MASt3R-SLAM / VGGT-SLAM 2.0 / Pi3 / LoGeR exported artifacts into gsplat training and browser-viewable .splat demos.

It ships external SLAM dry-run manifests, candidate-resolution traces, and eight public comparison scenes.

https://github.com/rsasaki0109/gs-mapper
```

### Community post

```text
GS Mapper is a small open-source bridge for turning visual geometry outputs into 3D Gaussian Splatting demos. It accepts image folders, robotics logs, and external SLAM artifacts, then trains or exports browser-viewable .splat files.

The current demo set compares supervised GNSS + LiDAR, DUSt3R, MASt3R, VGGT-SLAM 2.0, and MASt3R-SLAM outputs on outdoor robotics scenes. The external SLAM import path has dry-run manifests so missing trajectories, point clouds, and image-directory mismatches are caught before GPU training.

Live demo: https://rsasaki0109.github.io/gs-mapper/splat.html
Repo: https://github.com/rsasaki0109/gs-mapper
```

### Awesome-list entry

```text
- [GS Mapper](https://github.com/rsasaki0109/gs-mapper) - Converts photos, robotics logs, and MASt3R-SLAM / VGGT-SLAM / Pi3 / LoGeR artifacts into trainable 3DGS outputs and browser-viewable WebGL / WebGPU splat demos.
```

### Japanese announcement

```text
GS Mapper を公開しました。写真フォルダ、ロボティクスログ、MASt3R-SLAM / VGGT-SLAM 2.0 / Pi3 / LoGeR の出力を、学習可能な 3D Gaussian Splatting と WebGL / WebGPU ビューアにつなぐツールです。

Live demo: https://rsasaki0109.github.io/gs-mapper/splat.html
GitHub: https://github.com/rsasaki0109/gs-mapper
```

## Topics

`3d-gaussian-splatting`, `3dgs`, `slam`, `visual-slam`, `robotics`, `autonomous-driving`, `mast3r`, `dust3r`, `vggt-slam`, `webgl`, `webgpu`
