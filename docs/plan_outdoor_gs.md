# 屋外 3D Gaussian Splatting 開発計画 / 引継ぎメモ

更新日: 2026-04-20（OSS 顔整備 + MCD supervised 経路開通セッション後）

この文書は、`GS Mapper` リポジトリにおける屋外 3D Gaussian Splatting 対応の現在地を、**Claude / Copilot / その他のコーディングエージェント**がそのまま引き継げる粒度でまとめた handoff 文書です。リポジトリ直下の `CLAUDE.md` は開発コマンド早見、本書は **屋外パイプラインの文脈・判断・失敗の履歴**に重きを置きます。

## エージェント向け: 読み方

1. **初回**: §0 TL;DR → §6 blocker → §7 優先順位 → §8 触るファイル。
2. **データを手元で動かす**: §9 コマンド早見 → §10 GitHub Pages / 検証 → `CLAUDE.md`。
3. **同じ実験を繰り返さない**: §3 empirical findings（特に Autoware bag6 と COLMAP tuning の失敗）。
4. **品質の本丸**: §4 の MCD pose import と §10 の splat viewer。

## 0. TL;DR

- 公開名は `GS Mapper`、repo slug `rsasaki0109/gs-mapper`、Pages `https://rsasaki0109.github.io/gs-mapper/`
- Python import path は **まだ** `gs_sim2real`（rename しない）
- CLI は `gs-mapper`、旧 `gs-sim2real` は legacy alias として維持
- ローカル worktree `nerf-gs-playground` はそのまま
- 屋外向け 7 フェーズ（config / SH 全次数 / depth supervision / DUSt3R / LiDAR SLAM-GNSS-NMEA / dynamic mask / appearance-sky）は全て実装済み
- **2026-04 のセッションで Blocker 1（MCD pose import）と §16 の Pages デモ偽物問題は解決済み**（詳細は §4 / §10）
- **2026-04-20 セッションで MCDVIRAL calibration YAML を公式 Download page 本体で発見し、`scripts/download_mcd_calibration.sh` + `--mcd-static-calibration` フラグまで配線完了**（詳細は §4.3.3.c / §15 と PR #79 / #80）。残るは実際の supervised MCD training 本走のみ。

直近の最大の残課題は以下。

1. **Priority A**: MCD day session (`tuhh_day_04`) を supervised 経路 (`--method mcd` + GNSS + LiDAR depth + `--mcd-static-calibration`) で回して、bundled demo に 6 本目 `mcd-tuhh-day04-supervised.splat` を追加する。レシピは §4.3.3.c、pitfalls は §4.3.3.a。**calibration + downloader + CLI フラグが全部 merge 済み or pending なので、次セッションは新規コードほぼ不要、GPU 時間が要るだけ**。
2. Priority B: **Waymo 実データ E2E が未検証**
3. Priority C: NMEA / GNSS / IMU robustness、depth / appearance / sky の比較評価

## 1. リポジトリの現在の識別子

| 種別 | 現在値 | 備考 |
|------|--------|------|
| Public name | `GS Mapper` | README / Pages / docs で使う名前 |
| GitHub repo | `rsasaki0109/gs-mapper` | remote slug 変更済み |
| GitHub Pages | `https://rsasaki0109.github.io/gs-mapper/` | |
| Python package | `gs_sim2real` | **まだ rename しない** |
| Main CLI | `gs-mapper` | |
| Legacy CLI | `gs-sim2real` | 互換 alias |
| Local worktree | `nerf-gs-playground` | そのまま |

## 2. 実装ステータス

### 2.1 元の 7 フェーズ（完了）

屋外用 config、LiDAR depth supervision、DUSt3R、LiDAR SLAM / trajectory import、appearance embedding、dynamic object masking、sky / background model。

### 2.2 2026-04 セッションで追加されたもの

| 項目 | ファイル | 要点 |
|------|---------|------|
| MCD `/tf`, `/tf_static`, `/camera_info`, `/gnss/fix` 抽出 | `src/gs_sim2real/datasets/mcd.py`, `ros_tf.py` | |
| `MCDLoader.merge_lidar_frames_to_world()` | `mcd.py` | per-frame NPY + TUM ENU + `T_base_lidar` で world 点群を合成。identity quaternion の場合は ENU 動きから yaw 推定 |
| `MCDLoader.colorize_lidar_world_from_images()` | `mcd.py` | 各カメラ画像に world LiDAR 点を投影 → bilinearly サンプリングした RGB を累積平均。points3D.txt が grey(128) ではなく実色で seed される（**もや晴らしの本丸**） |
| `MCDLoader.export_lidar_depth_per_image()` | `mcd.py` | 各学習画像に world LiDAR を投影、closest z を per-pixel 保持した float32 (H, W) depth map を `depth/<subdir>/<stem>.npy` に保存 |
| LiDAR-seeded COLMAP sparse (色付き) | `cli.py` `_mcd_gnss_sparse_import` + `_mcd_colorize_seed` | `points3D.txt` を実 LiDAR 点 + 画像由来 RGB で seed。`--mcd-skip-lidar-colorize` で opt-out |
| Nx6 点群の `points3D.txt` 書き出し | `preprocess/lidar_slam.py`, `preprocess/depth_from_lidar.py` | `_write_colmap` / `_write_colmap_multiview` が Nx6（xyz + RGB）を受けて色を書く |
| 新 CLI フラグ | `cli.py` | `--mcd-lidar-frame` / `--mcd-skip-lidar-seed` / `--mcd-skip-lidar-colorize` / `--mcd-export-depth`（preprocess/run/demo すべて） |
| gsplat depth supervision (`render_mode=RGB+D`) | `train/gsplat_trainer.py` | `_render_gsplat(want_depth=True)` で depth を返し、`depth_loss_weight * L1(rendered_depth, gt_depth)` を valid pixel でトレーニングに加算 |
| PLY→.splat 変換 (`ply_to_splat`) | `src/gs_sim2real/viewer/web_export.py` | antimatter15/splat 形式（32 B/gauss）。`normalize_target_extent` で世界スケール→viewer 適合サイズに縮小、`min_opacity` / `max_scale` フィルタで霧除去 |
| WebGL Gaussian Splat viewer | `docs/splat.html`, `docs/splat-viewer/main.js` | antimatter15/splat (MIT) を vendor。gzip 対応 dynamic buffer、`defaultViewMatrix` は outdoor scale、downsample しきい値は 30k |
| `_densify_and_prune` の mask off-by-one 修正 | `src/gs_sim2real/train/gsplat_trainer.py` | clone → split の順で tensor 拡張する際に旧サイズ mask で indexing する IndexError を解消 |
| Three.js viewer のソフトスプライト化 / helper スケール | `docs/index.html` | `PointsMaterial` を radial-alpha disc + sizeAttenuation。grid/axes は scene bounds に合わせて動的に再構築 |
| `configs/training_depth.yaml` | `configs/` | `depth_loss_weight: 0.1` の outdoor + depth 学習プリセット |
| `configs/datasets.yaml` に bag4 / bag6 | `configs/datasets.yaml` | S3 公開バケットの `autoware_leo_drive_bag4` / `bag6` を `scripts/download_datasets.py` から取得可 |

## 3. Public data で「実際に verified された」もの

### 3.1 Autoware Leo Drive - ISUZU sensor data

公開元: <https://autowarefoundation.github.io/autoware-documentation/main/datasets/#leo-drive---isuzu-sensor-data>

- **bag1**: `MCD → COLMAP → gsplat` E2E 成功（`outputs/outdoor_smoke_autoware_seq/`、プラン §6.1 の known-good 条件）。
- **bag6**: プラン初版では **image-only COLMAP で 2 registered images** 止まり（§4.1）。pose-import セッションで **GNSS + /tf_static + LiDAR seed により 180 registered images (3 カメラ × 60 frames) / 100,000 world points** 達成。さらに **250 frames × 3 cam × colorize + 30k iter depth supervision** で 1,024,794 Gaussians まで拡張済み（`outputs/bag6_depth_train/`）。
- **bag4**: 別ルート。**240 frames × 3 cam = 720 registered images + colorize (196k/200k) + per-image depth (720 maps) + 30k iter depth-supervised training** で 932,243 Gaussians を生成し、現在 `/splat.html` のライブ配信に採用中（`outputs/bag4_full_train/`）。

### 3.2 Verified だが品質面は弱いもの

- bag6 / bag4 ともに colorize + depth で Gaussian 数は百万級に到達。ブラウザで trajectory 沿いの 3D 構造は視認できるが、「街並み」として読めるほどは解像していない。学習量/カメラ数/長時間軌跡が足りていないと見ている。

### 3.3 実装済みだが public 実データ E2E 未確認のもの

| 項目 | 実装 | 実データ E2E |
|------|------|-------------|
| Waymo frame / depth / mask extraction | 済み | 未確認 |
| Waymo `run --preprocess-method waymo` | 済み | 未確認 |
| NMEA import + training path | 済み | 未確認 |

## 4. 最近の empirical findings

### 4.1 bag6 の image-only COLMAP チューニングは効かない（プラン初版からの履歴）

`every_n = 1 / 2 / 3 / 5 / 10 / 20`, `matching = sequential / exhaustive` いずれも **2 registered images** 止まり。matcher tuning では解けなかった。

### 4.2 pose-seeded import が効いた

`/tf_static` の `base_link → camera_top/camera_link` 変換 + GNSS 軌跡 + LiDAR 点群 world merge（200k 点）で COLMAP text model を直接書き出し → 3 カメラで 180 frames 全部 registered、100k 点 seed。これが本セッションの突破口。

### 4.3 MCD (mcdviral.github.io) NTU session17 も image-only COLMAP は 2 registered 止まり

`scripts/download_mcd_session.sh` で 633 MB rosbag1 (NTU #17) を取得できるようになった（Drive 確認フォーム自動突破）。ただしこのセッションは handheld D455 ステレオ + IMU のみで **GNSS / /tf_static は含まれない**。150 frame サブサンプルで image-only COLMAP を走らせても Autoware bag6 と同じ「2 registered images / 36 points」症状で詰まる。gsplat 側は 5295 gauss まで学習するが点群が貧弱で demo 品質に達しない。MCD を本格的に使うなら: (a) SLAM 済みポーズを外部で流し込む、(b) もっと画像間重複が多いセッションを選ぶ、(c) DUSt3R pose-free 経路に切り替える、のいずれか。

### 4.3.1 DUSt3R pose-free 経路は本実装済み（PR #55）、ただし NTU #17 では退化

`src/gs_sim2real/preprocess/pose_free.py` の `_run_dust3r` は以前 stub → simple fallback にフォールスルーしていたが、PR #55 で本実装に差し替えた（`run_dust3r_inference` + `write_colmap_sparse`、`scripts/run_dust3r.py` は CLI ラッパ）。DUSt3R の pairwise pointmap + `PointCloudOptimizer` の MST-seeded global align を 300 iter 回し、per-image PINHOLE `cameras.txt` / c2w→w2c の `images.txt` / confidence-filtered `points3D.txt` を gsplat trainer が読める形で吐く。16 GB 級 GPU では inference 後に model weights を GPU から解放しないと aligner の stacked-pred tensor で OOM するので `del model` + `empty_cache()` を明示挿入。

NTU #17 にそのまま適用した場合、多くのフレームが translation ≈ 0 に退化する（image-only COLMAP と同じ parallax 不足症状）。代替として Autoware bag6 cam0 の 20 frame で回したところ 19/20 が非退化 trajectory を復元し、gsplat 3000 iter で 5.57M gauss まで収束。そこから antimatter15 splat format に 400k × 32 B = 12.8 MB で焼いた成果物が `docs/assets/outdoor-demo/outdoor-demo-dust3r.splat` で、`splat.html?url=...` 経由で GNSS+LiDAR supervised demo とトグル比較できる。GNSS も LiDAR も無しで image だけから出ている点が売り（L1 ≈ 0.15、supervised 版の 0.06〜0.08 には届かないが、ポーズなし構築の比較基準として bundled）。

### 4.3.2 MCD NTU #17 は image-only では詰む（3 通り試して全滅）— 原因は low inter-frame feature match

2026-04-19 に次の 4 経路を全部試した結果、NTU session17 は image-only で成立しない scene だと確定:

| 試行 | Frames | 結果 |
|------|--------|------|
| image-only COLMAP (150 frame 連続) | 150 color | 2 registered / 36 points（初回セッションで既知） |
| DUSt3R mono 30 frame complete | 30 color | 多数が translation=0、scene 非メトリック |
| DUSt3R stereo pair (infra1+infra2 × 10 pair) | 20 IR | 10/20 nonzero、scene extent ~0.2m（D455 baseline しか拾えず） |
| DUSt3R mono 18 frame spread (bag 全域) | 18 color | 3/18 nonzero（最悪。時間差が開くほど退化） |
| ORB-SLAM3 compact (`simple_visual_slam/run_mono --tum`, 600 frame) | 600 color | 初期化不能。`Initializer: Matches found: 2〜13` が全フレームで継続、>100 match に一度も到達せず |

最大要因は ORB の frame-to-frame match が常に <15 個しか取れないこと。原因候補: 歩行による motion blur、NTU キャンパスの繰り返し textures、handheld の揺れによる大きな viewpoint jump。いずれも image-only アプローチを根本的にブロックする。

**結論**: NTU #17 は pose injection 無しでは shippable demo にならない。次に MCD で demo を作るなら `(a)` 別 session 選定（GNSS 付き、重複多い Kirinyaga / Tuas など）、`(b)` MCDVIRAL の GT pose csv をダウンロードして `--trajectory-format tum` で直接食わせる、のどちらか。本セッションではここで撤退し、NTU #17 は "known-failing scene" として scratched。成果物は保持: `outputs/mcd_ntu17_stereo_dust3r/` など（retain しても demo 用途なし）。

### 4.3.3 GNSS+LiDAR 付き MCD session 候補（次セッション向け）

MCDVIRAL のダウンロードページを監査して GNSS (VN100/VN200) + LiDAR (Ouster OS1-64) + camera + /tf_static を全部揃えた session を size 順にリスト化した。NTU #17 撤退後の後継候補は次のとおり:

| Session | Size | Mount | Notes |
|---------|------|-------|-------|
| `tuhh_night_09` | **3.5 GB** | handheld | 最小。GNSS (VN200) + Ouster OS1-64 + D455b + D455t. 同セッションの `ltpb.bag` (312 KB) に calibration + /tf_static が入ると推測 |
| `tuhh_night_07` | 10.2 GB | handheld | |
| `tuhh_day_04`   | 12.5 GB | handheld | day light 条件 |
| `kth_night_05`  | 14.2 GB | handheld | |
| `ntu_day_02`    | **14.8 GB** | vehicle (ATV) | 最小の車載 session。Autoware bag 系と同種の GNSS + LiDAR + camera + /tf_static が期待できる |
| `tuhh_night_08` | 16.8 GB | handheld | |
| `ntu_night_13`  | 17.3 GB | vehicle | |
| ... | ... | ... | full list: https://mcdviral.github.io/Download.html |

ダウンロードは session 単位で Drive **folder** なので、単一 file ID しか扱わなかった `scripts/download_mcd_session.sh` ではなく `scripts/download_mcd_folder.sh` を使う（本 PR で追加）。folder は `https://drive.google.com/drive/folders/<ID>` の `<ID>` を渡せば `gdown --folder` で全 bag を再帰取得する:

```bash
scripts/download_mcd_folder.sh 1nEPiTXkVmLIhmBOVNpwSAEgnAXupnAxx data/mcd/tuhh_night_09/
```

取得後は既存の `--method mcd` + `--mcd-seed-poses-from-gnss` + `--mcd-tf-use-image-stamps` + `--mcd-export-depth` パイプラインがそのまま通る想定（Autoware bag 系で確認済）。ただし MCD handheld session は /tf_static の親 frame 名が Autoware と異なる可能性があるので、preprocess 実行時に `mcd.py` の TF lookup が拾えるか一度確認する（拾えなかった場合、`--mcd-reference-bag` 相当の frame override を足す）。

#### 4.3.3.a `tuhh_night_09` で分かった落とし穴 (2026-04-19)

実際に download してみて分かった MCD 側の注意点:

1. **handheld night session は GPS fix が取れていない**。`tuhh_night_09_vn200.bag` に NavSatFix は 73957 本入っているが、全部 `latitude=longitude=altitude=0.0`、`status.status=0` (NO_FIX)。夜間 + 屋内を渡り歩く取り方なので当然の挙動。`--mcd-seed-poses-from-gnss` は機能しない。**night session 全般を避け、day session (`tuhh_day_04` / `ntu_day_02` 以降) から選ぶ**。
2. **MCD の bag には `/tf` も `/tf_static` も入っていない**。sensor 間 extrinsics は session folder 外の calibration YAML で配布されており、bag 内だけで完結しない。**2026-04-20 更新**: この YAML は MCDVIRAL の Download page 本体に Google Drive 直リンクで置かれている (`mcdviral/*` という別リポジトリではない)。`scripts/download_mcd_calibration.sh handheld` (PR #79) で 6.6 KB 落ちる → `--mcd-static-calibration <calib.yaml>` (PR #80) で `_mcd_gnss_sparse_import` の TF lookup に注入可能。詳細 recipe は §4.3.3.c。
3. session folder は `lua_bagname/<bagname>_*.bag` という二重ディレクトリ構造で展開される（gdown の挙動）。CLI が `--mcd-session data/mcd/<name>/` を要求するなら `<name>/<name>_*.bag` の shape を受け付けるか、単ファイル列挙にするかを決めておく。

これらを踏まえて、次セッションの着手順序:

1. day session 1 本 DL（`tuhh_day_04` 12.5 GB か `ntu_day_02` 14.8 GB）
2. `/vn200/GPS` に valid fix があるか事前に spot-check（`status.status >= 0` が 1% 以上）
3. MCDVIRAL calibration YAML を入手、`--mcd-static-calibration` みたいなフラグを足して `mcd.py` の TF lookup を bypass できるようにする
4. preprocess → gsplat train → splat export → Pages bundle

#### 4.3.3.b `tuhh_day_04` で image-only DUSt3R 経路が通った (2026-04-19)

上の段で "GNSS + calibration YAML を揃えないと詰む" と書いたが、`tuhh_day_04` (12.6 GB 完本 DL) の d455b color を 5558 frame 中から等間隔 20 frame だけ抜いて DUSt3R complete graph で回したら、**20 frame 中 19 frame が非退化 trajectory** を復元した。歩行モーションの norm 列が 0.009 → 0.21 → ... → 1.87 → ... → 0.11 と綺麗なドーム形で、handheld が屋外を往復したと物理的に解釈できる結果。gsplat 3000 iter で 3.23M gauss まで学習収束。

そのまま `ply_to_splat` で 400k / 12.8 MB に焼いて `docs/assets/outdoor-demo/mcd-tuhh-day04.splat` として bundle、`splat.html?url=...mcd-tuhh-day04.splat` でブラウザから toggle できる。

つまり **day session の場合 image-only DUSt3R 一発で MCD demo を作れる**、GNSS seeding も calibration YAML も要らない。NTU #17 (night handheld / GPS denied / motion blur) との差は「昼の屋外 + 歩行速度 + repetitive なし texture」で DUSt3R 前提がそのまま通るかどうか。次 MCD demo を作るときは

1. day session の image bag 1 本だけ DL (5 GB 前後、LiDAR bag は省略可)
2. `photos-to-splat` 一撃 or `run_dust3r.py` → gsplat train → `export --format splat`

で 30 分以内で回る、という playbook が確立した。LiDAR depth supervision / appearance / BA を足して質を詰めるなら、そのとき初めて GNSS + calibration path に戻る。

#### 4.3.3.c `tuhh_day_04` を supervised に乗せる full recipe (2026-04-20、次セッション引継)

2026-04-20 セッションで §4.3.3.a の blocker 群が全部解消した:

1. **calibration YAML は公式 Download page 本体に掲載されている** — handheld rig (kth_/tuhh_ 用) が Drive file `1htr26EE-Y1sHS5J4zaSbauC1XFgIh3Ym`、ATV rig (ntu_ 用) が `1zVTBqh4cA1DciWBj5n7BGiexbfan1BBL`。`scripts/download_mcd_calibration.sh <handheld|atv> [out-path]` で取得可 (PR #79)。
2. **YAML を `mcd.py` の TF lookup に注入する CLI** — `--mcd-static-calibration <calib.yaml>` フラグが `preprocess` / `run` / `demo` 全部に wired (PR #80)。`gs_sim2real.datasets.ros_tf.load_static_calibration_yaml` が `body → <sensor>` edge の `StaticTfMap` を返し、`_mcd_gnss_sparse_import` で bag 由来の空 tree とマージされて `--mcd-camera-frame` / `--mcd-lidar-frame` の lookup を通す。
3. **sensor name 対応表 (handheld rig)**: YAML 側の child key がそのまま `--mcd-camera-frame` / `--mcd-lidar-frame` に渡せる。
   - color cameras: `d455b_color` (rostopic `/d455b/color/image_raw`)、`d455t_color` (`/d455t/color/image_raw`)
   - IR mono: `d455b_infra1/2`、`d455t_infra1/2`
   - IMUs: `d455b_imu`、`d455t_imu`、`vn200_imu` (body 基準はたいてい vn200_imu が identity)
   - LiDAR: `mid70` (Livox `/livox/lidar`)、`os_sensor` (Ouster OS1-64 `/os1_cloud_node/points`)、`os_imu`
4. **day session に valid GNSS fix があることは §4.3.3.b で既に確認済み** (`tuhh_day_04` の d455b color は 5558 frame / 19-20 frame で非退化 trajectory)。

これを踏まえた next-session の full incantation (GPU 環境、`data/mcd/tuhh_day_04/` に session folder 展開済み前提):

```bash
# 1) calibration YAML fetch (1 回きり)
scripts/download_mcd_calibration.sh handheld data/mcd/calibration_handheld.yaml

# 2) supervised preprocess (GNSS trajectory + LiDAR-seeded colored sparse + per-image depth)
gs-mapper preprocess \
  --images data/mcd/tuhh_day_04 \
  --output outputs/tuhh_day04_sup \
  --method mcd \
  --image-topic /d455b/color/image_raw \
  --mcd-camera-frame d455b_color \
  --lidar-topic /os1_cloud_node/points \
  --mcd-lidar-frame os_sensor \
  --imu-topic /vn200/imu \
  --gnss-topic /vn200/gps \
  --mcd-static-calibration data/mcd/calibration_handheld.yaml \
  --mcd-seed-poses-from-gnss \
  --mcd-tf-use-image-stamps \
  --mcd-export-depth \
  --extract-lidar \
  --max-frames 400 --every-n 1 \
  --matching sequential --no-gpu

# 3) depth-supervised gsplat training (`configs/training_depth_long.yaml` が bag4/bag6 で実績)
gs-mapper train --data outputs/tuhh_day04_sup --output outputs/tuhh_day04_sup_train \
  --method gsplat --iterations 30000 \
  --config configs/training_depth_long.yaml

# 4) 400k/12.8 MB 以下に export して bundle
gs-mapper export \
  --model outputs/tuhh_day04_sup_train/point_cloud.ply \
  --format splat \
  --output docs/assets/outdoor-demo/mcd-tuhh-day04-supervised.splat \
  --max-points 400000
```

その後の bundle 作業（`CONTRIBUTING.md` §"Bundled demo splats" の triplet）:
- `docs/scenes-list.json` に entry 追加
- `docs/splat.html` / `docs/splat_spark.html` / `docs/splat_webgpu.html` の `<select>` に `<option>` 追加
- `tests/test_pages_assets.py` に `test_mcd_tuhh_day04_supervised_splat_present` 相当を追加

**Pitfalls / 事前チェック**:

1. **CameraInfo の `frame_id` が YAML のセンサー名と一致しないかもしれない**。MCD bag は `/d455b/color/camera_info` の header.frame_id がたとえば `d455b_color_optical_frame` のような `_optical_frame` suffix 付きで publish される可能性があり、そのままだと `tf_map.lookup(base_frame, "d455b_color_optical_frame")` が miss する。その場合は `--mcd-camera-frame d455b_color` を明示的に渡して override すれば YAML 側の key と一致する。`_mcd_gnss_sparse_import` は `CameraInfo.frame_id` を優先し、空なら `--mcd-camera-frame` にフォールバックするので、明示的に frame name を渡すのが安全。
2. **`--mcd-base-frame` のデフォルトは `base_link`** で、`load_static_calibration_yaml(path, base_frame="base_link")` として YAML を読むので、YAML の `body:` key は内部的に `base_link` にリラベルされる。`--mcd-lidar-frame os_sensor` / `--mcd-camera-frame d455b_color` のような child name はそのまま引ける。もし別名を使いたい場合は `--mcd-base-frame body` にして lookup 側も `body` に揃える。
3. **GNSS fix の事前 spot-check**: `scripts/outdoor_smoke.sh mcd-list` 等で `/vn200/gps` の `status.status >= 0` サンプル率を確認。day session は取れているはずだが、`tuhh_night_09` のように全 sample fix=0 な bag は `--mcd-seed-poses-from-gnss` で即落ちる。
4. **session folder の二重ディレクトリ** — `gdown --folder` で展開すると `data/mcd/tuhh_day_04/tuhh_day_04_*.bag` という shape になる (§4.3.3.a #3)。`MCDLoader._find_bag_paths` は再帰的に `.bag` を拾うので `--images data/mcd/tuhh_day_04` のまま渡して OK。
5. **LiDAR bag (`*_ouster.bag` など) が無いと depth supervision が無効化される** — image-only bag しか DL していない場合は `--extract-lidar` が空振りし、`mcd-export-depth` は warning を出して skip する。LiDAR 無しで supervised 感は出ないので、session folder 全部 (~12 GB) DL する。

**成果物の期待値**（§3.1 bag4 の類推から）:
- registered frames: 240-400 × 1-2 cam = 240-800（day session は single cam でも十分）
- LiDAR world seed: 100k-200k 点 (colorized)
- trained gauss: 500k-1.5M、400k filter 後 12.8 MB
- L1: 0.08-0.12 (bag4 相当を期待、DUSt3R 版の 0.16 より一段良い)
- bundled 6 本目 = 既存 5 本と A/B 比較可能な「同じ scene の supervised vs pose-free」軸が tuhh_day_04 でも成立。bag6 で {supervised, DUSt3R, MAST3R}、MCD で {DUSt3R, MAST3R, supervised} の対称行列が閉じる。

### 4.4 densification は 100k 級初期点でも Stable に走るよう修正済み

以前は iter 500 以降の densify で `IndexError: mask [N] != tensor [N+num_clone]` が即発生していた。本セッションの修正で bag1 30k iter → 244k Gaussians まで完走。

## 5. 各 subsystem の到達点

- **Training** (`train/gsplat_trainer.py`): `scene_extent` / `scene_auto_scale` / LiDAR point cloud bootstrap / SH 全次数 / depth supervision / appearance / sky / mask-aware L1。densify mask off-by-one 修正済み。
- **Waymo** (`datasets/waymo.py`): frame / depth / mask 抽出 + `to_colmap_format()` + CLI 接続。実 Waymo data 未検証。
- **MCD** (`datasets/mcd.py`): rosbag1/2、image/lidar/imu、camera_info、/tf、/tf_static、NavSatFix、`merge_lidar_frames_to_world`、`extract_lidar(save_timestamps=True)`。bag6 で完全 E2E。
- **LiDAR-SLAM / GNSS / NMEA** (`preprocess/lidar_slam.py`): TUM / KITTI / NMEA / ENU 変換 / yaw 補間。`_write_colmap` / `_write_colmap_multiview` は src==dst の image copy を skip（bag6 single-dir パス対応）。
- **COLMAP glue** (`preprocess/colmap.py`): custom path、CPU-only、sequential / exhaustive、single_camera_per_folder。sparse preflight は `colmap_ready.py`。
- **Smoke script** (`scripts/outdoor_smoke.sh`): `waymo` / `mcd` / `e2e` / `mcd-list` / `nmea` / `verify-colmap`。

## 6. 現在の最大 blocker

### Blocker（残っているもの）

- **NMEA/GNSS/IMU robustness**: GGA / RMC / ENU は入っているが、IMU quaternion / angular velocity 融合や日跨ぎ、logger 時刻ずれが未対応。
- **性能面**: splat viewer のライブ fps が 5–7 程度。100k 超 gaussian + HiDPI で shader 負荷がボトルネック。将来は WebGPU (mkkellogg/Spark 等) or sort/rasterize の optimise を検討。

### 非対象（意図的に scope 外）

- **Waymo real-data E2E**: code path / tests / `scripts/check_waymo_e2e_prereqs.sh` は揃っているが、Python 3.12 環境での SDK build 失敗 + Waymo Open Dataset の利用規約同意 + 手動 tfrecord ダウンロードが必要。プロジェクト方針として公開同意不要データ（Autoware Leo Drive）を優先するため、本セッションで意図的に非対象とした。必要になったら Python 3.10 venv を別途立てて `WAYMO_DATA_DIR` を指定すれば即動く状態。

### 解決済み（2026-04 セッション）

- ~~Blocker 1: MCD calibration / pose import~~ → bag6 で 2→180 registered、bag4 で 720 registered、**5-bag fusion (bag1 + bag2 + bag3 + bag4 + bag6) で 5040 registered**。`merge_lidar_frames_to_world` + `_mcd_gnss_sparse_import` + 3-camera extrinsics + colorize で完結。
- ~~§16.2 デモが campus 共有バイナリ~~ → **5-bag fusion** 由来の 80k gaussian `.splat` を本物の WebGL GS viewer (`/splat.html`) で配信中（15 cameras × 5040 frames, 1.43M Gaussians）。
- ~~「画面が灰色のもや」~~ → 画像由来 RGB による LiDAR 点群初期化 + per-image LiDAR depth supervision + per-image appearance embedding (scale, bias) で color std を 0.06 → 0.15–0.22 に、trajectory 沿いの 3D 構造を可視化。L1 loss も 0.32 → 0.20 と 38% 改善。
- ~~densify bug~~ → `_densify_and_prune` の clone/split 順序を修正、30k–50k iter 学習が安定化。
- ~~Multi-bag fusion~~ → `reference_origin` 共有 + `scripts/merge_mcd_sparse.py` で任意の N bag を 1 つの sparse に結合可能。
- ~~appearance embedding~~ → per-image (scale, bias) RGB affine を trainer に追加、`configs/training_appearance.yaml` で有効化。

## 7. 次の担当者に引き継ぐ順序

### 優先度 A（残り）

1. **WebGPU 版 viewer**: 調査したところ script tag 単体で使える WebGPU GS 実装は現時点で皆無（Spark は ESM import、PlayCanvas Web Components は GS 非対応、cvlab/epfl 版も bundler 必須）。vite 等のビルドステップを導入するか、antimatter15/splat WebGL2 で継続するかの判断が要る。
2. **MCD (mcdviral.github.io)**: 公開・登録不要だが 1 セッション 3.5〜51 GB と巨大。downloader は `aws s3 sync --no-sign-request` ではなく Google Drive 経由なので `configs/datasets.yaml` に入れられない。rosbag 形式が rosbag1 or 2 不明なので `MCDLoader` の AnyReader 互換を実データで確認する必要あり。
3. **NMEA/GNSS/IMU robustness**: IMU quaternion 融合、logger 時刻ずれ、日跨ぎ RMC を堅牢化。

### 優先度 B

1. Waymo real-data E2E（ユーザが Waymo Open Dataset の Terms of Use に同意し、Python 3.10/3.11 環境を立てられる場合のみ）。プロジェクト方針として Autoware 系が主軸のため後回し。
2. Waymo dynamic mask / depth の実使用評価。

### 既に対応済み（refactor の余地はあるが動く）

- ~~画像由来 initial color~~ → `colorize_lidar_world_from_images` で完了
- ~~depth supervision~~ → `_render_gsplat(want_depth=True)` + `configs/training_depth.yaml` で完了
- ~~Multi-bag fusion~~ → `--mcd-reference-origin` / `--mcd-reference-bag` CLI + `scripts/merge_mcd_sparse.py` で実装完了。**5-bag fusion (bag1+2+3+4+6) をライブデモに採用**。1000 images 超で自動的に lazy image loading へ切り替わる
- ~~appearance embedding~~ → per-image (scale, bias) の trainer 実装 + `configs/training_appearance.yaml` で完了
- ~~joint pose refinement (BA)~~ → 各学習画像に 6-DOF (so3 + translation) delta を学習可能化、`joint_pose_start_iter` から有効化。`configs/training_ba.yaml` で depth + appearance + pose の全部 on

## 8. 具体的に触るファイル

### Waymo E2E をやるなら

**前提条件**:

- Python 3.10 or 3.11（`waymo-open-dataset-tf-2-12-0` は Python 3.12 で `pkgutil.ImpImporter` 削除により build 失敗）
- `pip install "waymo-open-dataset-tf-2-12-0"` (TensorFlow 2.12 依存)
- Waymo Open Dataset の利用同意 + `*.tfrecord` ダウンロード（<https://waymo.com/open/download/>）

**セットアップ確認**:

```bash
bash scripts/check_waymo_e2e_prereqs.sh
```

[OK] / [WARN] / [MISS] で Python バージョン、SDK、入力データの状態を表示。

**触るファイル**:

- `src/gs_sim2real/datasets/waymo.py`
- `src/gs_sim2real/cli.py`
- `scripts/outdoor_smoke.sh`
- `tests/test_waymo.py`, `tests/test_cli.py`, `tests/test_waymo_prereqs_script.py`

### bag6 の色改善をやるなら

- `src/gs_sim2real/datasets/mcd.py`（`merge_lidar_frames_to_world` の拡張として image-based colorize）
- `src/gs_sim2real/preprocess/lidar_slam.py`（`_write_colmap` の points3D.txt に RGB を書く経路）
- `src/gs_sim2real/viewer/web_export.py`（`ply_to_splat` は現状 SH DC から RGB、変更不要）

### Splat viewer を深くいじるなら

- `docs/splat-viewer/main.js`（antimatter15/splat の vendored JS）
- 当該ファイルは外部 upstream なので、local patch は最小に。gzip content-length 対応、downsample しきい値、defaultViewMatrix の 3 箇所だけ触っている（git log で差分確認）。

## 9. 参考コマンド集

### 9.1 Repo sanity

```bash
gs-mapper --help
gs-sim2real --help
```

### 9.2 bag6 multi-camera pose-seeded 再現（本セッション）

```bash
gs-mapper preprocess \
  --images data/autoware_leo_drive_bag6 \
  --output outputs/bag6_multicam \
  --method mcd \
  --image-topic "/lucid_vision/camera_0/raw_image,/lucid_vision/camera_1/raw_image,/lucid_vision/camera_2/raw_image" \
  --lidar-topic /sensing/lidar/concatenated/pointcloud \
  --imu-topic /sensing/imu/imu_data \
  --gnss-topic /gnss/fix \
  --mcd-seed-poses-from-gnss \
  --mcd-tf-use-image-stamps \
  --extract-lidar \
  --max-frames 60 \
  --every-n 2 \
  --matching sequential \
  --no-gpu

gs-mapper train --data outputs/bag6_multicam --output outputs/bag6_multicam_train \
  --method gsplat --iterations 15000 --config configs/training.yaml
```

### 9.3 PLY → .splat → Pages デプロイ

```python
from gs_sim2real.viewer.web_export import ply_to_splat
ply_to_splat(
    "outputs/bag6_multicam_train/point_cloud.ply",
    "docs/assets/outdoor-demo/outdoor-demo.splat",
    max_points=60000,
    normalize_target_extent=30.0,
)
```

## 10. GitHub Pages・デモ・E2E の現状

### 10.1 ライブ URL

- サイト: `https://rsasaki0109.github.io/gs-mapper/`
- Three.js 点ビューア（outdoor タブ）: `?scene=outdoor-demo`
- **本物の 3D Gaussian Splat viewer**: `/splat.html`（antimatter15/splat vendored）

### 10.2 Outdoor GS Demo の中身（2026-04 セッション後）

- `docs/assets/outdoor-demo/outdoor-demo.splat` は **5-bag fusion (bag1+2+3+4+6) の実学習結果 80k gaussians**（1.43M Gaussians をフィルタ: `min_opacity ≥ 0.3`, `max_scale ≤ 2 m`, 世界スケール 500 m を 30 単位に正規化）。image-projected RGB 初期化 + per-image LiDAR depth supervision + per-image appearance embedding (scale, bias)。
- `docs/assets/outdoor-demo/outdoor-demo.points.bin`（Three.js 用）は bag1 30k iter の密点群（244k → 60k subsample）。
- 「campus-gallery は 1 枚の写真から合成した擬似点群」という事実は変わっていないので、**ギャラリー系と outdoor-demo の性質差は大きい**。ユーザに説明する際は注意。

### 10.3 Splat viewer の限界

- 100k → 60k に絞ってもレンダ fps は HiDPI 環境で数〜数十 fps。`downsample=1/dpr` から `downsample=1` への切り替えで改善済み。
- 本物の 3D Gaussian rendering（SH 視点依存 / 異方性 2D splat）を行っているが、**scene が駅や道路として識別できるレベルではない**。これは学習量（60 frames × 3 cam / 15k iter）と初期色の質の問題で、viewer 側の問題ではない。

### 10.4 CI と Pages E2E

- Lint: `ruff` + `bash -n`。
- Playwright 系 E2E はデフォで skip（`PLAYWRIGHT_E2E=1` で有効化）。
- `.github/workflows/pages-e2e.yml` は手動実行時に `pages_url` を指定可能。

## 11. validation / test

**推奨の一回まわし（PR 前・引継ぎ時）**

```bash
ruff check src/ tests/ scripts/
ruff format --check src/ tests/ scripts/
pytest tests/ -v
```

2026-04 時点で main の pytest は 285 passed。

**屋外・データ関連で特に重要なファイル**

- `tests/test_waymo.py`, `tests/test_mcd.py`, `tests/test_lidar_slam.py`, `tests/test_colmap.py`
- `tests/test_pose_free.py`, `tests/test_cli.py`
- `tests/test_pages_assets.py`（outdoor-demo bundle の manifest と binary の存在を検証）
- `tests/e2e/test_github_pages_outdoor.py`（`PLAYWRIGHT_E2E=1` のときのみ）

## 12. 現在の環境メモ

- `COLMAP` は user-space にインストール済み。`--colmap-path` と `--no-gpu` が使える。
- system Python に `pip install --break-system-packages -e .` を利用。
- optional dependency: `dust3r`、`mcd`（rosbag ingest）、Waymo (`waymo-open-dataset-tf-2-12-0`)。

## 13. 今後の non-goal / 触らない方がよいもの

- `gs_sim2real` package path の rename
- local worktree directory 名の rename
- DreamWalker 系の別機能の大規模整理
- `gsplat_trainer.py` の全面分割
- antimatter15/splat の vendored コードへの大改修（upstream が sparse に続いていないので local patch は必要最小限に）

## 14. 関連ドキュメント（このファイルとの役割分担）

| ファイル | 内容 |
|----------|------|
| `CLAUDE.md` | 開発者向けコマンド早見（lint, pytest, E2E, verify-colmap 等）。**毎回の作業入口** |
| `README.md` | 公開向け概要、データセット表、Autoware smoke、ディレクトリ構成 |
| **本書** (`docs/plan_outdoor_gs.md`) | **屋外パイプラインの文脈・判断・失敗の履歴・長引きそうな blocker** |

エージェントは **実装の細部はソースと `CLAUDE.md`、方針と地雷は本書** を参照すると迷いにくい。

## 15. セッション 2026-04-19〜2026-04-20 の差分要約

Claude + Codex 併走で PR #55〜#75 の 21 本を merge。OSS としての顔と中身を両軸で整えたセッション。

**Pose-free パス (新機能)**: PR #55 で DUSt3R を `stub → real` に差し替え、`PoseFreeProcessor(method="dust3r")` + `scripts/run_dust3r.py` で CLI からも回せるように。PR #66 で MAST3R を第 2 backend として同 shape で wire、`scripts/run_mast3r.py` も PR #70 で対称化。`gs-mapper photos-to-splat` (PR #62) で JPG ディレクトリ → `.splat` が一撃。`gs-mapper export --format splat` も同 PR で追加。

**Bundled demo (5 scene)**:
1. `outdoor-demo.splat` — supervised GNSS + LiDAR 6-bag fused (既存)
2. `outdoor-demo-dust3r.splat` — PR #55、bag6 cam0 DUSt3R pose-free
3. `bag6-mast3r.splat` — PR #66、MAST3R metric + 15k iter (PR #70 で quality push)
4. `mcd-tuhh-day04.splat` — PR #64、MCD 非-Autoware day session DUSt3R
5. `mcd-tuhh-day04-mast3r.splat` — PR #67、MCD MAST3R 15k iter

**Viewer (3 本完全 symmetry)**: `splat.html` / `splat_spark.html` / `splat_webgpu.html` に PR #65 / #71 / #72 で scene picker 移植、PR #73 で `docs/scene-picker.js` + `docs/scenes-list.json` に DRY。Spark 2.0 の blank canvas は PR #71 で解決 (SparkRenderer 追加 + three r179 pin)。PR #74 で iPhone 14 viewport dogfood + `splat_spark.html` viewport meta 追加。

**MCD 経路の知見**:
- NTU #17 は image-only 全滅 (PR #58): COLMAP / DUSt3R ×3 / ORB-SLAM3 全部 frame-to-frame match 不足で詰む。night handheld + GPS-denied + repetitive texture の組み合わせが原因。
- `scripts/download_mcd_folder.sh` (PR #60) で folder 全体を `gdown --folder` で 1 本取得可。
- tuhh_night_09 は vn200 GPS が全 sample lat/lon/alt=0 (PR #61) — **night handheld は GPS 取れてない**。
- tuhh_day_04 は image-only DUSt3R で 19/20 非退化 + gsplat 収束 (PR #64) — day session なら GNSS/calibration 無しで demo 化可能。
- ~~MCDVIRAL calibration YAML は未公開、`--method mcd` の GNSS+LiDAR 経路を回すには別途入手が必要~~ → **2026-04-20 に公式 Download page 本体で発見** (PR #79)、`--mcd-static-calibration` フラグで CLI から食わせられる (PR #80)。詳細は §4.3.3.c。

**OSS 顔整備**:
- README hero GIF (PR #68): Playwright で scene picker cycle 録画 → 640×360 / 760 KB GIF。
- README 比較表 + A/B サムネイル (PR #67)、mobile 節 (PR #74)、Credits 節 (PR #73)、Benchmark 表 (PR #75)。
- CLI cleanup (PR #69): `experiment-*` 14 本を `gs-mapper experiment <lab>` nested subparser に隔離、legacy alias は argv rewriter で back-compat、`gs-mapper --help` が痩せた。
- Robotics smoke (PR #59): `scripts/robotics_smoke.py` で PLY → render → bridge payload を ROS なしで貫通、CI 対応。
- GGRt は upstream の CUDA extension 依存で integration 断念、PR #63 で docs を現実に揃えて "reference only" とラベリング。

**次セッション向け着手候補** (PR #75 当時):
- ~~MCD day session を GNSS + calibration YAML 経由で正規 `--method mcd` に乗せる~~ → **PR #79 / #80 で calibration downloader + `--mcd-static-calibration` までは揃った**。残りは GPU training run だけ、§15.1 / §4.3.3.c 参照。
- CoVLA の HF access 承認後、`gs-mapper photos-to-splat --preprocess mast3r` で demo 化。
- ~~WebXR (Enter VR) button を Spark viewer に露出~~ → PR #76 で完了 (`renderer.xr.enabled = true` + `VRButton.createButton`)、README にも反映済み。
- `docs/experiments.md` 系 lab の整理（14 本残存）。

## 15.1 セッション 2026-04-20（post-#75、OSS 顔 + MCD supervised 経路 開通）

Claude Opus 4.7 で PR #77〜#80 の 4 本。OSS 顔の残り整備 + §4.3.3.a の最大 blocker (calibration YAML 入手不能) を解消した短い session。

**OSS 顔 (housekeeping 2 本)**:
- PR #77 — `.github/dependabot.yml` (GHA + pip 週次、torch/vision/audio は gsplat/DUSt3R/MAST3R upstream と coupling するため pin で ignore) + `.github/PULL_REQUEST_TEMPLATE.md` + `.github/ISSUE_TEMPLATE/{bug_report,feature_request,config}.yml` を追加。CONTRIBUTING.md の lint/test incantation を PR 本文テンプレに pre-fill、issue 側は CONTRIBUTING.md §"Reporting issues" の要求項目 (OS / GPU / SDK 版 / minimal repro) を構造化。
- PR #78 — Spark 2.0 の LoD knob を `?lod=auto|high|low` URL param で露出。`LOD_PRESETS = { auto: {}, high: { lodSplatScale: 4.0, lodRenderScale: 0.5 }, low: { lodSplatScale: 0.5, lodRenderScale: 2.0 } }` を `new SparkRenderer({ renderer, ...lodPreset })` に spread。README の viewer 比較表と info pane にも文言追加、`tests/test_pages_assets.py::test_splat_spark_exposes_lod_url_param` で配線を pin。Spark 2.0 の view-dependent LoD / progressive streaming を「謳うだけ」から「触れる機能」にした。

**MCD supervised 経路の blocker 解消 (2 本)**:
- PR #79 — `scripts/download_mcd_calibration.sh <handheld|atv> [out-path]` + `tests/test_download_mcd_calibration_script.py` (5 件 smoke)。Google Drive file ID を MCDVIRAL 公式 Download page の raw HTML scraping で特定 (handheld `1htr26EE-Y1sHS5J4zaSbauC1XFgIh3Ym` / ATV `1zVTBqh4cA1DciWBj5n7BGiexbfan1BBL`)、`body: { <sensor>: { T:[…4×4…], intrinsics, distortion_*, rostopic, timeshift_cam_imu } }` 構造を確認。`curl -sL` で 6.4-6.6 KB、virus-scan confirm dance 不要。CC BY-NC-SA 4.0 なので repo には committee せず downloader + license reminder only。docs/plan_outdoor_gs.md §4.3.3.a の「別リポジトリ想定」推測を訂正、README Credits の MCDVIRAL 行に script 参照を追記。
- PR #80 — `--mcd-static-calibration <calib.yaml>` フラグを `preprocess` / `run` / `demo` の 3 subcommand に配線。実装は `src/gs_sim2real/datasets/ros_tf.py::load_static_calibration_yaml(path, base_frame=…)` が YAML の `body:` セクションから `StaticTfMap` (parent=`base_frame` → child=`<sensor>` edges) を構築、`merge_static_tf_maps(…)` で bag-derived map とマージ (後者が collision 時に勝つ — MCD では常に empty なので YAML が通る)。`_mcd_gnss_sparse_import` の `tf_map` 構築直後と `HybridTfLookup` 用の `static_topo` 構築直後の両方でマージして、`--mcd-tf-use-image-stamps` と併用しても正しい extrinsics が出るようにした。テスト 6 件: YAML parser (happy path / parent override / lookup round-trip / missing `body:` raises / malformed T skipped) + merge helper (後者が勝つ / 両側エッジ保存 / `None` 許容) + CLI flag presence on 3 subcommands。

**本セッションで変わった blocker 状態**:
- §6 の "非対象" には Waymo E2E が残るが、§4.3.3 の MCD supervised 経路はもう blocker が無い。**calibration 入手手段 + YAML 注入 CLI + day session 選定 + GNSS fix 確認が全部揃っている**。次セッションは (a) `scripts/download_mcd_calibration.sh handheld`、(b) §4.3.3.c の CLI incantation、(c) 30k iter training、(d) `ply_to_splat` で 400k/12.8 MB、(e) bundle triplet 更新 (CONTRIBUTING.md §"Bundled demo splats")、の 5 step が一直線。bag4/bag6 の supervised と同 recipe なので GPU 2-4 時間あれば回る見込み (bag4 は 30k iter で 932k gauss、L1 ~0.08)。
- PR #77 の dependabot が 2026-04-21 月曜に初回発火する。最初の週は pip 側で `urllib3` / `certifi` 等の patch bump が数本出る想定、GHA 側は `actions/setup-python` / `actions/checkout` の最新 minor があれば 1-2 本。マージ前に `ci.yml` / `pages.yml` / `publish.yml` が通ることを必ず確認。

**本セッションで明らかに touch しなかった / あえて残した判断**:
- `docs/experiments.md` (911 行) の prune は「公開 vs 内部 note の線引き」という価値判断が要るので auto mode では踏み込まなかった。次セッションでユーザと目線合わせしてから。
- BYO photos / CoVLA demo は外部依存 (ユーザの写真 / HF access 承認) 待ちで auto 環境で完結しない。
- `--renderer gsplat` の CUDA path は GPU smoke が要るので保留。

## 15.2 次セッションへの短冊 (2026-04-20 時点)

**Priority A (ship できる最短経路)**:

1. **`mcd-tuhh-day04-supervised.splat` を bundle の 6 本目に** — §4.3.3.c の full recipe。コード変更は bundle triplet 更新 (1 行 JSON + 3 HTML `<option>` + 1 test 関数) のみ、本丸は GPU training。PR #79 / #80 が merged 済みなら即着手可。想定 L1 0.08-0.12 で pose-free DUSt3R 版 (0.18) / MAST3R 版 (0.16) に対して明らかに sharper な比較対象になる。README Benchmark 表の MCD 行を 3 項目 (DUSt3R / MAST3R / supervised) に拡張できる。
2. **`docs/experiments.md` の prune** — ユーザ合意後。公開 vs 内部 note の線引き方針を先に決める (前セッションで ROI 低判定、次セッションで議論)。

**Priority B (blocker あり or ROI 中)**:

3. **BYO photos demo** — ユーザ自身の 20-40 枚写真で `photos-to-splat --preprocess mast3r` を回して claim の自己実証。ユーザが写真を提供する必要。
4. **CoVLA mini** — HuggingFace `turing-motors/CoVLA-Dataset-Mini` の access request 承認後 pipeline 実走。driving scene 第 6 demo 候補。
5. **Spark 2.0 LoD knob を scene-picker に promote** — PR #78 は URL param only。CONTRIBUTING.md §"Where to start" で示唆されているとおり、`docs/scene-picker.js` に LoD `<select>` を追加して 3 viewer 共通にしておくと将来 `?cameras=` preset も同じ DRY 経路で足せる。

**Priority C (refactor / 深堀り)**:

6. `--renderer gsplat` CUDA path の smoke テスト (gated、GPU 必要)。
7. README `apps/` / `projects/` プロトタイプ節の最新化 — DreamWalker Live / Robotics の現状と同期していない可能性あり。
8. NMEA / GNSS / IMU robustness (§6 の残 blocker)。日跨ぎ RMC、IMU quaternion 融合、logger 時刻ずれ。
