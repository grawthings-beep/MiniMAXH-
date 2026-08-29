# MiniMax H3 I2V for ephemeral RunPod Pods

MiniMax H3のI2Vを、永続ストレージなしのRunPod Podで毎回起動する構成です。
ベースモデル、アップスケーラー、creator LoRA、LightX2V Turbo LoRA、自動モザイク用モデルは
Docker imageへ埋め込まず、Pod起動時に並列・再開可能・整合性検証付きで取得します。

## UIに表示する3ワークフロー

ComfyUIへ表示するMiniMax H3ワークフローは、次の3本だけです。

| 名前 | H3サンプリング | 用途 |
|---|---|---|
| `01_MiniMax_H3_Quality_2x` | 20 steps / `res_multistep` | 最終品質と比較基準 |
| `02_MiniMax_H3_Fast_FBCache_2x` | 20 steps / FirstBlockCache Safe | 品質を大きく落とさず高速化 |
| `03_MiniMax_H3_Turbo_8step_2x` | 8 steps / Euler / shift 12・3 | 通常の高速生成 |

3本すべてが次に対応します。

- 1枚の開始画像によるI2V
- 任意の終了画像によるfirst/last-frame補間
- 起動時に取得したcreator LoRAの選択と強度変更
- `RealESRGAN_x2plus`による完成フレーム2倍化
- 完成動画だけを対象にするCPU自動モザイク
- H3ネイティブの24fpsステレオ音声

2xが不要なら`ImageUpscaleWithModel`をbypassします。モザイクは
`WanAutoMosaicVideo`ノードの`enabled`でON/OFFできます。LoRAは強度`0.0`で効果を
無効化できます。

旧I2V/R2V/Storyboard/EasyCache派生JSONは互換性・再生成テスト用としてrepo内に残しますが、
RunPodのワークフロー一覧へはコピーしません。起動時に旧`MiniMax_H3`配布ワークフローを消し、
上の3本だけを配置します。

## Runtime構成

Community Cloudのhost driver差を吸収するため、同じworkflowを2種類のimageで配布します。

| image tag | PyTorch | host driver | 特徴 |
|---|---|---|---|
| `community-cu128` | `2.9.1+cu128` | r525以上 | r550 L40S、r570 RTX 5090、r580以降で起動する互換優先版 |
| `fast-cu130` | `2.10.0+cu130` | r580以上 | ComfyKitchen CUDA INT8を使う高速版 |

- ComfyUI `v0.31.0`を固定
- MiniMax H3 FL2VA pruned INT8 ConvRot
- `community-cu128`はComfyKitchen eager fallbackを許可
- `fast-cu130`だけComfyKitchen CUDAを必須化
- FastはFirstBlockCacheの`H3 Safe`（threshold `0.08`、10〜95%、最大2連続hit）
- TurboはLightX2V FL2VA Turbo 8-step v1.0を強度`1.0`でcreator LoRAの前へ適用
- Turboのcreator LoRAは32GBで二重LoRA 208 patchesを避けるため初期値`0.0`
- Turboは公式推奨どおり`Euler`、`simple`、8 steps、video shift `12`、audio shift `3`
- EasyCacheとFirstBlockCacheは併用しない
- TurboとFirstBlockCacheも既定では併用しない
- `--fast-disk`はモデル退避I/Oを増やす場合があるため既定で使わない

FirstBlockCache custom nodeはcommit
`725973c3bfd9de6dce249bc93dc5fe27f820df31`を固定します。Turbo LoRAはHugging Face
revision `05ef678438e84933c406131b59abbf86919b3aac`、サイズ`1956193000` bytes、
SHA256 `2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e`を検証します。

## 必要容量と推奨マシン

| ファイル | 容量 |
|---|---:|
| FL2VA pruned INT8 | 20.97 GB |
| Qwen3-VL-32B NVFP4/AWQ | 15.69 GB |
| Video VAE | 5.21 GB |
| Audio VAE | 0.61 GB |
| Real-ESRGAN x2plus | 0.067 GB |
| creator LoRA 2本 | 0.620 GB |
| LightX2V Turbo 8-step LoRA | 1.956 GB |
| 自動モザイクモデル | 0.019 GB |
| 起動時取得合計 | 約45.1 GB |

- Container Disk: `120 GB`
- Volume Disk: `0 GB`
- HTTP Port: `8188`
- GPU VRAM: 32 GB以上を推奨。32GBでは0.4MP/5秒から開始、48GBでは0.6MP/5秒から開始
- System RAM: 64 GB以上推奨
- NVIDIA Driver: `community-cu128`はr525以上、`fast-cu130`はr580以上

## RunPod template

Docker image:

```text
ghcr.io/grawthings-beep/minimax-h3-i2v:community-cu128
```

r580以上を確認できるhostだけ高速版へ変更します。

```text
ghcr.io/grawthings-beep/minimax-h3-i2v:fast-cu130
```

環境変数は次を設定します。tokenの実値をテンプレートやログへ直接書かず、RunPod Secretを使います。
`CIVITAI_API_TOKEN`を省略した場合は`CIVITAI_TOKEN`を自動的に再利用します。

```text
HF_TOKEN={{ RUNPOD_SECRET_HF_TOKEN }}
CIVITAI_TOKEN={{ RUNPOD_SECRET_CIVITAI_TOKEN }}
CIVITAI_API_TOKEN={{ RUNPOD_SECRET_CIVITAI_TOKEN }}

ACCEPT_MINIMAX_H3_LICENSE=1
MINIMAX_H3_LICENSEE_IN_APPLICABLE_TERRITORY=1
MINIMAX_H3_SEPARATE_LICENSE=0

H3_LORA_REQUIRED=1
H3_LORA_SELECTION=all
H3_LORA_REPO_ID=uwgm/nikke-civitai-backup
H3_LORA_SOURCE_PATH=hmmotion_minimax-h3_epoch12.safetensors
H3_LORA_REVISION=main
H3_CIVITAI_LORA_URL=https://civitai.red/api/download/models/3206518?fileId=3088013

H3_TURBO_REQUIRED=1
H3_TURBO_REPO_ID=lightx2v/Minimax-h3-Turbo
H3_TURBO_SOURCE_PATH=minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors
H3_TURBO_REVISION=05ef678438e84933c406131b59abbf86919b3aac

AUTO_MOSAIC_REQUIRED=1
AUTO_MOSAIC_MANIFEST=/opt/minimax-h3/manifests/auto_mosaic.json

HF_XET_HIGH_PERFORMANCE=auto
HF_XET_CHUNK_CACHE_SIZE_BYTES=0
HF_DOWNLOAD_WORKERS=4
HF_HUB_DOWNLOAD_TIMEOUT=120
DOWNLOAD_RETRIES=3
MODEL_VERIFY=size
MODEL_MANIFEST=/opt/minimax-h3/manifests/minimax_h3_i2v_upscale.json
REQUIRE_COMFY_KITCHEN_CUDA=0
COMFYUI_ARGS="--disable-dynamic-vram --reserve-vram 4"
TINI_SUBREAPER=1
```

`fast-cu130`を使う場合だけ`REQUIRE_COMFY_KITCHEN_CUDA=1`へ変更します。旧Templateの
`--vram-headroom 2`は起動時に安定profileへ自動移行されます。`REQUIRE_COMFY_KITCHEN_CUDA=0`
だけを旧cu130 imageへ設定してもdriver非互換は解消しません。

MiniMax H3のライセンスを確認し、利用者本人または組織がApplicable Territoryを拠点とする場合だけ
`MINIMAX_H3_LICENSEE_IN_APPLICABLE_TERRITORY=1`を設定してください。別途MiniMaxから許諾を得た場合は
代わりに`MINIMAX_H3_SEPARATE_LICENSE=1`を使用します。

## Pod起動処理

起動時には以下を並列で実行します。

1. GPU、driver、VRAM、RAM、disk、ComfyKitchen CUDAを診断
2. H3ベースモデル4ファイルとReal-ESRGANを取得
3. HMMotion V1とHMNSFW AIO V2 creator LoRAを取得
4. LightX2V Turbo 8-step LoRAをHugging Face Xetで取得
5. YOLO11 segmentationモデルをCivitaiから再開可能download
6. サイズ、SHA256、safetensors header、ZIP CRCを対象ごとに検証
7. 全必須モデルとcustom nodeが揃った場合だけComfyUIを起動

毎回ダウンロードする構成なので、モデルはPod破棄時に消えます。生成物も必要ならPod停止前に
手元へ保存してください。

## 自動モザイク

`WanAutoMosaicVideo`は入力画像ではなく、`VAE Decode → Real-ESRGAN 2x`後の完成フレームへ
一度だけ適用されます。`enabled=false`なら検出モデルをロードせず、元のtensorをそのまま返します。

- CPU-only YOLO11 instance segmentation
- coverage `JUST`
- confidence `0.30`
- IoU `0.50`
- block size `0`（短辺÷50、最小10px）
- gap `3`、ループ境界をまたぐ短い検出抜けも補間
- 既定対象: `pussy,penis,testicles`
- `anus`は既定対象から除外

## LoRA

Quality/Fastではcreator LoRAだけが適用されます。Turboではモデルチェーンを次の順序に固定します。

```text
INT8 FL2VA → LightX2V Turbo 1.0 → 選択creator LoRA → SigmaShift → Scheduler/Guider
```

Turbo LoRAをcreator LoRAの後ろへ動かしたり、強度を変更したりしないでください。creator LoRAは
プルダウンでV1/V2を選び、強度を調整できます。Quality/Fastの既定は
`HMNSFW_AIO_V2.safetensors / 0.5`、Turboだけは安定性のため`0.0`です。

## 検証

```bash
python -m unittest discover -s tests -v
python scripts/verify_workflow.py --workflow workflows/minimax_h3_preset_01_quality.json --manifest manifests/minimax_h3_i2v_upscale.json --mode i2v --expect-upscale --expect-auto-mosaic --auto-mosaic-manifest manifests/auto_mosaic.json --expect-lora HMNSFW_AIO_V2.safetensors --expect-lora-strength 0.5
python scripts/verify_workflow.py --workflow workflows/minimax_h3_preset_02_fast_fbcache.json --manifest manifests/minimax_h3_i2v_upscale.json --mode i2v --expect-upscale --expect-auto-mosaic --auto-mosaic-manifest manifests/auto_mosaic.json --expect-lora HMNSFW_AIO_V2.safetensors --expect-lora-strength 0.5 --expect-first-block-cache
python scripts/verify_workflow.py --workflow workflows/minimax_h3_preset_03_turbo_8step.json --manifest manifests/minimax_h3_i2v_upscale.json --mode i2v --expect-upscale --expect-auto-mosaic --auto-mosaic-manifest manifests/auto_mosaic.json --expect-lora HMNSFW_AIO_V2.safetensors --expect-lora-strength 0.0 --expect-turbo
bash -n scripts/entrypoint.sh scripts/download_models.sh
```

実際の生成smoke testにはNVIDIA GPUと約45.1GBのダウンロードが必要です。

## Sources

- [ComfyUI MiniMax H3 guide](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)
- [ComfyUI v0.31.0](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.31.0)
- [MiniMax H3 weights](https://huggingface.co/Comfy-Org/MiniMax-H3)
- [LightX2V MiniMax H3 Turbo](https://github.com/ModelTC/Minimax-H3-Turbo)
- [MiniMax H3 FirstBlockCache](https://github.com/duckyshell/ComfyUI-MiniMaxH3-FirstBlockCache)
- [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
