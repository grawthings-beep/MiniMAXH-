# MiniMax H3 I2V for ephemeral RunPod Pods

MiniMax H3のI2Vを、永続ストレージなしのRunPod Podで毎回起動する構成です。
ベースモデル、アップスケーラー、creator LoRA、LightX2V Turbo LoRA、自動モザイク用モデルは
Docker imageへ埋め込まず、Pod起動時に並列・再開可能・整合性検証付きで取得します。

## キャラR2VAだけを使う場合（起動ダウンロード削減）

新imageで `H3_PROFILE=r2v` を指定すると、同一キャラ画像1〜8枚＋全文プロンプト1欄の04だけを配置します。
RefMod Full Reference対応。画像追加/削除/並べ替え、合計参照量制限、互換ファイル書き出し付き。
4欄に分けたりfullモードへ切り替えたりする必要はありません。新imageへ交換後、04を開き直してください。
公式INT8 Video VAEを使用し、基本モデルは約63.5GBから約40.1GBへ削減。
FL2VA本体・Turbo LoRA・追加LoRA一覧は取得しません。2xとCPUモザイクは維持します。
任意LoRAはModels列の「選択 / 強度 / ON-OFF」で操作可能（初期OFF）。
起動取得は `H3_R2V_LORA_SELECTION` に必要分だけ指定（既定 `none`）。
旧環境変数が残っていても専用profileが取得対象を固定します。
CUDA 13用設定は [runpod-template.r2v-cu130.example.json](runpod-template.r2v-cu130.example.json)、
使い方と制約は [R2VA専用ガイド](docs/r2v-only.md) を参照してください。
`H3_R2V_MODEL=dasiwa-v2` で **DaSiWa Hybrid v2 INT8＋作者推奨25step/simple/shift11,4** を使用可能。
`dasiwa-turbo-v2` は内蔵蒸留8step、`official` は従来版へ戻す設定です。
選んだ本体だけ取得し、モデルと設定が一致した04を1本生成します。環境変数省略時は公式のまま。
[作者workflowとの比較・採用判断・切替手順](docs/dasiwa-r2v.md)を確認してください。
旧構成は `H3_PROFILE=legacy`（省略時の既定）で引き続き使えます。

## 通常起動（legacy）でUIに表示する3ワークフロー

通常起動でComfyUIへ表示するMiniMax H3ワークフローは、次の3本です。

| 名前 | H3サンプリング | 用途 |
|---|---|---|
| `01_MiniMax_H3_Quality_2x` | 20 steps / `res_multistep` | 最終品質と比較基準 |
| `02_MiniMax_H3_Fast_FBCache_2x` | 20 steps / FirstBlockCache Safe | 品質を大きく落とさず高速化 |
| `03_MiniMax_H3_Turbo_4_8step_768p_2x` | 8-step v1.0 / 4-step v1.2選択、Euler、shift 6・3 | 通常の高速生成または最速draft |

3本すべてが次に対応します。

- 1枚の開始画像によるI2V
- 任意の終了画像によるfirst/last-frame補間
- 起動時に取得したcreator LoRAの選択と強度変更
- `RealESRGAN_x2plus`による完成フレーム2倍化（初期OFF）
- 完成動画だけを対象にするCPU自動モザイク
- H3 Attention・MLPの2分割と、ComfyUIの時間chunk Video VAE Decode
- H3ネイティブの24fpsステレオ音声

2xは初期状態でbypassです。サブグラフを開き、`Optional 2x (OFF)`を選択して
`Ctrl+B`で有効化できます。配布ファイル名の`_2x`は互換性のため残しています。モザイクは
`WanAutoMosaicVideo`ノードの`enabled`でON/OFFできます。LoRA操作はサブグラフ外の
紫色`LoRA CONTROLS`グループへ常時表示され、強度`0.0`ならLoRAファイルをロードしません。

旧I2V/R2V/Storyboard/EasyCache派生JSONは互換性・再生成テスト用としてrepo内に残しますが、
RunPodのワークフロー一覧へはコピーしません。起動時に旧`MiniMax_H3`配布ワークフローを消し、
上の3本だけを配置します。

### キャラ画像から自由な冒頭を作る：Character R2V（追加オプション）

`H3_CHARACTER_R2V=1` を指定すると、既存3本に加えて
`04_MiniMax_H3_Character_R2V_2x` を配置します。キャラ画像を開始フレームではなく
外見の参照として使い、「一部だけ見せる→待たせる→顔を見せる」などの演出を指定できます。
公式Ref2VA INT8モデルを起動時に追加取得（約21GB）し、共通のQwen/VAEは重複取得しません。
25 steps、近似キャッシュ・追加LoRAなし、モデル退避→タイルVAE→2x→CPUモザイク→MP4です。
すべての操作ノードをトップレベルに配置しています。

**初期値は動作・キャラ保持の確認用5秒/0.4MP。最高画質設定ではありません。**
生成解像度を0.6→0.98MPと上げる手順、演出の入力方法、限界、RunPod設定は
[Character R2Vガイド](docs/character-r2v.md)を参照してください。
新しいイメージが必要です。旧イメージへ変数だけ追加しても有効になりません。

## Runtime構成

Community Cloudのhost driver差を吸収するため、同じworkflowを2種類のimageで配布します。

| image tag | PyTorch | host driver | 特徴 |
|---|---|---|---|
| `community-cu128` | `2.9.1+cu128` | r525以上 | r550 L40S、r570 RTX 5090、r580以降で起動する互換優先版 |
| `fast-cu130` | `2.10.0+cu130` | r580以上 | ComfyKitchen CUDA INT8を使う高速版 |

- ComfyUI `v0.37.0`を固定
- MiniMax H3 FL2VA pruned INT8 ConvRot
- `community-cu128`はComfyKitchen eager fallbackを許可
- `fast-cu130`だけComfyKitchen CUDAを必須化
- FastはFirstBlockCacheの`H3 Safe`（threshold `0.08`、10〜95%、最大2連続hit）
- Turboは1つの`Turbo Mode`でLightX2V FL2VA 768pの8-step v1.0（既定）または4-step v1.2を選択
- 選択したTurbo LoRAだけを強度`1.0`でcreator LoRAの前へ適用し、stepsも自動連動
- Turboのcreator LoRAは32GBで二重LoRA 208 patchesを避けるため初期値`0.0`
- Turboは両モードとも`Euler`、`simple`、video shift `6`、audio shift `3`
- EasyCacheとFirstBlockCacheは併用しない
- TurboとFirstBlockCacheも既定では併用しない
- H3だけ`ModelAttentionBackend`でComfyKitchen INT8 Attentionを選択（未対応環境はComfyUIがPyTorchへfallback）
- KJNodesのH3専用Attentionをhead 2分割、MLPをtoken 2分割（4096 tokens超）で適用
- KJNodes commit `d3cfe21625e5170126ce06fbfcfe1d88108688c3`のH3モジュールとライセンスのみを組み込む
- `--fast-disk`が明示されていない場合、起動時に`--disable-fast-disk`を追加して自動disk offloadを抑制
- 全3プリセットを標準`VAEDecode`に戻し、v0.37.0の時間chunk入出力と必要時の自動tile fallbackを利用
- 毎回の全モデル解放は行わず、モデル配置と再利用はComfyUIに任せる。旧解放・tileノードは旧JSON互換用に残す

FirstBlockCache custom nodeはcommit
`725973c3bfd9de6dce249bc93dc5fe27f820df31`を固定します。Turbo LoRAはHugging Face
revision `2f015e66b37c585cea9dc4ae6f1850ea8788e742`を固定し、各`1956193000` bytesと次の
SHA256を検証します。8-step: `08cfe946033af7d27719b964b6e0a0e50c32138daabbd6ce4137e23df6bf9980`、
4-step: `c8168ebc17bbacc4296103dda2fec1ba85b24392fa08cf2bfbcef0cff0dc3cc8`。

## 必要容量と推奨マシン

| ファイル | 容量 |
|---|---:|
| FL2VA pruned INT8 | 20.97 GB |
| Qwen3-VL-32B NVFP4/AWQ | 15.69 GB |
| Video VAE | 5.21 GB |
| Audio VAE | 0.61 GB |
| Real-ESRGAN x2plus | 0.067 GB |
| creator LoRA 2本 | 0.620 GB |
| LightX2V Turbo 8-step/4-step LoRA | 3.912 GB |
| 自動モザイクモデル | 0.019 GB |
| 起動時取得合計 | 約47.1 GB |

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
H3_TURBO_8STEP_SOURCE_PATH=minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors
H3_TURBO_4STEP_SOURCE_PATH=minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors
H3_TURBO_REVISION=2f015e66b37c585cea9dc4ae6f1850ea8788e742

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
COMFYUI_ARGS=--lowvram --vram-headroom 2
TINI_SUBREAPER=1
```

`fast-cu130`を使う場合だけ`REQUIRE_COMFY_KITCHEN_CUDA=1`へ変更します。旧Templateの
誤って設定された旧v1.1 profile `--disable-dynamic-vram --reserve-vram 4` は、起動時に
`--lowvram --vram-headroom 2`へ自動修正されます。DynamicVRAMを無効化すると、CUDA 12.8の
INT8 eager fallbackが一時バッファを確保できず、24GB/32GB GPUでOOMしやすくなります。
`REQUIRE_COMFY_KITCHEN_CUDA=0`
だけを旧cu130 imageへ設定してもdriver非互換は解消しません。

MiniMax H3のライセンスを確認し、利用者本人または組織がApplicable Territoryを拠点とする場合だけ
`MINIMAX_H3_LICENSEE_IN_APPLICABLE_TERRITORY=1`を設定してください。別途MiniMaxから許諾を得た場合は
代わりに`MINIMAX_H3_SEPARATE_LICENSE=1`を使用します。

## Pod起動処理

### 2回目以降の生成でOOMになる場合

新イメージではまず`03 Turbo`、`8-step`、`0.4MP`、`5秒`、creator LoRA `0.0`、
2x OFFでseedを変えて2回生成してください。同一seedの再実行ではキャッシュが再利用され、
2回目のsamplingを検証できません。成功したら同じ条件の10秒、最後に2x ONを試します。
GPUと使用モデルの種類は維持します。Podは新規作成時に新イメージを指定し、従来どおり起動時にモデルを取得します。

ビルドでは実際のComfyUI/KJNodesを使ってFP32/BF16のAttentionとMLPをCPUで比較します。
A40上のINT8 CUDA実行、実VRAMピーク、5秒の連続生成、10秒以上の安定性・速度は別途実測が必要です。
分割の呼び出し回数が増えるため、全条件で高速化するとは限りません。

既定のキャッシュとモデル再利用は維持します。キャッシュ保持の影響を切り分ける診断として、
一時的に`--cache-none --disable-smart-memory`を指定できます。中間ノードの出力・オブジェクトを
生成間でキャッシュせず、処理終了時にはComfyUIのモデル管理経由でGPUモデルを退避します。
毎回モデルやテキスト条件を読み直す分、次の生成の準備時間は増えるため、自動では適用しません。
DynamicVRAMは有効のままです。速度を優先する場合はまず失敗したノード・OOMログを確認し、
必要な箇所の解放へ絞り込んでください。

すでに公開済みの`sha-90aeee2-fast-cu130`などでも、次回Pod作成時の環境変数を次のようにすれば
この診断用オプションを適用できます（イメージ更新は不要）。

```text
COMFYUI_ARGS=--lowvram --vram-headroom 2 --cache-none --disable-smart-memory
```

起動ログの`[comfyui] extra args:`で設定を確認し、同じ解像度・秒数でseedを変更して2回連続生成します。
診断終了後は追加した2つのフラグを外すと元の再利用設定へ戻ります。
この設定はキャッシュ保持の切り分け用で、すべてのOOMの解消を保証するものではありません。
失敗する場合は、エラーになったノード名、CUDA OOMの全文（allocated/reserved/freeを含む）、
解像度・秒数・選択LoRAを記録してください。プロセス自体が消える場合はGPU VRAMに加えて
ホストRAM不足も確認します。

### 起動時のダウンロード

起動時には以下を並列で実行します。

1. GPU、driver、VRAM、RAM、disk、ComfyKitchen CUDAを診断
2. H3ベースモデル4ファイルとReal-ESRGANを取得
3. HMMotion V1とHMNSFW AIO V2 creator LoRAを取得
4. LightX2V Turbo 8-step/4-step 768p LoRAをHugging Face Xetで同時取得
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

### URLを貼ってLoRAを追加する

**[manifests/extra_loras.txt](manifests/extra_loras.txt) をGitHubで編集し、URLを1行に1つ追加して
`main`へ保存してください。次回のPod起動時にGitHub上の最新版を取得して自動ダウンロードします。**
初回だけ、この機能を含むDockerイメージをビルド・公開してPodへ反映する必要があります。
その後はURL一覧の変更だけならイメージの再ビルドは不要です。既に動いているPodには自動反映しません。

```text
# 行頭が # の行と空行は無視されます。以下のID・パスは実際の値に置き換えてください。
https://civitai.com/api/download/models/VERSION_ID?fileId=FILE_ID
https://civitai.com/models/MODEL_ID/model-name?modelVersionId=VERSION_ID
https://huggingface.co/OWNER/REPO/resolve/main/lora.safetensors
```

- CivitaiのダウンロードURL、モデルページ、`civitai.red`にも対応します。
  ページURLに`modelVersionId`がない場合はAPIが返す先頭のバージョンを使います。
  バージョン内は条件に一致するSafeTensorのModelファイル（複数ならprimary）を選び、
  一意に決まらない場合はエラーにします。再現性が必要なら`fileId`付きのダウンロードURLを使ってください。
- Hugging Faceの`resolve`と`blob`リンク、その他のHTTPS直接ダウンロードURLにも対応します。
  一般サイトの紹介ページやZIP・ckptファイルは対象外です。
- 認証には既存のRunPod Secret `CIVITAI_TOKEN`（未設定なら`CIVITAI_API_TOKEN`）と`HF_TOKEN`を使います。
  **公開URL一覧にトークンを入れないでください。** トークン付きURLは拒否し、認証ヘッダーは対応するホストだけへ送信します。
- 保存先は`models/loras/extra/`です。同名ファイルの衝突を避けるためURL由来の短いIDをファイル名へ付けます。
  ComfyUI起動後、紫色の`OPTIONAL CREATOR LoRA`選択欄から`extra/…`を選び、強度を設定してください。
  **MiniMax H3対応LoRAを選んでください。ダウンロード成功はモデル互換性を保証しません。**
- safetensorsヘッダーとデータ範囲を検査し、CivitaiはAPIのSHA256とも照合します。
  その他のURLは初回取得時のSHA256を記録し、再実行時に既存ファイルを検査して再利用します。
  一般URLのリモート更新は自動検出しないため、更新時は別URL・revisionを使ってください。
  失敗時は最大`DOWNLOAD_RETRIES`回、先頭から再試行し、検証完了後にだけファイルを公開します。
- 一覧からURLを削除しても既存ファイルは削除しません。永続ディスクなしの新しいPodでは新しい一覧だけを取得します。

追加分の取得に失敗しても、既定ではエラーをログへ記録してComfyUIの起動を続けます。
追加分をすべて必須にする場合は`H3_EXTRA_LORA_REQUIRED=1`を設定してください。
既存のcreator/Turbo LoRAの取得設定とは独立しています。

| 環境変数 | 既定値・用途 |
| --- | --- |
| `H3_EXTRA_LORA_LIST_URL` | `https://raw.githubusercontent.com/grawthings-beep/MiniMAXH-/main/manifests/extra_loras.txt`。forkでは自分のリポジトリURLに変更 |
| `H3_EXTRA_LORA_LIST` | `/opt/minimax-h3/manifests/extra_loras.txt`。`H3_EXTRA_LORA_LIST_URL`を空にした場合のローカル一覧 |
| `H3_EXTRA_LORA_REQUIRED` | `0`。`1`なら一覧取得・追加LoRA取得の失敗で起動を停止 |
| `EXTRA_LORA_DOWNLOAD_TIMEOUT` | `120`秒（ネットワーク待ちのタイムアウト） |

起動済みPodへすぐ追加する場合は、GitHubで一覧を保存後、Podのターミナルで実行します。
終了後にComfyUIのノード定義を更新（Refresh）または画面を再読み込みすると選択欄へ反映されます。

```bash
python /opt/minimax-h3/scripts/download_extra_loras.py --required
```

ローカル一覧の構文だけを確認する、または別の一覧を手動取得する場合:

```bash
python scripts/download_extra_loras.py --list manifests/extra_loras.txt --check
python scripts/download_extra_loras.py --list /workspace/my_loras.txt --required
```

`--list`は環境変数のリモート一覧より優先します。`--check`はLoRA本体を取得せず、URLの形式だけを確認します。
実在・アクセス権・H3互換性は検査しません。

### 既存ワークフローのLoRA適用順

Quality/Fastではcreator LoRAだけが適用されます。Turboではモデルチェーンを次の順序に固定します。

```text
INT8 FL2VA → Turbo Mode（8-step/4-step＋steps出力）→ 選択creator LoRA → SigmaShift 6/3 → Scheduler/Guider
```

メインキャンバスの`TURBO LoRA`プルダウン以外の配線を変更する必要はありません。
creator LoRAも隣の`OPTIONAL CREATOR LoRA`でV1/V2を選び、強度を調整できます。
サブグラフを展開する必要はありません。Quality/Fastの既定は
`HMNSFW_AIO_V2.safetensors / 0.5`、Turboだけは安定性のため`0.0`です。

RTX 5090かつr580以上のhostでは`community-cu128`ではなく`fast-cu130`を使用してください。
`REQUIRE_COMFY_KITCHEN_CUDA=1`だけではCUDA 13 backendへ切り替わりません。Docker image自体を
`ghcr.io/grawthings-beep/minimax-h3-i2v:fast-cu130`へ変更する必要があります。

## 検証

```bash
python -m unittest discover -s tests -v
python scripts/verify_workflow.py --workflow workflows/minimax_h3_preset_01_quality.json --manifest manifests/minimax_h3_i2v_upscale.json --mode i2v --expect-upscale --expect-auto-mosaic --expect-h3-memory --auto-mosaic-manifest manifests/auto_mosaic.json --expect-lora HMNSFW_AIO_V2.safetensors --expect-lora-strength 0.5
python scripts/verify_workflow.py --workflow workflows/minimax_h3_preset_02_fast_fbcache.json --manifest manifests/minimax_h3_i2v_upscale.json --mode i2v --expect-upscale --expect-auto-mosaic --expect-h3-memory --auto-mosaic-manifest manifests/auto_mosaic.json --expect-lora HMNSFW_AIO_V2.safetensors --expect-lora-strength 0.5 --expect-first-block-cache
python scripts/verify_workflow.py --workflow workflows/minimax_h3_preset_03_turbo.json --manifest manifests/minimax_h3_i2v_upscale.json --mode i2v --expect-upscale --expect-auto-mosaic --expect-h3-memory --auto-mosaic-manifest manifests/auto_mosaic.json --expect-lora HMNSFW_AIO_V2.safetensors --expect-lora-strength 0.0 --expect-turbo
bash -n scripts/entrypoint.sh scripts/download_models.sh
```

実際の生成smoke testにはNVIDIA GPUと約47.1GBのダウンロードが必要です。

## Sources

- [ComfyUI MiniMax H3 guide](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)
- [ComfyUI v0.37.0](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.37.0)
- [MiniMax H3 weights](https://huggingface.co/Comfy-Org/MiniMax-H3)
- [LightX2V MiniMax H3 Turbo](https://github.com/ModelTC/Minimax-H3-Turbo)
- [MiniMax H3 FirstBlockCache](https://github.com/duckyshell/ComfyUI-MiniMaxH3-FirstBlockCache)
- [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
