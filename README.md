# MiniMax H3 I2V-first for ephemeral RunPod Pods

MiniMax H3の画像→動画とマルチモーダル参照→動画を、永続ストレージなしのRunPod Podで起動する構成です。
I2V、開始／終了フレーム、複数画像、参照動画の動き・カメラ・付随音声、ネイティブステレオ音声に対応します。

モデルをDockerイメージへ埋め込まず、Pod起動時にHugging Face Xetと公式GitHub Releaseから
I2Vに必要な42.54GBを高速取得します。R2Vは必要な場合だけ20.97GBを追加できます。
Podを破棄するとモデルと出力は消えます。

## 構成

- ComfyUI `v0.30.0`をコミットまで固定
- PyTorch `2.10.0+cu130`をイメージdigestまで固定
- MiniMax H3 `FL2VA` pruned INT8モデル（`REF2VA`は任意）
- Qwen3-VL-32B NVFP4/AWQテキストエンコーダー
- Video VAEと32kHzステレオAudio VAE
- Real-ESRGAN x2plusによる動画フレーム2倍化
- Hugging Face `hf_xet`のRAM連動High Performance／Adaptive切替
- 4つの大型Hugging Faceファイルの並列取得、3回の再試行、途中再開
- Xet失敗時のaria2並列Range Downloadフォールバック
- manifest固定サイズによる高速検証（公式LFS SHA256の完全検証も選択可能）
- GPU、ドライバー、VRAM、RAM、空き容量、ComfyKitchen CUDAの起動前診断
- GPU VRAMに応じた解像度プロファイルの提案
- I2VのQuality版、旧V1 LoRA版、V1/V2選択式LoRA版、EasyCache Fast版、それぞれの2x版を収録
- 2〜100枚を自由に追加・削除・並べ替えできる長尺Storyboard Quality/Fast版を収録
- 長尺2x出力は区間ごとに処理・一時MKV化してからffmpeg結合し、2x全フレームの同時RAM保持を回避
- R2Vを選択した場合だけR2Vワークフローも自動表示
- 全モデル、全ノード、EasyCache、2xアップスケール、動画・音声参照配線をビルド時に照合

既定のI2Vモデルの正確なリビジョン、サイズ、SHA256は
[`manifests/minimax_h3_i2v_upscale.json`](manifests/minimax_h3_i2v_upscale.json)に固定しています。
R2Vを含む[`manifests/minimax_h3_all.json`](manifests/minimax_h3_all.json)など、
モード別manifestも同じディレクトリに収録しています。

## 必要容量

| ファイル | 容量 |
|---|---:|
| FL2VA pruned INT8 | 20.97 GB |
| Qwen3-VL-32B NVFP4/AWQ | 15.69 GB |
| Video VAE | 5.21 GB |
| Audio VAE | 0.61 GB |
| Real-ESRGAN x2plus | 0.067 GB |
| I2V既定合計 | 42.54 GB（39.62 GiB） |
| HMMotion V1 LoRA | +0.310 GB |
| HMNSFW AIO V2 LoRA | +0.310 GB |
| I2V + LoRA 2本 | 43.16 GB |
| REF2VA pruned INT8（任意） | +20.97 GB |
| I2V + R2V + LoRA 2本 | 64.13 GB |

RunPodのContainer Diskは、再構築バッファ、入力動画、出力領域を含めて120GBを推奨します。
Volume DiskやNetwork Volumeは使用しません。

## 初回セットアップ

### 1. ライセンスを確認

モデル利用前に
[MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
を確認してください。ライセンスは米国、EU、英国、韓国をApplicable Territoryから除外しています。

確認後、RunPodテンプレートの環境変数を次のように変更します。

```text
ACCEPT_MINIMAX_H3_LICENSE=1
MINIMAX_H3_LICENSEE_IN_APPLICABLE_TERRITORY=1
```

`ACCEPT_MINIMAX_H3_LICENSE=1`はライセンスを確認・承諾したこと、
`MINIMAX_H3_LICENSEE_IN_APPLICABLE_TERRITORY=1`は利用者本人または組織が
Applicable Territoryを拠点としていることの自己確認です。いずれも`1`でなければモデルは
ダウンロードされません。

MiniMaxの[公式License Q&A](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/QA-about-License.md)は、
個別許諾が必要な対象を`persons based in these regions`と説明しています。ライセンス本文は
クラウド計算機の物理所在地を明示的に定義していないため、RunPodが自動設定する
`RUNPOD_DC_ID`は診断情報として表示します。米国、EU、英国、韓国の識別子では注意を表示しますが、
利用者所在地の自己確認後は停止しません。

利用者本人または組織が除外地域を拠点としている場合は、MiniMaxから別途許諾を得た場合だけ
`MINIMAX_H3_SEPARATE_LICENSE=1`を使用できます。この場合、
`MINIMAX_H3_LICENSEE_IN_APPLICABLE_TERRITORY=1`は設定しません。

このガードは法的判断を自動化するものではありません。利用者、配信先、Outputの利用にも
ライセンス条件が適用されます。第三者へ生成サービスを提供する場合の利用規約・安全対策・AI生成表示、
商用利用時の表示や一定規模以上の事前許諾など、ライセンス本文の追加条件も適用されます。

### 2. コンテナをビルド

```bash
docker build --platform linux/amd64 -t ghcr.io/OWNER/minimax-h3-i2v:0.7.0 .
docker push ghcr.io/OWNER/minimax-h3-i2v:0.7.0
```

タグ`v*`をpushするか、GitHub Actionsの`Build container`を手動実行してGHCRへ公開することもできます。
公開先に合わせて`runpod-template.example.json`の`imageName`を変更してください。

### 3. RunPodテンプレート

[`runpod-template.example.json`](runpod-template.example.json)を基準にCustom Templateを作成します。

- Container Disk: 120GB
- Volume Disk: 0GB
- Expose HTTP Port: `8188`
- Docker image: 上で公開した固定タグ
- `ACCEPT_MINIMAX_H3_LICENSE=1`
- 日本など許可地域を拠点とする利用者は`MINIMAX_H3_LICENSEE_IN_APPLICABLE_TERRITORY=1`

GPUは起動のたびに変更できます。CUDA 13.0の公式カーネルを使うため、NVIDIA Driver
`r580`以上のホストを選択してください。古いドライバーやCUDA 12.8へフォールバックしたPodは、
42.54GBのモデル取得前に起動を停止します。

### 4. 起動

Podログには次の順で状態が表示されます。

1. GPU・ドライバー・RAM・空き容量・ComfyKitchen CUDA診断
2. Xet並列ダウンロード
3. manifest固定サイズ検証
4. 実効ダウンロード速度
5. ComfyUI `8188`起動

RunPodの`Connect to HTTP Service [Port 8188]`からComfyUIを開き、
次の8ワークフローから選びます。高速化と2倍化の両方が必要なら
`MiniMax_H3_I2V_Fast_EasyCache_2x`を選びます。

- `MiniMax_H3_I2V_Quality`
- `MiniMax_H3_I2V_Fast_EasyCache`
- `MiniMax_H3_I2V_Quality_2x`
- `MiniMax_H3_I2V_Quality_HMMotion_LoRA_2x`
- `MiniMax_H3_I2V_Quality_Selectable_LoRA_2x`
- `MiniMax_H3_I2V_Fast_EasyCache_2x`
- `MiniMax_H3_Story_Quality_Selectable_LoRA_2x`
- `MiniMax_H3_Story_Fast_EasyCache_Selectable_LoRA_2x`

R2Vも使用する場合は、RunPodテンプレートのmanifestを次へ変更します。

```text
MODEL_MANIFEST=/opt/minimax-h3/manifests/minimax_h3_all.json
```

追加のREF2VA 20.97GBを取得し、次の4ワークフローも表示します。

- `MiniMax_H3_R2V_Quality`
- `MiniMax_H3_R2V_Fast_EasyCache`
- `MiniMax_H3_R2V_Quality_2x`
- `MiniMax_H3_R2V_Fast_EasyCache_2x`

### LoRA選択

`MiniMax_H3_I2V_Quality_HMMotion_LoRA_2x`は、正常動作を確認した
`MiniMax_H3_I2V_Quality_2x`から派生したプリセットです。H3のINT8 ConvRotモデルを
ComfyUI標準の`LoraLoaderModelOnly`へ通し、その出力を`BasicScheduler`と
`BasicGuider`の両方へ接続しています。初期強度は`1.0`です。LoRAノードの強度を
`0.0`にすると、配線を変えずに効果だけ無効化できます。

`MiniMax_H3_I2V_Quality_Selectable_LoRA_2x`は同じ配線で、新しい
`HMNSFW_AIO_V2.safetensors`を既定にした選択式プリセットです。LoRAノードの
`lora_name`プルダウンには起動時に取得した次のファイルが表示されます。

- `hmmotion_minimax-h3_epoch12.safetensors`: 旧V1、既定強度`1.0`
- `HMNSFW_AIO_V2.safetensors`: 新V2、既定強度`0.5`

V2のトリガーワードは`hmmotion`です。作者は強度`0.5`以下を推奨し、BF16モデルで
学習・生成したと説明しています。この構成のINT8 ConvRotモデルでの結果は同一seedで
Quality版と比較してください。

LoRAはコンテナへ同梱しません。Pod起動時にベースモデル、旧V1、新V2を並列取得します。
既定では両方を取得するため、Hugging FaceとCivitaiのread tokenをRunPod secretへ設定します。

```text
H3_LORA_REPO_ID=uwgm/nikke-civitai-backup
H3_LORA_SOURCE_PATH=hmmotion_minimax-h3_epoch12.safetensors
H3_LORA_REVISION=main
H3_LORA_REQUIRED=1
H3_LORA_SELECTION=all
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
CIVITAI_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

`HF_TOKEN`には非公開バックアップを読めるtoken、`CIVITAI_TOKEN`にはCivitai API tokenが
必要です。V2の作者がログイン必須ダウンロードを指定しているため、匿名取得はできません。
Bearer tokenはURLやログへ出さず、短命CDN URLへのリダイレクトにも転送しません。

```text
H3_LORA_SELECTION=all             # V1とV2を取得し、両方をプルダウン表示
H3_LORA_SELECTION=hmmotion_v1     # V1のみ。CIVITAI_TOKEN不要
H3_LORA_SELECTION=hmnsfw_aio_v2   # V2のみ。HF_TOKEN不要
H3_LORA_SELECTION=none            # LoRAを取得せずLoRAワークフローも非表示
```

両ファイルとも固定サイズ、固定SHA256、safetensorsヘッダー、tensor領域を検査してから
ワークフローを表示します。取得または検証に失敗した場合は、`H3_LORA_REQUIRED=1`なら
ComfyUIを起動しません。

### 任意枚数の長尺Storyboard

`MiniMax_H3_Story_Quality_Selectable_LoRA_2x`と
`MiniMax_H3_Story_Fast_EasyCache_Selectable_LoRA_2x`は、任意枚数のキーフレームを
順番どおりにつなぐI2V専用ワークフローです。既定LoRAはV2の強度`0.5`で、LoRAノードの
プルダウンから起動時に取得したV1/V2を選べます。

1. `MiniMax H3 Ordered Storyboard`の`＋ 画像を追加`で2〜100枚を一括アップロードします。
2. `↑`、`↓`、`×`で順番と枚数を変更します。
3. 各画像の下で、その画像から次の画像へ移る区間のプロンプトと秒数を設定します。
4. 必要なら`最後→最初を追加（ループ）`を有効にします。
5. Queueすると、通常はN枚からN−1区間、ループ時はN区間を順次生成し、最終2x MP4を保存します。

画像枚数は2〜100枚で可変ですが、Directorが全区間の元解像度tensorを保持するため、合計生成尺は
メモリ安全策として24fps・`17k+5`整列後で90秒までです。UI右上に実尺を表示し、超過時はQueue前に止めます。

既定の区間長`6.5秒`はH3の`17k+5`制約で158フレーム（約6.58秒）になります。
10枚・非ループなら9区間で、結合前は約59.25秒です。出力ノードは2区間目以降の重複境界を
1フレームずつ除くため、完成尺は約58.9秒になります。ループでは最後→最初の追加区間を生成し、
プレイヤーのループ時に同一フレームが二重にならないよう終端も処理します。
生成は9区間を順番に実行するため、所要時間は概ね「単一区間の時間×9＋2x処理」です。
5秒が142秒の環境ならQuality版の約1分は最低でも約21分に2x処理時間を加えた目安になります。

現行Directorはseedを全区間で1つだけ使用します。Storyboard内の区間Seedは将来互換用として
保存されますが、現在の生成を変えるのはDirectorノードの全体seedです。区間プロンプトと秒数は
個別に反映されます。

長尺版はDirectorを`segments`出力に固定し、各区間を16フレームずつ2x化して即一時MKVへ書き、解放してから
ffmpegで結合します。また、固定したDirectorへ全区間fp32 tensorの冗長結合を避けるローカルパッチを
適用しています。これにより2x全フレームを一括保持しませんが、モデルと元解像度の区間tensorは
必要です。映像は再圧縮せず結合し、区間音声はPCMで保持して最終段でAACへ1回だけ圧縮し、timestampも補正します。
約1分の0.4MP→2xではシステムRAM 64GBを下限、96GB以上を推奨します。
入力キーフレームはメモリ保持前に長辺1536pxへ縮小し、40MPを超える画像はQueue時に拒否します。

Directorの現行FL2V continuityには未修正のクラッシュ報告があるため、このプリセットは
`continuityEnabled=false`です。各隣接区間は同じ境界画像（A→B、B→C）を共有し、出力時に
重複フレームを除去します。continuityを手動でONにしないでください。

## 高速ダウンロード

現在のHugging Face HubはXetバックエンドを使用します。旧方式の
`HF_HUB_ENABLE_HF_TRANSFER=1`や`hf_transfer`は使用しません。

既定値：

```text
HF_XET_HIGH_PERFORMANCE=auto
HF_XET_CHUNK_CACHE_SIZE_BYTES=0
HF_DOWNLOAD_WORKERS=4
HF_HUB_DOWNLOAD_TIMEOUT=120
DOWNLOAD_RETRIES=3
MODEL_MANIFEST=/opt/minimax-h3/manifests/minimax_h3_i2v_upscale.json
MODEL_VERIFY=size
H3_LORA_SELECTION=all
```

新規ファイルだけを毎回取得する用途では、Xet chunk cacheを無効にした方が高速です。
4つの大型ファイルは同時に取得され、各ファイル内部でもXetがRange Downloadを並列化します。
`auto`では64GB級以上のRAMを検出した場合だけHigh Performanceを有効にし、
それ未満ではメモリ効率のよいAdaptive Concurrencyを使います。強制する場合は`1`または`0`を指定できます。

ベースモデルは公開されていますが、LoRAを`all`で取得する場合は`HF_TOKEN`と
`CIVITAI_TOKEN`が必要です。トークンをリポジトリやテンプレートJSONへ直接書かず、
RunPodのsecret環境変数として設定してください。約296MiBのLoRA 2本はベースモデルと並列取得します。

理論上のモデル取得時間は次の通りです。実測値はPodのネットワーク、NVMe、Hugging Face側の混雑で変わります。

| 実効回線速度 | 42.54GBの理論時間 |
|---:|---:|
| 1 Gbit/s | 約5分40秒 |
| 2 Gbit/s | 約2分50秒 |
| 5 Gbit/s | 約1分08秒 |
| 10 Gbit/s | 約34秒 |

既定の`MODEL_VERIFY=size`はmanifestへ固定した正確なファイルサイズを確認し、
42.54GBの全量再読込を省きます。Xet側の転送整合性検査も有効です。公式LFS SHA256まで
再検証したい場合は`MODEL_VERIFY=sha256`を指定できますが、ローカルNVMeの読み取り時間が追加されます。

## GPU別の初期値

起動時にVRAMを検出し、ログと
`user/default/minimax_h3_runtime_profile.json`へ推奨値を書き出します。

| VRAM | 初期プロファイル | Megapixels | 長さ |
|---:|---|---:|---:|
| 8–15GB | preview | 0.2 | 5秒 |
| 16–23GB | balanced | 0.4 | 5秒 |
| 24–31GB | high | 0.6 | 5秒 |
| 32GB以上 | native-768p | 0.98 | 5秒 |

ComfyUIのDynamic VRAMをそのまま使用します。`--novram`、`--disable-smart-memory`、
`--disable-pinned-memory`などはH3のDay-0実装で問題を起こし得るため、既定では付けません。
RunPodテンプレートでは`COMFYUI_ARGS=--vram-headroom 2`を設定し、デコード時のVRAM圧迫用に
2GBを余分に空けます。サンプリング側が遅くなるGPUでは、同じseedでこの値を外してA/B比較してください。

## INT8高速化とAttention

H3のpruned INT8 ConvRot重みはComfyUI標準の量子化形式です。別のINT8ローダーは追加せず、
ComfyKitchen `0.2.26`のCUDAバックエンドを使用します。従来のCUDA 12.8イメージではこの
バックエンドが`available: true, disabled: true`になり、INT8演算が低速なeager実装へ落ちていました。
このイメージはPyTorch `2.10.0+cu130`へ更新し、起動前診断で次を必須にします。

```text
acceleration.ready: true
comfy_kitchen_cuda.available: true
comfy_kitchen_cuda.disabled: false
```

`REQUIRE_COMFY_KITCHEN_CUDA=0`なら診断失敗を警告扱いにできますが、低速なため通常は使いません。
`--fast-disk`も既定では使用しません。これはモデル退避先をRAMよりディスク優先にするフラグで、
十分なシステムRAMがあるL40S Podではモデルの往復I/Oが増えて逆効果になり得ます。

[ComfyUI-INT8-Fast](https://github.com/BobJohnson24/ComfyUI-INT8-Fast)は、公式ComfyUIの
INT8対応後は作者自身が実質的にretireと案内しているため同梱しません。
[ComfyUI_sol-attn_Blackwell](https://github.com/KingGore/ComfyUI_sol-attn_Blackwell)は
現行版の高速化対象がRTX 5090などのSM120に限定され、L40SではSDPAフォールバックになるため
同梱しません。5090用はL40S版とは分離して、実機検証後に専用ワークフローとして追加します。

## ワークフロー

[`workflows/minimax_h3_i2v.json`](workflows/minimax_h3_i2v.json)は、
Comfy-Org公式I2Vテンプレートを固定コミットから無変更で収録しています。

- 入力画像1枚: 通常のI2V
- first frame + last frame: 開始／終了フレーム補間
- 24fps
- 映像とステレオ音声を同時生成
- 既定20 steps、`res_multistep`

[`workflows/minimax_h3_r2v.json`](workflows/minimax_h3_r2v.json)は、公式R2Vテンプレートへ
ComfyUI標準の`LoadVideo → GetVideoComponents`を接続したマルチモーダル版です。

- 2枚の参照画像を`<Picture 1>`、`<Picture 2>`として使用
- 参照動画の全フレームを`<Video 1>`として使用
- 参照動画の付随音声も同時にH3へ入力
- `res_multistep`、20 steps、`normal` scheduler
- `ref_image_size=match`を既定にして速度とVRAMを優先

`*_easycache.json`はComfyUI標準EasyCacheを使用するFast版です。既定値は
`reuse_threshold=0.20`、`start_percent=0.15`、`end_percent=0.95`、`verbose=true`です。
近似キャッシュにより一部のdiffusion stepを省略するため、最終品質ではQuality版と同一seedで比較してください。

`*_upscale.json`は、VAEデコード後の全フレームをComfyUI標準ノード
`UpscaleModelLoader → ImageUpscaleWithModel`に通し、`RealESRGAN_x2plus.pth`で2倍化する版です。
例えば768×1344は1536×2688になります。サンプリング後の処理なので、シャープさは改善しますが、
動きの破綻、人体の形、時間的なちらつきを修復するものではありません。フレーム数、fps、ステレオ音声は維持します。
また、2倍化の追加時間は必要なので、H3本体の生成速度そのものはEasyCacheで比較してください。

入力画像と出力のアスペクト比が一致しない場合、H3コアノードでは開始画像が引き伸ばされるため、
公式ワークフローの画像サイズ連動を維持してください。

## 2Kについて

公開されているローカルモデルはH3-Baseです。MiniMax公式品質の2Kは非公開の
`H3-Regenerate-2K`を使用するため、このリポジトリはローカル768p生成を正規ルートとします。
このリポジトリの2x版は、H3本体とは別のReal-ESRGANアップスケール工程として明示しています。

## ローカル検証

```bash
python -m unittest discover -s tests -v
python scripts/verify_workflow.py --workflow workflows/minimax_h3_i2v.json --manifest manifests/minimax_h3_i2v.json --mode i2v --comfyui-root /path/to/ComfyUI
python scripts/verify_workflow.py --workflow workflows/minimax_h3_i2v_easycache_upscale.json --manifest manifests/minimax_h3_i2v_upscale.json --mode i2v --expect-easycache --expect-upscale --comfyui-root /path/to/ComfyUI
python scripts/verify_workflow.py --workflow workflows/minimax_h3_i2v_hmmotion_lora_upscale.json --manifest manifests/minimax_h3_i2v_upscale.json --mode i2v --expect-upscale --expect-lora --comfyui-root /path/to/ComfyUI
python scripts/verify_workflow.py --workflow workflows/minimax_h3_i2v_selectable_lora_upscale.json --manifest manifests/minimax_h3_i2v_upscale.json --mode i2v --expect-upscale --expect-lora HMNSFW_AIO_V2.safetensors --expect-lora-strength 0.5 --comfyui-root /path/to/ComfyUI
python scripts/verify_workflow.py --workflow workflows/minimax_h3_story_quality_lora_2x.json --manifest manifests/minimax_h3_i2v_upscale.json --mode story --expect-upscale --expect-lora HMNSFW_AIO_V2.safetensors --expect-lora-strength 0.5 --comfyui-root /path/to/ComfyUI --custom-node-root /path/to/ComfyUI_MiniMaxH3_Director --custom-node-root custom_nodes/minimax_h3_ordered_storyboard
bash -n scripts/entrypoint.sh scripts/download_models.sh
python -m json.tool manifests/minimax_h3_all.json >/dev/null
python -m json.tool workflows/minimax_h3_i2v_easycache_upscale.json >/dev/null
```

実際のI2V生成テストにはNVIDIA GPUと約42.54GBのモデル取得が必要です。

## 調査根拠

- [ComfyUI公式MiniMax H3 I2V/R2Vガイド](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)
- [ComfyUI v0.30.0](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.30.0)
- [PyTorch 2.10 CUDA 13.0公式イメージ](https://hub.docker.com/layers/pytorch/pytorch/2.10.0-cuda13.0-cudnn9-runtime/images/sha256-1f57418aedd9a4d0d3a59646619e1d4f82cacc33817247cead4f749e1f452d4b)
- [ComfyKitchen 0.2.26のバックエンド要件](https://pypi.org/project/comfy-kitchen/0.2.26/)
- [Comfy-Org公式モデル配布](https://huggingface.co/Comfy-Org/MiniMax-H3)
- [Hugging Face Xetの性能設定](https://huggingface.co/docs/hub/en/xet/using-xet-storage)
- [RunPodの自動環境変数](https://docs.runpod.io/pods/templates/environment-variables)
- [RunPodのHTTPポート公開](https://docs.runpod.io/pods/configuration/expose-ports)
- [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
- [ComfyUI公式アップスケールガイド](https://docs.comfy.org/tutorials/basic/upscale)
- [Real-ESRGAN公式リポジトリ](https://github.com/xinntao/Real-ESRGAN)
- [Civitai V2モデル情報](https://civitai.com/api/v1/model-versions/3206518)
- [Civitai API認証](https://github.com/civitai/civitai-developer-docs/blob/main/site/guide/authentication.md)
- [AIMixer MiniMax H3 Director（固定元）](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/commit/a267324a9f88141ff4e4b0e8c1a6ed90b4e45db7)
- [Director FL2V continuity既知不具合 #26](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/26)
- [Director長尺結合OOM報告 #32](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/32)
