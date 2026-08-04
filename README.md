# MiniMax H3 I2V + R2V for ephemeral RunPod Pods

MiniMax H3の画像→動画とマルチモーダル参照→動画を、永続ストレージなしのRunPod Podで起動する構成です。
I2V、開始／終了フレーム、複数画像、参照動画の動き・カメラ・付随音声、ネイティブステレオ音声に対応します。

モデルをDockerイメージへ埋め込まず、Pod起動時にHugging Face Xetから約63.4GBを高速取得します。
Podを破棄するとモデルと出力は消えます。

## 構成

- ComfyUI `v0.30.0`をコミットまで固定
- PyTorch `2.10.0+cu130`をイメージdigestまで固定
- MiniMax H3 `FL2VA`と`REF2VA`のpruned INT8モデル
- Qwen3-VL-32B NVFP4/AWQテキストエンコーダー
- Video VAEと32kHzステレオAudio VAE
- Hugging Face `hf_xet`のRAM連動High Performance／Adaptive切替
- 5ファイルの並列取得、3回の再試行、途中再開
- Xet失敗時のaria2並列Range Downloadフォールバック
- 公式LFS SHA256による完全検証
- GPU、ドライバー、VRAM、RAM、空き容量、ComfyKitchen CUDAの起動前診断
- GPU VRAMに応じた解像度プロファイルの提案
- I2V/R2VのQuality版とEasyCache Fast版を収録
- 全モデル、全ノード、EasyCache配線、動画・音声参照配線をビルド時に照合

モデルの正確なリビジョン、サイズ、SHA256は
[`manifests/minimax_h3_all.json`](manifests/minimax_h3_all.json)に固定しています。
モード別の厳密な検証用manifestも同じディレクトリに収録しています。

## 必要容量

| ファイル | 容量 |
|---|---:|
| FL2VA pruned INT8 | 20.97 GB |
| REF2VA pruned INT8 | 20.97 GB |
| Qwen3-VL-32B NVFP4/AWQ | 15.69 GB |
| Video VAE | 5.21 GB |
| Audio VAE | 0.61 GB |
| 合計 | 63.44 GB（59.08 GiB） |

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
docker build --platform linux/amd64 -t ghcr.io/OWNER/minimax-h3-i2v:0.3.2 .
docker push ghcr.io/OWNER/minimax-h3-i2v:0.3.2
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
63.4GBのモデル取得前に起動を停止します。

### 4. 起動

Podログには次の順で状態が表示されます。

1. GPU・ドライバー・RAM・空き容量・ComfyKitchen CUDA診断
2. Xet並列ダウンロード
3. SHA256検証
4. 実効ダウンロード速度
5. ComfyUI `8188`起動

RunPodの`Connect to HTTP Service [Port 8188]`からComfyUIを開き、
次の4ワークフローから選びます。

- `MiniMax_H3_I2V_Quality`
- `MiniMax_H3_I2V_Fast_EasyCache`
- `MiniMax_H3_R2V_Quality`
- `MiniMax_H3_R2V_Fast_EasyCache`

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
MODEL_VERIFY=sha256
```

新規ファイルだけを毎回取得する用途では、Xet chunk cacheを無効にした方が高速です。
5つの大型ファイルは同時に取得され、各ファイル内部でもXetがRange Downloadを並列化します。
`auto`では64GB級以上のRAMを検出した場合だけHigh Performanceを有効にし、
それ未満ではメモリ効率のよいAdaptive Concurrencyを使います。強制する場合は`1`または`0`を指定できます。

公開モデルなのでトークンなしでも取得できますが、レート制限を避けるため、Hugging Faceのread tokenを
RunPodのsecret環境変数`HF_TOKEN`として設定することを推奨します。トークンをリポジトリや
テンプレートJSONへ直接書かないでください。

理論上のモデル取得時間は次の通りです。実測値はPodのネットワーク、NVMe、Hugging Face側の混雑で変わります。

| 実効回線速度 | 63.44GBの理論時間 |
|---:|---:|
| 1 Gbit/s | 約8分28秒 |
| 2 Gbit/s | 約4分14秒 |
| 5 Gbit/s | 約1分42秒 |
| 10 Gbit/s | 約51秒 |

完全SHA256検証には追加でローカルNVMeの読み取り時間がかかります。速度優先なら
`MODEL_VERIFY=size`を指定できますが、既定は安全側の`sha256`です。

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

入力画像と出力のアスペクト比が一致しない場合、H3コアノードでは開始画像が引き伸ばされるため、
公式ワークフローの画像サイズ連動を維持してください。

## 2Kについて

公開されているローカルモデルはH3-Baseです。MiniMax公式品質の2Kは非公開の
`H3-Regenerate-2K`を使用するため、このリポジトリはローカル768p生成を正規ルートとします。
将来ローカル2Kを追加する場合は、H3本体とは別のアップスケール工程として明示します。

## ローカル検証

```bash
python -m unittest discover -s tests -v
python scripts/verify_workflow.py --workflow workflows/minimax_h3_i2v.json --manifest manifests/minimax_h3_i2v.json --mode i2v --comfyui-root /path/to/ComfyUI
python scripts/verify_workflow.py --workflow workflows/minimax_h3_r2v_easycache.json --manifest manifests/minimax_h3_r2v.json --mode r2v --expect-easycache --require-video-reference --comfyui-root /path/to/ComfyUI
bash -n scripts/entrypoint.sh scripts/download_models.sh
python -m json.tool manifests/minimax_h3_all.json >/dev/null
python -m json.tool workflows/minimax_h3_r2v_easycache.json >/dev/null
```

実際の生成テストにはNVIDIA GPUと約63.4GBのモデル取得が必要です。

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
