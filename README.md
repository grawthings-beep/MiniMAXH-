# MiniMax H3 I2V for ephemeral RunPod Pods

MiniMax H3の画像→動画生成だけを、永続ストレージなしのRunPod Podで起動する構成です。
公式ComfyUI I2Vワークフロー、ネイティブステレオ音声、開始／終了フレーム制御に対応します。

モデルをDockerイメージへ埋め込まず、Pod起動時にHugging Face Xetから約42.5GBを高速取得します。
Podを破棄するとモデルと出力は消えます。

## 構成

- ComfyUI `v0.30.0`をコミットまで固定
- MiniMax H3 `FL2VA`のpruned INT8モデルのみ
- Qwen3-VL-32B NVFP4/AWQテキストエンコーダー
- Video VAEと32kHzステレオAudio VAE
- Hugging Face `hf_xet`のRAM連動High Performance／Adaptive切替
- 4ファイルの並列取得、3回の再試行、途中再開
- Xet失敗時のaria2並列Range Downloadフォールバック
- 公式LFS SHA256による完全検証
- GPU、VRAM、RAM、空き容量の起動前診断
- GPU VRAMに応じた解像度プロファイルの提案
- 公式I2Vワークフローの全モデルと全ノードをビルド時に照合

モデルの正確なリビジョン、サイズ、SHA256は
[`manifests/minimax_h3_i2v.json`](manifests/minimax_h3_i2v.json)に固定しています。

## 必要容量

| ファイル | 容量 |
|---|---:|
| FL2VA pruned INT8 | 20.97 GB |
| Qwen3-VL-32B NVFP4/AWQ | 15.69 GB |
| Video VAE | 5.21 GB |
| Audio VAE | 0.61 GB |
| 合計 | 42.47 GB（39.55 GiB） |

RunPodのContainer Diskは、再構築バッファと出力領域を含めて90GBを推奨します。
Volume DiskやNetwork Volumeは使用しません。

## 初回セットアップ

### 1. ライセンスを確認

モデル利用前に
[MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
を確認してください。ライセンスは米国、EU、英国、韓国をApplicable Territoryから除外しています。

確認後、RunPodテンプレートの環境変数を次のように変更します。

```text
ACCEPT_MINIMAX_H3_LICENSE=1
```

`1`でなければモデルはダウンロードされません。

Podは物理的な計算場所が許可地域にある必要があります。RunPodが自動設定する
`RUNPOD_DC_ID`を起動時に検査し、`US-*`、`EU-*`、英国・韓国の識別子では停止します。
現在のRunPod候補では、たとえば`AP-JP-1`、`CA-MTL-*`、`OC-AU-1`、
`EUR-IS-*`、`EUR-NO-1`が除外地域外です。実際に表示される所在地と最新のライセンスを必ず確認してください。

RunPod以外で`RUNPOD_DC_ID`が存在しない場合は、計算場所を確認したうえで
`MINIMAX_H3_DEPLOYMENT_ALLOWED=1`を設定します。除外地域についてMiniMaxから別途許諾を
得ている場合だけ`MINIMAX_H3_SEPARATE_LICENSE=1`を使用できます。

このガードで確認できるのは計算場所だけです。利用者、配信先、Outputの利用も許可地域内に
限定されます。第三者へ生成サービスを提供する場合の利用規約・安全対策・AI生成表示、
商用利用時の表示や一定規模以上の事前許諾など、ライセンス本文の追加条件も適用されます。

### 2. コンテナをビルド

```bash
docker build --platform linux/amd64 -t ghcr.io/OWNER/minimax-h3-i2v:0.1.0 .
docker push ghcr.io/OWNER/minimax-h3-i2v:0.1.0
```

タグ`v*`をpushするか、GitHub Actionsの`Build container`を手動実行してGHCRへ公開することもできます。
公開先に合わせて`runpod-template.example.json`の`imageName`を変更してください。

### 3. RunPodテンプレート

[`runpod-template.example.json`](runpod-template.example.json)を基準にCustom Templateを作成します。

- Container Disk: 90GB
- Volume Disk: 0GB
- Expose HTTP Port: `8188`
- Docker image: 上で公開した固定タグ
- `ACCEPT_MINIMAX_H3_LICENSE=1`
- Data center: MiniMax H3ライセンスの除外地域外

GPUは起動のたびに変更できます。CUDA 12.8対応イメージなので、対応ドライバーがあるNVIDIA GPUを選択してください。

### 4. 起動

Podログには次の順で状態が表示されます。

1. GPU・RAM・空き容量診断
2. Xet並列ダウンロード
3. SHA256検証
4. 実効ダウンロード速度
5. ComfyUI `8188`起動

RunPodの`Connect to HTTP Service [Port 8188]`からComfyUIを開き、
`MiniMax_H3_I2V`ワークフローをロードします。

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
4つの大型ファイルは同時に取得され、各ファイル内部でもXetがRange Downloadを並列化します。
`auto`では64GB級以上のRAMを検出した場合だけHigh Performanceを有効にし、
それ未満ではメモリ効率のよいAdaptive Concurrencyを使います。強制する場合は`1`または`0`を指定できます。

公開モデルなのでトークンなしでも取得できますが、レート制限を避けるため、Hugging Faceのread tokenを
RunPodのsecret環境変数`HF_TOKEN`として設定することを推奨します。トークンをリポジトリや
テンプレートJSONへ直接書かないでください。

理論上のモデル取得時間は次の通りです。実測値はPodのネットワーク、NVMe、Hugging Face側の混雑で変わります。

| 実効回線速度 | 42.47GBの理論時間 |
|---:|---:|
| 1 Gbit/s | 約5分40秒 |
| 2 Gbit/s | 約2分50秒 |
| 5 Gbit/s | 約1分08秒 |
| 10 Gbit/s | 約34秒 |

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

## ワークフロー

[`workflows/minimax_h3_i2v.json`](workflows/minimax_h3_i2v.json)は、
Comfy-Org公式I2Vテンプレートを固定コミットから無変更で収録しています。

- 入力画像1枚: 通常のI2V
- first frame + last frame: 開始／終了フレーム補間
- 24fps
- 映像とステレオ音声を同時生成
- 既定20 steps、`res_multistep`

入力画像と出力のアスペクト比が一致しない場合、H3コアノードでは開始画像が引き伸ばされるため、
公式ワークフローの画像サイズ連動を維持してください。

## 2Kについて

公開されているローカルモデルはH3-Baseです。MiniMax公式品質の2Kは非公開の
`H3-Regenerate-2K`を使用するため、このリポジトリはローカル768p生成を正規ルートとします。
将来ローカル2Kを追加する場合は、H3本体とは別のアップスケール工程として明示します。

## ローカル検証

```bash
python -m unittest discover -s tests -v
python scripts/verify_workflow.py --workflow workflows/minimax_h3_i2v.json --manifest manifests/minimax_h3_i2v.json --comfyui-root /path/to/ComfyUI
bash -n scripts/entrypoint.sh scripts/download_models.sh
python -m json.tool manifests/minimax_h3_i2v.json >/dev/null
python -m json.tool workflows/minimax_h3_i2v.json >/dev/null
```

実際の生成テストにはNVIDIA GPUと約42.5GBのモデル取得が必要です。

## 調査根拠

- [ComfyUI公式MiniMax H3 I2Vガイド](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)
- [ComfyUI v0.30.0](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.30.0)
- [Comfy-Org公式モデル配布](https://huggingface.co/Comfy-Org/MiniMax-H3)
- [Hugging Face Xetの性能設定](https://huggingface.co/docs/hub/en/xet/using-xet-storage)
- [RunPodの自動環境変数](https://docs.runpod.io/pods/templates/environment-variables)
- [RunPodのHTTPポート公開](https://docs.runpod.io/pods/configuration/expose-ports)
- [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
