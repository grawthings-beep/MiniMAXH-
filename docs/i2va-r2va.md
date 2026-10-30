# 画像の使い方を選ぶ: 09 I2VA / 04 REF2VA

ユーザー提供の `DasiwaMinimaxH3WorkflowsT2VA_cMMH3V33.json` を検討した結果、
必要な **I2VAとREF2VA** を現行の小さなRunPod構成へ統合しました。
DaSiWaノードpackとAIMixerの `MiniMaxH3Director` は同じノードIDを登録するため、
作者の多機能JSONをそのまま同居させていません。V33の継続生成・RTX・v4 Turbo等は対象外です。
この追加は新規09だけで、既存04のR2VA配線・既存07のT2VA配線は変更しません。

|  | 09 I2VA | 04 REF2VA |
|---|---|---|
| 画像の意味 | 完成動画の0秒フレーム | 顔・髪・衣装・画風の参照 |
| 冒頭構図 | 入力画像と同じ | プロンプトで別の構図にできる |
| 画像枚数 | 最初の1枚 | 同一キャラ1〜8枚 |
| モデル | DaSiWa Hybrid v2 INT8共通 | 選んだR2VAモデル |
| 入力欄 | 画像、全文プロンプト、尺、解像度 | 参照画像、全文プロンプト、尺、解像度 |
| 仕上げ | INT8 VAE→RealESRGAN 2x→CPUモザイク | `H3_R2V_VAE`で選択した従来04の仕上げ |

## RunPod設定

既存のCUDA 13用R2VAイメージに、ビルド・公開された新しいタグを指定してから、
次の環境変数を設定します。古いイメージへ環境変数だけ追加しても09は現れません。
HF/CivitAIはRunPod Secretから渡してください。ライセンスの確認値は自身で確認して設定してください。

```dotenv
HF_TOKEN="{{ RUNPOD_SECRET_HF_TOKEN }}"
CIVITAI_API_TOKEN="{{ RUNPOD_SECRET_CIVITAI_TOKEN }}"
ACCEPT_MINIMAX_H3_LICENSE=1
MINIMAX_H3_LICENSEE_IN_APPLICABLE_TERRITORY=1
MINIMAX_H3_SEPARATE_LICENSE=0
H3_PROFILE=r2v
H3_R2V_MODEL=dasiwa-v2
H3_R2V_VAE=x2-detail
H3_I2VA=1
H3_R2V_LORA_SELECTION=all
H3_FAST_VAE=1
AUTO_MOSAIC_REQUIRED=1
AUTO_MOSAIC_MANIFEST=/opt/minimax-h3/manifests/auto_mosaic.json
HF_XET_HIGH_PERFORMANCE=auto
HF_XET_CHUNK_CACHE_SIZE_BYTES=0
HF_DOWNLOAD_WORKERS=4
HF_HUB_DOWNLOAD_TIMEOUT=120
DOWNLOAD_RETRIES=3
MODEL_VERIFY=size
REQUIRE_COMFY_KITCHEN_CUDA=1
COMFYUI_ARGS="--lowvram --vram-headroom 2 --fast fp16_accumulation"
TINI_SUBREAPER=1
```

CUDA 12.8の互換イメージでは `REQUIRE_COMFY_KITCHEN_CUDA=0`。
既存のLoRAを使わないなら `H3_R2V_LORA_SELECTION=none` で起動取得を減らせます。
09には04のLoRAを自動適用しません。`H3_I2VA=0` で配布09だけを非表示にします。
Pod内で別名保存したワークフローやモデルは削除しません。

## 操作

1. 画像をそのまま動画の最初の画にしたい場合は、Workflowsから
   `09_MiniMax_H3_I2VA_First_Frame_2x` を開く。左上で画像をアップロード。
   FULL PROMPTでその後の動きと音を指定する。最初は5秒・0.4MP・seed固定。
2. 最初は人物を見せず、途中でフレームインさせたい場合は、従来の
   `04_MiniMax_H3_Character_R2V_2x` を開く。画像は外見参照で、0秒の画を強制しない。
3. 09の入力画像はモザイク処理しません。生成・デコード後の完成フレームだけに、
   RealESRGAN 2xの後でCPU輪郭モザイクを一度適用します。

09のnative MiniMax H3ノードは最初の画像をキャンバス寸法へ**ストレッチ**してencodeします。
縦横比が大きく違う画像では人物が変形し得るため、生成比率に近い画像を使ってください。
09で顔や衣装が変わる場合、冒頭フレーム保持と動画全体の一貫性は別問題です。
04のREF2VAも完全な同一性を保証しません。

## ダウンロードと検証範囲

09は04のDaSiWa v2本体、Qwen、音声VAEを共有。INT8映像VAEとRealESRGAN x2は
04が `x2-detail` の場合だけ追加取得（約2.88GB）。既存の並列・再開ダウンロードとモデル検証に統合します。
起動時にモデルmanifestとノード・配線を検証し、不足ならComfyUIを起動しません。
ローカル自動テストとDocker CPU smokeはGPU実生成を代替しません。RunPodではまず5秒・0.4MPを
連続2回生成し、VRAM、冒頭画像、人物維持、音声、2x/モザイクの結果を確認してください。
