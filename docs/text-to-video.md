# 07 · 画像なしT2VAで演出の元動画を作る

文章だけで映像＋音声を生成し、良い元動画を06でキャラ置換するための追加ワークフロー。
参照画像・動画・時刻ごとの構図画像は不要。通常04、比較05、人物置換06は変更しません。

## 起動設定

CUDA 13: `ghcr.io/grawthings-beep/minimax-h3-i2v:t2va-master-fast-cu130`

ビルド・公開成功後に使えるタグです。古いイメージへ環境変数だけ追加しても07は増えません。
CUDA 12.8は `t2va-master-community-cu128`、その場合 `REQUIRE_COMFY_KITCHEN_CUDA=0`。
CUDA 13対応はGPU名だけでなくホストドライバにも依存します。

環境変数全文。ライセンスの同意・適格性を自身で確認済みの場合のみ該当項目を1にしてください。
比較05を外し、04・06・07の3本を表示する構成です。比較が必要なら `H3_R2V_COMPARE=1`。

```dotenv
HF_TOKEN="{{ RUNPOD_SECRET_HF_TOKEN }}"
CIVITAI_API_TOKEN="{{ RUNPOD_SECRET_CIVITAI_TOKEN }}"
ACCEPT_MINIMAX_H3_LICENSE=1
MINIMAX_H3_LICENSEE_IN_APPLICABLE_TERRITORY=1
MINIMAX_H3_SEPARATE_LICENSE=0
H3_PROFILE=r2v
H3_R2V_MODEL=dasiwa-v2
H3_R2V_VAE=x2-detail
H3_R2V_COMPARE=0
H3_CHARACTER_SWAP=1
H3_T2VA=1
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

既存の任意LoRA取得指定は維持しますが07には適用しません。04用LoRAが不要なら
`H3_R2V_LORA_SELECTION=none` で起動取得を省けます。06の人物置換LoRAは独立して取得されます。
永続volumeなし、disk120GB、HTTP8188、start commandなしは従来と同じです。

## 操作

1. Workflowsから `07_MiniMax_H3_Text_to_Video` を新しく開く。画像アップロード欄はありません。
2. 左のFULL PROMPTへ、映像・音・BGMを英語の全文で入力。4分割やモード切替は不要。
   `<Picture 1>` / `<Video 1>` / `<Audio 1>`は参照がないためエラーで止めます。
3. 初期は5秒指定、9:16、約0.4MP、25steps。DaSiWa v2本体 / simple / res_multistep / shift11,4。
   H3の尺は17n+5フレーム単位。5秒指定は124frames=5.167秒。最大15秒指定は362frames。
   元動画は0.8MP以下。07/06は同じ1MP=1,000,000画素の予算を使用し、初期9:16は448×832です。
   標準ResolutionSelectorの1024²換算とは異なり、06へ渡す際の意図しない再縮小を避けます。
   長尺や大解像度でのOOM回避は保証しません。
4. 実行。右端に元解像度MP4を保存。2xは行わず、音声生成とCPU輪郭モザイクは維持。
5. 良い結果だけダウンロードし、06の元動画欄へアップロード。初期124framesの動画なら
   区間は開始0秒・長さ5.2秒にする。5.0秒に縮めると107framesへ切り詰められます。
   長い元動画を全部使うなら06の区間を実際の尺に合わせて延長。両方の生成解像度も合わせる。
6. 06へ置換キャラ画像・置換プロンプトを設定して実行。仕上げは選択済みVAEで2x。
   音声sourceなら07の生成音を使いますが、置換で動きが変われば音の同期もずれます。

初期プロンプトは「一度右へ退出→空の背景→近くへ素早く再登場」。
人物・背景はオリジナルの簡単な例で、参考CMの映像・音声を使用しません。
同じseedは比較用。別候補はseed変更、量産はRandomNoiseのcontrol after generateをrandomizeにして
複数回キューへ追加してください。1回の巨大なフレームバッチで候補を並列生成する方式ではありません。

## モデルと取得

- この07は非TurboのDaSiWa Hybrid v2を共有します。第二の大型チェックポイントは取りません。
- 本repoの`official`はRef2VA専用です。公式FL2VAと混同しないよう、07では拒否します。
  `dasiwa-turbo-v2`も今回の検証対象外として起動前に拒否します。
- 07はINT8 Video VAEで元解像度保存。06/比較05で既に取得していれば追加取得なし。
  X2だけで使っていた場合はINT8 VAE 2,811,065,184 bytesを追加。manifestはパスで重複排除。
- HF/Civitaiの既存並列・再開・サイズ/SHA検証を使います。追加の有料生成APIはありません。
  RunPod料金は発生し、T2VAと人物置換の2回分を計算します。
- `H3_T2VA=0` で配布07だけ非表示に戻せます。個人が別名保存したworkflowやモデルは削除しません。

## 確認した範囲・限界

公式H3は[画像なしT2VA](https://www.minimax.io/news/minimax-h3-open-source)に対応。
実装は固定ComfyUI v0.37.0の `MiniMaxH3ImageToVideo` を両keyframeなしで呼びます。
RefModや置換LoRA、元絵が隠れて入力される経路はありません。文字数・token数・サイズを検査します。
生成後は既存H3退避→INT8タイルVAE→CPUモザイク→24fps MP4。GPU tensorを独自cacheに保持しません。

自動テストは全リンク両端、モデル依存、既存04/05/06不変、配置の非重複、各起動設定、
native tokenizer/AV latent、実MP4の保存と06への124frame引き渡しを確認します。
DockerはCUDA12.8/13でCPU契約テスト・実entrypoint smokeを行います。
**モデル重みを用いたGPU生成、DaSiWaのT2VA映像品質、VRAM使用量、速度は未実測です。**
完全退出後の再登場で別人化する、空白の長さが違う、置換後の動きが変わる可能性があります。
タイミングや同一性を保証せず、まず短い元動画1本とその置換結果を確認してください。
