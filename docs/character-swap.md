# 06 · 人物置換（作者のRef2VA構成）

`H3_CHARACTER_SWAP=1` で `06_MiniMax_H3_Character_Swap.json` を追加します。04/05/07は維持します。
元動画と参照画像は利用権のある素材を使い、実在人物の置換には本人の同意を得てください。

今回の06は[LoRA作者の公開ワークフロー](https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA/blob/62407e0cc8089c363abd9ce4b0b27662abb237af/examples/H3%20Character%20Swap%20v1%20Ref2VA.json)に主要設定を合わせています。
公式Ref2VA INT8本体、`h3_character_swap_pro4500_1000.safetensors` 強度1.0、`res_multistep` / `simple` / 20 steps、Turboなし、ComfyUI標準 `MiniMaxH3ReferenceToVideo` で画像と動画の両方をエンコードします。
RefMod画像の独自注入・保存ノード、人物検出マスク、承認コード、元映像コピー判定は06から外しました。完成フレームへの既存の自動モザイクは独立して残します。

作者はこのLoRAを**試験的**と説明し、動作タイミング・表情・カットでの不安定さも明記しています。[モデルカード](https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA)の成功例と同じ設定でも、置換を保証しません。生成完了と置換成功は別です。

## RunPod

新しいイメージでPodを起動し、Workflowsから06を**新しく開き直してください**。古いキャンバスは自動更新されず、削除したマスクノードが残ったままになります。
CUDA 13対応ホストなら `ghcr.io/grawthings-beep/minimax-h3-i2v:character-swap-fast-cu130`、それ以外のCommunity Cloudなら `character-swap-community-cu128` を使用。固定digest/tagを優先してください。

主な環境変数:

```dotenv
HF_TOKEN="{{ RUNPOD_SECRET_HF_TOKEN }}"
CIVITAI_API_TOKEN="{{ RUNPOD_SECRET_CIVITAI_TOKEN }}"
ACCEPT_MINIMAX_H3_LICENSE=1
MINIMAX_H3_LICENSEE_IN_APPLICABLE_TERRITORY=1
MINIMAX_H3_SEPARATE_LICENSE=0
H3_PROFILE=r2v
H3_R2V_MODEL=dasiwa-v2
H3_R2V_VAE=x2-detail
H3_CHARACTER_SWAP=1
H3_SWAP_MODEL=official
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

ライセンス項目は同意・適格性を自分で確認した場合のみ1にしてください。CUDA 12.8版は `REQUIRE_COMFY_KITCHEN_CUDA=0`。`H3_SWAP_MODEL`を未指定にしても今は`official`が既定ですが、旧設定の`shared`が残っているとDaSiWaを使うため、明示しておくのが確実です。06用の公式チェックポイントは04/07のDaSiWaとは別に起動時ダウンロードされ、起動時間・ディスク使用量が増えます。人物置換LoRAも起動時に取得・サイズ/SHA256検証します。トークンや重みをイメージへ埋め込みません。

## 使い方

1. 元動画をアップロード。最初は人物1人、カットなし、4〜5秒程度のショット。
2. `Source clip`で開始時刻・尺・解像度を設定。既定は0秒、5.2秒、0.4MP。30/60fps素材もPTSに基づいて24fpsへ時間変換し、H3の`17n+5`フレームへ末尾を切り詰めます。5.2秒以上の動画なら124frames（5.167秒）が目安。静止フレームの水増しはしません。
3. `Character images`へ置換先の同一キャラを入れる。まず顔・髪・衣装が明瞭な1枚。複数枚なら同一人物の別角度のみ、最大8枚。`<Picture 1>`から入力順です。
4. `Replacement prompt`で元動画中の**どの人物**かを服・位置で特定。`<Video 1>`が元動画です。LoRAは既定ON・強度1.0。他の速度LoRAは重ねずに確認してください。
5. `Audio`は`source`（既定）/`generated`/`mute`を選択。元音声はモデル条件ではなく完成動画への後付けなので、映像が変わると口形や効果音がずれることがあります。
6. 出力後、元人物の顔・髪・衣装が残っていないか、フレームアウトと再登場を含め目視確認する。チェックを自動でパスしたという意味ではありません。

基本プロンプト例（服・位置を実素材に合わせて置換）:

```text
Swap the woman in the red jacket on the left in <Video 1> with the character in <Picture 1>. Use the replacement character's face, hair, outfit and art style. Match her position, scale, pose and movement. Preserve the camera, background, lighting and other people.

Soundscape: natural sounds matching the visible actions. No added dialogue.
Non-diegetic music: N/A.
```

全身の向きや細部の表情まで厳密に転写する機能ではありません。長い禁止文・強い表情指示を足すと逆に置換が弱まる場合がある、と作者は報告しています。元動画を短い単一ショットへ分け、対象人物を明確に指すことを優先します。

## モデル・仕上げ・検証範囲

06だけ公式Ref2VAを使用しても04/07の`H3_R2V_MODEL=dasiwa-v2`は変わりません。`H3_SWAP_MODEL=shared`なら従来のDaSiWaを06でも使えますが、作者が学習した公式本体ではなく、置換成績も同一とみなせません。
参照画像と元動画のencodeには公式INT8映像VAEを使います。`H3_R2V_VAE=x2-detail`は**生成後のdecode/2x**で、人物置換の強さを保証する設定ではありません。CPUモザイクは完成フレームにのみかかります。音声decodeと元音声選択も従来どおりです。

ローカル/CIの自動テストはグラフ配線、必要重み、ノード登録、CPU動画・音声読み込み、標準Ref2VAへの参照順序、実LoRAのH3キー/形状適合を検証します。実GPUでの置換品質やVRAMピークはこの検証だけでは証明できません。
