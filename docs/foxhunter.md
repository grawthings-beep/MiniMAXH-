# FoxFurEssence H3 Seed Hunter v3

This is an isolated, first-priority RunPod image for the user-supplied workflow.
It does **not** overwrite the existing stable 04/09 workflows or their Docker
tags. Open `00_FoxFur_Seed_Hunter_v3` from ComfyUI Workflows after the pod starts.

## What is included

- The exact selected [CivitAI Eros Max beta5 file 3185144](https://civitai.com/api/download/models/3294059?fileId=3185144):
  `10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized.safetensors`.
  It is fetched from the creator's pinned [Hugging Face copy](https://huggingface.co/TenStrip/10Eros-Max/blob/e4753986789e11aa50dd3eea15a774d4ef773a5a/10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized.safetensors),
  whose SHA256 matches the selected CivitAI file. Its Turbo delta is baked in,
  so the old Silver Turbo LoRA is not stacked. This W4A8 edition is the exact
  file requested, although its creator recommends the separate INT8 edition
  as the beta5 default; treat output quality as experimental until rendered.
- Qwen3-VL NVFP4 encoder, INT8 video VAE, FP32 audio VAE, and
  `MysticXXX_MMH3-V5-ref2va.safetensors` from the requested private
  `uwgm/illustrious` repository. Mystic V5 is enabled at 0.5 strength in the
  visible Pixaroma LoRA loader. The first pass is Euler/simple at 8 steps,
  with video/audio sigma shifts 12/7 per the Eros beta5 creator's guidance.
- LBH 3D latent upscaler v1 BF16, TAE preview weights, and RIFE v4.26 weights.
  LBH moved the same v1 weights to a versioned folder, so the saved widget is
  updated to the **current filename**. Its optional upscale/RIFE nodes remain
  bypassed exactly as they were in the supplied JSON; use its control panel to
  enable them deliberately.
- Pinned AnimeCadence, Fantastic Media Loader, Pixaroma, Prompt IDE, RefMods,
  Use Everywhere, VideoHelperSuite, rgthree, KJNodes, and ComfyUI-VFI. The two
  Easy-Use IDs used by this graph are supplied by a small local compatibility
  package to avoid installing that pack's unrelated inference dependencies.

The default author-character example was replaced by a neutral <Picture 1>
prompt. The first-run duration was reduced from 14 to 5 seconds to avoid an
unintentional long, multi-pass run. The author's local output preview paths
were removed. No model weights or secrets are embedded in the image. All eight
assets total **33,962,725,035 bytes (~31.6 GiB)** and are downloaded at each
ephemeral Pod start with resumable parallel transfers and pinned size/SHA256
validation. Plan for a 150 GB container disk, no persistent volume.

## First run

1. Select the Community CUDA 12.8 image unless the host driver supports CUDA
   13 / r580+. Set `HF_TOKEN` from a RunPod Secret and explicitly confirm the
   MiniMax license/territory flags. `HF_TOKEN` also needs read access to
   `uwgm/illustrious`. The checkpoint is byte-identical to the requested
   CivitAI file but downloaded from its author's Hugging Face repo, so no
   `CIVITAI_API_TOKEN` is needed for this profile.
2. Upload a new image in the Fantastic Media Loader. Replace the neutral
   Prompt IDE text with the desired character/scene. The supplied `lunalovegood.wav`
   audio-upload node is muted; upload your own audio before enabling that branch.
3. Keep the initial 5-second setting and optional latent upscale/RIFE branches
   off for a first render. The saved workflow makes three parallel seed
   candidates, then offers an optional second-pass upscale;
   24/32 GB VRAM is **not** guaranteed. Raise duration or enable optional
   passes one at a time after measuring VRAM and output quality.

The build checks that ComfyUI can register all executable node IDs on CPU and
that the pinned ComfyUI/ComfyKitchen stack recognizes asymmetric W4A8.
That check is **not** a full GPU-generation test; only a live Pod render can
establish image quality and practical VRAM limits.

RunPod examples: `runpod-template.foxhunter-community-cu128.example.json` and
`runpod-template.foxhunter-fast-cu130.example.json`.

## 画像×プロンプトを一括登録する（サンプル2専用）

画像1枚と全文プロンプト1本で、**サンプル2 → latent upscale → 第2パス → 音声付きMP4**を1件だけ実行します。サンプル1・3、低解像度プレビュー保存、RIFEは実行しません。映像と音は第2パスから取得します。元のワークフローは変更しません。

最初に `00_FoxFur_Seed_Hunter_v3` を開き、通常の画像参照モードで **Preview #2 = ON、Final Upscale Pass = ON、Single Pass = OFF** にしてください。解像度・尺・LoRA・シードは普段動いている設定を使います。画像と全文プロンプトはジョブごとに差し替えるため、元の画面で画像を入れ替える必要はありません。専用の動画継続・音声参照モードとの併用は対象外です。

- **現在稼働している従来イメージ（Windows）:** 上の設定をComfyUIで「Export (API)」して保存（通常のワークフローJSONでは不可）。PCで `FoxHunter_Batch.cmd` を開き、PodのComfyUI接続URLとAPI JSONを指定。次に **「② 一覧ファイルを開く」** で下記形式のUTF-8テキストを選び、「③ 未完了の行を順番に生成」を押します。画像と一覧が同じフォルダなら画像を選ぶ作業も不要です。別の場所なら画像フォルダを1回指定します。ファイルの代わりに「コピーした一覧を一括登録」も使えます。従来の個別編集、画像と同名の `.txt` 自動入力にも対応。PCとランチャーは完了まで起動したままにしてください。Python 3.10以降＋Tkが必要で、pip追加・環境変数変更は不要です。認証なしで直接接続できる個人用Pod向けです。
- **iPhone / Safari（Podだけで一括生成）:** 上記のワークフローを開き、画面右下の **「画像を一括登録」** をタップ。「ファイル」で `ラピ.png` と `ラピ.txt` のような画像＋同名UTF-8テキストを、必要な組数まとめて**同時に**選びます（PNG/JPEG/WebP、最大100組。写真アプリではなく「ファイル」を使用）。表示された対応を確認し「登録した順に生成」を1回タップ。テキストはSafari内で読み取り、生成先のPodにだけ送ります。登録完了まではSafariを前面に保ち、**「Pod受付済み」** が出たら画面ロック・タブを閉じても処理は続きます。PC・手動貼り付け・API JSONエクスポートは不要です。デスクトップでもFoxHunterメニューか `H3 Prompt IDE` ノードの右クリックから同じ画面を開けます。進捗は「直近の一括生成の進捗」で確認。Pod自体を停止・再作成すると待機リストは復元されません。

**Windows用の一覧例（prompts.txt）：**

```text
[ラピ.png]
The character turns toward the camera as the wind lifts her hair.

[scene 02.webp]
The character watches a paper boat drift downstream.
Soundscape: running water and distant birds.
```

見出しは実際のファイル名（拡張子込み）。本文は複数行可。空のリストから取り込むと一覧順、すでに画像を登録している場合は画像の登録順で、名前で照合します。画像名の重複や未入力がある場合は反映せず、生成も開始しません。

出力はPodの `ComfyUI/output/foxhunter_batch/<実行ID>/` 配下で、ファイル名に行番号と画像名が入ります。前の行のMP4保存が履歴で確認できるまで次の行を投入しません。キューが空なら行間にComfyUIへキャッシュ・モデルの解放も要求します。1本自体が収まらない設定はOOMになるため、まず実績のある単体設定で試してください。OOM・中断・保存失敗・通信エラー時はそこで停止し、自動で再送しません。通信切断時は二重生成を避けるためPodの履歴を確認してから未完了分を再登録してください。単独利用のPod向けです。
