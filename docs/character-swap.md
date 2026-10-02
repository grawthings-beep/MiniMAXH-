# 06 · 元動画の人物置換

元動画の人物を、参照画像のキャラへ置き換えるR2VAワークフロー。
`H3_CHARACTER_SWAP=1` の起動時だけ `06_MiniMax_H3_Character_Swap.json` を追加します。
通常04と任意比較05は変更しません。元動画と参照画像は自分が利用権を持つ素材を使い、実在人物の場合は同意を得てください。

## 今のDaSiWa構成から

新イメージ: `ghcr.io/grawthings-beep/minimax-h3-i2v:character-swap-fast-cu130`

CUDA 13対応ドライバのホストが必要です。GPU機種名だけでは判断できません。
このタグはDockerビルド成功後に公開されます。古いPodのキャンバスは自動更新されないので、
新イメージで起動してWorkflowsから06を新しく開いてください。
CUDA 12.8用は `character-swap-community-cu128`、その場合 `REQUIRE_COMFY_KITCHEN_CUDA=0`。

現在のユーザー設定を維持した環境変数全文（ライセンス項目は同意・適格性を自身で確認済みの場合のみ1）：

```dotenv
HF_TOKEN="{{ RUNPOD_SECRET_HF_TOKEN }}"
CIVITAI_API_TOKEN="{{ RUNPOD_SECRET_CIVITAI_TOKEN }}"
ACCEPT_MINIMAX_H3_LICENSE=1
MINIMAX_H3_LICENSEE_IN_APPLICABLE_TERRITORY=1
MINIMAX_H3_SEPARATE_LICENSE=0
H3_PROFILE=r2v
H3_R2V_MODEL=dasiwa-v2
H3_R2V_VAE=x2-detail
H3_R2V_COMPARE=1
H3_CHARACTER_SWAP=1
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

比較不要なら `H3_R2V_COMPARE=0`。04用の任意creator LoRAが不要なら
`H3_R2V_LORA_SELECTION=none` で追加取得を減らせます。人物置換LoRAはこの指定と独立し、
`H3_CHARACTER_SWAP=1` で必須取得されます。04用LoRAを06へ自動で重ねません。
不要時は `H3_CHARACTER_SWAP=0` に戻せば06のみ非表示に戻り、ユーザー命名のワークフローは残します。

## 左から右へ操作

1. **元動画をアップロード**: 動画を選ぶ。まず人物1人・カットなしの短いショット。
2. **Source clip**: 開始時刻、使う尺、生成解像度を指定。初期値0秒・5.2秒・0.4MP。
3. **Character images**: 置き換え先の同一キャラを1〜8枚。通常1枚から。
4. **Replacement prompt**: 全文1欄。`<Video 1>` が元動画、`<Picture 1>`以降が左の画像順。
   複数人物なら「左の赤いジャケットの人物」のように、**元動画側の対象**を具体的に記述。
5. **人物置換LoRA**: 初期ON・強度1.0。選択／強度／OFFが外に見える。まず他のLoRAは重ねない。
6. **Audio**: `source`=元動画の同じ区間（初期）、`generated`=H3生成音、`mute`=無音。
7. 実行。2xの完成動画をSave Videoから保存。入力画像/入力動画そのものへモザイクはかけません。

基本プロンプト例（服や位置は元動画に合わせて変更）：

```text
Replace the person wearing a red jacket on the left of <Video 1> with the same character shown in <Picture 1>. Preserve the reference character's facial proportions, hairstyle, outfit and rendering style. Follow that person's position, actions and body orientation throughout the shot. Keep the other people, background, camera movement and scene composition unchanged. Maintain the replacement character consistently from beginning to end.
```

他人物・背景・カメラ保持は指示であり、厳密な保証ではありません。
複数参照画像は同一キャラの補助参照であり、複数人物へ個別割当するUIではありません。

## 尺・音声・VRAMの扱い

- 元動画は必要区間だけCPUで読み込み、PTS（表示時刻）に基づいて24fpsへ変換。
  30/60fpsを24fps扱いしてスローにする方式ではありません。VFRにも対応。
- H3の `17n+5` フレームへ**切り詰め**。末尾静止画で水増ししません。
  5.000秒のソースは107frames=4.458秒、5.2秒以上の区間は124frames=5.167秒。
  実際の尺・解像度・元音声有無を実行後のPreview as Textに表示します。
- 区間上限15.1秒／最大362frames。これはUI上限であり、品質・VRAM上の推奨尺ではありません。
  まず約4〜5秒・0.4MP、顔保持と背景を確認してから増やす。元画像のみの04より参照動画ぶん重くなります。
  長尺一括・複数カットは推奨しません。GPU実測やOOM回避は未保証。
- 元映像の縦横比を基準に32px単位へ縮小。参照動画・生成解像度の上限は0.8MP。
  既存の2x処理後は各辺が2倍。HDRソースはこのRGB SDR経路ではHDR維持を保証しません。
- 元音声は48kHz stereoへ変換し、選んだ時刻から実出力尺だけ切り出して再エンコード。
  元音声はモデル条件へ入れず、完成動画へ戻します。元映像の動きが変わると口形/効果音がずれることがあります。
  音声なし素材のsourceは無音。元音声の発話内容がキャラに合わせて変わる機能ではありません。
- 映像は `参照image/video INT8 encode → 20step生成 → H3退避 → 選択VAE/2x → CPUモザイク → MP4`。
  X2 VAE選択時も**参照encodeは公式INT8 VAE**。音声decodeは既存経路を維持し、source/muteでも実行されます。

## モデル・採用判断

[akatz-ai作者資料](https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA)
と固定revision `62407e0cc8089c363abd9ce4b0b27662abb237af` のネイティブ例を確認。
専用LoRA `h3_character_swap_pro4500_1000.safetensors` を1.0、res_multistep/simple20stepsで開始。
作者は公式Ref2VA INT8および一部hybridで評価。DaSiWa v2で同等の品質を保証する資料ではありません。
既存のDaSiWaを維持し、公式を二重ダウンロードせず選択済み本体を共有します。
DaSiWaのshift11/4は維持、06のみ20steps。`dasiwa-turbo-v2`＋人物置換ONは起動前に拒否します。

追加LoRAは155,110,320 bytes（約148MiB）。固定URL・サイズ・SHA256必須。
既存の並列・再開可能ダウンローダーへ統合し、`MODEL_VERIFY=size`でもこのLoRAはSHA256検証します。
X2単独運用で未取得なら参照用INT8 VAE 2,811,065,184 bytesも追加。
比較05を有効にして取得済みならINT8 VAEの追加ダウンロードはありません。
MiniMax H3 Community Licenseと既存適格性チェックを維持。トークン/モデルはイメージへ埋め込みません。

## 検証範囲

自動検証: 全profile/仕上げの配線・重なり・依存・モデル重複防止、実entrypoint分岐、
24/30/60fps/VFRの実CPU decode、区間/audio/MP4、native tokenizerとimage/video RefModの順序、
1/複数画像、実LoRAのSHA256・全キー/shapeとnative H3の適合・ON/OFF。
VAE/拡散モデルの実重みを使ったGPU動画生成・DaSiWa品質・ピークVRAM測定は含みません。
