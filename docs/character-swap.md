# 06 · 元動画の人物置換

元動画の人物を、参照画像のキャラへ置き換えるR2VAワークフロー。
`H3_CHARACTER_SWAP=1` の起動時だけ `06_MiniMax_H3_Character_Swap.json` を追加します。
通常04と任意比較05は変更しません。元動画と参照画像は自分が利用権を持つ素材を使い、実在人物の場合は同意を得てください。

**完全置換を毎回保証する機能ではありません。** 改訂06は人物領域を指定して元人物の外見情報を弱め、
元動画のほぼ再現になった出力を保存前に止める試験的な経路です。
DaSiWaで成功した従来経路も `mode=original` で残します。マスク版のGPU生成品質は未検証です。

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
H3_SWAP_MODEL=shared
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
4. **Swap target**: 初期値 `masked_replace`。CPU人物検出で変更する領域を作る。
   初回は `mask_approval` を空にして実行。全フレームのシアン色マスクを確認する段階で停止する。
   `MASK REVIEW REQUIRED` は意図した停止で、この段階ではH3サンプリングを実行しない。
   マスクノードのプレビューを拡大して、顔・髪・衣装、退場と再登場を全区間確認する。
   問題がなければ表示された12文字のコードを **Character Swap encoder の mask_approval** へ貼り、再実行する。
   動画・マスク・ぼかし設定を変えるとコードが変わる。古いコードでは生成を続けない。
5. **Replacement prompt**: 全文1欄。`<Video 1>` が元動画、`<Picture 1>`以降が左の画像順。
   マスク版の自動検出は人物1人用。複数人物を検出した場合は対象を勝手に選ばず停止する。
   複数人物なら対象だけの動画MASKを接続するか、`original` に切り替えて対象の服・位置をプロンプトに書く。
6. **人物置換LoRA**: 初期ON・強度1.0。選択／強度／OFFが外に見える。まず他のLoRAは重ねない。
7. **Audio**: `source`=元動画の同じ区間（初期）、`generated`=H3生成音、`mute`=無音。
8. 実行後、顔・髪・衣装・再登場を目視してから公開する。入力画像/入力動画そのものへモザイクはかけません。

### マスク確認は省略しない

- 検出器はYOLO11s-seg / COCO person、**CPU実行**。アニメ絵・顔アップ・遮蔽・再登場で検出を落とす場合がある。
  実素材でも再登場後の検出抜けを観測した。未検出フレームを「誰もいない」と自動で断定しない。
  人物がいるのにシアンがない、顔や髪が欠ける、別人物を覆う場合は承認しない。
- confidenceを調整しても直らなければ、別の方法で用意した対象専用MASKを `target_mask` へ接続する。
  MASKは **Source clip処理後と同じフレーム数・高さ・幅、CPUの0〜1バッチ** が必要。
  単一画像MASKの全フレーム使い回しや未整列の動画MASKは受け付けない。
  用意できない場合は `original` を使う。従来の成功済み保存ワークフローもそのまま読み込める。
- `grow=32` は髪・衣装変更の余白。H3の空間32pxセルと因果的な時間区切りに整列するため、
  プレビューは輪郭そのものより広くなる。これは編集用マスクであり、後段モザイクのJUST輪郭とは別物。
- `source_retention=0.1` は人物領域に残す元画像の割合。残りは `blur_radius=48` のぼかし参照。
  小さくすると元の顔や衣装を弱めるが、動作・接触・小道具の手掛かりも失う可能性がある。
  参照キャラ画像はぼかさない。入力ファイル自体も変更しない。
- `original` はマスク編集・ぼかし・背景合成・保存前チェックをすべてバイパスする。
  旧保存ワークフローは旧プロンプトと設定まで保持する。新06のモードだけを戻しても新プロンプトは残る。

### 保存前チェックの意味

`Swap check` は背景ではなく検出人物の領域を元動画と比較する。
判定対象フレームの80%以上が相関0.985以上なら `SWAP NOT CONFIRMED` でMP4保存を止める。
閾値は調整可能で、`reject_unchanged` を明示的にOFFにすれば比較判定による停止だけを解除できる。
**相関値は顔認証でも参照キャラとの一致率でもない。** 正しい置換の誤検知も、異なるキャラや破綻動画の通過もあり得る。
チェックを通っただけで成功とは呼ばない。自動再生成・自動LoRA増強は行わない。
マスク外は元動画を出力サイズへ拡大して合成し、変更範囲を限定する。境界・遮蔽物・影は目視確認が必要。
元の人物輪郭より大きく衣装・髪型・体格が変わる場合はマスクを調整する。

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
- マスク版は `CPUマスク確認 → 人物領域ぼかし参照 + image/video INT8 encode → masked 20step生成 → H3退避 → 選択VAE/2x → 元動画コピー判定/背景復元 → CPUモザイク → MP4`。
  X2 VAE選択時も**参照encodeは公式INT8 VAE**。音声decodeは既存経路を維持し、source/muteでも実行されます。

## モデル・採用判断

[akatz-ai作者資料](https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA)
と固定revision `62407e0cc8089c363abd9ce4b0b27662abb237af` のネイティブ例を確認。
専用LoRA `h3_character_swap_pro4500_1000.safetensors` を1.0、res_multistep/simple20stepsで開始。
作者は公式Ref2VA INT8および一部hybridで評価。DaSiWa v2で同等の品質を保証する資料ではありません。
初期値 `H3_SWAP_MODEL=shared` は既存のDaSiWaを維持し、本体を共有します。
比較したい場合のみ `H3_SWAP_MODEL=official` にすると06だけ公式Ref2VA INT8を使います。
04/05/07の選択は変えません。公式の大型チェックポイントが別途追加ダウンロードされるため、通常はsharedのままにします。
DaSiWaのshift11/4は維持、06のみ20steps。`dasiwa-turbo-v2`＋人物置換ONは起動前に拒否します。

追加LoRAは155,110,320 bytes（約148MiB）。固定URL・サイズ・SHA256必須。
既存の並列・再開可能ダウンローダーへ統合し、`MODEL_VERIFY=size`でもこのLoRAはSHA256検証します。
X2単独運用で未取得なら参照用INT8 VAE 2,811,065,184 bytesも追加。
比較05を有効にして取得済みならINT8 VAEの追加ダウンロードはありません。
人物検出器は20,669,228 bytes。`manifests/swap_guard.json` にURL・サイズ・SHA256を固定し、
既存並列/再開ダウンローダーを使って `models/swap_detection/yolo11s-seg.pt` へ配置します。
`MODEL_VERIFY=size` でもSHA256必須。モデル欠損/改変時に実行を拒否し、検出ノード内部では勝手に取得しません。
Docker buildのCPU検証では一時取得・検証後に削除し、最終イメージへ重みを含めません。
Ultralyticsは既存の8.4.104を使用。検出重み/実装は [Ultralyticsライセンス](https://www.ultralytics.com/license) に従います。
MiniMax H3 Community Licenseと既存適格性チェックを維持。トークン/モデルはイメージへ埋め込みません。

領域編集と元人物情報抑制の参考:
[Fantastic MiniMaxH3 PromptBuilderのmasked editing](https://github.com/Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder#editing-a-clip-with-a-mask)。
パッケージ全体やSAMを追加せず、固定ComfyUIのH3 AV noise_mask仕様へローカルノードで接続しています。

## 検証範囲

自動検証: 全profile/仕上げの配線・重なり・依存・モデル重複防止、実entrypoint分岐、
24/30/60fps/VFRの実CPU decode、区間/audio/MP4、native tokenizerとimage/video RefModの順序、
1/3/8枚、実LoRAのSHA256・全キー/shapeとnative H3の適合・ON/OFF、
承認前のlazy encode停止/動画変更による承認失効、22/124/362framesのマスク形状、
native AV noise_maskのpack/再整形/H3条件への受け渡し、CPU検出器のhash/import/推論、
近似コピー停止・旧経路バイパス・マスク外背景保持・06だけ公式モデル選択。
VAE/拡散モデルの実重みを使ったGPU動画生成・DaSiWa品質・ピークVRAM測定は含みません。

### 2026-10-03 ローカル確認

- 全168テスト成功。実Git Bash entrypointの隔離テストを含む（実コンテナ起動とは別）。
- 固定ComfyUIで上記CPU runtime smoke成功。実YOLO重みのSHA256とCPU推論も確認。
- ユーザー提供の失敗出力04は、検出人物91フレーム中87フレームが近似コピーとして保存拒否になった。
  提供された成功出力02は拒否しなかった。2例の確認であり、一般的な検出精度を示すベンチマークではない。
- 元素材の検出を全フレームで試すと、参考CMの再登場後にも検出抜けがあった。
  この素材は自動マスクを無確認で承認してはいけない。手動MASKまたは従来経路が必要。
- この時点でGPUによる新しいマスク版の動画生成とDocker buildは未実施。
