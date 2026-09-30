# R2VA専用 / INT8 Video VAE

同一キャラ画像1〜8枚＋全文プロンプト1欄の04だけを使う、永続ディスク不要な起動設定です。
構図画像・Add Guideは追加しません。2x upscaler・CPUモザイクは既存04と同じ。
Video VAEは公式INT8版です。`H3_R2V_MODEL` で公式/DaSiWa v2/DaSiWa Turbo v2から
本体1つと対応サンプリングを選べます。比較判断と詳細は [DaSiWaガイド](dasiwa-r2v.md)。

## RunPod

この変更を含む新しい `fast-cu130` imageを使い、
`runpod-template.r2v-cu130.example.json` を設定の基準にしてください。
旧imageに環境変数だけを加えてもR2VA専用にはなりません。

```text
H3_PROFILE=r2v
H3_R2V_MODEL=dasiwa-v2
H3_R2V_LORA_SELECTION=none
H3_FAST_VAE=1
COMFYUI_ARGS=--lowvram --vram-headroom 2 --fast fp16_accumulation
```

`HF_TOKEN` と `CIVITAI_API_TOKEN` はRunPod Secretで渡します。
モデルライセンスの同意・適用地域の申告は従来どおり必要です。
`CIVITAI_TOKEN` が残っている場合は従来のAPI_TOKENへの補完も働きます。

専用profileは以下を強制します。古い環境変数が残っていても取得対象は増えません。

- 基本モデルは `minimax_h3_r2v_int8_upscale.json` を基準にした5個のみ。
  `H3_R2V_MODEL` に応じて本体1個だけを差し替えます。省略時は従来どおり公式版。
- `MODEL_MANIFEST` は選択から生成したmanifestへ置換し、旧I2V/全モデルmanifestをマージしません。
- `H3_CHARACTER_R2V` は従来profile用です。r2v profileでは不要。
- FL2VA本体、FL2VA Turbo 4/8-step、追加LoRA URL一覧は取得・検証待ちをしません。
  creator LoRAも既定は未取得。新しい `H3_R2V_LORA_SELECTION` で明示した分だけ並列取得します。
- 配置する配布workflowは `04_MiniMax_H3_Character_R2V_2x` の1本だけ。
  01/02/03の既知の配布ファイルだけを置き換え対象にし、ユーザー名のworkflowは残します。
- 既存のモデルファイルや出力動画は削除しません。使わないモデルを新規取得しないだけです。
- Director/FBCの起動前チェックは行いません。R2VAで使うローカルノードは確認します。
- 基本モデルとモザイクモデルを並列取得。DaSiWaの場合は本体も別系統で並行取得。
  既存の `hf download` / Xet・aria2再開方式・
  MODEL_VERIFY=size/sha256 と必須ファイル確認を維持し、失敗時はComfyUIを起動しません。

公式のダウンロード対象は40,140,903,884 bytes（約40.14GB / 37.38GiB）＋小さなモザイクモデルです。
DaSiWaも約40.14GBです。本体が増えるわけではありません。
従来のI2V+R2V基本セット63,508,026,812 bytesから23,367,122,928 bytes、約36.8%減です。
creator/Turbo/追加LoRAの削減はこの差分とは別です。GPU・ネットワーク・初回image pullなどで
所要時間は変わり、削減率と起動時間の短縮率が一致する保証はありません。
120GBコンテナディスク・永続volume 0GBをテンプレートの開始値として維持しています。

## 全文プロンプト・RefModの使い方

- 04をWorkflowsから開き直す。旧ブラウザーキャンバスや別名保存したworkflowは自動変換しません。
- 左の「画像を追加」で同じキャラを1枚、または複数枚まとめて選択。↑↓で並べ替え、外すで削除。
  異なる縦横比をそのまま扱い、開始フレームには固定しません。動画・アニメGIFは対象外。
- `FULL PROMPT`へ演出・キャラ保持・音・BGMを含む全文を入力。他の3欄やmode切替はありません。
  公式6セクションでも通常の文章でも入力可能。コピーした単一コードフェンス/BOM/改行を整え、
  内容は勝手に要約・分割・翻訳しません。`<Picture 1>`などは左の画像順に対応します。
  複数画像が同一人物であることを記述してください。存在しないPicture/Video/Audio参照、空欄、
  16,000文字超または4,096 text tokens超は重いQwen実行前に説明付きエラーで止めます。
- 過去のfullモードエラーの該当tracebackは特定できていません。旧4欄ノードを新しい04から除外し、
  単一全文からnative tokenizer/conditioningまでをテストする変更です。未知のエラーを直ったとは断定しません。
- 各画像を独立した**Full Reference**としてVAE encodeし、上流の`H3RefMod`オブジェクトへ格納。
  Qwenは元の縮小画像を受け取り、DiTにはRefModを一度だけ付与。画像を時間方向に積み上げたり、
  最初の画像へ合わせて他を切り抜いたりしません。1枚を変換するだけで画質が上がるという保証はありません。
- 参照長辺1024px、合計2048 tokensが初期上限。枚数が増えた場合は全画像を同じ係数で縮小して収めます。
  UI上限は長辺1536px/合計4096 tokens。枚数や上限を増やすと遅延/OOMリスクが増えます。
  処理済み参照とlatentはCPUへ保持し、入力が同じ場合はComfyUIのノードキャッシュで再利用します。
- 保存ノードから`output/refmods/character_<hash>.safetensors`をダウンロードできます。
  上流v5の独立image-member bundle形式。別のRefMod対応workflowで再利用可能です。
  **今回の04の入力UIは画像用**です。Podを削除すると入力画像も保存RefModも消えるため、事前にダウンロード。
  新Podでは元の画像を再投入して作成するか、保存ファイルを対応するRefMod Loaderで使用してください。

RefModのcore.py/bundle.pyとMITライセンスだけを固定commit
`f9462081e28794389b5a6c5067eb327412ad8ee7`からimageへ組み込みます。
参照元: https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod
別パッケージのUI/音声/グローバルパッチは読み込みません。生成モデルの追加download、学習、有料APIなし。
起動前にコード・UI・revisionの存在を確認し、不足時はモデルdownload前に停止します。

テストは1→複数→1枚、入力変更時のキャッシュ無効化、native H3 tokenizer、空latent生成、
RefModの保存/上流読込、全リンク両端、配置非重複、DaSiWa sampler/shift維持を対象にします。
CPU smokeのQwen重み/VAE重みはfixtureです。実GPUでの顔保持率や連続生成OOMは未検証です。
開発時にはComfyUI v0.37.0 / frontend 1.52.7の実画面でも、2枚同時追加・並べ替え・1枚への削減・
再読込・保存後の画像順保持を確認しました。内部JSON欄は非表示にして全文入力だけを見せます。

## 任意LoRA（DaSiWa / RefModを維持）

04のModels列、モデルローダーのすぐ下に `LoRA · 選択 / 強度 / ON-OFF` を常時表示します。
サブグラフ展開は不要です。初期値は未選択・強度0.4・OFFなので、従来の結果を基準に比較できます。

1. 起動時に取得するファイルを下表から指定。旧 `H3_LORA_SELECTION` だけではR2VAの取得は増えません。
2. 新imageで起動後、Workflowsから04を開き直し、`lora_name` で取得済みファイルを選ぶ。
3. `strength` を設定し `enabled` をON。OFFまたは強度0ならLoRAファイルを読み込まず、入力MODELをそのまま返します。
   ON→OFFでもこのノードのLoRAキャッシュを解放します。選択名はOFFにしても残ります。

| H3_R2V_LORA_SELECTION | 起動時の取得対象 |
| --- | --- |
| `none`（既定） | なし |
| `hmnsfw_aio_v2` | `HMNSFW_AIO_V2.safetensors`（約310MB） |
| `hmmotion_v1` | `hmmotion_minimax-h3_epoch12.safetensors`（約310MB） |
| `all` または `hmmotion_v1,hmnsfw_aio_v2` | 上記2本のみ。UIで使うのは選択した1本 |

V2は既存 `CIVITAI_API_TOKEN`（または `CIVITAI_TOKEN`）、V1は読み取り権限付き `HF_TOKEN` を使います。
RunPod Secretの渡し方は変更不要。既存の取得スクリプト・再開方式・サイズ/SHA256検証を再利用し、
明示したLoRAが取得/検証できなければ `H3_LORA_REQUIRED=0` が残っていても起動を止めます。
取得しただけでは適用されません。UIはOFFから開始します。手動で `models/loras` へ置いたファイルも選択可能です。

配線は `DaSiWa → 任意LoRA → SigmaShift 11/4 → SchedulerとGuider`。
25 steps、Full Prompt、1〜8枚RefMod、INT8 VAE、モデル退避、2x、CPUモザイクは維持。
旧FL2VA Turbo LoRAの自動追加、追加URL一覧の一括取得、複数LoRAの積み重ねはしません。
作者によるV2の強度目安は0.5以下ですが、DaSiWa＋R2VAの画質保証ではありません。
参照の顔や衣装が変わる場合は強度を下げるかOFFで同じseedと比較してください。LoRAでVRAM使用が増える可能性もあります。
実GPUでの個別LoRA互換性・画質・連続生成OOMは未検証です。
ComfyUI v0.37.0の実UIでは、選択メニュー・強度変更・ON→OFFをサブグラフ展開なしで確認済み。
Docker buildでは小さな合成重みを使って、CoreのH3キー変換・LoRA読み込み・強度計算・OFF復帰も検証します。

この変更を含むimageへの更新が必要です。古いimageへ環境変数だけ追加してもノードは増えません。

## VAEと比較

`minimax_h3_video_vae_int8_convrot.safetensors` は2,811,065,184 bytes。
HF revision `bf92c4091e333e69b8ca1998e0a669f15cb0832b`、SHA256
`52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e` を固定しています。
公式選択のRef2VA、共通のQwen/audio VAEは従来manifestと同じサイズ・SHA256です。

ComfyUI v0.37.0の対応コードを使います。`H3_FAST_VAE=1` は既存COMFYUI_ARGSに
`--fast fp16_accumulation` がなくても補います。`H3_FAST_VAE=0` は自動追加を止めますが、
ユーザーがCOMFYUI_ARGSへ書いたフラグは消しません。比較時は両方から外してください。
このフラグはVAEだけの専用スイッチではなく、対応する他のFP16演算にも作用し得ます。
メモリ保護の生成モデル退避とタイルdecodeは維持しています。

VAEは演出やキャラ保持を改善するモデルではありません。速度・数値精度・実GPUでの
連続生成OOMは別途確認が必要です。GPUがないCIの成功を画質ベンチマークとは扱いません。
試す際は同じ画像・prompt・seed・steps・画素数・尺で比較してください。

旧構成へ戻す場合は `H3_PROFILE=legacy`。既存H3_CHARACTER_R2V/LoRA設定が再び有効になります。
FP16版の旧04と旧manifestもrepo内に残しています。

Sources:
- https://blog.comfy.org/p/making-the-minimax-h3-video-vae-2x
- https://huggingface.co/Comfy-Org/MiniMax-H3/blob/bf92c4091e333e69b8ca1998e0a669f15cb0832b/vae/minimax_h3_video_vae_int8_convrot.safetensors
