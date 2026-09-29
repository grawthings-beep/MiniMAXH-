# R2VA専用 / INT8 Video VAE

キャラ画像1枚＋プロンプトの04だけを使う場合の、永続ディスク不要な起動設定です。
構図画像・Add Guideは追加しません。参照画像上限・2x upscaler・CPUモザイクは既存04と同じ。
Video VAEは公式INT8版です。`H3_R2V_MODEL` で公式/DaSiWa v2/DaSiWa Turbo v2から
本体1つと対応サンプリングを選べます。比較判断と詳細は [DaSiWaガイド](dasiwa-r2v.md)。

## RunPod

この変更を含む新しい `fast-cu130` imageを使い、
`runpod-template.r2v-cu130.example.json` を設定の基準にしてください。
旧imageに環境変数だけを加えてもR2VA専用にはなりません。

```text
H3_PROFILE=r2v
H3_R2V_MODEL=dasiwa-v2
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
- FL2VA本体、creator LoRA、FL2VA Turbo 4/8-step、追加LoRA URL一覧は取得・検証待ちをしません。
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
