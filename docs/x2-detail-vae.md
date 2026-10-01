# X2 Detail VAE を試す

2026-10-01。R2V専用04で動画VAEを切り替える機能です。
X2 VAEのモデル重み・実GPUを使う生成テストは未実施です。

## 変更される処理

`H3_PROFILE=r2v` と `H3_R2V_VAE=x2-detail` を指定すると、
動画VAEが `MiniMax-H3-X2-Detail-v1.safetensors` になり、
作者指定の `MiniMax H3 VAE Decode (fast)` で縦横2倍にデコードします。
RealESRGANの取得・処理は外します。0.4MPの縦長設定は生成480×864 → 保存960×1728です。

公式Ref2VA / DaSiWa v2 / DaSiWa Turbo v2のどれでも同じVAEを選べます。
本体とそのサンプリング設定、参照画像リスト、プロンプト、音声、モザイク、任意LoRA操作は維持します。
追加LoRAは初期OFF。従来の01～03、legacyの04には適用しません。

VAEは参照画像のエンコードにも接続されます。今回の試験は配布元の「Mode 1: 2X Decode」です。
RGB参照画像を先に補強する「Mode 2: Detailed Upscale / B32」は含みません。
同時に追加するとVAE交換と参照変更の影響が混ざるためです。

## 起動方法

**この変更を含むDockerイメージのビルドと公開が先に必要です。**
古い公開イメージへ環境変数だけを追加しても、この機能は入りません。
`runpod-template.r2v-x2-detail-cu130.example.json` は、その新イメージ用のテンプレートです。
初期モデルはDaSiWa v2です。比較時は現在使う `H3_R2V_MODEL` と同じ値へ合わせてください。

GitHubへ変更を反映した後、Actionsの「Build container」を実行します。
成功した新しい `ghcr.io/grawthings-beep/minimax-h3-i2v:x2-detail-fast-cu130` を使用します。
再現性のため、公開後はそのイメージdigestをRunPodへ指定することを推奨します。
この説明はビルド・公開・GPU試験が既に完了したことを意味しません。

既存のライセンス確認・認証情報・GPU選択を引き継ぎ、環境変数を次のように設定します。

```text
H3_PROFILE=r2v
H3_R2V_VAE=x2-detail
```

起動後、Workflowsから `04_MiniMax_H3_Character_R2V_2x.json` を開き直します。
ブラウザーに残った旧キャンバスは自動的に置き換わりません。
VAE名が `MiniMax-H3-X2-Detail-v1.safetensors`、デコードが `MiniMax H3 VAE Decode (fast)`、
RealESRGANノードがないことを確認します。ログには `VAE=x2-detail` と表示されます。

元へ戻すには `H3_R2V_VAE=int8` で再起動し、04を開き直します。
未指定時も従来のINT8 VAEになります。ユーザーが別名保存したworkflowや既存モデルファイルは削除しません。

## 初回の比較

1. 従来の04とその入力画像・プロンプト・seed・本体モデルを記録する。
2. 同じ本体・同じサンプリング・5秒・0.4MP・同じseedでX2版を試す。
3. 顔、2Dの細線、3Dの質感、ちらつき、格子、動き、音声、完成までの時間を比較する。
4. 連続実行でも問題がないことを確認してから0.6MPへ上げる。尺は同時に増やさない。

初期設定は空間タイルON、256px、overlap64px、出力CPU、追加の時間分割OFFです。
配布元の基準設定に合わせています。GPU出力・大きなタイルはVRAM負担を増やします。
本体をVRAMから退避してからデコードする既存の処理は維持します。
メモリ不足や画質悪化が出たらINT8へ戻して比較を止めます。

元の04はINT8 VAE＋RealESRGAN、X2版はX2 VAEだけです。これは同じ保存寸法での実用比較であり、
VAEだけの厳密比較ではありません。VAE単体を調べる場合は同じ保存済みlatentを両経路へ渡し、
同じ表示サイズで比較してください。参照エンコードも変わるため、seedだけでは完全な同一条件になりません。

## 固定先と検証範囲

- [VAE配布元](https://huggingface.co/speach1sdef178/MiniMax-H3-X2-Detail-VAE/tree/af8c92d267c6849fec5032c35a65d5766737b338)
  の5,246,877,148 bytesを取得。SHA256 `2296840f4acedcaa976688e7d7b97f7bf570b136e400385d3f46224011897aac`。
  `MODEL_VERIFY=size` でも、このファイルはSHA256まで検証します。
- [専用デコーダー](https://github.com/TripleHeadedMonkey/ComfyUI-MiniMaxH3_LatentUpscaler/tree/2e568dfe3e4f5e81da178bc845e05dcfbb64d55b)
  の `vae_decode.py` と `utils.py` のみ導入。latentアップスケーラーの重みやWeb機能は追加しません。
- 自動検査はモデル/VAEの組合せ、配線、二重拡大防止、ハッシュ不一致、起動前検査、INT8への復帰を対象にします。
  Dockerビルドには実PyTorchのCPU合成テンソルを用いたPixelShuffle・反復・例外後復元の検査も含めます。
  これらは実際のVAE重みを使った画質・OOM・速度の検証を代替しません。
- ComfyUIは0.37.0のままです。0.38.0のタイル合成修正は別の比較対象とし、このVAE試験と同時には変更しません。

VAEは約5.25GBで従来INT8の約2.81GBより大きく、出力画素数は4倍になります。
32GB GPUでの最大VRAM・処理時間・画質改善は未確認です。
