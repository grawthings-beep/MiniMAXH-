# 同じ生成結果で仕上げ3方式を比較する

通常04はそのまま残し、必要なときだけ05を追加します。DaSiWa、本体の選択、
参照画像1〜8枚、Fullプロンプト、外に出した任意LoRA選択・強度・OFFは維持します。

## 起動・操作

この変更を含む新イメージが必要です。CUDA13の場合は、ビルド成功後の
`ghcr.io/grawthings-beep/minimax-h3-i2v:upscale-compare-fast-cu130` を指定します。
CUDA12.8版は `upscale-compare-community-cu128`。GPUホストの互換条件は従来どおりです。
公開済みかはActionsのBuild container結果で確認してください。この文書自体はビルド完了の証明ではありません。

認証、ライセンス、DaSiWa、LoRAなど既存の環境変数はそのままにし、次を設定します。

```text
H3_PROFILE=r2v
H3_R2V_COMPARE=1
```

`H3_R2V_VAE` は通常04用なので、現在の値を維持して構いません。
比較05の参照エンコードは必ずINT8に固定されます。

1. Pod起動後、Workflowsから `05_MiniMax_H3_Upscale_Compare.json` を開く。旧キャンバスは自動更新されません。
2. 左側に参照画像とFullプロンプトを設定。まず5秒・0.4MP、seedをfixedにする。
3. 通常どおりQueueを1回実行。H3のサンプリングは1回で、仕上げをA→B→Cの順に処理する。
4. 右側の比較ノードに3本のプレビューとMP4保存リンクが出る。「3本を先頭から再生」で比較。
   音声は同じなのでAだけ再生。共通シークで止めて比べる。ブラウザ同期は近似で、フレーム精度の保証はありません。
5. `output/video` に3本のMP4と同じ名前の処理時間JSONが残る。永続なしPodを削除する前に保存する。

| 出力 | 仕上げ | 最終寸法 |
|---|---|---|
| A_X2_VAE | X2 Detail VAEの専用デコード | 縦横2倍 |
| B_INT8_SPAN | 通常INT8 VAE → 2x NomosUni SPAN | 縦横2倍 |
| C_INT8_AnimeSharp | 通常INT8 VAE → 2x AnimeSharp V4 RCAN | 縦横2倍 |

3本とも同じlatent・音声・24fps・H264 CRF18・8bit sRGB。
0.4MP縦長なら480×864から960×1728です。X2の後ろに追加の2xは重ねません。
通常04のINT8版は従来どおりRealESRGANであり、今回のBはSPANです。

右側の `mosaic_enabled` は3本共通。標準ONで、各最終フレームへCPUモザイクを一度だけ適用し、その後MP4化。
既存のJUST・confidence .30・IoU .50・block自動・gap3・anus対象外を維持します。
輪郭検出結果は各方式の映像に応じて変わり得るため、モザイク領域そのものの差はアップスケール品質と混同しないでください。
今回の05にRIFEは追加しません。

## 比較条件と見る場所

これは**サンプリング済みlatentから先の仕上げ比較**です。04の `x2-detail` は参照エンコードもX2になるため、
04と05のAを同seedで回しても、完全に同じ生成結果になるとは限りません。
05内の3方式は同一latentを使います。X2作者のMode 2参照補強やlatent再サンプリングは比較対象外です。

縮小プレビューだけで決めず、保存MP4を同じ表示サイズで比較してください。
顔・目の形、髪の細線、肌や材質、輪郭のハロー、タイル境界、動いた際のちらつきが観察対象です。
静止画のシャープさと動画の時間的一貫性は別に判断します。どの方式が最良かは実素材で確認が必要です。

JSONには各方式のモデル読込＋デコード、拡大＋退避、モザイク、エンコード、合計秒数とCUDA allocatedピークを記録します。
サンプリング時間・起動ダウンロード時間は含みません。GPU値はPyTorch割当で、総VRAM・予約VRAM・CPU RAMではありません。
初回読込やOSキャッシュの影響があります。順番は固定A→B→Cなので、厳密な速度ベンチマークではありません。

## メモリとダウンロード

H3をGPUから退避後、各方式を順番に実行して保存・解放します。アップスケールは1フレームずつ。
3本の高解像度IMAGEをComfyUIキャッシュへ返しません。ただし完成フレームバッチとモザイクの作業コピーはCPU RAMを使います。
定メモリのストリーミングではなく、長尺・高解像度でOOMしない保証はありません。

必要なモデルは既存の並列・再開可能ダウンローダーで起動時取得。共通ファイルは重複取得しません。
通常04がINT8ならX2約5.25GB＋アップスケーラー約35.5MBが増えます。
通常04がX2ならINT8約2.81GB＋約35.5MBが増えます。モデルはDockerイメージへ含めません。
X2と追加アップスケーラーは `MODEL_VERIFY=size` でもSHA256まで検証します。不足・破損・取得失敗時は起動を止めます。
途中の方式が失敗した場合、完成済みMP4は残し、JSONをfailedとして保存します。成功を装った3本比較は表示しません。

比較不要になったら `H3_R2V_COMPARE=0` で再起動。配布05だけ非表示になり、追加取得を止めます。
個人保存したworkflowやダウンロード済みモデル、完成動画は削除しません。

## 出典・利用条件

- [NomosUni SPAN](https://openmodeldb.info/models/2x-NomosUni-span-multijpg): Helaman、CC-BY-4.0。
  固定HFファイル・サイズ・SHA256は `manifests/upscale_compare.json`。
- [AnimeSharp V4](https://openmodeldb.info/models/2x-AnimeSharpV4): Kim2091、**CC-BY-NC-SA-4.0（非商用）**。
  [作者のRCAN版リリース](https://github.com/Kim2091/Kim2091-Models/releases/tag/2x-AnimeSharpV4)を使用。
  Fast RCAN-PU版ではありません。商用利用許可をこの機能が与えるものではありません。
- X2 Detail VAEおよび専用デコーダーの出典・固定先は [X2ガイド](x2-detail-vae.md)。

## 検証範囲

自動検査: モデル3種×通常VAE2種の起動、opt-in/OFF復帰、モデル重複防止、配線の両端、
同一latent/音声/寸法、画面配置、UI再生・シーク契約、実SPAN/RCANのCPU2倍処理、H264＋音声保存、例外後の解放と失敗レポート。
Dockerビルドの両CUDA版にもCPU実行検査を入れています。生成用H3/X2実重み・GPUを使った画質や速度、OOM耐性は別途実測が必要です。
