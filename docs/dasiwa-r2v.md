# DaSiWa R2VA: モデルと作者ワークフローの比較・採用判断

確認日: 2026-09-29。目的は「キャラ画像1枚＋演出から、元絵を保ちながら自由な冒頭の動画」。
結論は **DaSiWa Hybrid v2 INT8を試すための設定を追加し、R2VAの小さなグラフは維持**。
公式が最良とも、DaSiWaなら必ず改善するとも断定しません。実画像によるGPU A/Bは未実施です。

## 調べたものと判断

一次資料:

- 作者の[モデルAPI](https://civitai.com/api/v1/models/2877206)と
  [Hybrid v2](https://civitai.com/models/2877206?modelVersionId=3314675)、
  [Hybrid Turbo v2](https://civitai.com/models/2877206?modelVersionId=3314686)。
- 作者workflow最新スナップショット
  [C-MMH3-25](https://github.com/darksidewalker/dasiwa-comfyui-workflows/blob/39b3427e6a67ff2e2c98fb5afabc7ce28117dec7/C-MMH3/DaSiWa%20MiniMaxH3%20MythicAlchemy%20C-MMH3-25.json)。
- Directorノードの[説明](https://github.com/darksidewalker/ComfyUI-DaSiWa-Nodes/blob/53e2630a6970a4f052e34fd75ef580add7c7c44a/docs/minimax_h3_director.md)と
  [Guide実装](https://github.com/darksidewalker/ComfyUI-DaSiWa-Nodes/blob/53e2630a6970a4f052e34fd75ef580add7c7c44a/nodes/nodes_minimax_h3_director_guide.py)。

作者はv2をREF2VA/FL2VA対応のHybrid、動き・視覚的一貫性等を調整した派生として説明しています。
これは作者の説明であり、このrepoでの改善実測ではありません。非蒸留版は
res_multistepまたはeuler / simple、20–25step、video shift 10–12、audio shift 3–5が推奨。
TurboはeulerまたはLCM / simple、4/8step。モデルページのTurbo shift範囲は6–12/3–5、
workflowの説明はさらに狭い6–8/4–5のため、両方の範囲内の8/4を開始値にします。

| 項目 | 既存04 | 作者C-MMH3-25 | 今回の採用 |
|---|---|---|---|
| 参照処理 | native MiniMaxH3ReferenceToVideo | Director Guideから同じnativeノードを呼ぶ | nativeを直接使う |
| 入力 | 1画像・4つの文章欄 | 画像/動画/音声タイムライン、複数モード | 現状維持。追加の構図画像なし |
| 初期モード | R2VA | 保存JSONはI2VA | R2VAのまま |
| 参照予算 | 長辺1024px | Autoは短辺2048pxまで | 長辺1024px。必要ならユーザーが調整 |
| サンプリング | res_multistep / normal / 25 | simple / 25 / shift 11,4 | DaSiWa通常版に作者設定を採用 |
| 仕上げ | INT8 VAE・退避・タイル・2x・CPUモザイク | 複数の任意upscale/補間/保存機能 | 今の2xとモザイクを維持 |
| キャッシュ/補助モデル | なし | Cache/RTX/latent upscale等は任意 | 追加なし |

作者版には複数参照・音声参照・継続生成など明確な利点があります。しかし今回はその機能の
要求ではなく、画像1枚の演出と品質が対象です。Directorへ置き換えるだけで別の生成方式に
なるわけではありません。作者JSONの初期I2VA・大きめ参照・多機能依存を丸ごと移すと、
今の「冒頭を参照画像に固定しない」「毎回の起動を短く」「入力欄を見やすく」に合いません。
作者workflow自体はコピーせず、native APIで同じサンプリング設定を組み直しました。

なお作者JSONのモデル欄自体も公式FL2VA/Ref2VAを参照しています。「作者のworkflowを開く」だけでは
DaSiWa本体へ自動的に変わりません。今回の起動profileは本体・サンプラー・steps・shiftを
セットにし、取り違えを検証で止めます。

## 選択方法: 表示workflowは1本

新imageで `H3_PROFILE=r2v` と `H3_R2V_MODEL` を指定します。

| H3_R2V_MODEL | 取得する本体1つ | 初期設定 |
|---|---|---|
| `dasiwa-v2` | Hybrid v2 INT8 / version3314675 file3203130 | res_multistep / simple / 25step / video11 audio4 |
| `dasiwa-turbo-v2` | Hybrid Turbo v2 INT8 / version3314686 file3203135 | euler / simple / 8step / video8 audio4 |
| `official` | 従来の公式Ref2VA INT8 | 従来04そのまま / normal / 25step |

テンプレートは品質比較用の `dasiwa-v2` を選択済み。環境変数を省略した既存環境は
互換性のため `official` のままです。Turboは速度優先の任意選択で、追加LoRAではありません。
INT4版や旧FL2VA Turbo LoRAは取得しません。Turboでも重みの大きさはほぼ同じなので、
step減少はVRAM不足の解決を保証しません。

選択を変えたらPodを再起動し、**Workflowsから04を開き直す**。
ブラウザーに残る旧キャンバスは自動更新されません。モデルノードのDaSiWa名とSamplingを確認。
4入力欄・画像・解像度・seed・shiftはトップレベルにあり、サブグラフ展開は不要です。
自分のプロンプト等は別名保存/コピーしてください。配布04のみ起動時に再生成します。

## 起動・ダウンロード

- 本体1個 + 同じQwen/INT8 video VAE/audio VAE/x2 upscaler = 計5個。通常DaSiWa版は
  **40,138,193,428 bytes**、Turboは40,138,193,436 bytes。別に小さなCPUモザイクモデル。
- 公式本体とDaSiWa本体の両方を新規ダウンロードしません。既存の未選択ファイルは削除しません。
- HFの4共通ファイルは既存Xet/aria2。DaSiWaはCivitai認証後の署名CDN URLへaria2の
  最大16並列rangeで取得。HF・DaSiWa・モザイクの3系統が並行します。
- `.part` + aria2制御ファイルで再開。**DaSiWaはMODEL_VERIFY=sizeでもSHA256必須**。
  ハッシュとsafetensors構造を確認してから正規ファイル名へatomic renameします。
- Civitai APIのversion/fileId・サイズ・SHA256は `manifests/r2v_profiles.json` に固定。
  APIの「最新」やデフォルトファイルには追従しません。
- `CIVITAI_API_TOKEN` はRunPod Secretからのみ渡します。API認証ヘッダーはCDNへ転送せず、
  aria2にアカウントトークンを渡しません。署名URLも引数/ログへ出さずstdinで渡します。
- 401/403、サイズ/ハッシュ不一致、ノード/配線不足ならComfyUI起動前に停止。
  この調査時も匿名配布URLは401だったため、実トークンのダウンロード権限が必要です。
- 新しい有料APIはありません。RunPod利用料・通信・image pullの時間は従来どおりです。
- MiniMaxの既存ライセンス確認は維持。作者の個別許諾が利用者へ自動移転するわけではありません。
  作者ファイルを再配布せず、Podで直接取得し、埋め込みmetadataを改変しません。

設定全体は `runpod-template.r2v-cu130.example.json`。コンテナdisk120GB、永続volume0GB、HTTP8188。

## テスト範囲と比較手順

CI: 3選択のmodel/manifest一致、全リンク両端、ノード/グループの非重複、shiftの両経路、
取得対象の限定、既存04の不変、SHA破損・途中再開・認証情報漏洩防止、起動停止を検査。
Docker build: pinned ComfyUIでnative sigma-shiftとschedulerを実行(CPU)、3profileの実entrypoint
をモデル取得前までsmoke test。CUDA12.8/13の両imageで同じ検査を実施します。

**フルモデルをロードしたGPU生成・音声品質・2回目のOOM・顔の保持率・実download速度は未計測**。
まず同じ参照画像、4入力欄、seed42、5秒、0.4MP、同じ2x/モザイクでofficialとdasiwa-v2を比較。
本体と作者推奨設定を一緒に変える実用比較なので、モデル単体の優劣を切り分ける実験ではありません。
顔/衣装/動き/冒頭の自由度/音声/時間/連続2回のメモリを確認した後に解像度や尺を上げてください。
