# code/ — OLR のコード(GPU サーバで実行したもの)

サーバ(4× RTX 2080 Ti)で実行したファイルの写しです。公開にあたって次の 2 点だけを変えています。

- プロジェクトのルートと Python インタプリタのパスを、一般的なものに置き換えた(`../tools/subs.py`)。
- 本文で扱わない手法(開発の物差し)を計算する行・名前を挙げる行を除いた(開発の `gx_eval*.py`、評価の `ol_final.py`、いくつかのコメント)。

このため、これらのファイルの md5 は `../protocols/PROTOCOL_OLR.md` に記録した値(実行した版)とは一致しません。一致しないのは、上の 2 点を変えたファイルだけです。

## 手法(凍結した実装)

| ファイル | 内容 |
|---|---|
| `ol_score.py` | OLR のスコア(I68 で凍結)。ブロック A までの振幅の並び a(x) を 1 回の順伝播で読み、`s(x) = -(a - m*)^T P (a - m*)`, `m* = m + 8 (m - m_o)`, `P = (S + 1e-3·(分散の平均)·I)^-1` を計算する。比較用の `d2`, `l`, 全ブロック版 `sa` も保存する。 |
| `dgcore.py`, `ct_extract_defs.py` | 共通部分:モデルの読み込み、共有のノイズ画像、5 つのモデルと読むブロックの規則(`SPEC`, `PATS`)。 |
| `admwrap.py`, `dgnet.py` | ADM 256×256 のラッパーと、FashionMNIST 用に学習したモデルの定義。 |
| `rfx_run.py` | タグの変種(`<id>_eq50` 縮小の履歴をそろえた対照、`<id>_rinit` / `<id>_rshuf` 重みを初期化 / 並べ替え、`<id>_by_<model>` 別のデータのモデル)で、スクリプトをそのまま走らせる。 |
| `run_hires2.py`, `e11hr_run.py` | 256×256 の集合(in256c, in256d)用の実行ラッパー。 |

## 評価と分析

| ファイル | 内容 | 出力(`paper/gen/data/ol/`) |
|---|---|---|
| `run_ol.sh` | 凍結したスコアの実行(dev: val2 での確認、final: 6 つのテスト分割)。 | `runs/ido/ol/*.npz` |
| `ol_devcheck.py` | 凍結した実装が val2 で開発時の値を再現することの確認(I68 (4))。 | `ol_devcheck_s0.log` |
| `ol_final.py` | 1 回だけの評価(I68–I69)。公開版は OLR の行と H4 だけを計算する。 | `ol_final.json` |
| `boot.py` | ブートストラップ AUROC(重みつき、並べ直しなし)。 | |
| `run_olx.sh`, `olx_eval.py` | 評価の後の分析 X1–X5, X7(画質、モデルの変種、CFS との比較、256×256、集合ごとの値)。 | `olx_dq.json`, `olx_var.json`, `olx_cfs.json`, `olx_hr.json`, `olx_sets.json`, `olx_info.json` |
| `olx_dev.py` | val2 での分析 X6(定数 c、向き、枚数、統計量、定数のない形)。 | `olx_dev.json` |
| `olx_cos.py`, `olx_diag.py`, `olx_fall.py` | 診断 X8, X8b, X9, X10(方向の余弦、振幅は下がるか、向きを変えた並びの広がり)。 | `olx_cos.json`, `olx_diag.json`, `olx_fall.json` |

## 開発(val2 での 8 回。I59–I67)

| ファイル | 内容 |
|---|---|
| `mk_gx_sets.py`, `mk_gx_neg.py`, `mk_gx_neg2.py`, `mk_rfx_files.py` | 開発の集合と、分布内の画像の写し(劣化させた写し、回転・切り貼り・左右反転した較正画像)を作る。 |
| `gx_extract.py`, `gx_extract_c.py`, `run_gx*.sh` | 振幅(とログ標準偏差)をブロックごと・チャネルごとに取り出す。 |
| `gx_eval.py` … `gx_eval8.py` | 各回の候補(計 62)と、候補にしない変種(35)を val2 で評価する。 |

## 入っていないもの

- 本文の表 2 の組ごとの比較(`ol_vs_all.py`)と CPU 時間の計測スクリプト:どちらも、本文で扱わない手法を含むスクリプトから生成したもののため。結果は `paper/gen/data/ol/ol_vs_all_test2.json`, `bench_ol.jsonl` にある。
- データの準備(各データセットの配列 `data/npy/*.npy`)、比較手法の実行と保存したスコア(CFS など)、モデルの重み(`ckpt/`)、外部のコード(`ext/edm`, `ext/edm2`, `ext/improved-diffusion`, `ext/guided-diffusion`)。
- 集合の定義(どの画像がどの分割に入るか)は `../runs/` にある。

使い方の例: `PY=python bash code/run_ol.sh 0 final edmc10:0`(リポジトリを `~/olr` に置き、`data/npy/`, `ckpt/`, `ext/` を用意したとき)。
