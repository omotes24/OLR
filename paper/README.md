# 向きを変えた分布内画像による拡散モデルの分布外検知 (method name OLR is a working name; title in body0.tex)

26f WIP 中間発表のレジュメ(MIRU 2025 クラス, pLaTeX).

## ビルド
    python3 gen/ol_gen.py && python3 assemble.py && latexmk -r latexmkrc main.tex

- `pre.tex`, `body0.tex`〜`body3.tex`, `texts/OL*.tex` → `assemble.py` → `main.tex`
- 数値と表の本体(`numbers_ol.tex`, `tab_*_ol.tex`, `supp_*_ol.tex`)は `gen/ol_gen.py` が `gen/data/ol/` から作る. 手で書いた数値はない. 本文の言い回しが数値に依る箇所は `gen/ol_gen.py` の assert で確かめている.
- 図1は `figs/fig_method.tex`(TikZ, 模式図).
- 手法の名前は `pre.tex` の `\Ours`, 題名は `body0.tex` で変えられる.

## 手法(凍結した実装: サーバの code/ol_score.py, PROTOCOL_IDO.md I68)
ブロックAまでの振幅の並び a について

    score = -(a - m*)' P (a - m*),   m* = m + 8 (m - m_o),   P = (S + 1e-3 * 平均分散 * I)^-1

m, S は val1 の分布内の画像の並びの平均と共分散, m_o は同じ画像を残り7つの向きにしたときの並びの平均.
表では「OLR(Ours)」を太字で書く(pre.tex の \Ours). C1(I68 で書き直したもの): 勾配による学習なし, 追加のネットワークなし, 人手のラベルなし, 分布外の画像なし. 「学習した分類器なし」とは書かない.

## データ(gen/data/ol, GPUサーバからの写し)
| ファイル | 中身 |
|---|---|
| ol_final.json | 1回だけの評価(I68-I69). サーバでの分割名 test7 = 本文の test6, test6〜test3 = test5〜test2, test2 = test. 元の画像と,縮小の履歴をそろえた対照 |
| ol_devcheck_s0.log | 凍結した実装が val2 で開発時の値を再現することの確認 |
| olx_sets.json | 評価のスコアからの集合ごとの AUROC, FPR95(I70 X7) |
| ol_vs_all_test2.json / .log | test での先行研究との組ごとの比較(勝/分/負, ブートストラップ. I70 X3) |
| olx_cfs.json | test での CFS との比較. 元の画像と対照(I70 X3) |
| cfs_n.json | CFS を OL と同じ val1 の画像だけに当てはめた値(test. I48, I52) |
| olx_dq.json | 分布内の画像と,それを劣化させた写しの区別(I70 X1) |
| olx_var.json | 乱数の重み,並べ替えた重み,別のデータで学習したモデル(I70 X2) |
| olx_hr.json | 256x256(in256c)と 112 ピクセル経由の対照(in256d)(I70 X4) |
| bench_ol.jsonl | CPU時間(1セッション. I70 X5) |
| olx_dev.json | val2 での分析: 定数 c, 向き, 枚数, 定数のない形(I70 X6) |
| olx_cos.json | 診断: OLR の方向と, val2 の分布外集合で決めた方向の余弦(seed 0. I72 X8) |
| olx_diag.json | 診断(val2, seed 0-2, ブロックAまで): 振幅が下がるブロックの割合(符号), OLR の方向と分布外集合で決めた方向(余弦, AUROC), 向きを変えた並びの共分散の比(I73 X9, X8b, X10) |
| olx_fall.json | 診断: 0.2標準偏差の幅で数えた, 振幅が下がる/上がるブロックの割合とブロックAでの z(I73) |
| olx_info.json | 当てはめた量(p, n, Delta) |
| gx_val2*_s0.json, gx_val2h_s1/2.json | val2 での開発の8回(候補62通り, 候補にしない変種35通り. I59-I67) |
| hl_meta.json, fk_ablate.json | モデルのブロック, val1 の枚数 |
| ssb_diag.json, ssb_diag2.json, bench_rf256.jsonl | 256x256 の集合の元ファイルの量, 256 での順伝播の時間 |

## 本文に書いてある制約
- 定数 c = 8 は val2 で決めた(8 と 16 の間は平ら). 尤度比の共分散の比 1+1/c = 1.125 は測った値(0.83〜1.65)と合わない. 題名の尤度比はこの比を固定したモデルでのもの.
- val2 で 62 通りの式を試した.
- 6つのテスト分割は,以前に他の手法で結果を見ている. この式の選択には使っていない.
- 評価の前に書いた主張は5つですべて成り立った. 本文に関わるのは「6分割の平均が 95.0% 以上」. 残りの4つは本文で扱わない別の手法との比較.
