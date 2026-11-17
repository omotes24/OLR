# runs/ — 集合の定義(サーバの写し)

どの画像がどの集合・分割に入るかを決めるファイルです。各行は `[集合の名前, 配列の名前, 画像の番号の列, 前処理]` で、
配列は `data/npy/<配列の名前>.npy`(各データセットを並べた uint8 の配列。リポジトリには入っていない)を指します。
`code/ol_score.py` は次の順にこれらを読み、同じ名前の集合は先に読んだものを使います。

| ファイル | 中身 |
|---|---|
| `ok/sets_<id>.json` | `devall-`(val1 の分布内の画像。較正に使う)、`sel-`(val1 の分布外。OLR では読まない)、`fresh-`(val2。開発用)ほか |
| `ok/sets_<id>_eq50.json` | 縮小の履歴をそろえた対照の較正画像 |
| `e11/sets_<id>.json` | サーバの `test2` = 本文の test |
| `hx/sets1_<id>.json`, `hx/sets3_<id>.json` | 開発用の集合と、サーバの `test3` = 本文の test2 |
| `ido/sets4〜7_<id>.json` | サーバの `test4`〜`test7` = 本文の test3〜test6 |
| `ido/sets8_in256d.json`, `ido/probe_in256d.json` | 256×256 の 112 ピクセル経由の対照 |
| `ido/sets9_<id>.json` | test6 の分布内の画像を劣化させた写し(X1) |
| `ido/gx_<id>.json`, `ido/gx_<id>_eq50.json` | 開発(val2)の集合と、その縮小の履歴をそろえた版 |
| `hx/s0/<tag>/meta.json` | 各モデルのブロック(フックの名前、エンコーダ/デコーダ、チャネル数、出力の大きさ)の順 |

`<id>`: edmc10 = CIFAR-10, l6svhn = SVHN, dpceleba = CelebA, in64 = ImageNet-64, fmnist = FashionMNIST。in256c / in256d は 256×256 の ImageNet。
6 つのテスト分割は、OLR より前に他の手法で結果を見たものです(本文に記載)。
