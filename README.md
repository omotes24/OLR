# OLR:向きを変えた分布内画像による拡散モデルの分布外検知

26f WIP 中間発表のレジュメ(全版の TeX と PDF)、図解レジュメ、スライド、実験のコードと記録です。OLR は作業上の手法名です。

## 手法

学習済みの拡散モデルを凍結したまま、テスト画像を 1 回だけ通します(ノイズの大きさ 0.01)。勾配で学習するものなし、ネットワークなし、人手のラベルなし、分布外画像なし。

1. ブロック $h$ の出力 $f_{h,j,u}$(チャネル $j$、位置 $u$)から振幅を求め、ブロック A(16×16 を出す最初のデコーダのブロック)までを並べてプロファイル $\vec{a}(\vec{x})$($p\times 1$)とします。$C_h$ はブロック $h$ のチャネル数、$\bar{f}_{h,j}$ はチャネル $j$ の位置についての平均です。

```math
a_h(\vec{x})=\frac{1}{C_h}\sum_{j=1}^{C_h}\log\max_{u}\left|f_{h,j,u}-\bar{f}_{h,j}\right|
```

2. val1 の分布内の画像から、プロファイルの平均 $\vec{\mu}$ と共分散 $\mathbf{S}$ を求めます。同じ画像を残り 7 つの向き(上下反転、左右反転、180°・90°・270° の回転、2 本の対角線での反転)にしたときの平均を $\vec{\mu}_o$ とします。

3. スコアは次の式です($c=8$、$\lambda=10^{-3}$、$\bar{s}$ は $\mathbf{S}$ の対角成分の平均)。

```math
s(\vec{x})=-\{\vec{a}(\vec{x})-\vec{\mu}^{*}\}^{\top}\mathbf{P}\{\vec{a}(\vec{x})-\vec{\mu}^{*}\}
```

```math
\vec{\mu}^{*}=\vec{\mu}+c\,(\vec{\mu}-\vec{\mu}_o)
```

```math
\mathbf{P}=(\mathbf{S}+\lambda\bar{s}\mathbf{I})^{-1}
```

展開すると $s=-d^2+2c\ell-c^2\Delta^2$ です。$d^2$ は白色化した距離、$\ell=(\vec{\mu}-\vec{\mu}_o)^{\top}\mathbf{P}\{\vec{a}(\vec{x})-\vec{\mu}\}$ は内積の項、$\Delta^2$ は定数です。実装は `code/ol_score.py`(凍結した版)。

## 結果(6 つのテスト分割、ノイズの種 0〜2 の平均、自然画像の分布外集合に対する AUROC)

| 分布内 | CIFAR-10 | SVHN | CelebA | ImageNet-64 | FashionMNIST | 5 つの平均 |
|---|---|---|---|---|---|---|
| OLR | 95.1 | 96.9 | 100.0 | 85.9 | 99.7 | **95.5**(分割ごとに 95.3〜95.7) |

- 縮小の履歴をそろえた対照:91.9(3.6 下がる)。上下逆にした分布内の画像:84.8(弱点)。
- CPU で 1 枚 0.068 秒(ブロック A までの順伝播が 0.066 秒)。
- 256×256(ADM)では向きを変えてもプロファイルがほとんど動かず($\Delta=0.55$)、76.3 にとどまります(手法の限界としてレジュメに記載)。
- 先行研究との比較、画質への反応、モデルの変種などはレジュメ(`paper/main.pdf`)の表にあります。

本文に書いてある注意:6 つのテスト分割は、OLR より前に他の手法で結果を見たものです(OLR の選択には使っていません)。$c=8$ は val2 で決め、val2 で 62 通りの候補を試しました(候補にしない変種 35 通り)。評価の前に「6 分割の平均が 95.0 以上」と書き、成り立ちました。経緯は `protocols/PROTOCOL_OLR.md` にあります。

## 中身

| フォルダ | 内容 |
|---|---|
| `paper/` | 最新版 ol7 のソース(pLaTeX、MIRU 2025 のクラス)、数値と表の生成スクリプト `gen/ol_gen.py`、その元データ `gen/data/ol/` |
| `versions/` | ol1〜ol7 の PDF と変更の一覧。各版のソースはタグ `ol1`〜`ol7` |
| `zukai/` | 図解レジュメ(手法を手順ごとに 1 つの図で説明。5 ページ) |
| `slides/` | 中間発表のスライド(pptx と PDF)と、それを作るスクリプト |
| `code/` | GPU サーバで実行したコード(スコア、1 回だけの評価、評価後の分析、val2 での開発) |
| `runs/` | 集合の定義(どの画像がどの集合・分割に入るか) |
| `protocols/` | 実験の記録 I59〜I80(計画を結果より先に書いた記録。時刻は UTC) |
| `environment/` | 実行環境(GPU 側の pip freeze と、解析側の依存) |
| `tools/` | 公開時に置き換えたパス |

## 作り方

    cd paper && python3 gen/ol_gen.py && python3 assemble.py && latexmk -r latexmkrc main.tex    # レジュメ(TeX Live の platex)
    cd zukai && python3 gen/mk_data.py && latexmk -r latexmkrc main.tex                           # 図解レジュメ
    cd slides && python3 build/mk_deck.py 26f_WIP_中間発表_OLR_10p.pptx                           # スライド(python-pptx)

GPU 側のコードを動かすには、データセットの配列、モデルの重み、外部のコード(NVlabs の EDM / EDM2、improved-diffusion、guided-diffusion)が別に要ります(`code/README.md`)。

## 公開版について

- 著者のメールアドレスを、すべての版のソースと PDF から除きました。PDF は各版のソースから組み直したもので、ほかの文面と数値・表は届けた版と同じです。
- 開発の物差しに使った、本文で扱わない手法の項目を、結果ファイル・記録・コードから除きました(どこを除いたかは `code/README.md` と `protocols/PROTOCOL_OLR.md` の I80)。レジュメの出力は変わりません。
- `code/` のファイルは、サーバのパスを置き換え、上の行を除いたため、記録にある md5(実行した版)とは一致しません。

## ライセンス

MIT(`LICENSE`)。ただし `paper/` と `zukai/` の `miru2025j.cls`, `miru2025j-unsrt.bst` は MIRU 2025 の配布物、`slides/build/logo.png` は研究室のロゴで、MIT の対象外です。
