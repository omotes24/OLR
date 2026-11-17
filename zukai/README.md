# 図解レジュメ（OLR の手法、2026-11-17）

本編のレジュメ ol7 の 3 節（提案手法）を、手順ごとに 1 つの図で説明する。MIRU 2025 のクラス（1 段組。クラスの onecolumn オプション）。

- 図 1：流れ（3 つの手順）。図 2：手順 1（画像 1 枚から p 個の数）。図 3：向きを変えた画像。図 4：プロファイルの空間（CIFAR-10 の Δ=1.768、c=8 で縮尺どおり）。図 5：形の確認（1×p、p×p、p×1）。図 6：2 つの項（d² と ℓ）。
- 数値は本編の numbers_ol.tex と gen/data/ol（olx_info.json、hl_meta.json）の CIFAR-10 の値。点の雲、チャネルの地図、図 3(b) の折れ線は模式図（gen/mk_data.py）。

## 作り方
    python3 gen/mk_data.py
    latexmk -r latexmkrc main.tex      # platex + dvipdfmx
