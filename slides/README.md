# slides/ — 26f WIP 中間発表のスライド(OLR)

- `26f_WIP_中間発表_OLR_10p.pptx`:本編 10 枚と付録(計 25 枚)。`26f_WIP_中間発表_OLR_10p.pdf` は Keynote で書き出したもの。
- 数値はすべてレジュメ(`../paper`)から読む。手で書いた数値はない。

## 作り方

    python3 build/mk_tabs.py ../paper tex          # 表の TeX(レジュメの tab_*_ol.tex, supp_*_ol.tex の行から)
    # tex/t_*.tex を lualatex で組み(Times とヒラギノ明朝。macOS)、600 dpi の PNG にしたものが png/
    python3 build/mk_deck.py 26f_WIP_中間発表_OLR_10p.pptx   # python-pptx。数式は PowerPoint の数式オブジェクト
    zsh build/kn_run.sh 26f_WIP_中間発表_OLR_10p     # (macOS)Keynote で開いてフォントを確かめ、PDF に書き出す

`build/` の画像:`logo.png`(研究室のロゴ)、`bird.png`(CIFAR-10 の画像 1 枚)、`fig_*.png`, `strip_*.png`(レジュメの図と、向きを変えた画像の帯)。
