# PROTOCOL_OLR.md -- the record of OLR (the orientation likelihood-ratio score, working name OL), entries I59-I80

The project keeps one log, `PROTOCOL_IDO.md`, written while the work was done: every plan before its numbers, every result after
it, with times in UTC.  That log also records earlier methods of the same family, which are not part of this paper.  This file
publishes the OLR entries (I59 onward), **rewritten so that they do not describe those methods**: comparisons with them, their
numbers and their parts are left out, and the yardstick of the development is given by its two values.  Everything about OLR is as
in the log: the times, the plans, the candidates and their numbers, the freeze, the single evaluation and the analyses.  Entry
numbers are kept, so "PROTOCOL_IDO.md Ixx" in `code/` and `paper/README.md` refers to the entry of the same number here.

## Constraints (as they apply to OLR)
- **C1** (worded by the author in I68): one forward pass per test image (sigma_1 = 0.01) through the frozen diffusion model; nothing
  trained by gradient descent, no further network, no human label, no OOD image.  ("No trained classifier" is not claimed.)
- **C2**: no OOD image in any choice.  Per-ID quantities come only from ID images: the val1 ID images (`devall-<ID>`, the
  calibration images) and transformed copies of them.  The val1 OOD sets (`sel-`) are never read.
- **C3**: the rule is the same for the five ID datasets (universal constants and ID-only statistics only).
- **Goal**: five-ID natural-OOD mean AUROC above 95.0.

## Splits (server names; the paper renames them)
- val1: `devall-<ID>` (calibration images, 1,500-2,000 per ID) and `sel-` (val1 OOD, not read).  val2: `fresh-` (development).
- Test splits: server `test2` = the paper's *test*; server `test3` ... `test7` = the paper's *test2* ... *test6*.  All six had been
  looked at with other methods before OLR existed (disclosed in the paper); nothing of OLR was chosen on them.

## Earlier entries cited below (not published; what they made)
- I47: the seven degraded copies of the ID images (blur 0.5 / 1.0 px, JPEG 90 / 75 / 50, bicubic to 3/4 and back, 2x box reduction and
  bicubic back) and the model-variant tags (`<id>_rinit` freshly initialised weights, `<id>_rshuf` permuted weights, `<id>_by_<model>`
  the trained model of another dataset); `code/mk_rfx_files.py`, `code/rfx_run.py`.
- I49 / I50: the identical-history versions of the images (every image reduced to half its side with a 2x box filter and brought back
  bicubic; tag `<id>_eq50`) and CFS on them (`runs/ido/cfseq`).
- I48 / I52: CFS with both heads fitted on the val1 ID images only (`cfs_n.json`).
- I54 / I56 / I57: the 256x256 tier and its controls; file size, compression and chroma subsampling separate its sets.

## Log

I59 2026-11-16 10:29 UTC.  EXPLORATION OF A SIMPLER SCORE (asked by the author at 10:18 UTC: "a more elegant method above 95%").
    PLAN, written before any number.  Development only: val2 and the val1 ID images for every parameter; no test split is read, the
    sel- sets are not read, nothing is frozen and no claim is made here.
    WHY: the amplitude at block A alone is strong where the ID images are complex and weak where they are simple (val2: CIFAR-10 94.9,
    SVHN 63.0); the whitened distance of the whole profile is the reverse (73.5, 98.8).  Hypothesis: the complexity of the image is
    mixed into the response, and the response RELATIVE TO THE INPUT (a gain) is the single quantity; it should also react less to the
    image quality, which shows on the input side too.
    DATA (code/mk_gx_sets.py, code/gx_extract.py -> runs/ido/gx/<tag>_s0_<label>.npz; noise seed 0, sigma_1 = 0.01, the shared noise
      image): per image, the amplitude of every block and of the network input itself (x + sigma eps), for the image and for its
      upside-down copy; the same with the log standard deviation in place of the amplitude (secondary).  Sets: the calibration images
      (devall-<ID>), the val2 sets; the seven degraded copies of I47 made from the val2 ID images (dqv-); the identical-history versions
      (I49) of the val2 sets with the identical-history calibration images; the val2 sets through the untrained twin (<id>_rinit).
    CANDIDATES (code/gx_eval.py; a = amplitude of the upright image per block, b = of the upside-down copy, a0 = of the input, A = the
      first decoder block with a 16x16 output, D = a_A - b_A; every mean, covariance, regression and rank from the calibration images):
      G1  a_A - a0                                             G2  the residual of a_A after a least-squares line in a0
      G3  the residual of a_A after a regression on a0 and the encoder blocks        G4  -(G3 standardised)^2  (does the sign matter?)
      C1  -Mahalanobis^2 of the decoder part of the profile GIVEN a0 and the encoder part     C2  -Mahalanobis^2 of a0 and the encoder part
      C3  the residual of the mean decoder amplitude           S1  -Mahalanobis^2 of (a_A, b_A) given a0
      S2  G2 + D, S3  G3 + D  (each standardised on the calibration images, summed)
      G5  a_A(trained) - a_A(untrained twin)                   G6  the residual of a_A(trained) after a line in a_A(untrained twin)
    REPORTED for every row: natural mean AUROC per ID and the five-ID mean, the upside-down ID set, on val2; the same under the identical
      history; the AUROC of the val2 ID images against their degraded copies (mean of the seven).  All rows are reported.
    RULE FOR GOING ON (fixed now): a candidate is taken further only if it is a single formula without a combination of ranks, its val2
      natural mean is at least 95.07, AND its val2 natural mean under the identical history is at least 90.33 (the two bars of the
      development).  If none qualifies, that is the result.  A candidate that qualifies still needs one evaluation on test images.

I60 2026-11-16 10:41 UTC.  ROUND 1 SEEN (runs/ido/final/gx_val2_s0.json; gx_eval.py run once; 15 extractions rc=0).  val2, noise seed 0,
    amplitude; natural mean of the five IDs | upside-down || the same under the identical history || ID against its degraded copies:
      G1  84.25 | 75.9 || 77.40 || 54.0     G2  88.54 | 84.2 || 83.26 || 70.1     G3  87.67 | 81.9 || 87.20 | 79.7 || 54.3     G4  82.43
      C1  86.86 | 65.3 || 77.96 || 79.6     C2  84.24     C3  78.56     S1  81.18     S2  88.81 | 91.7 || 84.26 || 67.4
      S3  88.68 | 90.9 || 87.26 | 88.6 || 58.5        G5  81.57     G6  88.41        (log std: the best candidate 88.38)
    RULE OF I59: no candidate reaches the bar on val2 (best single formula 88.5; with D added 88.8) and none reaches it under the
    identical history (best 87.3 against 90.33).  NONE QUALIFIES; nothing is frozen, no test split was read.
    What the rows say: (1) the input amplitude does not explain the response at A (G2 88.5 against 88.2 for a_A alone; the untrained twin
    does not either, G6), so "response relative to the input" in its simplest form is not the single quantity.  (2) relative to the
    encoder path (G3) the picture changes against a_A alone: SVHN 63.4 -> 89.2, but CIFAR-10 94.6 -> 88.5 and ImageNet-64 83.5 -> 61.1
    (Places365 96.0 -> 44.0, CelebA 98.0 -> 34.4, Textures 96.8 -> 64.4).  (3) G3 is nearly blind to the image quality (49-58% on the
    seven degradations) and hardly moves with the resampling history (87.67 -> 87.20).  So about 8 points of the bar on val2 are not
    available to a quantity that does not react to image quality, and under the identical history the gap is 3 points.  This is a
    statement about what the benchmark rewards, not a method.

I61 2026-11-16 10:47 UTC.  EXPLORATION, ROUND 2 (the author at 10:45 UTC: "do not stop until it is exceeded").  PLAN before any number;
    val2 and the val1 ID images only, as I59; nothing frozen, no test split read.  The more is tried on val2, the less a val2 number
    proves: every row is reported, the count of rows tried is kept (round 1: 12 candidates), and a qualifier is re-run with noise seeds
    1 and 2 before anything else.
    PART I (the stored statistics of I59; code/gx_eval2.py -> runs/ido/final/gx_val2b_s0.json).  a_A alone and the whitened distance
      differ in the METRIC: the whitened distance makes the directions in which ID images vary little weigh most, and a shift of the
      overall level is one direction of 34; a_A is one coordinate in its own units.  One formula that covers both may exist between them.
      M1  -sum_h z_h^2 (each block standardised, no covariance)        M2  -|a - mean|^2 (no standardisation at all)
      M3  +sum_h z_h (one-sided)        M4  the mean amplitude over the blocks (one-sided)        M5  -(a-mu)^T Sigma^(-1/2) (a-mu)
      K   the SHAPE instead of the size: c_h = amplitude - log std of block h (log of peak over spread, free of the overall gain):
          K1  c_A (and its sign reported)    K2  -Mahalanobis^2 of c    K3  -Mahalanobis^2 of (a, log std) jointly    K4  -sum z^2 of c
      P   both orientations in one formula:  P1  -Mahalanobis^2 of (a, b)    P2  -sum z^2 over (a, a-b)    P3  +sum z over (a, a-b)
      diagnostics (to understand, not candidates): the one-sided AUROC of every single block (a_h, and a_h - b_h); the principal
      components of the calibration profile.
    PART II (new extraction, code/gx_extract_c.py -> runs/ido/gx/<tag>_s0_c.npz; code/gx_eval3.py -> runs/ido/final/gx_val2c_s0.json):
      block A read per CHANNEL instead of as one number (the pass stopped there would cost 0.066 s per image), image and upside-down copy.
      H1  +mean_c z_c of the channel amplitudes (one-sided)        H2  -sum_c z_c^2 (amplitudes)       H2b  -sum z^2 of (mean, std), as CFS's diagonal head
      H3  -Mahalanobis^2 of the channel amplitudes        H4  -Mahalanobis^2 of the channel log stds
      H5  +mean_c z_c of (amplitude of the image - amplitude of the upside-down copy)      H6  H1 + H5 (each standardised)
      H7  -Mahalanobis^2 of (channel amplitudes of the image, of the upside-down copy)
    RULE (as I59): single formula, at least 95.07 on val2 AND at least 90.33 under the identical history.  A qualifier is repeated with
    noise seeds 1, 2; only then a freeze and one evaluation on the test splits are planned in a new entry.

I62 2026-11-16 10:50 UTC.  ROUND 2, PART I SEEN (runs/ido/final/gx_val2b_s0.json; run once).  val2, seed 0, natural mean | under the
    identical history:  M1 90.74 | 83.76   M2 88.57   M3 70.58   M4 69.83   M5 91.06 | 85.18   K1 81.56   K2 80.69   K3 89.11   K4 83.12
    P1 88.49   P2 90.30 | 82.14   P3 76.20.   NONE QUALIFIES (24 candidates tried so far).
    Diagnostics, per block alone (one-sided, natural mean): the response is most informative in the late 8x8 and first 16x16 decoder
    blocks (CIFAR-10 91-95 with the peak AT block A; CelebA 98-100; FashionMNIST 99-100; ImageNet-64 82-86), the same blocks that tell
    the image from its upside-down copy (97-100).  The shallow blocks behave like image complexity: ID LOWER than OOD for the simple IDs
    (SVHN 6-7, FashionMNIST 9-13, i.e. 87-94 the other way), no information for CIFAR-10 (46-53), ID higher for ImageNet-64 (57-63).
    So the benchmark needs a two-sided reading of the shallow / output side and a one-sided reading of the deep decoder; the whitened
    distance mixes them in 34 directions and is diluted, a_A reads one deep block.
    ROUND 3, PLAN before any number (the stored statistics; code/gx_eval4.py -> runs/ido/final/gx_val2d_s0.json; rule as I59):
      fewer numbers, one metric (against the dilution):
      N2  -Mahalanobis^2 of (first block, block A): two numbers, the pass stopped at A      N3  the same with the last block added
      ST  -Mahalanobis^2 of one number per STAGE (encoder / middle / decoder, resolution)    STs  the same with the stages up to block A
      the direction of the loss of response learned from the ID images themselves (their upside-down copies have the same image
      statistics and no response; no OOD image, no label):
      L1  Fisher direction between the calibration profiles of the upright and of the upside-down images, applied to the upright profile
      L1s the same on the blocks up to A        LQ  log-likelihood ratio of the two Gaussians (upright against upside-down profiles)
      L1d the same direction applied to a(x) - a(Fx) (two passes)
      LN  -Mahalanobis^2 of (first block, L1): the complexity side two-sided and the learned response direction, two numbers

I63 2026-11-16 10:52 UTC.  ROUND 3 SEEN (runs/ido/final/gx_val2d_s0.json; run once).  val2, seed 0, natural mean | upside-down ||
    identical history || degraded copies:
      N2 88.86   N3 90.01   ST 90.67   STs 90.02   LQ 86.27   L1d 87.76 | 95.9   LN 88.78
      L1  93.06 | 89.0 || 87.33 || 69.9        L1s 93.60 | 88.0 || 88.64 || 68.4   (CIFAR-10 94.65, SVHN 90.28, CelebA 100.00, ImageNet-64
      83.42, FashionMNIST 99.66)
    NONE QUALIFIES (33 candidates so far).  But L1s is the first single formula near the bar: ONE linear functional of the profile up to
    block A, one pass stopped at A, the direction taken from the calibration images and their upside-down copies (no OOD image, no label,
    a closed form: two mean vectors and one covariance).  It beats a_A alone (88.2) by 5 points and sees the upside-down images (88.0)
    without a second pass.  Its weak ID is SVHN (90.3), where the whitened distance does the work.
    NOTE for the author: a Fisher direction is a linear discriminant fitted in closed form on the ID images and their upside-down copies;
    whether this is within C1 is the author's decision (decided in I68, where C1 was reworded).
    ROUND 4, PLAN before any number (rule as I59).
    One formula that contains both the whitened distance and L1.  With mu the mean and Sigma the covariance of the upright calibration
    profiles and mu_f the mean of the upside-down ones:   A(c) = -(a - mu*)^T Sigma^(-1) (a - mu*),   mu* = mu + c (mu - mu_f).
    Expanding, A(c) = -d2 + 2c L1 + const: c = 0 is the whitened distance, large c is L1, and the weight of the response term grows by
    itself with the distance between the upright and the upside-down profiles.
    The a-priori choice is c = 1: mu* is the mirror image of the upside-down mean through the ID mean.  Rows (code/gx_eval5.py ->
    runs/ido/final/gx_val2e_s0.json): A1s (c = 1, blocks up to A, one stopped pass) = THE CANDIDATE; A1 (c = 1, all blocks); Ts (c = 0,
    blocks up to A); for understanding only, not eligible: c = 0.5, 2, 4 on the blocks up to A, and A1s with the pooled covariance.
    PART II of I61 (block A per channel): H1-H7 as planned, plus LC (Fisher direction on the channel amplitudes) and AC (A(1) on them).

I64 2026-11-16 11:06 UTC.  ROUND 4 AND PART II OF ROUND 2 SEEN (runs/ido/final/gx_val2e_s0.json, gx_val2c_s0.json; each run once).
    val2, seed 0, natural mean | upside-down || identical history || degraded copies:
      A1s (THE CANDIDATE) 91.50 | 73.6 || 86.18 || 79.8      A1 92.01 | 73.1 || 85.34 || 83.8      Ts 87.84 | 63.5 || 82.95 || 78.8
      not eligible, for understanding:  A0.5s 90.03   A2s 93.27 | 78.2 || 88.13   A4s 94.68 | 82.7 || 90.09 || 77.6   A1sp 91.69
      (per ID, CIFAR-10 SVHN CelebA ImageNet-64 FashionMNIST: A1s 87.98 92.91 100.00 77.29 99.33; A1 85.45 99.07 100.00 76.18 99.35;
       A4s 94.44 95.75 100.00 83.54 99.68)
      block A per channel:  H1 87.80   H2 87.77   H2b 86.84   H3 76.02   H4 85.97   H5 85.06 | 93.2   H6 88.36 | 91.7   H7 77.31
      LC 88.89 | 91.5 || 84.52   AC 78.11   (stronger ridge, for understanding: LCr 89.74, ACr 78.17)
    NONE QUALIFIES (46 candidates so far, and 6 rows for understanding).  What the rows say: (1) the a-priori anti-point (c = 1) is too
    close to the whitened distance; the score rises with c over the whole range tried (87.8, 90.0, 91.5, 93.3, 94.7) and at c = 4 it is
    0.4 below the bar on val2 and 0.2 below it under the identical history, with one formula and one stopped pass, but c = 4 was not
    chosen beforehand.  (2) all blocks instead of the blocks up to A: SVHN 92.9 -> 99.1, the other IDs 1-2.5 points down at c = 1.
    (3) the 256-512 channels of block A carry less than the 34 numbers over depth (best 89.7): the profile over depth is the
    representation to keep.
    A(c) is the Gaussian log-likelihood ratio between the ID profiles N(mu, Sigma) and the upside-down ones with an inflated covariance
    N(mu_f, kappa Sigma), c = 1 / (kappa - 1): kappa -> infinity is the whitened distance, kappa = 1 the Fisher direction.
    ROUND 5, PLAN before any number (rule as I59; code/gx_eval6.py -> runs/ido/final/gx_val2f_s0.json; stored statistics of I59).
    With d2 = (a - mu)^T Sigma^-1 (a - mu), l = (mu - mu_f)^T Sigma^-1 (a - mu), Delta^2 = (mu - mu_f)^T Sigma^-1 (mu - mu_f), p = blocks:
    A-PRIORI CANDIDATES (no constant chosen on val2):
      Z1s, Z1a   (p - d2) / sqrt(2p) + l / Delta: each term in units of its own standard deviation over ID images under the Gaussian model
                 (= A(c) with c = sqrt(p/2) / Delta); s = blocks up to A (one stopped pass), a = all blocks
      TPs, TPa   the one-sided Mahalanobis distance -(d2 - max(l, 0)^2 / Delta^2)
    FOR UNDERSTANDING (not eligible): l alone (Ls, La); A(c) on all blocks for c = 2, 4, 8, 16 and on the blocks up to A for c = 8, 16,
      64; RKs, RKa = log rank of -d2 + log rank of l (what a rank combination of the two would give).
    ORACLE (NOT a method and never a candidate: it reads val2 OOD images with their set labels; two folds, even / odd images): linear and
      quadratic Gaussian discriminants between the val2 ID images and each val2 OOD set / all OOD sets of the ID pooled.  Purpose: to know
      whether any formula on these numbers can be far above 95.
    EXTRACTION ONLY, no score (code/mk_gx_neg.py, code/run_gxn.sh -> runs/ido/gx/<tag>_s0_n.npz): further copies of the calibration
      images that keep the pixels and lose the familiar arrangement: rotated by 90 and by 270 degrees, the four quadrants exchanged
      diagonally, the 4 x 4 patches permuted, and the left half of one image joined to the right half of the next.

I65 2026-11-16 11:09 UTC.  ROUND 5 SEEN (runs/ido/final/gx_val2f_s0.json; run once; mk_gx_neg.py stopped once at fmnist on an assertion
    about the number of axes of the grey arrays, fixed and run for fmnist only).  val2, seed 0, natural mean | upside-down || identical
    history || degraded copies:
      A-PRIORI CANDIDATES:  Z1s 93.92 | 80.1 || 89.36 || 79.9     Z1a 94.04 | 80.3 || 88.63 || 84.0     TPs 88.58     TPa 89.71
      NONE QUALIFIES (50 candidates so far).
      for understanding (c read on val2, not candidates):  blocks up to A:  A8s 95.26 | 85.5 || 91.17 || 75.4    A16s 95.12 || 91.08
      A64s 93.99    Ls (l alone) 93.16 || 88.29        all blocks:  A2a 93.40   A4a 94.69   A8a 95.42 || 90.35   A16a 95.47 || 90.61   La 92.39
      rank combination of -d2 and l:  RKs 94.64 || 89.83    RKa 94.94 || 89.26
      (per ID: A8s 95.12 96.11 100.00 85.39 99.70;  A16a 93.81 99.39 100.00 84.64 99.50)
    So with c between 8 and 16 the ONE formula A(c) = -d2 + 2 c l is at or above both bars, on the blocks up to A (one stopped pass) as
    on all blocks; it is above the rank combination of its own two parts.  But c was read on val2 here: this is a development result,
    not a qualification.  Against the degraded copies it is at 75, on the upside-down set at 86.  The self-normalised weight (Z1) puts
    too little on the response.
    ORACLE (not a method): one linear direction per ID fitted WITH the val2 OOD sets gives 97.25 (CIFAR-10 96.84, SVHN 99.53, ImageNet-64
    89.92), one per OOD set 98.44 (CIFAR-10 98.61, ImageNet-64 93.74); quadratic and the added statistics change little.  CIFAR-100 from
    CIFAR-10: 91.8 at most (the response direction alone has 91.0).  So the 34 numbers hold about 2 points more than any score so far, in
    the DIRECTION; part of it is the file history (SSB-hard 86.0 for a set whose content is ImageNet-like).  What is missing is a better
    direction without OOD images.
    ROUND 6, PLAN before any number (rule as I59; code/gx_eval7.py -> runs/ido/final/gx_val2g_s0.json; runs/ido/gx/*_n.npz of I64).
    Kinds of rearranged copies of the calibration images: v (upside-down), r90, r270, j2, j4, ch.  l_k = (mu - mu_k)^T Sigma^-1 (a - mu)
    with mu_k the mean profile of kind k and mu, Sigma from the untouched calibration images; lbar = the mean of the six l_k.  c = 8 is
    taken from round 5 (read on val2 with the upside-down copy only; stated as such).
    CANDIDATES:   M8s, M8a   -d2 + 16 lbar  (s = blocks up to A, a = all blocks)          LMs, LMa   lbar alone
                  Q8s        -d2 + 16 min_k l_k                                            PKs        the log posterior probability of
                  "untouched" among the seven Gaussians with the common covariance (no constant)
    FOR UNDERSTANDING: each kind alone (l_k), also the two diagonal reflections; -d2 + 16 lbar with the three geometric kinds only (v, r90,
      r270) and with the three rearrangements only (j2, j4, ch); c = 4 and 16 with the six kinds; A8s again (must reproduce 95.26); the
      cosine (in the Sigma metric) between each direction and the ORACLE direction of round 5 (diagnostic, not a method).

I66 2026-11-16 11:12 UTC.  ROUND 6 SEEN (runs/ido/final/gx_val2g_s0.json; run once; the calibration statistics of the two extractions
    agree to 0.002).  val2, seed 0, natural mean | upside-down || identical history || degraded copies:
      CANDIDATES:  M8s 92.59 || 81.81    M8a 91.89 || 80.89    LMs 88.92    LMa 87.92    Q8s 91.32 || 83.63    PKs 94.67 | 85.5 || 90.74 || 65.5
      NONE QUALIFIES (56 candidates so far; PKs is above the bar under the identical history and 0.4 below it on val2).
      for understanding:  each kind alone (l_k):  v 93.16   r90 94.45   r270 94.38   diagonal reflections 94.43, 94.34   j2 84.99   j4 82.85
      ch 83.70;   -d2 + 16 lbar with v, r90, r270 only:  MG8s 95.61 | 85.4 || 91.76 || 76.8  (per ID 95.39 97.36 100.00 85.51 99.78);
      with j2, j4, ch only: 86.66;   six kinds, c = 4: 93.28, c = 16: 91.32;   A8s 95.26 (reproduced).
      cosine with the oracle direction (CIFAR-10, SVHN, CelebA, ImageNet-64, FashionMNIST):  v .85 .60 .87 .72 .81;  r90 .86 .72 .85 .73 .75;
      j2 .68 .51 .84 .03 .61;  j4 .42 .61 .77 -.05 .72;  ch .53 .60 .68 -.03 .89.
    What the rows say: the copies that change the ORIENTATION of the whole image give the direction (a quarter turn is better than
    upside-down: SVHN 96.0 against 91.5, where an upside-down digit is still digit-like); the copies that cut the image into pieces do not
    (their direction points at the cut edges; for ImageNet-64 it is orthogonal to the oracle direction).  The three orientation copies with
    c = 8 are above both bars (95.61, 91.76), but that set of kinds was read here, not planned.
    ROUND 7, PLAN before any number (rule as I59; code/mk_gx_neg2.py, code/run_gxn2.sh -> runs/ido/gx/<tag>_s<seed>_n2.npz;
    code/gx_eval8.py -> runs/ido/final/gx_val2h_s0.json).  The canonical set, with nothing chosen: ALL seven other orientations of the
    square (the symmetry group of the square without the identity): upside-down v, mirrored h, half turn r180, quarter turns r90, r270,
    the two diagonal reflections t1, t2.  New extraction: the mirrored copies of the calibration images (h, and with the upside-down pass
    r180).  mu, Sigma from the untouched calibration images; mu_k the mean profile of orientation k; l_k, lbar (mean of the seven).
    CANDIDATES (s = blocks up to A, one stopped pass on the test image; a = all blocks):
      O8s, O8a    -d2 + 16 lbar   (c = 8 from round 5, as stated in I65)
      PKo, PKoa   the log posterior probability of "upright" among the eight orientations, Gaussians with the common covariance Sigma:
                  -log(1 + sum_k exp(-(l_k + Delta_k^2 / 2)))   (no constant at all)
      JFo         PKo - d2 / 2: the log density of "this profile and upright" (no constant)
      JRo         the same with d2 restricted to the subspace spanned by the seven directions (no constant)
    FOR UNDERSTANDING: lbar alone; c = 4 and 16; the five orientations without h and r180; l_h and l_r180 alone; MG8s again (95.61).
    Noise seeds 1 and 2: the extractions are started now; no score is computed from them unless a candidate qualifies with seed 0.

I67 2026-11-16 11:16 UTC.  ROUND 7 SEEN (runs/ido/final/gx_val2h_s0.json; run once).  val2, seed 0, natural mean | upside-down ||
    identical history || degraded copies:
      CANDIDATES:  O8s 95.54 | 85.1 || 91.66 || 77.2      O8a 95.34 | 84.9 || 90.72 || 80.5      PKo 94.45 | 87.1 || 90.19 || 68.6
                   PKoa 93.19 || 89.75      JFo 91.16 || 85.85      JRo 91.12 || 85.22
      (per ID, CIFAR-10 SVHN CelebA ImageNet-64 FashionMNIST: O8s 95.38 97.29 100.00 85.24 99.80)
      for understanding:  lbar alone 94.43 || 90.08;  c = 4: 94.95 || 90.43;  c = 16: 95.58 || 91.92;  five orientations (without mirrored
      and half turn) 95.59 || 91.68;  v, r90, r270: 95.61 (reproduced);  l_h alone 78.80 (CIFAR-10 52.2, ImageNet-64 49.9: a mirrored
      natural image is an ID image; Delta_h 0.14, 0.08);  l_r180 alone 92.67.
    RULE OF I59: O8s (planned in I66 with the canonical set of seven orientations and c = 8 stated beforehand) is a single formula
    without ranks, 95.54 >= 95.07 on val2 and 91.66 >= 90.33 under the identical history.  O8s QUALIFIES ON SEED 0 (so does O8a; O8s was
    the first-listed and is the cheaper one: one pass stopped at block A).  62 candidates were tried on val2 to get here; that is the
    reason the next two steps exist.  The versions without any constant (PKo, JFo, JRo) do not qualify.
    NEXT (I61): the same script with noise seeds 1 and 2 (-> runs/ido/final/gx_val2h_s1.json, _s2.json).  O8s holds if its three-seed
    means are at least the two bars.  Only then: a new entry with the frozen code, its check against these development numbers, and ONE
    evaluation on the test splits (which have been seen with other methods, never with this score; no unused test images remain for
    CIFAR-10).

I68 2026-11-16 11:32 UTC.  SEEDS 1 AND 2 SEEN; C1 REWORDED BY THE AUTHOR; FREEZE OF THE NEW SCORE; PLAN OF ITS SINGLE EVALUATION.
    (1) Seeds (code/gx_eval8.py run once per seed).  val2 natural mean || identical history, seeds 0, 1, 2:
        O8s  95.54, 95.55, 95.56 -> 95.55   ||  91.66, 91.88, 91.78 -> 91.77        upside-down 85.1, 85.2, 85.2
        (O8a 95.34, 95.37, 95.33 || 90.72, 91.09, 90.91)
        O8s HOLDS on the three seeds.
    (2) C1.  The author (11:29 UTC), asked whether a direction taken in closed form from the ID images and their re-oriented copies fits
        C1: if C1 means nothing trained by gradient descent, no network, no human label, no OOD image, it is within C1, and the wording
        of C1 is changed to that.  From here on C1 reads:
        C1  One forward pass per test image (sigma_1 = 0.01) through the frozen diffusion model; nothing trained by gradient descent, no
            further network, no human label, no OOD image.  ("No trained classifier" is not claimed, in the protocol or in the paper.)
    (3) FREEZE.  code/ol_score.py 8c43c1d6d2c4: the profile a of the amplitudes of the blocks up to A (one pass, stopped at A in a
        deployment); m, C from the val1 ID images (ridge 1e-3 x the mean variance; blocks with zero variance left out); m_o = the mean
        over the seven other orientations of the square of the mean profile of the same images;  score = -(a - m*)^T C^-1 (a - m*),
        m* = m + 8 (m - m_o).  Everything in it was fixed on val2 and the val1 ID images: the statistic, sigma_1 and the rule for block A
        come unchanged from the author's earlier work; c = 8 was read on val2 (round 5) and stated before round 7; the set of
        orientations is the whole group, not a selection.  Working name: OL.
        Evaluation code: code/ol_final.py 4fa29b1f13d1, code/ol_devcheck.py daaae4fd164f, code/run_ol.sh 3662ddbd1a74.
    (4) CHECK BEFORE ANY TEST IMAGE: ol_score.py on the val2 sets (label dev, seed 0, original and identical history) must reproduce rows
        O8s and O8a of runs/ido/final/gx_val2h_s0.json within 0.05 (code/ol_devcheck.py).  If it does not, the code is corrected and
        frozen again here before any test image is read.
    (5) THE SINGLE EVALUATION (code/ol_final.py, run once -> runs/ido/final/ol_final.json): the six test splits (server test7 ... test2),
        noise seeds 0-2, the original images and the identical history.  DISCLOSED: every one of these splits has been looked at with
        other methods; none with this score, and nothing of this score was chosen on them.  No unused test images are left (CIFAR-10),
        so this is the cleanest evaluation available; it is weaker evidence than a fresh split, and it is said so wherever the numbers are
        used.  Five statements were registered beforehand; four of them compare with methods outside this paper.  The one about OLR alone:
          H4  six-split mean of OL >= 95.0 (original images).
        Expected weakness, stated now: the upside-down ID set (about 85 on val2).
        62 candidates were tried on val2 before this one; the test splits are the only numbers of this score that were not selected.

I69 2026-11-16 12:01 UTC.  THE CHECK AND THE SINGLE EVALUATION OF I68 SEEN.
    Check (code/ol_devcheck.py): ol_score.py reproduces rows O8s and O8a of the development on val2, original and identical history,
    largest difference 0.003.  Then the 30 runs (5 IDs x seeds 0-2 x original / identical history; rc = 0 each) and code/ol_final.py,
    run once (runs/ido/final/ol_final.json).
    Five-ID natural mean AUROC, mean of seeds 0-2; server split names (paper: test7 = test6, ..., test3 = test2, test2 = test):
                          ORIGINAL IMAGES               IDENTICAL HISTORY
                          OL      upside-down           OL      upside-down
        test7            95.54     85.1                91.96     81.1
        test6            95.26     84.3                91.57     80.3
        test5            95.64     85.1                92.09     80.8
        test4            95.31     84.9                91.70     80.9
        test3            95.70     84.8                91.98     80.2
        test2            95.62     84.7                92.02     80.3
        six-split mean   95.51     84.8                91.88     80.6
        ablations, six-split mean:  all blocks 95.39 / 91.13;  l alone 94.39 / 90.11;  -d2 alone 87.53 / 83.38
    REGISTERED STATEMENTS: all five held; H4 (about OLR alone) is true: 95.51 >= 95.0.
    WHAT MUST BE SAID WITH THESE NUMBERS: all six splits had been seen with other methods; nothing of OL was chosen on them, and 62
    candidates were tried on val2.  The upside-down ID set is the weak point (84.8).

I70 2026-11-16 12:06 UTC.  THE PAPER IS WRITTEN AROUND OL (the author at 12:03 UTC: "write the paper; it is a standalone paper").
    PLAN of the analyses the paper needs, before any of their numbers.  They are made AFTER the single evaluation of I69; no number of
    that evaluation changes, nothing is selected with them, every row is reported.
    X1  image quality: the ID images of test7 against their seven degraded copies (I47), ol_score.py label dq, seeds 0-2; CFS and CFS-kNN
        from the stored runs/ido/cfsdq (I54).
    X2  what the model contributes: the model with freshly initialised weights, with shuffled weights, and the trained model of another
        dataset (the 16 variant tags of I47), test7, seed 0, label x.
    X3  CFS on the same images: test (server test2; the only test split with stored CFS scores, runs/cfs_l7) with the original images and
        under the identical history (runs/ido/cfseq, I50); paired bootstrap (2,000, rng 20261121, mean of the seeds) of OL - CFS and
        OL - CFS-kNN, per ID and five-ID mean, wins / ties / losses over the pairs; and OL against every published row of the paper's
        table on test (ol_vs_all_test2.json).
    X4  256x256 (ADM): in256c and its 112-px control in256d, seeds 0-2, label hr; CFS and CFS-kNN from the stored scores; paired bootstrap
        (rng 20261122).  The cautions of I56-I57 about this tier (file size, compression and chroma subsampling separate the sets) stay.
    X5  CPU time: the timing script of the shared code base, the same session and protocol as for CFS; method ol = one pass stopped at A,
        the amplitudes, one Mahalanobis distance.
    X6  development analyses on val2 from the stored profiles (code/olx_dev.py; seeds 0-2): c; the set of orientations; the number of
        calibration images; the statistic (amplitude / log std); blocks up to A / all; the versions without a constant.  val2 only.
    X7  per-set AUROC and FPR95 of OL on the six splits from the stored scores of the single evaluation (tables of the supplement).
    Code: code/run_olx.sh, code/olx_eval.py (parts dq, var, cfs, hr, sets), code/olx_dev.py.  The paper states the reworded C1, that c = 8
    was read on val2, that 62 candidates were tried on val2, and that the six test splits had been seen with other methods.

I71 2026-11-16 12:30 UTC.  THE ANALYSES OF I70 SEEN, EXCEPT X4 (256x256, still running).  All after the single evaluation; nothing selected.
    X3  test (server test2), same images: OL 95.62, CFS-kNN 87.99, CFS 85.63; OL - CFS-kNN +7.63 [+6.99, +8.28] (ol_vs_all_test2.json, the
        51 pairs, W/T/L 37/14/0), OL - CFS +9.99 [+9.29, +10.66] (42/9/0).  Against the eight other published rows OL wins 34-50 of 51
        pairs; it loses 11 pairs, all with SVHN as ID (NLL, TT, MSMA*, DiffPath near 99-100 there; SVHN upside-down against EigenScore).
        Identical history (olx_cfs.json): OL 92.02, CFS-kNN 84.40, CFS 82.45; OL - CFS-kNN +7.62 [+6.85, +8.42], W/T/L 36/3/4 of 43 (the
        four losses are the four FashionMNIST pairs, 98.9 against 99.9).  The drop is the same for both (3.6).
    X1  ID images of test7 against their degraded copies, five-ID mean (blur 0.5, 1.0, JPEG 90, 75, 50, down 3/4, 1/2):
        OL 62.7 90.0 69.0 74.9 79.8 72.5 86.7;  CFS-kNN 64.0 88.7 64.7 68.5 76.6 74.4 84.8;  CFS 57.4 88.9 69.7 73.0 73.0 65.8 81.9;
        the two parts of OL: l 58.7-81.4 (mean 70.1), -d2 66.2-91.6 (mean 78.4).  OL reacts to image quality about as CFS does (higher
        than CFS-kNN on 5 of 7); ImageNet-64 separates JPEG 90 at 98.5.
    X2  test7, seed 0: trained 95.5; freshly initialised weights 72.0; shuffled weights 67.9 (l: 94.5 -> 57.9; -d2: 87.3 -> 73.1).  The
        trained model of another dataset, six pairs: 84.2 (their own models 97.0); l 64.1, -d2 84.9; CIFAR-10 through the SVHN model 60.0,
        SVHN through the CIFAR-10 model 99.4 (the distance term separates digits from natural images with any model).
    X5  CPU, one session (bench_ol.jsonl): OL 0.0680 s per image (forward to A 0.0662), CFS stopped 0.0866, CFS-kNN stopped 0.0902,
        unstopped 0.181 / 0.185; measured while other GPU jobs ran (load about 3 of 24 cores).
    X6  val2, seeds 0-2 (olx_dev.json): c = 0, 1, 2, 4, 8, 16, 32, 64, inf: 87.85 91.93 93.63 94.95 95.55 95.60 95.30 94.99 94.48;
        orientations (c = 8): upside-down only 95.25, three flips 94.99, quarter turns and diagonals 95.53, six without the mirror 95.61,
        seven 95.55; all blocks 95.35; log std 95.19; n = 50, 100, 200, 500, 1000: 93.28 93.76 94.41 94.95 95.61; without a constant:
        posterior 94.50, with the density 91.16, in the span 91.15.
    X7  olx_sets.json (per set).  olx_info.json: p = 23, 8, 25, 23, 22; Delta = 1.77, 1.10, 8.84, 0.90, 2.78.
    NOTE  olx_eval.py part cfs stopped once in the identical-history branch (index convention of the processed sets) before writing
        anything, and was corrected.  Two runs of ol_score.py on the same images do not give bit-identical scores (GPU arithmetic,
        cudnn.benchmark): the ID scores of the dq runs differ from those of the evaluation by at most 0.6% of their standard deviation
        (Spearman >= 0.999994; exactly equal in 5 of the 15 runs).  No reported AUROC mixes scores of two runs.

I72 2026-11-16 13:07 UTC.  THE PAPER (resume) IS WRITTEN AROUND OL; a draft (ol1) was delivered without the 256x256 numbers (X4 running).
    The paper states: the six test splits were looked at before with other methods (none of OL's choices was made on them), c = 8 was
    read on val2, 62 candidates were tried on val2 (and variants for understanding), the amplitude, sigma_1 and the rule for block A come
    unchanged from the author's earlier work, and the registered statement "six-split mean >= 95.0" held.  The development criterion is
    described by its two values.
    PLAN X8, before its number (code/olx_cos.py -> runs/ido/final/olx_cos.json; diagnostic, val2, seed 0, stored statistics): the cosine in
    the S metric between OL's own direction P (m - m_o) on the blocks up to A and the linear discriminant direction between the val2 ID
    images and the pooled val2 OOD sets, per ID; and the mean of the cosines with the directions fitted on each OOD set.

I73 2026-11-16 13:23 UTC.  REVIEW OF THE DRAFT ol1 BY THE AUTHOR (13:20 UTC) and the plan that follows, written before the new numbers.
    The author: (1) give numbers for "the response becomes weaker": the share of blocks where the mean of the re-oriented images is below
    the ID mean, and where the OOD mean is; (2) the text on the seven orientations contradicts itself: write it as the candidate of the 8th
    round; (3) measure the comparison of directions with OLR's own direction (the mean over the seven orientations, blocks up to A); (4)
    fill the placeholders (4.4, limitation 7: 256x256); (5) optional: check the word "likelihood ratio" by measuring whether the
    covariance of the re-oriented profiles is about 1.125 times that of the ID profiles; (6) rename the method OLR, written "OLR(Ours)"
    in bold in the tables.
    PLAN (code/olx_diag.py -> runs/ido/final/olx_diag.json; val2, seeds 0-2, stored statistics, blocks up to A; descriptive):
      X9   share of blocks with m_o,h < m_h, with the mean of each val2 OOD set below m_h (mean, min, max over the sets), with the val2 ID
           mean below m_h (reference); the same at block A.
      X8b  cosine (S metric) between P (m - m_o) and the pooled LDA direction (val2 ID against the pooled val2 OOD sets) and the mean
           cosine with the per-set LDA directions; AUROC on two folds of OLR, of its direction term l, of the pooled and per-set LDA.
      X10  kappa_tr = tr(P C) / p and kappa_det = (det C / det S)^(1/p) for C = the covariance of the seven orientations pooled around m_o,
           the mean within-orientation covariance, and the val2 ID images (reference); the implied c = 1 / (kappa - 1).
      Whatever X10 shows is written in the paper; if kappa is far from 1.125 the paper says that the likelihood ratio is that of the
      model with kappa fixed to 1 + 1/c, not of a fitted model, and the question of the title is put to the author.
I73 continued, 2026-11-16 13:24 UTC.  olx_diag.py run once (runs/ido/final/olx_diag.json).  Seen:
      X9 (share of blocks up to A below the ID mean; re-oriented, OOD sets mean (min-max), val2 ID as reference): CIFAR-10 .91 .80 (.61-1.00)
      .94; SVHN .62 .31 (.25-.71) .00; CelebA .60 .80 (.68-1.00) .56; ImageNet-64 .80 .88 (.65-1.00) .17; FashionMNIST .59 .69 (.59-.82) .59.
      At block A the re-oriented mean and the mean of EVERY val2 OOD set are below the ID mean, for all five IDs and all seeds.
      The reference shows that the sign over the blocks is decided by shifts far below the spread (val2 ID .94 and .00).  Hence, BEFORE
      their numbers: code/olx_fall.py counts with a margin (z = difference / SD of the ID amplitude at the block; falls: z < -0.2, rises:
      z > +0.2) and gives z at block A.  The paper reports these, with the plain shares and the reference in the supplement.
      X8b (blocks up to A, two folds, seeds 0-2): OLR 95.48 97.30 100.00 85.19 99.79; its direction term 94.94 95.79 100.00 81.93 99.75;
      pooled LDA 96.70 98.97 100.00 89.41 99.97; per-set LDA 98.48 99.50 100.00 92.54 99.99; cosine OLR direction / pooled LDA .87 .71
      .86 .73 .82.
      X10 (kappa_tr pooled, within, val2 ID; kappa_det pooled): CIFAR-10 .87 .83 .98; .83  SVHN .96 .92 1.00; .93  CelebA 1.65 .88 .95; .94
      ImageNet-64 .83 .82 .99; .84  FashionMNIST 1.00 .95 1.00; .79.  The re-oriented profiles are about as spread as the ID ones (0.8-1.0;
      CelebA pooled 1.65 from the spread between the orientations), not 1.125 times: the likelihood ratio of the paper is that of a model
      with kappa fixed to 1 + 1/c (c a weight), not of a fitted model.  This is written in the paper; the title question goes to the author.

I74 2026-11-16 13:30 UTC.  X4 and the addendum of I73 SEEN; the resume ol2 delivered (method renamed OLR; "OLR(Ours)" in bold in the tables).
    X4 (olx_hr.json; seeds 0-2): in256c OLR 76.30 (near 70.81, far 79.95), upside-down 60.42; all blocks 82.67, l 60.27, -d2 76.42; CFS
    73.40, CFS-kNN 66.19; OLR - CFS +2.90 [+1.31, +4.41] (4/0/2), OLR - CFS-kNN +10.11 [+8.17, +12.00].  112-px control in256d: OLR 78.94,
    CFS 71.03, CFS-kNN 63.35 (+7.92, +15.59).  Delta = 0.55 at 256 (0.9-8.8 for the five models): the score is nearly the distance term.
    The file facts of the sets still separate them from ImageNet (AUROC >= 95.3), so the paper does not read the 256 result as content
    detection.
    olx_fall.json (margin 0.2 SD): re-oriented images: the amplitude falls in 38-65% of the blocks up to A and rises in at most 21%, z at A
    -7.6 to -0.6 (all five fall); OOD sets: falls 26-81%, rises 6-55% (SVHN: more blocks rise than fall), z at A below 0 for all 38 sets,
    below -0.2 for 36; val2 ID images: no block beyond the margin, |z at A| <= 0.08.
    Text changes from the review of I73: "the response becomes weaker" is now "the amplitude falls" with these numbers (section 6.1,
    Table 7; the plain sign shares in Table A4); the seven orientations are described as the candidate of round 8; section 6.5 and Table
    A6 use OLR's own direction on the blocks up to A (X8b); 4.4 and limitation (7) filled; the likelihood ratio is stated as that of the
    model with the covariance ratio fixed at 1.125, with the measured ratios 0.83-1.65 (Table A5).  Title unchanged.  ol2: 17 pages.

I75 2026-11-16 13:46 UTC.  SECOND REVIEW BY THE AUTHOR (13:42 UTC) and the resume ol3 (text only; no new computation).
    (a) The title is changed to Japanese, without "likelihood ratio": 「向きを変えた分布内画像で振幅の下がる方向を知る / 拡散モデルによる
    分布外検知」.  The method name stays OLR.
    (b) The 256x256 result (Delta 0.55, direction term alone 60.3, upside-down 60.4, the score nearly the distance term, the sets separable
    by their file facts) is now stated in the abstract, the introduction and the conclusion as a limit of the method: it works where
    re-orienting the images moves the profile; Delta is computed from ID images only.
    (c) A.3 and Table A2: Table A2 now lists, per round, the candidates (62) and the variants written as non-candidates (35, repeats left
    out) separately with their best values; rounds 6, 7 and 8 had variants above both bars (A16a 95.47 / 90.61, MG8s 95.61 / 91.76, O8s5
    95.59 / 91.68); they were not adopted, and the text says which candidates of the next round were set after seeing them (c = 8 after
    round 6; all seven orientations after round 7).  ol3: 17 pages.

I76 2026-11-16 21:55 UTC.  COPY-EDIT FOR LENGTH (the author at 21:38 UTC: main text plus references within 9 pages; the prose is too long).
    Text and layout only; no new computation; every number unchanged (all generated tables and numbers_ol.tex identical to ol3 except the
    row labels of tab_var_ol.tex).  Main text 18,937 -> 11,506 characters; main text plus references end on page 7; ol4: 14 pages.
    Details moved to the supplement, not dropped; kept: c = 8 set on val2; 62 candidates and 35 variants; the test splits were seen before
    with other methods; the sensitivity to image quality and resize history; the 256x256 limit; the covariance ratio 0.83-1.65.

I77 2026-11-16 21:59 UTC.  TITLE SHORTENED (the author chose the proposed one): 「向きを変えた分布内画像による拡散モデルの分布外検知」.
    Text only.  ol5: 14 pages.

I78 2026-11-17 04:19 UTC.  TERM RENAMED (the author at 04:04 UTC: a 1x1 matrix is a scalar, not a vector; the term meant is 「方向の項 ℓ」).
    ell = (mu - mu_o)^T P (a - mu) is a product of 1xp, pxp and px1 matrices, i.e. a scalar; "direction term" suggested a vector.  Renamed
    方向の項 -> 内積の項 everywhere (23 places) and 方向の手がかり -> 向きの手がかり; one sentence added after eq. 11 (d^2, ell and Delta^2 are
    1x1 matrices, i.e. scalars).  Text only; every number unchanged.  ol6: 14 pages.  The slides carry the same rename.

I79 2026-11-17 04:50 UTC.  NOTATION FIXES AND AN ILLUSTRATED RESUME (the author at 04:05 UTC: a resume that explains the method with
    pictures; the text is hard to follow).
    (a) ol7 (text only; every number unchanged): the channel index of eqs. (1)-(2) is j, not c (c is the constant c = 8); the profile is a
    column, a(x) = {a_1(x), ..., a_p(x)}^T (p x 1).  ol7: 14 pages.  Slide 3 (eq. 1) uses j as well.
    (b) Illustrated resume (zukai/; MIRU class, one column, 5 pages, 6 figures, 2 tables): one figure per step, CIFAR-10 values only.
    Figure 4 draws the profile space to scale in whitened coordinates (mu_o at -Delta, mu* at c Delta = 14.1); the clouds, the channel map
    and the curves of Figure 3(b) are schematic and say so.  The expansion s = -d^2 + 2c ell - c^2 Delta^2 is derived line by line.

I80 2026-11-17 05:42 UTC.  THE PUBLIC REPOSITORY (the author at 05:08 UTC: put the code, the versions and the TeX into
    github.com/omotes24/OLR; then, asked: leave the e-mail address out; OLR only; leave out everything about the methods used as
    yardsticks of the development).  Nothing is recomputed.
    (a) paper/ = ol7; versions/ol1.pdf ... ol7.pdf; the git tags ol1 ... ol7 hold each version's source.  Every PDF is rebuilt from its
    own source without the e-mail address of the title block; every generated number and table is byte-identical to the delivered
    version (checked for all seven), and the texts differ only in that address.  The numbers file of the shared timing session is
    reduced to the three numbers the resume uses (numbers_bench.tex); the output is unchanged.
    (b) Result files of paper/gen/data/ol: the entries about methods outside the paper are removed (their rows in the development files,
    their rows and summaries in ol_final.json, whose registered statements are reduced to H4; one line of bench_rf256.jsonl); the log of
    the single evaluation is left out (each of its lines compares with them).  The generator then checks H4 only; every output unchanged.
    (c) code/: the OLR scripts as run, with the server's project root and interpreter replaced (tools/subs.py), and with the lines that
    computed or named the yardsticks removed (the development scripts, the evaluation script ol_final.py reduced to its OLR part, a few
    comments); the md5 sums recorded above are those of the files as run.  The comparison with the published rows (ol_vs_all.py) and the
    timing script are not published: both are generated from scripts that also contain methods outside the paper.  Their results are in
    paper/gen/data/ol.
    (d) slides/ and zukai/: as delivered; the paths of the slide scripts are made relative to the repository (the deck built from
    the repository is part-for-part identical to the delivered one).
    (e) This file replaces, for the public, the entries I59-I80 of the log.
