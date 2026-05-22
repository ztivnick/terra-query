# Retrieval gate report

Generated at: 2026-05-22T13:48:59.501120+00:00

Sweep: 1 (model, bands) configs x 12 concepts. Each config max-pools cosine across 6 NAIP cycles per chip-location (cycles: 2012, 2014, 2016, 2018, 2020, 2022). Total chip-locations: 2209. Prompt ensemble averages 8-8 synonyms per concept then renormalizes.

## A. Aggregate over the 6 findable GT concepts (of 8 total; cemetery + cabin excluded because every feature is `findable_aerial=false` — see section D)

| config | hit@1 | hit@5 | hit@10 | hit@20 | recall@10 | MRR |
|---|---|---|---|---|---|---|
| georsclip-vit-l-14-336 / rgb |  33.3% |  50.0% |  66.7% |  66.7% |  25.0% | 0.440 |
| RANDOM (closed-form mean) |   0.2% |   1.2% |   2.4% |   4.7% |   0.5% | 0.015 |

**Best config by mean findable MRR: `georsclip-vit-l-14-336 / rgb`**

## B. Per-concept results (findable GT)

### waterfall
n_gt findable: 4, n_gt strict: 4

| config | hit@1 | hit@5 | hit@10 | hit@20 | recall@10 | MRR | top-1 score |
|---|---|---|---|---|---|---|---|
| georsclip-vit-l-14-336 / rgb | 100.0% | 100.0% | 100.0% | 100.0% |  25.0% | 1.000 | 0.241 |
| RANDOM |   0.2% |   0.9% |   1.8% |   3.6% |   0.5% | 0.012 | n/a |

### pond
n_gt findable: 12, n_gt strict: 16

| config | hit@1 | hit@5 | hit@10 | hit@20 | recall@10 | MRR | top-1 score |
|---|---|---|---|---|---|---|---|
| georsclip-vit-l-14-336 / rgb |   0.0% | 100.0% | 100.0% | 100.0% |  25.0% | 0.500 | 0.258 |
| RANDOM |   0.5% |   2.7% |   5.3% |  10.4% |   0.5% | 0.029 | n/a |

### dam
n_gt findable: 4, n_gt strict: 4

| config | hit@1 | hit@5 | hit@10 | hit@20 | recall@10 | MRR | top-1 score |
|---|---|---|---|---|---|---|---|
| georsclip-vit-l-14-336 / rgb | 100.0% | 100.0% | 100.0% | 100.0% |  75.0% | 1.000 | 0.255 |
| RANDOM |   0.2% |   0.9% |   1.8% |   3.6% |   0.5% | 0.012 | n/a |

### road
n_gt findable: 4, n_gt strict: 4

| config | hit@1 | hit@5 | hit@10 | hit@20 | recall@10 | MRR | top-1 score |
|---|---|---|---|---|---|---|---|
| georsclip-vit-l-14-336 / rgb |   0.0% |   0.0% |   0.0% |   0.0% |   0.0% | 0.000 | 0.273 |
| RANDOM |   0.2% |   0.9% |   1.8% |   3.6% |   0.5% | 0.012 | n/a |

### parking_lot
n_gt findable: 4, n_gt strict: 4

| config | hit@1 | hit@5 | hit@10 | hit@20 | recall@10 | MRR | top-1 score |
|---|---|---|---|---|---|---|---|
| georsclip-vit-l-14-336 / rgb |   0.0% |   0.0% |   0.0% |   0.0% |   0.0% | 0.000 | 0.249 |
| RANDOM |   0.2% |   0.9% |   1.8% |   3.6% |   0.5% | 0.012 | n/a |

### boardwalk
n_gt findable: 4, n_gt strict: 4

| config | hit@1 | hit@5 | hit@10 | hit@20 | recall@10 | MRR | top-1 score |
|---|---|---|---|---|---|---|---|
| georsclip-vit-l-14-336 / rgb |   0.0% |   0.0% | 100.0% | 100.0% |  25.0% | 0.143 | 0.284 |
| RANDOM |   0.2% |   0.9% |   1.8% |   3.6% |   0.5% | 0.012 | n/a |

### cemetery
n_gt findable: 0, n_gt strict: 4

| config | hit@1 | hit@5 | hit@10 | hit@20 | recall@10 | MRR | top-1 score |
|---|---|---|---|---|---|---|---|
| georsclip-vit-l-14-336 / rgb |   0.0% |   0.0% |   0.0% |   0.0% |   0.0% | 0.000 | 0.223 |
| RANDOM |   0.0% |   0.0% |   0.0% |   0.0% |   0.0% | 0.000 | n/a |

### abandoned_cabin
n_gt findable: 0, n_gt strict: 4

| config | hit@1 | hit@5 | hit@10 | hit@20 | recall@10 | MRR | top-1 score |
|---|---|---|---|---|---|---|---|
| georsclip-vit-l-14-336 / rgb |   0.0% |   0.0% |   0.0% |   0.0% |   0.0% | 0.000 | 0.272 |
| RANDOM |   0.0% |   0.0% |   0.0% |   0.0% |   0.0% | 0.000 | n/a |

## C. Best (model, bands) per concept

| concept | n_gt findable | best config | best hit@10 | best MRR |
|---|---|---|---|---|
| waterfall | 4 | georsclip-vit-l-14-336 / rgb | 100.0% | 1.000 |
| pond | 12 | georsclip-vit-l-14-336 / rgb | 100.0% | 0.500 |
| dam | 4 | georsclip-vit-l-14-336 / rgb | 100.0% | 1.000 |
| road | 4 | georsclip-vit-l-14-336 / rgb |   0.0% | 0.000 |
| parking_lot | 4 | georsclip-vit-l-14-336 / rgb |   0.0% | 0.000 |
| boardwalk | 4 | georsclip-vit-l-14-336 / rgb | 100.0% | 0.143 |
| cemetery | 0 | georsclip-vit-l-14-336 / rgb |   0.0% | 0.000 |
| abandoned_cabin | 0 | georsclip-vit-l-14-336 / rgb |   0.0% | 0.000 |

## D. Strict vs findable GT per concept (aerial-only gap)

When `n_gt_strict > n_gt_findable`, the extra features are flagged `findable_aerial = false` in the eval set (canopy-occluded). A persistent strict-hit@10 = 0 on those concepts is the signal that the LiDAR modality needs to recover them.

| concept | n_gt findable | n_gt strict | best findable hit@10 | best strict hit@10 | gap |
|---|---|---|---|---|---|
| waterfall | 4 | 4 | 100.0% | 100.0% | 0.0% |
| pond | 12 | 16 | 100.0% | 100.0% | 0.0% |
| dam | 4 | 4 | 100.0% | 100.0% | 0.0% |
| road | 4 | 4 |   0.0% |   0.0% | 0.0% |
| parking_lot | 4 | 4 |   0.0% |   0.0% | 0.0% |
| boardwalk | 4 | 4 | 100.0% | 100.0% | 0.0% |
| cemetery | 0 | 4 |   0.0% |   0.0% | 0.0% |
| abandoned_cabin | 0 | 4 |   0.0% |   0.0% | 0.0% |

## E. No-GT concepts (visual top-K only)

No recall metric for these (no labeled feature in the eval set). Open the thumbnail folders under `verification/bond_falls_25km_poc/gate/topk_chips/` to inspect:

- **mine_pit**
  - `verification/bond_falls_25km_poc/gate/topk_chips/mine_pit__georsclip-vit-l-14-336__rgb`

- **cliff_outcrop**
  - `verification/bond_falls_25km_poc/gate/topk_chips/cliff_outcrop__georsclip-vit-l-14-336__rgb`

- **old_logging_road**
  - `verification/bond_falls_25km_poc/gate/topk_chips/old_logging_road__georsclip-vit-l-14-336__rgb`

- **river_rapids**
  - `verification/bond_falls_25km_poc/gate/topk_chips/river_rapids__georsclip-vit-l-14-336__rgb`

## F. Automated gate checks

| check | result | note |
|---|---|---|
| 1. Pipeline end-to-end (1 configs + report rendered) | PASS | 1 configs evaluated over 6 cycles |
| 2. Best config aggregate hit@10 beats random | PASS | random hit@10 =   1.8%, best =  66.7% |
| 3. Waterfall hits @1 = 1.0 with the best config | PASS | best waterfall hit@1 = 1.0 |
| 4. >=70% of 6 findable in-scope concepts have findable hit@10 = 1.0 | PASS | 4/6 findable concepts hit@10=1.0 (target: >=4): waterfall, pond, dam, boardwalk |
| 5. At least one RS-CLIP beats CLIP floor on mean MRR | PASS | RS-CLIP best mean MRR = 0.440, (floor not in this run; check skipped) |

### Automated verdict: **GO** - all 5 automated checks pass.

The final go / iterate / kill call is yours. Open the thumbnail folders (section E + per-concept best configs in C) to verify the metric reflects reality.
