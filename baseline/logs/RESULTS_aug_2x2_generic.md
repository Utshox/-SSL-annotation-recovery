# Augmentation × SSL: pooled 5-fold pancreas CT Dice

Root: `<repo root>`

Expected complete coverage: 281 unique out-of-fold volumes per arm.
Means below pool cases, not fold means. Fold means are from the JSON files.

| Arm | Folds found | Folds missing | n cases | Mean Dice | Per-fold mean Dice |
|---|---|---|---:|---:|---|
| A0 | 0, 1, 2, 3, 4 | none | 281 | 0.487588 | 0: 0.488672; 1: 0.521464; 2: 0.494925; 3: 0.479596; 4: 0.453262 |
| C0 | 0, 1, 2, 3, 4 | none | 281 | 0.583130 | 0: 0.583505; 1: 0.621049; 2: 0.559384; 3: 0.574695; 4: 0.577011 |
| A1 | 0, 1, 2, 3, 4 | none | 281 | 0.514996 | 0: 0.565366; 1: 0.565756; 2: 0.484645; 3: 0.461866; 4: 0.496446 |
| C1 | 0, 1, 2, 3, 4 | none | 281 | 0.614356 | 0: 0.629016; 1: 0.623306; 2: 0.610667; 3: 0.611218; 4: 0.597311 |

## Paired contrasts

Each contrast uses only case IDs shared by all involved arms.
95% percentile paired bootstrap CIs: 20,000 resamples.
Two-sided sign-flip tests: 20,000 flips, p = (extreme + 1) / 20,001.
Seed 0 per contrast; improved means a strictly positive difference (ties excluded).

| Contrast | Definition | n | Mean difference | 95% CI | Improved / n | p-value |
|---|---|---:|---:|---|---:|---:|
| aug gain (supervised) | A1 - A0 | 281 | 0.027408 | [0.012717, 0.042197] | 152 / 281 | 0.000300 |
| aug gain (SSL) | C1 - C0 | 281 | 0.031226 | [0.018262, 0.044208] | 180 / 281 | 0.000050 |
| SSL gain without aug | C0 - A0 | 281 | 0.095542 | [0.083695, 0.107771] | 235 / 281 | 0.000050 |
| SSL gain with aug (her main contrast) | C1 - A1 | 281 | 0.099360 | [0.087648, 0.111074] | 250 / 281 | 0.000050 |
| interaction | (C1 - A1) - (C0 - A0) | 281 | 0.003818 | [-0.011570, 0.018978] | 142 / 281 | 0.621769 |
| total | C1 - A0 | 281 | 0.126768 | [0.110993, 0.142635] | 243 / 281 | 0.000050 |

## Reading

- aug gain (supervised): positive.
- aug gain (SSL): positive.
- SSL gain without aug: positive.
- SSL gain with aug (her main contrast): positive.
- interaction: null (CI spans 0).
- total: positive.
