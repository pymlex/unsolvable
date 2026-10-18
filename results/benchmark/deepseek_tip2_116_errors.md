# DeepSeek tip-2 errors on gold 116

Model: `deepseek/deepseek-v4-flash-0731` via Zveno. Protocol: zero-edit, temperature 0, answers `1.`…`5.`.

Exact match: **97/116 = 0.836**. Mean Jaccard: **0.940**. Invalid predictions: **0**.

DeepSeek справился с задачей хорошо. Он устойчиво разбирает тип 2 на реальных заданиях Решу ЕГЭ, так что его можно использовать для генерации синтетики.

## Where it failed

There are **19** exact-match misses. Most keep high Jaccard: the model usually finds part of the gold set and adds an extra option number or drops one.

| id | gold | pred | Jaccard | extra | missing |
| --- | --- | --- | ---: | --- | --- |
| 46255 | 35 | 25 | 0.333 | 2 | 3 |
| 51823 | 123 | 1235 | 0.750 | 5 |  |
| 51850 | 25 | 5 | 0.500 |  | 2 |
| 52977 | 35 | 345 | 0.667 | 4 |  |
| 58081 | 35 | 5 | 0.500 |  | 3 |
| 60208 | 35 | 235 | 0.667 | 2 |  |
| 60235 | 24 | 124 | 0.667 | 1 |  |
| 45114 | 13 | 134 | 0.667 | 4 |  |
| 45121 | 24 | 2 | 0.500 |  | 4 |
| 45126 | 145 | 1245 | 0.750 | 2 |  |
| 45137 | 34 | 134 | 0.667 | 1 |  |
| 45138 | 34 | 134 | 0.667 | 1 |  |
| 45139 | 234 | 2345 | 0.750 | 5 |  |
| 45155 | 24 | 245 | 0.667 | 5 |  |
| 45157 | 145 | 45 | 0.667 |  | 1 |
| 45158 | 23 | 234 | 0.667 | 4 |  |
| 50647 | 123 | 1235 | 0.750 | 5 |  |
| 60975 | 35 | 1345 | 0.500 | 14 |  |
| 50665 | 235 | 35 | 0.667 |  | 2 |

## Error pattern

- Extra option only: **13**
- Missing option only: **5**
- Both extra and missing: **1**

Typical failure is over-inclusion: the model keeps the gold digits and appends one more number from 1–5. Under-inclusion is rarer. Full swaps are almost absent, so sense understanding is mostly intact.

## Convergence

Final exact Wilson 95% CI width: **0.134**. Final Jaccard CI width: **0.052**.

Plots: `results/benchmark/plots/deepseek_tip2_116_convergence.png`, `results/benchmark/plots/deepseek_tip2_116_ci_width.png`.
