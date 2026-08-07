# Metric Improvement Plan

## Current Promotion State

`public_synthetic_resnet18_v1` passes every predeclared grouped synthetic
confirmation gate. It is the public champion for the independently generated
eight-class benchmark only.

## Weakest Confirmed Boundary

QuadrantFailure has 83.0% recall and 88.3% precision. Most errors route to
Scratch, EdgeEffect, or CenterCluster. The next model experiment should improve
geometric boundary representation without weakening calibration or class-tail
recall.

## Next Experiments

| Priority | Experiment | Predeclared promotion gate |
|---:|---|---|
| 1 | Add shape-aware augmentation and compare frozen layer4 embeddings | Macro F1 improves by at least 0.01 and every class remains at or above 83% recall on a new confirmation set |
| 2 | Expand pattern-family variation for quadrant/scratch ambiguity | At least 2,000 new disjoint confirmation wafers; minimum class recall at least 85% |
| 3 | Add abstention using validation-only confidence thresholds | Unsafe accepted error below 2% while retaining at least 80% coverage |
| 4 | Rebuild WM-811K evidence from an approved licensed source | Source hash, license record, duplicate/group split, untouched holdout, full metrics, and model hashes required before any real-data claim |
| 5 | Replace the partial STDF parser with a specification-tested library path | Golden little/big-endian and multi-wafer fixtures pass before STDF is exposed in the API |
| 6 | Add a separately trained yield-regression task | MAE, calibration, subgroup error, and production-domain validation required before using the word predictor |

## Non-Goals

- Do not tune on `public_synthetic_evaluation.json` confirmation families.
- Do not relabel synthetic evidence as WM-811K or production behavior.
- Do not treat local latency as a production SLO.
- Do not expose the current hand-written STDF parser through the confirmed API.
- Do not infer wafer yield from the pattern classifier.
