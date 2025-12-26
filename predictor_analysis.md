# predictor.py review

## Intended workflow
* Loads daily parquet/CSV symbol files (optionally via GUI picker) to build a "panel" of engineered features, or resumes from cached panel/processed lists to avoid recomputation.
* Trains LightGBM-based models for 1D (classification with calibration or regression), 3D, and 5D horizons with checkpointing and early stopping, then writes out-of-sample predictions.
* Generates watchlists, SHAP explanations, WhatWorked condition analytics, actionable 5D overlays, and summary Excel/CSV reports.

## Fragile points / failure risks
* Large global state is mutated inside `main` (e.g., CV splits, embargo, estimators) which can make re-entrancy brittle if functions are reused elsewhere.
* Many helper steps catch and suppress exceptions (status writes, GUI fallbacks, cleanup, reports), so silent failures may hide data-quality or write issues until downstream steps misbehave.
* Resume logic assumes presence/consistency of cached artifacts (`panel_cache.csv`, `processed_symbols.txt`, model/joblib and OOS CSVs); partial or corrupted outputs can trigger subtle misalignments when rerunning.
* Condition-combo evaluation iterates over combinations up to `max_combo_size` with limited guardrails; an overly large limit could explode runtime/memory on high symbol counts.

## Performance considerations (5 years × ~2400 symbols)
* Panel construction loads and concatenates per-symbol daily files; ensure `--limit-files`/`--symbols-like` are used for smoke tests and consider pre-filtering illiquid symbols early to reduce memory.
* Increase `--chunk-size` only if I/O permits; smaller chunks reduce peak memory but raise write overhead when writing panel chunks and processed lists.
* Reduce LightGBM tree counts (`--n-estimators-*`) and early stopping rounds for quicker training; 1D/3D/5D models default to 300 trees and 50 rounds, which will scale linearly with symbol/time rows.
* Keep SHAP computation scoped: limit `--shap-max-symbols` or omit horizons from `COMPUTE_SHAP_FOR` to avoid heavy per-symbol explainer calls on large panels.
* WhatWorked analysis thins overlapping rows for 3D/5D; keep `--thin-inference=true` and conservative `--max-combo-size` to cap combinatorial explosion on multi-year panels.

## Implemented improvements
* CSV ingestion now parses timestamps during read and enforces deterministic file ordering before dispatching work to the thread pool, improving prep speed and reproducibility when resuming large universes.
* Checkpointing utilities lazily require `joblib`, so lightweight commands such as `--help` or panel-only runs no longer fail if the dependency is missing; installation guidance is emitted only when model I/O is invoked.
