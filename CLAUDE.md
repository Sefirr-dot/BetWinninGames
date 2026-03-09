# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the app

```bash
# Auto-detect weekend window (Fri–Mon), writes visualizador/data/predictions.js
python main.py

# Single specific date
python main.py --date 2026-03-07

# Filter to one league
python main.py --league PL   # PL · PD · BL1 · FL1

# Force re-fetch (ignore SQLite cache)
python main.py --no-cache

# Regenerate tracker_data.js without calling the API
python tracker.py --no-update

# Walk-forward backtest (writes backtest_YYYY-MM-DD.txt + backtest_data.js)
python backtest.py --league PL --seasons 2023 2024
python backtest.py --league ALL --seasons 2023 2024        # all 4 leagues at once
python backtest.py --league PL --seasons 2024 --min-train 150

# Seed picks_history.db from backtest results (activates calibrator + weight optimizer)
# WARNING: do NOT re-train meta_learner after --seed-db (see meta_learner section)
python backtest.py --league ALL --seasons 2023 2024 --seed-db
python tracker.py --no-update   # re-trains calibrator after seeding

# Re-train meta_learner manually (only on real live picks)
python -c "from algorithms.meta_learner import train; print(train('cache/picks_history.db'))"
```

After running, open `visualizador/index.html` directly in a browser (no server needed — data is injected via `<script src>`).

## Dependencies

Python 3.14+. Install with:
```bash
pip install -r requirements.txt
```
Pinned to `>=` bounds (not exact versions) because numpy<2.0 and scipy<1.13 don't support Python 3.14.

## API keys

All keys live in `config.py`:
- `API_KEY` — football-data.org (historical match data, free tier)
- `ODDS_API_KEY` — the-odds-api.com (bookmaker odds, free tier: 500 req/month). Skip-if-fresh protects quota: CSVs younger than `CACHE_TTL_HOURS` are never re-fetched.
- `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` — optional; silently skipped when empty. `TELEGRAM_MIN_STARS` (default 4) filters which picks are sent.
- `OLLAMA_MODEL` / `OLLAMA_BASE_URL` — local Ollama for AI Advisor (default `qwen3.5:9b` at `http://localhost:11434`).

`MODEL_WEIGHTS` is validated at import time — raises `ValueError` immediately if it doesn't sum to 1.0.

## Architecture

### Data flow (main.py)

1. **Fetch historical** — `fetcher.py` loads past seasons from `football-data.org/v4`. Cached indefinitely in `cache/football_data.db`. Live match lookups use a 6 h TTL.
2. **Augment** — `fdco_fetcher.augment_historical()` appends seasons 2020–2022 from football-data.co.uk (CSV, no auth). Dedupes by `(homeTeam, awayTeam, utcDate)`.
3. **Enrich with xG** — `understat_fetcher.enrich_with_xg()` adds `_xg_home`/`_xg_away` in-place. Cached in `cache/understat_xg.db`; fetched once per (league, season).
4. **Fit models** — `dixon_coles.fit_per_league()` + `elo.build_ratings()` + `elo.build_split_ratings()` run once on the full corpus.
5. **Fetch odds** — `odds_fetcher.fetch_window()` writes `odds/YYYY-MM-DD.csv` per date. Manual CSV placement is also supported.
6. **Predict** — `ensemble.predict_match()` per match → `rank_predictions()`.
7. **AI Advisor** — `ai_advisor.enrich_predictions()` fetches Google News RSS per match and calls Ollama to detect injuries/suspensions. Runs for picks ≥ `AI_ADVISOR_MIN_STARS` stars. Requires `ollama serve` running in background. Fails silently when Ollama is unreachable.
8. **Report** — `value_detector.find_edges()` → `reporter.generate_js()` → `db_picks.save_picks()` → `tracker.run_tracker()`.

Predictions are wrapped as `{"match_info": <api_dict>, "prediction": <ensemble_dict>}`. `rank_predictions()` accesses score via `p.get("prediction", p)`.

### algorithms/ sub-models

| Module | Weight | Notes |
|---|---|---|
| `dixon_coles` | 45% | Bivariate Poisson MLE + Sarmanov theta. `fit_per_league()` returns `{"PL": params, ..., "_global": params}`. `predict()` returns `lambda_`, `mu_`, `over25`, `most_likely_score`, `prob_matrix` (numpy array). |
| `elo` | 20% | Home bonus applied at predict time only. `build_split_ratings()` → venue-specific home/away dicts. League-specific bonus via `ELO_HOME_BONUS_BY_LEAGUE`. |
| `form` | 20% | Exponential decay over last `FORM_WINDOW` matches, SoS-adjusted via Elo. Blends 60% actual goals + 40% xG. |
| `h2h` | 5% | Conditional on `n_h2h >= H2H_MIN_MATCHES`. Uses **time-based decay**: `H2H_YEARLY_DECAY^(days_elapsed/365)` — a match 2 years ago weights 0.49×. Accepts `reference_date` param (passed by ensemble). |
| `btts` | 5% | Poisson exact using fatigue-adjusted λ/μ. |
| `corners` | 5% | Proxy driven by λ/μ. |
| `fatigue` | n/a | Multiplicative penalty on λ/μ. ≥7 days → 1.0; ≤1 day → 0.82. |
| `cards` | n/a | Display-only linear proxy. Not tracked for accuracy. |
| `meta_learner` | override | XGBoost (25 features, 80/20 train/val split, early stopping 25 rounds). **Only trains on `source='live'` picks** — never on backtest seeds. Activated when `cache/meta_learner.pkl` exists. When active, skips Platt calibration. |

### Calibrator and weight optimizer (auto-loading)

Loaded at `ensemble.py` import time:
- `cache/calibrator.json` → Platt scaling post-blend (requires ≥200 resolved picks). Safe to train from backtest seeds.
- `cache/model_weights.json` → overrides `MODEL_WEIGHTS` (requires ≥50 resolved picks). Safe to train from backtest seeds.

**meta_learner WARNING**: If predictions look wrong after `--seed-db`, delete `cache/meta_learner.pkl` to revert to the weighted ensemble. The meta_learner must only be trained with real `source='live'` picks to avoid distribution shift from historical seasons.

### Picks persistence (`db_picks.py` + `tracker.py`)

`db_picks.py` manages `cache/picks_history.db`. Schema:

```sql
picks (match_id PK, run_date, match_date, home_team, away_team, league,
       prob_home, prob_draw, prob_away, stars, best_outcome, best_prob,
       over25, btts, fair_odds, market_odds,
       actual_result, actual_over25, actual_btts, result_fetched_at,
       sub_preds TEXT,   -- JSON: {dc, elo, form, h2h, context}
       source TEXT)      -- 'live' (main.py) | 'backtest' (--seed-db)
```

- `save_picks(..., source='live')` — called by `main.py` (default).
- `save_picks(..., source='backtest')` — called by `backtest.seed_picks_db()`.
- `get_real_picks()` — returns only `source='live'` rows; used by `meta_learner.train()`.
- `init_db()` runs migrations automatically (adds `sub_preds`, `source` if absent).

`tracker.run_tracker()` resolves pending picks, then calls `maybe_fit_calibrator()`, `maybe_optimize_weights()`, and `maybe_train_meta_learner()` (real_only=True).

### Backtest (`backtest.py`)

Walk-forward engine with no data leakage. New features:
- `--league ALL` — runs all 4 leagues sequentially, combines results, writes global + per-league metrics.
- `compute_fold_metrics(results)` — groups by `fold_id` (tagged per result), returns per-fold accuracy/ROI/Brier. Included in `backtest_data.js` as `folds: [...]`.
- `generate_backtest_js()` emits `folds` and `per_league` keys for the visualizer.

### AI Advisor (`ai_advisor.py` + `news_fetcher.py`)

- `news_fetcher.fetch_match_news(home, away)` — Google News RSS, no API key, returns up to 6 headlines.
- `ai_advisor.enrich_predictions(predictions, date_str)` — for each qualifying pick: fetches news → builds prompt → calls Ollama (`/api/chat`, `stream:True`, `think:False`) → parses JSON → applies probability adjustments (capped at `AI_ADVISOR_MAX_ADJ` = ±8%) → adds `_ai_note` / `_ai_factors` to prediction dict.
- `think:False` is required for Qwen3 models to disable extended reasoning (otherwise response never arrives in non-streaming mode).
- `reporter.py` writes `aiNote` / `aiFactors` into `predictions.js` and the `.txt` report.

### Key config knobs (`config.py`)

`MODEL_WEIGHTS` must sum to 1.0 (validated at import). Key thresholds: `HIGH_CONFIDENCE_THRESHOLD` (≥62% → 4–5★), `MEDIUM_CONFIDENCE_THRESHOLD` (≥55% → 3★). `DC_XI_BY_LEAGUE` controls temporal decay. `H2H_YEARLY_DECAY = 0.70` (annual decay for H2H time-weighting). `MARKET_BLEND_WEIGHT = 0.20` blends bookmaker implied prob. `VALUE_BET_EDGE_THRESHOLD_BY_LEAGUE` sets minimum edge per league (PL/PD/FL1=8%, BL1=12%).

### Value bet detector

`algorithms/value_detector.py` compares `model_prob` vs `1/bookmaker_odds`. Reports bets where `edge >= threshold`. Dynamic threshold: `VALUE_BET_EDGE_THRESHOLD + (5 - stars) * VALUE_BET_EDGE_STEP`. Kelly fraction capped at 25%. Team name matching: `_normalize()` (strips "FC", "CF" etc.) with exact-then-substring fallback.

### Output files

| File | Written by | Content |
|------|-----------|---------|
| `visualizador/data/predictions.js` | `reporter.generate_js()` | `ALL_MATCHES`, `DATES`, `DATE_LABELS` JS globals |
| `visualizador/data/tracker_data.js` | `tracker.generate_tracker_js()` | `TRACKER_PICKS`, `TRACKER_METRICS` JS globals |
| `visualizador/data/backtest_data.js` | `backtest.generate_backtest_js()` | `BACKTEST_DATA` with metrics + bankroll curve + `folds` + `per_league` |
| `backtest_YYYY-MM-DD.txt` | `backtest.generate_report()` | Human-readable metrics + per-fold table |
| `odds/YYYY-MM-DD.csv` | `odds_fetcher.py` | Bookmaker 1X2 odds (`home_team,away_team,odds_1,odds_x,odds_2`) |

### ALL_MATCHES schema (predictions.js)

Each object: `date`, `home`/`away`/`homeShort`/`awayShort`, `league`, `time`, `stars`, `prob1`/`probX`/`prob2` (%), `over25`/`over25yn`, `btts`/`bttsyn`, `corners`/`cornersOver`, `cards`/`cardsHome`/`cardsAway`/`cardsOver`, `goalsHome`/`goalsAway`, `eloConf`, `eloHome`/`eloAway`, `formHome`/`formAway`, `bestScore`/`bestScoreProb`, `fairOdds`, `marketOdds`, `h2h`, `valueBets`, `subModels`, `scoreGrid` (6×6 DC prob matrix), `fatigueHome`/`fatigueAway`, `homePos`/`awayPos`, `aiNote` (string or null), `aiFactors` (array or null).

### Visualizer (`visualizador/index.html`)

Single-file static app. State: `activeDate` (date string / `"ALL"` / `"BEST"` / `"VALUE"` / `"TRACK"` / `"BACK"` / `"WIN"`), `activeLeague`, `sortMode`.

**Best Bets view** (`activeDate === "BEST"`): `getSuggestedParlays(bets)` generates up to 4 tiered parlays ordered safest→riskiest:
- 🟢 **Doble Segura** — 2 legs, prefers 4★+
- 🟡 **Triple Media** — 3 legs, 3★+
- 🟠 **Cuádruple Arriesgada** — 4 legs, 3★+ (falls back to 2★)
- 💰 **Valor EV+** — legs with positive edge vs bookmaker (when value bets exist)

Each parlay shows combined probability, combined fair odds, and risk color coding.

Modal (`renderMatchModal(m)`) shows sub-model breakdown, score grid heatmap, H2H, value bets, and AI advisor note when present.

### Weekend date logic (`main.py: weekend_dates()`)

- Mon–Thu → upcoming Fri–Mon
- Fri/Sat/Sun → this Fri–Mon
- Mon (edge case) → Mon only

## Notes

### Windows / Unicode
All entry-point scripts call `sys.stdout.reconfigure(encoding="utf-8")` at startup. Use `->` not `→` in new print statements targeting stdout.

### Bankroll curve staking
`tracker.py` and `backtest.py` use **proportional staking**: `unit_stake = 1 / n_bets`. Keeps the bankroll curve near 1.0. Never switch to flat 1-unit stakes.

### v4.0 roadmap
See `V4_PLAN.md` for the full prioritised task list. Start with **Fase 1** (tracker per-league/market/stars breakdown) before touching any model code — it is the diagnostic foundation everything else depends on.

### meta_learner distribution shift
Training `meta_learner` on `--seed-db` data (historical seasons) causes distribution shift and degrades current-season predictions. The `source` column in `picks_history.db` separates `'live'` (main.py) from `'backtest'` (seed_picks_db). `meta_learner.train(real_only=True)` enforces this. If predictions look wrong: `rename cache\meta_learner.pkl cache\meta_learner.pkl.bak`.
