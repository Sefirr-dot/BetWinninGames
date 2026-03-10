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
python backtest.py --league ALL --seasons 2023 2024
python backtest.py --league PL --seasons 2024 --min-train 150

# Seed picks_history.db + retrain calibrator + weight optimizer
python backtest.py --league ALL --seasons 2023 2024 --seed-db
python tracker.py --no-update

# Re-train meta_learner manually (only on real live picks, never on seeds)
python -c "from algorithms.meta_learner import train; print(train('cache/picks_history.db'))"

# Pre-train draw model from backtest results (runs automatically at end of backtest)
python backtest.py --league ALL --seasons 2023 2024   # → also writes cache/draw_model.json

# Re-train draw model manually on live picks (>= 50 resolved required)
python -c "from algorithms.draw_model import train; print(train('cache/picks_history.db'))"

# Re-train over25 calibrator manually (>= 50 resolved required)
python -c "from algorithms.over25_model import train; print(train('cache/picks_history.db'))"

# Regenerate results.js for the frontend bankroll tracker (auto-runs inside tracker.py)
python -c "import db_picks, tracker; picks=db_picks.get_all_picks('cache/picks_history.db'); tracker._save_results_js(picks)"
```

After running, open `visualizador/index.html` directly in a browser (no server needed).

## Dependencies

Python 3.14+. Install with `pip install -r requirements.txt`.
Pinned to `>=` bounds because numpy<2.0 and scipy<1.13 don't support Python 3.14.

## API keys

All keys live in `config.py` (gitignored — copy from `config.example.py`):
- `API_KEY` — football-data.org (historical + lineups, free tier)
- `ODDS_API_KEY` — the-odds-api.com (500 req/month). CSVs younger than `CACHE_TTL_HOURS` are never re-fetched.
- `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` — optional; silently skipped when empty.
- `OLLAMA_MODEL` / `OLLAMA_BASE_URL` — local Ollama for AI Advisor (default `qwen3.5:9b`).

`MODEL_WEIGHTS` is validated at import time — raises `ValueError` if it doesn't sum to 1.0.

## Architecture

### Data flow (main.py)

1. **Fetch historical** — `fetcher.py` loads past seasons from football-data.org/v4. Cached indefinitely in `cache/football_data.db`.
2. **Augment** — `fdco_fetcher.augment_historical()` appends seasons 2020–2022 from football-data.co.uk CSVs. Also parses referee names (`_referee`), corner counts (`_hc`/`_ac`/`_total_corners`) and yellow cards (`_home_yellow`/`_away_yellow`).
3. **Enrich with xG** — `understat_fetcher.enrich_with_xg()` adds `_xg_home`/`_xg_away`. Cached in `cache/understat_xg.db`.
4. **Fit models** — `dixon_coles.fit_per_league()` + `elo.build_ratings()` + `elo.build_split_ratings()`.
5. **Fetch odds** — `odds_fetcher.fetch_window()` writes `odds/YYYY-MM-DD.csv` and saves snapshots to `cache/odds_history.db` for movement tracking. Auto force-refreshes (near-closing snapshot) when run after 12:00 UTC on a match day. `odds_fetcher.fetch_pinnacle_snapshots()` also fetches Pinnacle-specific odds to `cache/pinnacle/YYYY-MM-DD.csv` as sharp-line CLV reference.
6. **Predict** — `ensemble.predict_match()` per match → `rank_predictions()`. Lineup check fires automatically when kickoff < 3h.
7. **AI Advisor** — `ai_advisor.enrich_predictions()` calls Ollama with Google News headlines. `think:False` required for Qwen3 models.
8. **Report** — `value_detector.find_edges()` → `reporter.generate_js()` → `db_picks.save_picks()` → `tracker.run_tracker()`.

**Performance notes**: `dixon_coles.fit_per_league()` caches fitted params to `cache/dc_params_{hash}.json` — same-day reruns skip fitting entirely. `odds_fetcher` uses `ThreadPoolExecutor` to fire all 8 API calls (4 regular + 4 Pinnacle) in parallel.

### algorithms/ sub-models

| Module | Blend weight | Notes |
|---|---|---|
| `dixon_coles` | **58.5%** | Bivariate Poisson MLE + Sarmanov theta. `fit_per_league()` returns `{"PL": params, ..., "_global": params}`. |
| `elo` | **26.5%** | Venue-specific home/away ratings via `build_split_ratings()`. Per-league bonus via `ELO_HOME_BONUS_BY_LEAGUE`. |
| `form` | **0%** | Backtest confirmed near-zero predictive value. Still called for display (`form_string`) and context features; weight is 0 in the blend. `FORM_WINDOW=6` (short recent form), `FORM_DECAY=0.95`. |
| `h2h` | 5% | Time-decayed (`H2H_YEARLY_DECAY=0.70`). Only applied when `n_h2h >= H2H_MIN_MATCHES`. |
| `btts` | 5% | Poisson exact blended with per-league BTTS prior (`BTTS_PRIOR_BLEND=0.25`, `BTTS_RATE_BY_LEAGUE`). Accepts `league_code` param. |
| `corners` | 5% | Proxy driven by λ/μ. |
| `simulate` | n/a | Monte Carlo (50k sims, vectorized numpy) for secondary markets: Over 1.5/3.5/4.5, BTTS+Over, Asian HCap ±0.5/±1. Called after DC to compute `mc` dict. |
| `motivation` | n/a | Adjusts λ/μ ±8% based on table position. `from_standings(home_pos, away_pos, league_code)` → multipliers + tags (`six_pointer`, `dead_rubber`, `must_win`). |
| `referee` | n/a | Referee profiles built from fdco data. `get_adjustments(referee, league)` → home_bias correction (±3% on ph/pa) + expected cards. Profiles cached in `cache/referee_stats.json`. |
| `lineup_impact` | n/a | Fetches confirmed lineups from football-data.org when kickoff < 3h. Estimates positional absence impact on λ/μ. |
| `fatigue` | n/a | Multiplicative penalty on λ/μ. ≥7 days → 1.0; ≤1 day → 0.82. |
| `cards` | n/a | Display-only linear proxy. |
| `meta_learner` | override | XGBoost. **Only trains on `source='live'` picks** (never on backtest seeds). Activated when `cache/meta_learner.pkl` exists. Skips Platt calibration when active. |
| `draw_model` | n/a | Logistic regression (scipy) draw classifier. Features: `dc_draw`, `elo_draw`, `h2h_draw`, `mkt_draw`. **Replaces** the hand-tuned draw nudge in ensemble when `cache/draw_model.json` exists. Pre-trains from backtest automatically; fine-tunes on live picks (≥50) via tracker. `source` field in JSON distinguishes `backtest_pretrain` vs `live` — live model is never overwritten by backtest. |
| `over25_model` | n/a | Logistic regression calibrator for Over 2.5. Features: `mc_over25` (MC raw), `lam+mu`, `btts_prob`. **Replaces** raw MC over25 output when `cache/over25_model.json` exists. Same train/pretrain/source-guard pattern as draw_model. |
| `match_context` | n/a | `classify(elo_pred, form_pred, h2h_pred, home_pos, away_pos)` → tags list. Tags: `even_match`, `top6_clash`, `relegation_6ptr`, `home_in_form`, `away_in_form`, `h2h_dominant`. Combined with `motivation` tags in `_tags`. |

### Calibrator and weight optimizer (auto-loading)

Loaded at `ensemble.py` import time:
- `cache/calibrator.json` → Platt scaling post-blend (requires ≥200 resolved picks).
- `cache/model_weights.json` → overrides `MODEL_WEIGHTS` for DC/Elo (Form fixed at 0). Requires ≥50 resolved picks.

`weight_optimizer.py` now optimises **only DC and Elo** (Form permanently excluded). Both are safe to train from backtest seeds.

**meta_learner WARNING**: If predictions look wrong after `--seed-db`, delete `cache/meta_learner.pkl`. Distribution shift from historical seeds degrades current-season predictions.

### Picks persistence (`db_picks.py` + `tracker.py`)

`db_picks.py` manages `cache/picks_history.db`. Full schema:

```sql
picks (match_id PK, run_date, match_date, home_team, away_team, league,
       prob_home, prob_draw, prob_away, stars, best_outcome, best_prob,
       over25, btts, fair_odds, market_odds,
       actual_result, actual_over25, actual_btts, result_fetched_at,
       sub_preds TEXT,      -- JSON: {dc, elo, form, h2h, context}
       source TEXT,         -- 'live' | 'backtest'
       match_tags TEXT,     -- JSON array of context+motivation tags
       our_implied_prob REAL,  -- 1/fair_odds at prediction time (for CLV)
       closing_odds REAL,   -- filled by update_clv() when closing line available
       clv REAL)            -- our_implied_prob - 1/closing_odds
```

`init_db()` runs migrations automatically on every call.

`tracker.compute_metrics()` returns global metrics plus breakdowns:
- `per_league` — `{PL: {n, accuracy, roi, brier, accuracy_over25, accuracy_btts}}`
- `per_stars` — `{"3": {...}, "4": {...}, "5": {...}}`
- `per_market` — `{1x2: {accuracy, roi}, over25: {accuracy}, btts: {accuracy}}`
- `per_tag` — `{tag: {n, accuracy, roi, brier}}` — ROI by match context
- `avg_clv`, `avg_clv_by_league` — Closing Line Value tracking
- `hindsight_edge_by_league/stars` — retrospective edge validation

`tracker._save_metrics_json()` persists a snapshot to `cache/tracker_metrics.json` after each run — consumed by `value_detector.py` for dynamic Kelly sizing.

`tracker._save_results_js()` writes `visualizador/data/results.js` (`var RESOLVED_RESULTS = {...}`) keyed by `"home|away|date"` — consumed by the frontend bankroll tracker for automatic bet settlement.

### Dynamic Kelly by league

`algorithms/value_detector.py` loads `cache/tracker_metrics.json` at import time. If a league has ≥20 resolved picks, it computes:
`kelly_multiplier = max(0.30, min(1.50, 1.0 + league_roi))`

Leagues with positive ROI get a higher Kelly fraction; underperforming leagues get a lower one. Falls back to ×1.0 when file absent.

### Odds movement tracking

`odds_fetcher.save_odds_history()` snapshots each fetch to `cache/odds_history.db`. `get_odds_movement(home, away, date)` returns opening/closing ratios. A ratio ≥ 1.10 on the model's predicted outcome triggers a `sharp_money=True` flag and adds `edge_bonus=0.02` in `find_edges()`.

### Backtest (`backtest.py`)

Walk-forward with no data leakage. `compute_metrics()` returns:
- Standard: `accuracy_1x2`, `brier_score`, `log_loss`, `roi_flat`, `vb_n/roi/accuracy`
- Calibration: `calibration` (1X2), `calibration_over25`, `calibration_btts`
- Per-league secondary markets: `per_league_over25`, `per_league_btts`
- Corners (when fdco HC/AC available): `corners_mae`, `corners_accuracy`, `corners_n`

### Telegram notifications (`telegram_notifier.py`)

Sends **5 messages** after each `main.py` run:
1. Top picks (≥`TELEGRAM_MIN_STARS`) with outcome + probability + fair odds
2. Value bets with outcome, edge%, market odds, ⚡ sharp money flag
3. Doble Segura + Triple Media parlays (prob/odds combinados)
4. Cuádruple Arriesgada + Valor EV+ parlays
5. Quiniela La Liga — best pick per PD match (1/X/2/O25/BTTS)

Parlay logic mirrors `calcBestBets()` + `getSuggestedParlays()` in `index.html`: bets scored as `stars² × prob / 100`, deduplicated per match.

### ALL_MATCHES schema additions (predictions.js)

Beyond the original fields, each match object now includes:
- `contextTags` — array of strings (`even_match`, `top6_clash`, `six_pointer`, `home_must_win`, etc.)
- `over15`, `over35`, `over45` — from Monte Carlo simulation (%)
- `bttsAndOver25` — combined BTTS+Over 2.5 probability (%)
- `ahHomeMinus1Win`, `ahHomeMinus1Push`, `ahAwayPlus1Win` — Asian handicap -1/+1
- Value bet objects now include `sharpMoney` (bool), `oddsMovement` (ratio), `pinnacleProb` (%), `clvVsPinnacle` (%) — the last two populated when Pinnacle odds are available

### Visualizer (`visualizador/index.html`)

Single-file static app (~3600 lines). No build step — open directly in browser.

**Views** (`activeDate`): date string / `"ALL"` / `"BEST"` / `"VALUE"` / `"TRACK"` / `"BACK"` / `"WIN"` / `"BETNOW"` / `"SLIP"`.

State persisted via `localStorage`:
- `bwg_state` — `{activeDate, activeLeague, sortMode}` — navigation state
- `bwg_bankroll` — `{initial, current, setAt}` — persistent bankroll (set once on first visit)
- `bwg_slip` — array of pending picks to bet (cleared after `placeBets()`)
- `bwg_history` — full bet history with settlement status and P&L

**SLIP view** (v5.0 bankroll tracker):
- First visit: modal asks for initial bankroll, saved to `bwg_bankroll`, never asked again
- Each match card has "+ Local / + Empate / + Visitante / + Over 2.5 / + BTTS" buttons → `addToSlip()`
- Slip tabs: **Activo** (edit odds/stake per pick, Kelly warning when >1.5×), **Historial** (P&L curve + bet list), **Combinada** (parlay builder with auto combined odds)
- Auto-settlement: on page load, crosses `RESOLVED_RESULTS` (from `results.js`) against `bwg_history`; resolves simple bets and combinadas automatically
- **Ollama risk analysis**: button calls `http://localhost:11434/api/chat` directly from the browser (no server needed) with `think:false, stream:false`

Match modal shows: sub-model breakdown, score grid heatmap, context tags, stats (Over 1.5/2.5/3.5/4.5, BTTS+O25, Asian HCap), H2H, value bets with sharp money flag + Pinnacle CLV, AI advisor note.

TRACK view shows: global metrics, per-league table, per-stars table, **ROI per context tag**, bankroll curve, calibration diagram, sub-model accuracy.

**Adding a new view**: add a container div in HTML, add an item to `buildSidebar()` items array, add dispatch in `render()`, add `"VIEWID"` to the `activeDate` validation list in state init.

### Scheduler (Windows Task Scheduler)

`run_weekend.bat` → `python main.py` (Fri+Sat 10:00)
`run_tracker.bat` → `python tracker.py` (Mon 10:00)

Both log to `logs/` with date-stamped filenames.

## Key config knobs

`MODEL_WEIGHTS`: DC=0.58, Elo=0.27, Form=0.00, BTTS=0.05, Corners=0.05, H2H=0.05.
`BTTS_PRIOR_BLEND=0.25`, `BTTS_RATE_BY_LEAGUE` — per-league BTTS historical rates.
`FORM_WINDOW=6`, `FORM_DECAY=0.95` — short recent form window.
`DC_XI_BY_LEAGUE` — temporal decay (BL1=0.006 after tuning, was 0.007).
`ELO_HOME_BONUS_BY_LEAGUE` — BL1=92 after tuning (was 80).
`DRAW_RATE_BY_LEAGUE` — BL1=0.248 after tuning (was 0.230).
`MARKET_BLEND_WEIGHT=0.20` — reduced to 25%/50% for stale odds (>6h/>2h old).
`VALUE_BET_EDGE_THRESHOLD_BY_LEAGUE` — PL/PD/FL1=8%, BL1=15% (raised from 12% due to -6.5% backtest ROI).
`ANTIDRAW_SQUEEZE_THRESHOLD=0.05`, `ANTIDRAW_SQUEEZE_FACTOR=0.40`, `ANTIDRAW_EDGE_BONUS_MAX=0.04` — when market draw prob exceeds model draw by >5%, home/away bets get up to +4% edge bonus (exploits draw_model mkt weight=-0.62).

## Notes

### Windows / Unicode
All entry-point scripts call `sys.stdout.reconfigure(encoding="utf-8")` at startup. Use `->` not `→` in new print statements targeting stdout.

### Bankroll curve staking
`tracker.py` and `backtest.py` use **proportional staking**: `unit_stake = 1 / n_bets`. Never switch to flat 1-unit stakes.

### Backtest-derived calibration (current baseline: 2023+2024 seasons)
Global: Accuracy=51.8%, Brier=0.5927, ROI=-2.9% (at fair odds).
Per league: PL=-1.7%, PD=-1.3%, BL1=-6.5%, FL1=-2.7%.
ROI is negative at fair odds by design — real edge comes from value bets against market odds.

Draw model pretrain (6595 matches): tasa_draw=0.249, loss=0.5577.
Weights: bias=-1.86, dc=+2.29, elo=+1.61, h2h=+0.10, mkt=-0.62.
Interpretation: DC and Elo draw probs are the strongest predictors; market draw implied prob slightly corrects downward (market overprices draws).

Over25 model pretrain (6595 matches): tasa_over25=~0.52, loss converged.
Weights: bias=-0.896, mc_over25=+4.087, lam_plus_mu=-0.206, btts_prob=-1.009.
Interpretation: MC over25 is the dominant predictor; btts_prob=-1.009 corrects downward
when both teams are likely to score (1-1 style game has BTTS but not Over2.5).

### meta_learner distribution shift
`source='live'` vs `source='backtest'` column separates real picks from seeds. `meta_learner.train(real_only=True)` enforces this. If predictions look wrong: `rename cache\meta_learner.pkl cache\meta_learner.pkl.bak`.

### over25_model / draw_model source guard
`cache/draw_model.json` and `cache/over25_model.json` both have a `"source"` field: `"backtest_pretrain"` or `"live"`. Running backtest again will NOT overwrite a live-trained model. To reset: delete the file and re-run backtest.

### DC params cache
`cache/dc_params_{hash16}.json` — fitted params keyed by md5(match_count + ref_date + last 100 match dates). Invalidated automatically when new matches are fetched. Delete manually to force a cold re-fit.

### Anti-draw squeeze
`value_detector.find_edges()` computes `mkt_draw_clean - model_draw`. If gap > `ANTIDRAW_SQUEEZE_THRESHOLD` (5%), home/away bets for that match get up to `ANTIDRAW_EDGE_BONUS_MAX` (4%) added to effective edge. Only applies to home/away outcomes, not draw/over25/btts. Exposed as `antidraw_squeeze` field in value bet dicts.

### Live pick counts (as of v5.0)
75 live picks total, 38 resolved. Models requiring live data:
- `draw_model`: needs ≥50 resolved → **active** (36 live resolved at pretrain; retrained from backtest)
- `over25_model`: needs ≥50 resolved → **active** (pretrained from backtest)
- `calibrator`: needs ≥200 resolved → **inactive**
- `meta_learner`: needs ≥200 resolved → **inactive**
- `model_weights` optimizer: needs ≥50 resolved → **may activate soon**
