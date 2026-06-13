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
python backtest.py --league ALL --seasons 2023 2024 2025
python backtest.py --league PL --seasons 2024 --min-train 150

# Seed picks_history.db + retrain models (current baseline: 3 seasons)
python backtest.py --league ALL --seasons 2023 2024 2025 --seed-db
python tracker.py --no-update

# Re-train meta_learner manually (only on real live picks, never on seeds)
python -c "from algorithms.meta_learner import train; print(train('cache/picks_history.db'))"

# Pre-train draw model from backtest results (runs automatically at end of backtest)
python backtest.py --league ALL --seasons 2023 2024 2025   # → also writes cache/draw_model.json

# Re-train draw/over25 models manually on live picks (>= 500 resolved required;
# below that the backtest pretrain on ~8k matches stays authoritative)
python -c "from algorithms.draw_model import train; print(train('cache/picks_history.db'))"
python -c "from algorithms.over25_model import train; print(train('cache/picks_history.db'))"

# Regenerate results.js for the frontend bankroll tracker (auto-runs inside tracker.py)
python -c "import db_picks, tracker; picks=db_picks.get_all_picks('cache/picks_history.db'); tracker._save_results_js(picks)"

# Data freshness / realness audit (season coverage, model state, sample picks)
python scripts/verify_data.py

# DC/Elo weight sensitivity — Brier/accuracy/CLV across the full weight sweep
# (use to sanity-check whether the CLV-optimised weights generalise vs the prior)
python scripts/weight_sensitivity.py

# Backtest filtered to high-confidence picks only (per-stars ROI can also be
# derived directly from the seeded DB — see "Backtest-derived calibration")
python backtest.py --league ALL --seasons 2023 2024 2025 --min-stars 5
```

Open the visualizer via the local HTTP server (required for Ollama CORS):
```bash
run_visualizer.bat   # starts visualizador/server.py on port 8080, opens browser
# or manually:
python visualizador/server.py 8080
```
> Opening `index.html` directly as `file://` blocks Ollama AI risk analysis due to CORS.

`visualizador/server.py` is a custom `SimpleHTTPRequestHandler` subclass that serves all responses with `Cache-Control: no-cache, no-store`. All data JS files (`predictions.js`, `tracker_data.js`, etc.) are also loaded with a `?v=Date.now()` cache-busting query string baked into `index.html` — the browser always fetches fresh data after each `main.py` run.

## First-time setup

```bash
pip install -r requirements.txt          # Python 3.14+ required
cp config.example.py config.py           # then fill in API_KEY and ODDS_API_KEY

# Seed historical data + pre-train all models (one-time, ~15 min)
python backtest.py --league ALL --seasons 2023 2024 --seed-db
python tracker.py --no-update            # retrain calibrator + weight optimizer with seeds
```

After this, `cache/` contains all trained models. `main.py` is ready to run.

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
2. **Augment** — `fdco_fetcher.augment_historical()` appends seasons 2020–2022 from football-data.co.uk CSVs. Also parses referee names (`_referee`), corner counts (`_hc`/`_ac`/`_total_corners`) and yellow cards (`_home_yellow`/`_away_yellow`). `fdco_fetcher.enrich_with_odds()` then attaches market odds (`_bk_h`/`_bk_d`/`_bk_a`, Pinnacle→B365) to odds-less 2023+ matches by matching them against the same-season fdco CSVs on **(date ±1 day, final score)** with name similarity as tiebreak — used by the backtest to compute value-bet ROI at real odds.
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
| `simulate` | n/a | Monte Carlo (50k sims, vectorized numpy) for secondary markets: Over 1.5/3.5/4.5, BTTS+Over, Asian HCap ±0.5/±1/**quarter-ball** (±0.25/±0.75/±1.25). Quarter-ball uses `_quarter_ah` (half WIN) or `_quarter_ah_half_loss` (half LOSS) depending on which bracket pushes. |
| `motivation` | n/a | Adjusts λ/μ ±8% based on table position. `from_standings(home_pos, away_pos, league_code)` → multipliers + tags (`six_pointer`, `dead_rubber`, `must_win`). |
| `referee` | n/a | Referee profiles built from fdco data. `get_adjustments(referee, league)` → home_bias correction (±3% on ph/pa) + expected cards. Profiles cached in `cache/referee_stats.json`. |
| `lineup_impact` | n/a | Fetches confirmed lineups from football-data.org when kickoff < 3h. Estimates positional absence impact on λ/μ. |
| `fatigue` | n/a | Multiplicative penalty on λ/μ. ≥7 days → 1.0; ≤1 day → 0.82. |
| `cards` | n/a | Display-only linear proxy. |
| `meta_learner` | override | XGBoost. **Only trains on `source='live'` picks** (never on backtest seeds). Activated when `cache/meta_learner.pkl` exists. Skips Platt calibration when active. |
| `draw_model` | n/a | Logistic regression + **L2=0.01** draw classifier. Features: `dc_draw`, `elo_draw`, `h2h_draw`, `mkt_draw`. **Replaces** the hand-tuned draw nudge in ensemble when `cache/draw_model.json` exists. L2 acts as implicit pick filter — shrinks feature weights toward zero, outputting ~24% draw constant, which concentrates 5★ picks on clearest home/away wins (+6.6pp ROI gain). Pre-trains from backtest automatically; fine-tunes on live picks (≥50) via tracker. `source` field: `backtest_pretrain` vs `live`. |
| `over25_model` | n/a | Logistic regression + **L2=0.01** calibrator for Over 2.5. Features: `mc_over25` (MC raw), `lam+mu`, `btts_prob`. After L2, `lam+mu` dominates over raw MC. **Replaces** raw `mc["over25"]` output when `cache/over25_model.json` exists. Same source-guard pattern as draw_model. |
| `match_context` | n/a | `classify(elo_pred, form_pred, h2h_pred, home_pos, away_pos)` → tags list. Tags: `even_match`, `top6_clash`, `relegation_6ptr`, `home_in_form`, `away_in_form`, `h2h_dominant`. Combined with `motivation` tags in `_tags`. |
| `weather_model` | n/a | Converts Open-Meteo weather to λ/μ multipliers. Wind >55km/h → -8% goals. Rain >5mm/h → -7% goals. Temp <0°C → -3%. Max cap: -12% goals + +4pp draw_boost. `weather_fetcher.py` fetches from Open-Meteo (free, no key). Stadium coords for ~80 teams in `STADIUM_COORDS`. Cached 3h. |
| `squad_depth` | n/a | Estimates λ/μ impact from player availability via `/teams/{id}/persons` (football-data.org TIER_TWO — returns None on free tier, silently skipped). Missing attackers reduce λ, missing defenders increase μ. Cached 12h in `cache/squad_depth/`. |
| `sharpness` | n/a | `compute_sharpness_score()` → int 0-100. Components: CLV vs Pinnacle (25pts) + steam/movement (25pts, bug-fixed: capped per-component not total) + edge magnitude (25pts) + model agreement+MIS (25pts). Exposed as `sharpness_score` on each value_bet dict. |
| `portfolio` | n/a | Markowitz QP optimizer using `scipy.optimize.minimize(SLSQP)`. `optimize_portfolio(value_bets)` → optimal stake fractions. `simulate_weekend_pnl()` → Monte Carlo P5/P50/P95 distribution. Correlation matrix from same-match + cross-league/day heuristics. `kelly_portfolio` field added to each value_bet. |

### Calibrator and weight optimizer (auto-loading)

Loaded at `ensemble.py` import time:
- `cache/calibrator.json` → Platt scaling post-blend (requires ≥200 resolved picks).
- `cache/model_weights.json` → overrides `MODEL_WEIGHTS` for DC/Elo (Form fixed at 0). Requires ≥50 resolved picks.

`weight_optimizer.py` optimises **only DC and Elo** (Form permanently excluded). Primary objective: **CLV vs Pinnacle** (`WEIGHT_OPTIMIZER_OBJECTIVE="clv"`). Falls back to Brier when fewer than `CLV_OPTIMIZER_MIN_SAMPLES=60` picks have `closing_odds` populated. **Important**: `closing_odds` is populated by `_try_update_clv()` in tracker.py reading from `cache/pinnacle/YYYY-MM-DD.csv` after each resolved pick. The CLV objective uses `np.argmax(blend)` dynamically (not the historical `best_outcome` label) to avoid circular bias.

**Prior shrinkage (low-sample regularisation):** the raw CLV optimum overfits with few picks (85 CLV picks pin the unconstrained optimum at the DC=0.10/Elo=0.90 bound). After optimisation the weights are **linearly shrunk toward the `MODEL_WEIGHTS` prior**: `w_final = s·prior + (1−s)·w_opt`, with `s = clamp((300 − n)/(300 − WEIGHT_OPTIMIZER_MIN_SAMPLES), 0, 1)` — fully prior at `n=50`, fully data at `n≥300`. Shrinkage is applied **after** `minimize`, not as an in-objective penalty, so it behaves identically for the CLV (~0.02 scale) and Brier (~0.6 scale) objectives — a single additive penalty term could not regularise both consistently. The applied `s` is stored as `_prior_shrinkage` in `model_weights.json`.

**Brier veto (CLV reality check):** the CLV objective rewards *confidence* (prob mass on our argmax vs the closing line), not calibrated accuracy. `scripts/weight_sensitivity.py` sweeps the DC/Elo split over all 7920 resolved picks and shows the CLV optimum (w_dc≈0.05, "dump Dixon-Coles") sits in the **worst** Brier region while the config prior (w_dc≈0.68) sits **at the Brier optimum** — i.e. at this sample size CLV is anti-correlated with predictive accuracy. So after shrinkage, CLV-chosen weights are only kept if their full-sample 1X2 Brier is within `2e-3` of the prior's; otherwise they fall back to the prior. Recorded as `_clv_brier_veto` in `model_weights.json`. Net effect today: `s=0.86` shrinkage already lands at w_dc≈0.60 (Brier ≈ prior), so the veto does not fire — but it is a hard backstop once CLV sample count grows enough to weaken shrinkage. **Bottom line: until far more Pinnacle closing lines accrue, the hand-set prior is effectively optimal; the optimizer is a guard, not a source of edge.**

**Platt calibrator order (v7 fix):** Applied AFTER market blend in `ensemble.py` (not before). This matches the distribution it was trained on (picks_history.db stores post-blend probabilities).

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

### Web app (`web/` — Vite + React + TypeScript + Tailwind v4)

The new frontend. Source in `web/src/`; `npm run build` (inside `web/`) outputs to
`visualizador/app/`, which `visualizador/server.py` serves at `http://localhost:8080/app/`
— same origin as the legacy page, so the `bwg_*` localStorage (state, bankroll, slip,
history, profile) carries over unchanged. **Never rename the `bwg_*` keys or change
their shapes** — they hold the user's real betting history and must stay readable by
both frontends. Data loading keeps the legacy contract: script-injection of
`/data/{predictions,tracker_data,backtest_data,results}.js` globals with cache-busting.
Dev: `npm run dev` (port 5173, proxies `/data` to :8080 — server.py must be running).
Views are hash-routed (`#/ALL`, `#/SLIP`, …). i18n via `src/i18n.ts` `_t()` (es/en).

### Legacy visualizer (`visualizador/index.html`)

Single-file static app (~4,700 lines). Kept as fallback during the React migration —
do not add features here; it will be deleted once the React app reaches full parity.
No build step — serve via local HTTP server (see above).

**Views** (`activeDate`): date string / `"ALL"` / `"BEST"` / `"VALUE"` / `"TRACK"` / `"BACK"` / `"WIN"` / `"BETNOW"` / `"SLIP"`.

State persisted via `localStorage`:
- `bwg_state` — `{activeDate, activeLeague, sortMode}` — navigation state
- `bwg_bankroll` — `{initial, current, setAt}` — persistent bankroll (set once on first visit)
- `bwg_slip` — array of pending picks to bet (cleared after `placeBets()`)
- `bwg_history` — full bet history with settlement status and P&L

**SLIP view** (v5.0 bankroll tracker):
- First visit: modal asks for initial bankroll, saved to `bwg_bankroll`, never asked again
- Each match card has `+` outcome buttons → `addToSlip()`
- Slip tabs: **Active** (edit odds/stake, Kelly warning >1.5×), **History** (personal stats + expandable P&L), **Parlay** (auto combined odds)
- Auto-settlement on page load: crosses `RESOLVED_RESULTS` (from `results.js`) against `bwg_history`
- **Ollama risk analysis**: browser calls `localhost:11434/api/chat` directly (`think:false, stream:false`)
- Top Picks and Quiniela views have "Add to bankroll" buttons that load parlays directly into the Parlay tab

**i18n**: `const LANG = navigator.language.startsWith('es') ? 'es' : 'en'` at script top. `T{}` dict + `_t(key)` helper. Translations cover sidebar, buttons, badges, outcome labels, stats headers. All new UI strings must use `_t()` — never hardcode Spanish or English text directly in template literals.

Match modal shows: sub-model breakdown, score grid heatmap, context tags, stats (Over 1.5/2.5/3.5/4.5, BTTS+O25, Asian HCap), H2H, value bets with sharp money flag + Pinnacle CLV, AI advisor note.

TRACK view shows: global metrics, per-league table, per-stars table, **ROI per context tag**, bankroll curve, calibration diagram, sub-model accuracy.

**Adding a new view**: add a container div in HTML, add an item to `buildSidebar()` items array, add dispatch in `render()`, add `"VIEWID"` to the `activeDate` validation list in state init.

### Scheduler (Windows Task Scheduler)

`run_weekend.bat` → `python main.py` (Fri+Sat 10:00)
`run_tracker.bat` → `python tracker.py` (Mon 10:00)

Both log to `logs/` with date-stamped filenames.

## Key config knobs

`MODEL_WEIGHTS`: DC=0.58, Elo=0.27, Form=0.00, BTTS=0.05, Corners=0.05, H2H=0.05.
`DC_XI_BY_LEAGUE` — temporal decay: PL=0.0075, PD=0.0055, BL1=0.006, **FL1=0.0055** (lowered from 0.0065 — more history smooths PSG outlier effect).
`ELO_HOME_BONUS_BY_LEAGUE` — PL=90, PD=110, BL1=92, FL1=95.
`MARKET_BLEND_WEIGHT=0.20` — reduced to 25%/50% for stale odds (>6h/>2h old).

**Value bet thresholds** (re-derived 2026-06 from the clean 2023-25 reseed):
`VALUE_BET_EDGE_THRESHOLD_BY_LEAGUE` — PL/PD=8%, BL1=10%, FL1=12%.
`VALUE_BET_MIN_STARS_BY_LEAGUE` — 3★ in all four leagues. The old BL1/FL1=5★ rule
came from the duplicated-seed era; clean seeds show ≥3★ positive everywhere.
`MIN_STARS_SAVE=3` — picks below 3★ are never persisted to picks_history.db (live
history showed 1-2★ at ~40% accuracy vs 60% for ≥3★). The visualizer still shows
all matches; `PREDICTIONS_META.minStarsTracked` tells the frontend the cutoff.

`ANTIDRAW_SQUEEZE_THRESHOLD=0.05`, `ANTIDRAW_SQUEEZE_FACTOR=0.40`, `ANTIDRAW_EDGE_BONUS_MAX=0.04` — when market draw prob exceeds model draw by >5%, home/away bets get up to +4% edge bonus.

## Notes

### No test suite / no linter
There are no unit tests and no linting configuration (`pytest`, `flake8`, `black`, etc. are not installed). Validation is done by running `backtest.py` and inspecting output. Do not attempt to run tests or add linting config.

### Windows / Unicode
All entry-point scripts call `sys.stdout.reconfigure(encoding="utf-8")` at startup. Use `->` not `→` in new print statements targeting stdout.

### Bankroll curve staking
`tracker.py` and `backtest.py` use **proportional staking**: `unit_stake = 1 / n_bets`. Never switch to flat 1-unit stakes.

### Backtest-derived calibration (current baseline: 2023+2024+2025 seasons, 7954 matches, leak-free reseed 2026-06-13)

> These numbers come from the **leak-free** reseed (post `_disable_posthoc_models` guard).
> The previous baseline was inflated by the draw_model leakage — see "Backtest leakage
> guard" above. **Fair-odds ROI is NOT real betting profit**: a calibrated model scores
> ROI≈0 at fair odds by construction. Real edge lives in `vb_roi` (value bets at *market*
> odds, where you must beat the bookmaker margin), not here.

**All picks (fair odds):**
Global: Accuracy=52.1%, Brier=0.5914, ROI=−0.2%.
Per league: PL=+1.7%, PD=+0.1%, BL1=−3.4%, FL1=+0.3%.

**≥3★ picks (fair-odds ROI from the seeded DB — sanity/calibration check, not profit):**
PL: n=684, acc=67.3%, ROI=+1.1% · PD: n=645, acc=68.8%, ROI=+4.5%
BL1: n=535, acc=66.5%, ROI=+0.0% · FL1: n=546, acc=63.7%, ROI=−1.8%

**Value-bet ROI at REAL market odds (Pinnacle/B365 from fdco, the only number that
reflects betting profit) — 2026-06-13, the definitive result:**
Global **−6.7%** (n=1570). PL +0.2% · PD −6.5% · BL1 −11.8% · FL1 −9.9%.
Higher model edge → *worse* ROI (edge≥5% −6.4%, ≥10% −7.8%, ≥15% −15.9%): the textbook
signature of a model with **no real edge over the market**. No star/edge/league subset is
robustly profitable (the few positive pockets are n<55 noise). **Conclusion: the system
does not beat the closing line — it is well-calibrated (≈ as good as the market), which
means it cannot profit betting against sharp books after their margin.** Market odds are
attached by `fdco_fetcher.enrich_with_odds()` (matches by date+score, see its docstring)
and persisted to the seeded DB as `market_odds` (best_outcome).

Contrast with the old contaminated figures (PL +19.7%, PD +22.4%, BL1 +18.5%, FL1 +10.2%)
to see the magnitude of the leak: it inflated ROI ~15–20pp and accuracy ~3–4pp, and roughly
halved the ≥3★ count (the leaked draw model distorted the confidence distribution).

Per-stars ROI can be derived directly from the seeds without re-running backtest:
`SELECT league, stars, COUNT(*), AVG(best_outcome=actual_result), ... FROM picks WHERE source='backtest'`.

Draw model pretrain (7954 matches, L2=0.01): tasa_draw=0.249, loss=0.5613.
Weights: bias=-1.14, dc=+0.11, elo=+0.05, h2h=+0.03, mkt=0.0 (bias-dominated ~24% draw —
acts as implicit quality filter concentrating 5★ on clear home/away wins).

Over25 model pretrain (7954 matches, L2=0.01): tasa_over25=0.540, loss=0.6860.
Weights: bias=-0.42, mc_over25=+0.09, lam_plus_mu=+0.49, btts_prob=+0.12 (lam+mu dominant).

### meta_learner distribution shift
`source='live'` vs `source='backtest'` column separates real picks from seeds. `meta_learner.train(real_only=True)` enforces this. If predictions look wrong: `rename cache\meta_learner.pkl cache\meta_learner.pkl.bak`.

### over25_model / draw_model source guard
`cache/draw_model.json` and `cache/over25_model.json` both have a `"source"` field: `"backtest_pretrain"` or `"live"`. Running backtest again will NOT overwrite a live-trained model. To reset: delete the file and re-run backtest. Live retrain requires **≥500 resolved live picks** (raised from 200 — a 200-sample fit must not displace a ~8k-sample pretrain).

### Seed determinism + live-only analytics (fixed 2026-06-12)
- `fdco_fetcher._synthetic_id()` uses **md5** (was built-in `hash()`, randomized per
  process — every `--seed-db` run duplicated all fdco matches under new IDs; the DB
  reached 36,836 backtest rows for 7,342 unique matches). Seeds are now idempotent.
- `tracker.run_tracker()` computes ALL analytics (metrics, PSI drift, calibrator,
  meta-learner gate, tracker_data.js export) from **live picks only**. Seeds stay in
  the DB solely as training data for the weight optimizer and draw/over25 pretrains.
  Never feed seeds back into per-league ROI (dynamic Kelly) or the Platt calibrator.

### Backtest leakage guard — post-hoc models (fixed 2026-06-13)
`ensemble.py` loads `cache/{draw_model,over25_model,calibrator}.json` + `meta_learner.pkl`
into module globals **at import time**. Those models are trained on the FULL dataset, so
if present in cache they would be applied to every walk-forward fold — leaking future folds
into early-fold predictions. `draw_model` in particular *replaces* the blended draw
probability, re-normalising ph/pa and changing `best_outcome`/`best_prob` →
**contaminating the per-stars ROI baseline**. `backtest._disable_posthoc_models()` (called
at the top of `run_backtest`) now sets those globals to `None` before the walk-forward; the
draw/over25 pretrain still runs at the END of `main()` on the clean results, so live models
are unaffected. Fingerprint of past contamination: stored `prob_draw` std collapses from the
natural ~0.050 (raw DC/Elo blend) to ~0.009 (near-constant L2 draw_model output).
**The seeded DB was refreshed leak-free on 2026-06-13** (`seed_picks_db` now deletes existing
`source='backtest'` rows first, so a reseed actually replaces stale rows instead of being
ignored). To reseed again: `python backtest.py --league ALL --seasons 2023 2024 2025 --seed-db`
then `python tracker.py --no-update`. (The weight optimizer is unaffected by the leak — it
reads the raw `sub_preds` dc/elo vectors, computed before the draw_model replacement.)

### DC params cache
`cache/dc_params_{hash16}.json` — fitted params keyed by md5(match_count + ref_date + last 100 match dates). Invalidated automatically when new matches are fetched. Delete manually to force a cold re-fit.

### Anti-draw squeeze
`value_detector.find_edges()` computes `mkt_draw_clean - model_draw`. If gap > `ANTIDRAW_SQUEEZE_THRESHOLD` (5%), home/away bets for that match get up to `ANTIDRAW_EDGE_BONUS_MAX` (4%) added to effective edge. Only applies to home/away outcomes, not draw/over25/btts. Exposed as `antidraw_squeeze` field in value bet dicts.

### v7 new modules (2026-03-11)

**Critical bug fixes applied in v7:**
- `draw_model.py`: target label was `"D"` (always False) → fixed to `"draw"`. Delete `cache/draw_model.json` and retrain if upgrading.
- `over25_model.py`: `lam+mu` normalized by `_LAM_MU_NORM=3.0` — invalidates existing model. Delete and retrain.
- `simulate.py`: quarter AH effective formula — `_quarter_ah_half_loss()` (push = half LOSS) vs `_quarter_ah()` (push = half WIN). Affects AH -0.25, -1.25 home and AH +0.75 away.
- `ensemble.py`: `away_depth_impact` lam/mu was swapped — away weak attack → reduce μ, away weak defence → increase λ.
- `ensemble.py`: Platt calibrator now applied AFTER market blend (correct distribution match).
- `tracker.py`: drift alert fixed to use `ks_home/draw/away` keys (KS test replaced PSI).
- `value_detector.py`: Kelly hard cap `min(kelly_base * mis, 0.25)` enforced after MIS scaling.
- `sharpness.py`: `movement_pts = min(10.0, ...)` (was no-op identity min).

**New config knobs:** `WEIGHT_OPTIMIZER_OBJECTIVE="clv"`, `CLV_OPTIMIZER_MIN_SAMPLES=60`, `PSI_ALERT_THRESHOLD=0.2`, `PSI_LOOKBACK_RECENT=20`, `SUBMODEL_ACCURACY_WINDOW=30`.

**New value_bet fields:** `sharpness_score` (0-100), `mis`, `corr_discount`, `kelly_portfolio`, `match_date`.

**New sub-models/modules:**
- `weather_fetcher.py` + `algorithms/weather_model.py` — Open-Meteo API (free), stadium coords for ~80 teams, cached 3h.
- `algorithms/squad_depth.py` — `/teams/{id}/persons` API (TIER_TWO only, silently skipped on free tier), cached 12h.
- `algorithms/sharpness.py` — Sharpness Score 0-100 per value_bet.
- `algorithms/portfolio.py` — Markowitz QP (SLSQP), `kelly_portfolio` field, MC P&L distribution.

**Drift detection:** KS 2-sample test (Bonferroni α=0.0167) replaces PSI in `compute_psi()`.

**CLV pipeline active:** `_try_update_clv()` reads `cache/pinnacle/YYYY-MM-DD.csv` post-resolution → populates `closing_odds`/`clv` → enables CLV optimizer objective.

### Live pick counts (as of 2026-06-12)
203 live resolved picks, 0 pending (2025-26 season closed). Models requiring live data:
- `draw_model` / `over25_model`: live retrain needs ≥500 resolved → **active on backtest pretrain** (7954 matches, L2=0.01)
- `model_weights` optimizer: **active on CLV objective** (85 Pinnacle CLV picks ≥ 60 min). DC=0.511, Elo=0.339 (raw CLV optimum DC=0.10/Elo=0.90 shrunk toward prior at s=0.86; Brier veto did not fire). Weight-sweep finding: the config prior is already at the Brier optimum — the optimizer adds no edge at this CLV sample size, it only guards against drift.
- `calibrator` (Platt): needs ≥200 live resolved → **active** (203 live picks, live-only since 2026-06-12)
- `meta_learner`: needs ≥500 with sub_preds → **inactive** (165 usable)
Note: with `MIN_STARS_SAVE=3`, live volume accrues slower next season (only ≥3★ picks are saved).

### Security
`run_visualizer.bat` binds to `127.0.0.1` only — not accessible from local network.
`visualizador/data/results.js` is gitignored (added in v5.0 alongside other generated data files).
All API calls go through Python — no keys ever reach the browser.
