# BetWinninGames

A statistical football prediction engine for the top 4 European leagues (Premier League, La Liga, Bundesliga, Ligue 1), built around Dixon-Coles Poisson modelling, Elo ratings, and a suite of calibrated secondary models. Includes a full browser-based visualizer with bankroll tracking, bet slip, and AI-assisted risk analysis via local Ollama.

---

## Prerequisites

- **Python 3.14+** — `pip install -r requirements.txt`
- **API keys** (free tiers):
  - [football-data.org](https://www.football-data.org) — historical data + lineups
  - [the-odds-api.com](https://the-odds-api.com) — bookmaker odds (500 req/month free)
- **Optional**:
  - Telegram bot token — for push notifications after each run
  - [Ollama](https://ollama.ai) running locally with `qwen3.5:9b` — for AI Advisor and bankroll risk analysis

---

## Setup

```bash
# 1. Clone and install dependencies
pip install -r requirements.txt

# 2. Configure API keys
cp config.example.py config.py
# Edit config.py: set API_KEY and ODDS_API_KEY

# 3. Seed historical data + pre-train models (one-time, ~10 min)
python backtest.py --league ALL --seasons 2023 2024 --seed-db
python tracker.py --no-update

# 4. Open the visualizer
run_visualizer.bat        # Windows: starts local server + opens browser
# or manually: cd visualizador && python -m http.server 8080
```

> **First visit**: the visualizer will ask you to set your initial bankroll. This is saved locally in the browser and never sent anywhere.

---

## Weekly workflow

```
Friday / Saturday morning
  └── python main.py
        Fetches upcoming matches, fits models, downloads odds (Pinnacle + best market),
        generates predictions, sends Telegram picks if configured.
        → writes visualizador/data/predictions.js

Monday morning (or whenever weekend results are in)
  └── python tracker.py
        Resolves match results, updates model metrics, retrains draw/over25 calibrators,
        auto-settles your tracked bets.
        → writes visualizador/data/tracker_data.js + results.js

Then open the visualizer → pending bets settle automatically on page load.
```

**Windows Task Scheduler** (pre-configured):
- `run_weekend.bat` → `python main.py` at 10:00 on Friday and Saturday
- `run_tracker.bat` → `python tracker.py` at 10:00 on Monday

---

## Visualizer

```bash
run_visualizer.bat    # starts Python HTTP server on :8080, opens browser
```

> Running via localhost (not `file://`) is required for the Ollama AI risk analysis — the browser calls `localhost:11434` directly, which is blocked by CORS on `file://`.

### Views

| View | Description |
|---|---|
| **🎯 My Bankroll** | Bet slip, parlay builder, auto-settling P&L history, personal stats, AI risk analysis |
| **💰 Bet Now** | Curated high-confidence value bets with Kelly sizing vs your tracked bankroll |
| **🏆 Top Picks** | Best picks by category + suggested parlays (Double / Triple / Quad / EV+) with one-click add to bankroll |
| **💎 Value Bets** | All picks with positive edge vs market, Pinnacle CLV reference |
| **📊 Performance** | Model accuracy, ROI by league/stars/context tag, bankroll curve, calibration |
| **📈 Backtest** | Walk-forward backtest results by fold and league |
| **🇪🇸 Quiniela** | Best pick per La Liga match + one-click add to bankroll |

### Bankroll tracker

- Set your initial bankroll once on first visit (stored in `localStorage`, never uploaded)
- Add picks to your slip from any match card with the `+` outcome buttons
- Enter your actual bookmaker odds — they differ from model fair odds
- Adjust stakes; Kelly warnings fire when you exceed 1.5× the suggested fraction
- Place as simple bets or build a parlay in the **Parlay** tab
- Results resolve automatically the next time you open the visualizer after `tracker.py` runs
- **AI risk analysis**: sends your slip to local Ollama for a natural-language risk assessment (requires Ollama running)

---

## Architecture overview

### Prediction pipeline (`main.py`)

```
Historical data (football-data.org + football-data.co.uk)
    → xG enrichment (Understat)
    → Dixon-Coles per-league fit  [params cached between same-day runs]
    → Elo ratings (venue-split home/away)
    → Odds fetch (parallel: 4 leagues × best market + Pinnacle)
    → Ensemble prediction per match
         DC (58%) + Elo (27%) + H2H (5%) + BTTS (5%) + Corners (5%)
         + draw classifier (logistic regression)
         + Over 2.5 calibrator (logistic regression)
         + motivation / fatigue / referee / lineup adjustments
         + market blend (20% bookmaker implied)
    → Value bet detection (edge vs market + anti-draw squeeze + sharp money)
    → AI Advisor (Ollama + Google News headlines)
    → predictions.js + picks_history.db + Telegram
```

### Sub-models

| Model | Role | Trained on |
|---|---|---|
| `dixon_coles` | 58% blend — Bivariate Poisson MLE + Sarmanov theta | All historical matches per league |
| `elo` | 27% blend — venue-split home/away ratings | All historical matches |
| `draw_model` | Replaces hand-tuned draw nudge | Backtest pre-train → fine-tunes on live picks (≥50) |
| `over25_model` | Calibrates raw Monte Carlo Over 2.5 estimate | Backtest pre-train → fine-tunes on live picks (≥50) |
| `meta_learner` | XGBoost override of full 1X2 blend | Live picks only (≥200 required) |
| `calibrator` | Platt scaling post-blend | Live picks (≥200 required) |

### Self-improving loop

Each `tracker.py` run automatically retrains calibration models as more resolved picks accumulate. Models with a `"source"` field (`backtest_pretrain` vs `live`) are **never overwritten** by backtest runs — live data always takes priority.

---

## Configuration (`config.py`)

| Key | Default | Notes |
|---|---|---|
| `MODEL_WEIGHTS` | DC=0.58, Elo=0.27, Form=0, BTTS=0.05, Corners=0.05, H2H=0.05 | Must sum to 1.0 |
| `VALUE_BET_EDGE_THRESHOLD_BY_LEAGUE` | PL/PD/FL1=8%, BL1=15% | BL1 stricter due to -6.5% backtest ROI |
| `MARKET_BLEND_WEIGHT` | 0.20 | Bookmaker implied prob weight; reduced for stale odds |
| `ANTIDRAW_SQUEEZE_THRESHOLD` | 0.05 | Min gap (market_draw − model_draw) to add home/away edge bonus |
| `DC_XI_BY_LEAGUE` | PL=0.0075, PD=0.0055, BL1=0.006, FL1=0.0065 | Temporal decay per league |
| `ELO_HOME_BONUS_BY_LEAGUE` | PL=90, PD=110, BL1=92, FL1=95 | Home advantage per league |
| `OLLAMA_MODEL` | `qwen3.5:9b` | Any Ollama model; `think:False` required for Qwen3 |
| `TELEGRAM_MIN_STARS` | 4 | Minimum star rating to send via Telegram |

---

## Backtest baseline (2023 + 2024 seasons, 6,595 matches)

| League | Accuracy | ROI at fair odds |
|---|---|---|
| Premier League | 54.0% | −1.7% |
| La Liga | 51.9% | −1.3% |
| Bundesliga | 50.9% | −6.5% |
| Ligue 1 | 50.3% | −2.7% |
| **Global** | **51.8%** | **−2.9%** |

ROI is negative at fair odds by design — the real edge comes from value bets placed at bookmaker odds above the model's fair price.

---

## Project structure

```
main.py              — weekly prediction run (Fri/Sat)
tracker.py           — results resolution + model retraining (Mon)
backtest.py          — walk-forward backtester
config.py            — all configuration (gitignored, copy from config.example.py)
algorithms/          — sub-models: dixon_coles, elo, ensemble, draw_model,
                       over25_model, meta_learner, value_detector, …
visualizador/
  index.html         — single-file browser app (~4000 lines, no build step)
  data/              — generated JS data files (predictions, tracker, results)
cache/               — SQLite databases + trained model files (gitignored)
odds/                — downloaded bookmaker odds CSVs
run_visualizer.bat   — starts local HTTP server + opens browser
run_weekend.bat      — scheduled: python main.py (Fri + Sat 10:00)
run_tracker.bat      — scheduled: python tracker.py (Mon 10:00)
```
