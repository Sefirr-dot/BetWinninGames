"""
BetWinninGames Configuration — EXAMPLE FILE
Copy this to config.py and fill in your API keys.
Set your API key from football-data.org (free registration).
"""

# --- API ---
API_KEY = "YOUR_FOOTBALL_DATA_ORG_KEY"  # football-data.org
BASE_URL = "https://api.football-data.org/v4"

ODDS_API_KEY = "YOUR_ODDS_API_KEY"  # the-odds-api.com (free tier: 500 req/month)


# --- Leagues (free tier) ---
LEAGUES = {
    "PL":  2021,  # Premier League
    "PD":  2014,  # La Liga
    "BL1": 2002,  # Bundesliga
    "FL1": 2015,  # Ligue 1
}

# --- Seasons to load for historical data (football-data.org) ---
HISTORY_SEASONS = [2023, 2024, 2025]  # 2022 gives 403 on free tier; fdco covers it

# --- Understat xG seasons (includes older fdco seasons for xG enrichment) ---
UNDERSTAT_SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]

# --- Rate limiting ---
RATE_LIMIT_CALLS_PER_MINUTE = 10
RATE_LIMIT_SLEEP = 60.0 / RATE_LIMIT_CALLS_PER_MINUTE  # seconds between calls

# --- Cache ---
CACHE_DIR = "cache"
CACHE_DB = "cache/football_data.db"
CACHE_TTL_HOURS = 6  # re-fetch after this many hours

# --- Market blend ---
MARKET_BLEND_WEIGHT = 0.20   # fraction of market implied prob blended into final prediction
                              # 0.0 = disabled, 1.0 = use only market. Applied only when odds CSV exists.

# --- Model weights (must sum to 1.0) ---
MODEL_WEIGHTS = {
    "dixon_coles": 0.45,
    "elo":         0.20,
    "form":        0.20,
    "btts":        0.05,
    "corners":     0.05,
    "h2h":         0.05,  # only applied when n_h2h >= H2H_MIN_MATCHES
}

# --- H2H parameters ---
H2H_MIN_MATCHES = 3       # minimum head-to-head matches to trust the H2H model
H2H_YEARLY_DECAY = 0.70   # time-based weight decay per year (match 1yr ago → 0.70×, 2yr → 0.49×)

# --- Dixon-Coles parameters ---
DC_XI = 0.0065      # temporal decay factor (per day) — global fallback
DC_XI_BY_LEAGUE = {
    "PL":  0.0075,  # Premier League: more rotation -> downweight old data faster
    "PD":  0.0055,  # La Liga: more consistent -> historical data stays relevant longer
    "BL1": 0.0060,  # Bundesliga: volatile league -> use more history for smoother model
    "FL1": 0.0065,  # Ligue 1: same as global
}
DC_RHO = 0.1        # low-score correction factor
HOME_ADVANTAGE = 1.25  # multiplicative home advantage for expected goals

# --- Elo parameters ---
ELO_INITIAL = 1500
ELO_K = 32
ELO_HOME_BONUS = 100        # added to home team rating before probability calc — global fallback
ELO_HOME_BONUS_BY_LEAGUE = {
    "PL":  90,   # Premier League: moderate home advantage
    "PD": 110,   # La Liga: high home advantage
    "BL1": 92,   # Bundesliga: raised to match real home advantage data
    "FL1": 95,   # Ligue 1: moderate
}
ELO_GOAL_DIFF_EXP = 0.8     # exponent for goal-difference multiplier: (1+diff)^exp
ELO_SEASON_REGRESSION = 0.85  # fraction of deviation from 1500 kept each new season

# --- Form parameters ---
FORM_DECAY = 0.95       # exponential decay per match — near-equal weights for recent streak
FORM_WINDOW = 6         # short window: captures current form, not seasonal average

# --- BTTS calibration ---
BTTS_PRIOR_BLEND = 0.25
BTTS_RATE_BY_LEAGUE = {
    "PL":  0.548,
    "PD":  0.529,
    "BL1": 0.565,
    "FL1": 0.510,
}

# --- Confidence thresholds for picks ---
HIGH_CONFIDENCE_THRESHOLD = 0.62   # >= 62% for a single outcome
MEDIUM_CONFIDENCE_THRESHOLD = 0.55

# --- Output ---
OUTPUT_DIR = "."
JS_OUTPUT_PATH = "visualizador/data/predictions.js"

# --- Draw model ---
DRAW_RATE_BY_LEAGUE = {
    "PL":  0.235,  # Premier League historical draw rate
    "PD":  0.265,  # La Liga
    "BL1": 0.248,  # Bundesliga: raised to reflect real draw rate (~24-25%)
    "FL1": 0.270,  # Ligue 1
}

# --- Value Bet Detector ---
VALUE_BET_EDGE_THRESHOLD = 0.05   # minimum edge to flag a value bet (5-star picks) — global fallback
VALUE_BET_EDGE_STEP = 0.005       # extra edge required per star below 5
VALUE_BET_MIN_STARS = 3           # only flag value bets for picks with >= this many stars
VALUE_BET_MIN_ODDS = 1.40         # ignore value bets below this bookmaker odds (market too efficient)
VALUE_BET_EDGE_THRESHOLD_BY_LEAGUE = {
    "PL":  0.08,
    "PD":  0.08,
    "BL1": 0.12,
    "FL1": 0.08,
}
ODDS_DIR = "odds"                  # folder where user places YYYY-MM-DD.csv files

# --- Backtesting ---
BACKTEST_MIN_TRAIN  = 100   # minimum matches before first test fold
BACKTEST_BATCH_SIZE = 30    # matches per walk-forward fold

# --- High-probability correction (backtest-derived) ---
HIGH_PROB_CORRECTION_ALPHA     = 0.80
HIGH_PROB_CORRECTION_THRESHOLD = 0.60

# --- Tracker / Calibration ---
CALIBRATION_MIN_SAMPLES      = 200
WEIGHT_OPTIMIZER_MIN_SAMPLES = 50
PICKS_DB         = "cache/picks_history.db"
TRACKER_JS_PATH  = "visualizador/data/tracker_data.js"
BACKTEST_JS_PATH = "visualizador/data/backtest_data.js"

# --- Understat xG ---
UNDERSTAT_LEAGUES = {
    "PL":  "EPL",
    "PD":  "La_liga",
    "BL1": "Bundesliga",
    "FL1": "Ligue_1",
}
UNDERSTAT_XG_DB = "cache/understat_xg.db"

# --- Telegram notifications (optional) ---
# 1. Create a bot via @BotFather on Telegram -> get BOT_TOKEN
# 2. Start a chat with the bot, then fetch:
#    https://api.telegram.org/bot<BOT_TOKEN>/getUpdates  -> copy "chat"."id"
TELEGRAM_BOT_TOKEN = ""   # e.g. "123456:ABC-DEF..."
TELEGRAM_CHAT_ID   = ""   # e.g. "-100123456789" or "123456789"
TELEGRAM_MIN_STARS = 4    # only send picks with this many stars or more

# --- AI Advisor (Ollama local) ---
OLLAMA_BASE_URL          = "http://localhost:11434"
OLLAMA_MODEL             = "qwen3.5:9b"   # change if your model has a different name (ollama list)
AI_ADVISOR_ENABLED       = True
AI_ADVISOR_MIN_STARS     = 3
AI_ADVISOR_MAX_ADJ       = 0.08
AI_ADVISOR_TIMEOUT       = 120

# --- Config validation ---
_w_sum = sum(MODEL_WEIGHTS.values())
if abs(_w_sum - 1.0) > 0.001:
    raise ValueError(
        f"MODEL_WEIGHTS debe sumar 1.0, actualmente suma {_w_sum:.4f}. Revisa config.py."
    )
