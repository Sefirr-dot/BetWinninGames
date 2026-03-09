"""
SQLite persistence layer for picks history.

Database: cache/picks_history.db (same folder as football_data.db)

Schema
------
picks table stores one row per match prediction, keyed by the API match_id.
Results (actual_result, actual_over25, actual_btts) are NULL until tracker.py
resolves them by calling the football-data.org /matches/{id} endpoint.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone

_SCHEMA = """
CREATE TABLE IF NOT EXISTS picks (
    match_id          INTEGER PRIMARY KEY,
    run_date          TEXT,
    match_date        TEXT,
    home_team         TEXT,
    away_team         TEXT,
    league            TEXT,
    prob_home         REAL,
    prob_draw         REAL,
    prob_away         REAL,
    stars             INTEGER,
    best_outcome      TEXT,
    best_prob         REAL,
    over25            REAL,
    btts              REAL,
    fair_odds         REAL,
    market_odds       REAL,
    actual_result     TEXT,
    actual_over25     INTEGER,
    actual_btts       INTEGER,
    result_fetched_at TEXT,
    sub_preds         TEXT,
    source            TEXT DEFAULT 'live'
)
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return sqlite3.connect(db_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_db(db_path: str) -> None:
    """Create picks table if it doesn't exist, and run any pending migrations."""
    with _connect(db_path) as conn:
        conn.execute(_SCHEMA)
        existing = {row[1] for row in conn.execute("PRAGMA table_info(picks)").fetchall()}
        if "sub_preds" not in existing:
            conn.execute("ALTER TABLE picks ADD COLUMN sub_preds TEXT")
        if "source" not in existing:
            conn.execute("ALTER TABLE picks ADD COLUMN source TEXT DEFAULT 'live'")
        if "match_tags" not in existing:
            conn.execute("ALTER TABLE picks ADD COLUMN match_tags TEXT")
        if "our_implied_prob" not in existing:
            conn.execute("ALTER TABLE picks ADD COLUMN our_implied_prob REAL")
        if "closing_odds" not in existing:
            conn.execute("ALTER TABLE picks ADD COLUMN closing_odds REAL")
        if "clv" not in existing:
            conn.execute("ALTER TABLE picks ADD COLUMN clv REAL")
        conn.commit()


def save_picks(
    predictions_list: list[dict],
    date_str: str,
    run_ts: str,
    db_path: str,
    value_bets: list[dict] | None = None,
    source: str = "live",
) -> int:
    """
    Insert predictions into the DB. Uses INSERT OR IGNORE so re-runs don't
    create duplicates.

    Parameters
    ----------
    predictions_list : list of {match_info: ..., prediction: ...} dicts
    date_str         : YYYY-MM-DD of the match day (fallback when utcDate absent)
    run_ts           : ISO timestamp of when main.py ran
    db_path          : path to picks_history.db
    value_bets       : optional list from value_detector.find_edges(); used to
                       populate market_odds for each match

    Returns number of new rows inserted.
    """
    init_db(db_path)

    # Build market-odds index keyed by (home_name, away_name, outcome)
    odds_index: dict[tuple, float] = {}
    if value_bets:
        for vb in value_bets:
            key = (vb.get("home_name", ""), vb.get("away_name", ""), vb.get("outcome", ""))
            odds_index[key] = vb.get("bk_odds")

    inserted = 0
    with _connect(db_path) as conn:
        for entry in predictions_list:
            mi   = entry.get("match_info", {})
            pred = entry.get("prediction", {})

            match_id = mi.get("id")
            if match_id is None:
                continue

            # Derive match date from utcDate field (YYYY-MM-DDTHH:MM:SSZ)
            utc_date   = mi.get("utcDate", "")
            match_date = utc_date[:10] if utc_date else date_str

            best_prob    = pred.get("best_prob", 0)
            fair_odds    = round(1.0 / best_prob, 2) if best_prob > 0.01 else None
            best_outcome = pred.get("best_outcome")
            home_name    = mi.get("homeTeam", {}).get("name", "")
            away_name    = mi.get("awayTeam", {}).get("name", "")
            market_odds  = odds_index.get((home_name, away_name, best_outcome))

            # Serialize sub-model probabilities + extra features for meta-learner
            sub = {}
            for model_key in ("dc", "elo", "form", "h2h"):
                raw = pred.get(model_key) or {}
                if not raw:
                    continue
                probs = {k: round(raw[k], 6) for k in ("prob_home", "prob_draw", "prob_away") if k in raw}
                if probs:
                    if model_key == "h2h":
                        if not raw.get("sufficient"):
                            continue
                        probs["sufficient"] = True
                    # Extra features for XGBoost meta-learner
                    if model_key == "dc":
                        for extra in ("lambda_", "mu_", "over25"):
                            if extra in raw:
                                probs[extra] = round(raw[extra], 6)
                    if model_key == "elo":
                        for extra in ("rating_home", "rating_away"):
                            if extra in raw:
                                probs[extra] = round(raw[extra], 2)
                    sub[model_key] = probs
            ctx = pred.get("_context") or {}
            if ctx:
                sub["context"] = {k: round(v, 6) for k, v in ctx.items()}
            sub_preds_json = json.dumps(sub) if sub else None
            tags           = pred.get("_tags") or []
            tags_json      = json.dumps(tags) if tags else None
            # Store our implied probability at prediction time for CLV tracking
            our_implied    = round(1.0 / fair_odds, 4) if fair_odds else None

            cur = conn.execute(
                """INSERT OR IGNORE INTO picks
                   (match_id, run_date, match_date, home_team, away_team, league,
                    prob_home, prob_draw, prob_away, stars, best_outcome, best_prob,
                    over25, btts, fair_odds, market_odds, sub_preds, source,
                    match_tags, our_implied_prob)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    match_id,
                    run_ts,
                    match_date,
                    home_name,
                    away_name,
                    mi.get("_league_code", ""),
                    pred.get("prob_home"),
                    pred.get("prob_draw"),
                    pred.get("prob_away"),
                    pred.get("stars", 1),
                    best_outcome,
                    best_prob,
                    pred.get("over25"),
                    pred.get("btts_prob"),
                    fair_odds,
                    market_odds,
                    sub_preds_json,
                    source,
                    tags_json,
                    our_implied,
                ),
            )
            inserted += cur.rowcount
        conn.commit()

    return inserted


def update_clv(match_id: int, closing_odds: float, db_path: str) -> None:
    """
    Store the closing market odds and compute CLV for a resolved pick.
    CLV = our_implied_prob - (1 / closing_odds)
    Positive CLV means our prediction was better than the closing market.
    """
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT our_implied_prob FROM picks WHERE match_id=?", (match_id,)
        ).fetchone()
        if row and row[0]:
            clv = round(row[0] - (1.0 / closing_odds), 4)
            conn.execute(
                "UPDATE picks SET closing_odds=?, clv=? WHERE match_id=?",
                (closing_odds, clv, match_id),
            )
            conn.commit()


def get_unresolved(before_date: str, db_path: str) -> list[dict]:
    """
    Return picks without a result where match_date <= before_date.

    Parameters
    ----------
    before_date : YYYY-MM-DD upper bound (inclusive); use today's date to
                  include all past matches that haven't been resolved yet.
    """
    if not os.path.exists(db_path):
        return []
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT * FROM picks
               WHERE actual_result IS NULL AND match_date <= ?
               ORDER BY match_date""",
            (before_date,),
        ).fetchall()
    return [dict(r) for r in rows]


def update_result(
    match_id: int,
    actual_result: str,
    actual_over25: int,
    actual_btts: int,
    db_path: str,
) -> None:
    """Mark a pick as resolved with its actual result."""
    now = datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as conn:
        conn.execute(
            """UPDATE picks
               SET actual_result=?, actual_over25=?, actual_btts=?, result_fetched_at=?
               WHERE match_id=?""",
            (actual_result, actual_over25, actual_btts, now, match_id),
        )
        conn.commit()


def get_all_picks(db_path: str) -> list[dict]:
    """Return all picks ordered by match_date DESC (for the visualiser)."""
    if not os.path.exists(db_path):
        return []
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM picks ORDER BY match_date DESC, home_team"
        ).fetchall()
    return [dict(r) for r in rows]


def get_real_picks(db_path: str) -> list[dict]:
    """
    Return only picks generated by main.py (source='live'), resolved or not.

    Used by meta_learner.train() so it never trains on seeded backtest data,
    which comes from different seasons and causes distribution shift.
    """
    if not os.path.exists(db_path):
        return []
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM picks WHERE source = 'live' ORDER BY match_date DESC"
        ).fetchall()
    return [dict(r) for r in rows]
