"""
football-data.co.uk — secondary historical data source.

Provides free CSV downloads for seasons blocked by football-data.org free
tier (2022 and earlier give 403).  Covers seasons 2020, 2021, 2022.

URL pattern
-----------
    https://www.football-data.co.uk/mmz4281/{SSYY}/{code}.csv
    e.g. "2122/E0.csv" = Premier League 2021/22

Match dict format produced
--------------------------
Compatible with every downstream consumer (Dixon-Coles, Elo, Form, H2H,
Fatigue) — same keys as football-data.org API dicts.

Team ID resolution
------------------
fdco uses short team names ("Man City") while football-data.org uses full
names ("Manchester City FC").  A registry built from the existing fd.org
matches (which have real API IDs) is used to map fdco names → IDs via:

  1. Exact match after normalization
  2. Fuzzy match (SequenceMatcher ≥ 0.72)
  3. Synthetic negative ID (never collides with fd.org positive IDs)

Teams with synthetic IDs still improve DC / Elo indirectly (opponent
strength), but their direct form/H2H records won't link to predictions.
"""

import csv
import hashlib
import io
import os
import sqlite3
import time
from difflib import SequenceMatcher

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_BASE_URL      = "https://www.football-data.co.uk/mmz4281"
_REQUEST_DELAY = 1.5   # polite delay between downloads

_FDCO_LEAGUE_CODES = {
    "PL":  "E0",
    "PD":  "SP1",
    "BL1": "D1",
    "FL1": "F1",
}

_FDCO_DB      = "cache/fdco_data.db"
_FDCO_SEASONS = [2020, 2021, 2022]   # seasons to download (2020 = 2020/21, etc.)

# Common short-name aliases used by fdco that differ from fd.org names
_ALIASES: dict[str, str] = {
    # Premier League
    "man united":      "manchester united",
    "man city":        "manchester city",
    "tottenham":       "tottenham hotspur",
    "spurs":           "tottenham hotspur",
    "newcastle":       "newcastle united",
    "wolves":          "wolverhampton wanderers",
    "west brom":       "west bromwich albion",
    "sheffield utd":   "sheffield united",
    "sheffield weds":  "sheffield wednesday",
    "brighton":        "brighton hove albion",
    "luton":           "luton town",
    "nott'm forest":   "nottingham forest",
    "norwich":         "norwich city",
    "watford":         "watford",
    "burnley":         "burnley",
    "brentford":       "brentford",
    # La Liga
    "atletico madrid": "atletico madrid",
    "betis":           "real betis",
    "espanol":         "espanyol",
    "vallecano":       "rayo vallecano",
    "sociedad":        "real sociedad",
    "alaves":          "alaves",
    "valladolid":      "real valladolid",
    "granada":         "granada",
    # Bundesliga
    "m'gladbach":      "borussia monchengladbach",
    "gladbach":        "borussia monchengladbach",
    "ein frankfurt":   "eintracht frankfurt",
    "hertha":          "hertha berlin",
    "dusseldorf":      "fortuna dusseldorf",
    "bielefeld":       "arminia bielefeld",
    "greuther furth":  "greuther furth",
    # Ligue 1
    "psg":             "paris saint-germain",
    "paris sg":        "paris saint-germain",
    "st etienne":      "saint-etienne",
    "marseille":       "olympique marseille",
    "lyon":            "olympique lyonnais",
    "nice":            "nice",
    "nantes":          "nantes",
    "metz":            "metz",
    "troyes":          "troyes",
    "clermont":        "clermont foot",
}


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    """Light normalisation: lowercase, strip ' FC'/' CF', remove punctuation."""
    n = name.lower().strip()
    # Apply alias first
    n = _ALIASES.get(n, n)
    # Strip trailing legal suffixes
    for sfx in (" fc", " cf", " afc", " sfc", " sc", " ac", " if"):
        if n.endswith(sfx):
            n = n[: -len(sfx)].strip()
    # Transliterate common accented chars
    for a, b in [("ó","o"),("é","e"),("ü","u"),("ö","o"),("ä","a"),
                 ("á","a"),("í","i"),("ú","u"),("ñ","n"),("ß","ss"),
                 ("'",""),("'",""),("&","and"),("."," ")]:
        n = n.replace(a, b)
    return " ".join(n.split())   # collapse whitespace


def _fuzzy(target: str, candidates: list[str], threshold: float = 0.72) -> str | None:
    best_r, best_c = 0.0, None
    tn = _norm(target)
    for c in candidates:
        r = SequenceMatcher(None, tn, _norm(c)).ratio()
        if r > best_r:
            best_r, best_c = r, c
    return best_c if best_r >= threshold else None


def _synthetic_id(league: str, name: str) -> int:
    """Deterministic negative ID for teams not found in the fd.org registry.

    Uses md5 instead of built-in hash() — str hashing is randomized per
    process (PYTHONHASHSEED), which made these IDs change on every run and
    caused duplicate seed rows in picks_history.db across --seed-db runs.
    """
    key = f"fdco:{league}:{_norm(name)}"
    digest = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:12], 16)
    return -(digest % 900_000 + 100_001)


# ---------------------------------------------------------------------------
# Team registry
# ---------------------------------------------------------------------------

def build_registry(fd_matches: list[dict]) -> dict[str, int]:
    """
    Build {normalized_name: team_id} from football-data.org match dicts.
    Also stores the original (un-normalised) names so fuzzy can reach them.
    """
    registry: dict[str, int] = {}
    for m in fd_matches:
        for side in ("homeTeam", "awayTeam"):
            t = m.get(side, {})
            tid  = t.get("id")
            name = t.get("name") or t.get("shortName") or ""
            if tid and name:
                registry[_norm(name)] = tid
    return registry


def _resolve_id(
    fdco_name: str,
    league: str,
    registry: dict[str, int],
    registry_keys: list[str],
) -> int:
    """Map an fdco team name to a fd.org team ID (or a synthetic negative ID)."""
    norm = _norm(fdco_name)
    if norm in registry:
        return registry[norm]
    best = _fuzzy(fdco_name, registry_keys)
    if best:
        return registry[best]
    return _synthetic_id(league, fdco_name)


# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_FDCO_DB), exist_ok=True)
    return sqlite3.connect(_FDCO_DB)


def _init_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fdco_matches (
                league     TEXT    NOT NULL,
                season     INTEGER NOT NULL,
                match_date TEXT    NOT NULL,
                home_name  TEXT    NOT NULL,
                away_name  TEXT    NOT NULL,
                home_goals INTEGER NOT NULL,
                away_goals INTEGER NOT NULL,
                xg_home    REAL,
                xg_away    REAL,
                b365_h     REAL,
                b365_d     REAL,
                b365_a     REAL,
                psh        REAL,
                psd        REAL,
                psa        REAL,
                PRIMARY KEY (league, season, match_date, home_name, away_name)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fetched_seasons (
                league TEXT    NOT NULL,
                season INTEGER NOT NULL,
                PRIMARY KEY (league, season)
            )
        """)
        conn.commit()
        # Migrate existing DBs: add odds columns if absent, force re-download
        _migrate_odds_columns(conn)


def _migrate_odds_columns(conn: sqlite3.Connection) -> None:
    """Add B365/PS odds columns to existing DBs and force re-download if needed."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(fdco_matches)").fetchall()}
    new_cols = [c for c in ("b365_h", "b365_d", "b365_a", "psh", "psd", "psa") if c not in cols]
    if new_cols:
        for col in new_cols:
            conn.execute(f"ALTER TABLE fdco_matches ADD COLUMN {col} REAL")
        # Clear season cache so data is re-downloaded with odds columns
        conn.execute("DELETE FROM fetched_seasons")
        print("    [fdco] Schema actualizado (cuotas B365/Pinnacle) — re-descargando datos...")
        conn.commit()


def _is_cached(league: str, season: int) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM fetched_seasons WHERE league=? AND season=?",
            (league, season),
        ).fetchone()
    return row is not None


def _save_to_cache(league: str, season: int, rows: list[dict]) -> None:
    with _connect() as conn:
        for r in rows:
            conn.execute(
                """INSERT OR IGNORE INTO fdco_matches
                   (league, season, match_date, home_name, away_name,
                    home_goals, away_goals, xg_home, xg_away,
                    b365_h, b365_d, b365_a, psh, psd, psa)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (league, season, r["match_date"], r["home_name"], r["away_name"],
                 r["home_goals"], r["away_goals"], r.get("xg_home"), r.get("xg_away"),
                 r.get("b365_h"), r.get("b365_d"), r.get("b365_a"),
                 r.get("psh"),   r.get("psd"),    r.get("psa")),
            )
        conn.execute(
            "INSERT OR IGNORE INTO fetched_seasons VALUES (?,?)",
            (league, season),
        )
        conn.commit()


def _load_from_cache(league: str, season: int) -> list[dict]:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT match_date, home_name, away_name,
                      home_goals, away_goals, xg_home, xg_away,
                      b365_h, b365_d, b365_a, psh, psd, psa
               FROM fdco_matches WHERE league=? AND season=?""",
            (league, season),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# CSV download & parse
# ---------------------------------------------------------------------------

def _season_code(year: int) -> str:
    """2021 → '2122', 2022 → '2223', 2020 → '2021' ..."""
    return f"{str(year)[2:]}{str(year + 1)[2:]}"


def _parse_date(raw: str) -> str | None:
    """Parse DD/MM/YY or DD/MM/YYYY → YYYY-MM-DD."""
    raw = raw.strip()
    if not raw:
        return None
    for fmt in ("%d/%m/%y", "%d/%m/%Y"):
        try:
            from datetime import datetime
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _download_csv(league: str, season: int) -> list[dict]:
    """Download CSV from football-data.co.uk and return raw parsed rows."""
    code = _FDCO_LEAGUE_CODES.get(league)
    if not code:
        return []
    url = f"{_BASE_URL}/{_season_code(season)}/{code}.csv"
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20,
        )
        resp.raise_for_status()
    except Exception as exc:
        print(f"    [fdco] WARNING: {league} {season} no disponible: {exc}")
        return []

    rows = []
    reader = csv.DictReader(io.StringIO(resp.text))
    for row in reader:
        home = row.get("HomeTeam", "").strip()
        away = row.get("AwayTeam", "").strip()
        if not home or not away:
            continue
        match_date = _parse_date(row.get("Date", ""))
        if not match_date:
            continue
        try:
            hg = int(row["FTHG"])
            ag = int(row["FTAG"])
        except (KeyError, ValueError):
            continue

        entry: dict = {
            "match_date": match_date,
            "home_name":  home,
            "away_name":  away,
            "home_goals": hg,
            "away_goals": ag,
        }

        # xG columns — present in most leagues from ~2019/20 onward
        try:
            entry["xg_home"] = float(row["HomeXG"]) if row.get("HomeXG") else None
            entry["xg_away"] = float(row["AwayXG"]) if row.get("AwayXG") else None
        except (ValueError, TypeError):
            pass

        # Corners — HC (home corners) / AC (away corners)
        try:
            hc = int(row["HC"]) if row.get("HC") else None
            ac = int(row["AC"]) if row.get("AC") else None
            if hc is not None and ac is not None:
                entry["home_corners"] = hc
                entry["away_corners"] = ac
        except (ValueError, TypeError):
            pass

        # Referee
        ref = row.get("Referee", "").strip()
        if ref:
            entry["referee"] = ref

        # Cards — HY (home yellow) / AY (away yellow)
        try:
            hy = int(row["HY"]) if row.get("HY") else None
            ay = int(row["AY"]) if row.get("AY") else None
            if hy is not None:
                entry["home_yellow"] = hy
            if ay is not None:
                entry["away_yellow"] = ay
        except (ValueError, TypeError):
            pass

        # Bookmaker odds — B365 (most coverage) and Pinnacle/PS (sharpest)
        for src_h, src_d, src_a, dst_h, dst_d, dst_a in [
            ("B365H", "B365D", "B365A", "b365_h", "b365_d", "b365_a"),
            ("PSH",   "PSD",   "PSA",   "psh",    "psd",    "psa"),
        ]:
            try:
                h = float(row[src_h]) if row.get(src_h) else None
                d = float(row[src_d]) if row.get(src_d) else None
                a = float(row[src_a]) if row.get(src_a) else None
                if h and d and a and h > 1.0 and d > 1.0 and a > 1.0:
                    entry[dst_h], entry[dst_d], entry[dst_a] = h, d, a
            except (ValueError, TypeError, KeyError):
                pass

        rows.append(entry)

    return rows


# ---------------------------------------------------------------------------
# Convert raw rows → internal match dicts
# ---------------------------------------------------------------------------

def _to_match_dict(
    row: dict,
    league: str,
    season: int,
    registry: dict[str, int],
    registry_keys: list[str],
) -> dict:
    """Build a match dict compatible with the rest of the pipeline."""
    home_id = _resolve_id(row["home_name"], league, registry, registry_keys)
    away_id = _resolve_id(row["away_name"], league, registry, registry_keys)

    m: dict = {
        "id":     _synthetic_id(league, f"{row['match_date']}{row['home_name']}{row['away_name']}"),
        "utcDate": f"{row['match_date']}T12:00:00Z",
        "status": "FINISHED",
        "season": {"id": season},
        "_league_code": league,
        "_fdco": True,   # marker so downstream can identify fdco records if needed
        "homeTeam": {"id": home_id, "name": row["home_name"]},
        "awayTeam": {"id": away_id, "name": row["away_name"]},
        "score": {
            "fullTime": {
                "home": row["home_goals"],
                "away": row["away_goals"],
            }
        },
    }

    # Inject xG directly (skips Understat lookup for these matches)
    if row.get("xg_home") is not None:
        m["_xg_home"] = row["xg_home"]
        m["_xg_away"] = row["xg_away"]

    # Inject corners for backtest validation
    if row.get("home_corners") is not None and row.get("away_corners") is not None:
        m["_hc"]             = row["home_corners"]
        m["_ac"]             = row["away_corners"]
        m["_total_corners"]  = row["home_corners"] + row["away_corners"]

    # Inject referee and cards for referee model
    if row.get("referee"):
        m["_referee"]     = row["referee"]
    if row.get("home_yellow") is not None:
        m["_home_yellow"] = row["home_yellow"]
    if row.get("away_yellow") is not None:
        m["_away_yellow"] = row["away_yellow"]

    # Inject bookmaker odds for value bet backtesting
    # Prefer Pinnacle (sharpest), fall back to B365 (most coverage)
    if row.get("psh") and row.get("psd") and row.get("psa"):
        m["_bk_h"] = row["psh"]
        m["_bk_d"] = row["psd"]
        m["_bk_a"] = row["psa"]
        m["_bk_source"] = "PS"
    elif row.get("b365_h") and row.get("b365_d") and row.get("b365_a"):
        m["_bk_h"] = row["b365_h"]
        m["_bk_d"] = row["b365_d"]
        m["_bk_a"] = row["b365_a"]
        m["_bk_source"] = "B365"

    return m


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_season(league: str, season: int) -> list[dict]:
    """
    Return raw cached rows for one league+season.
    Downloads from football-data.co.uk on first call; uses cache thereafter.
    """
    _init_db()
    if _is_cached(league, season):
        return _load_from_cache(league, season)

    print(f"    [fdco] Descargando {league} {season}...")
    rows = _download_csv(league, season)
    if rows:
        _save_to_cache(league, season, rows)
        print(f"    [fdco] {league} {season}: {len(rows)} partidos cacheados.")
    else:
        print(f"    [fdco] {league} {season}: sin datos.")
    time.sleep(_REQUEST_DELAY)
    return rows


def augment_historical(
    fd_matches: list[dict],
    league_filter: str | None = None,
) -> list[dict]:
    """
    Download football-data.co.uk data for _FDCO_SEASONS and prepend to
    fd_matches (older data first so temporal models see correct order).

    Parameters
    ----------
    fd_matches    : existing list from fetcher.load_historical_data()
    league_filter : if set, only augment that league (mirrors --league flag)

    Returns
    -------
    Combined list: fdco matches (2020-2022) + original fd.org matches (2023+)
    """
    leagues = (
        {league_filter: _FDCO_LEAGUE_CODES[league_filter]}
        if league_filter and league_filter in _FDCO_LEAGUE_CODES
        else _FDCO_LEAGUE_CODES
    )

    # Build registry from existing fd.org data
    registry = build_registry(fd_matches)
    registry_keys = list(registry.keys())

    fdco_matches: list[dict] = []
    total_raw = 0

    for league in leagues:
        for season in _FDCO_SEASONS:
            rows = fetch_season(league, season)
            total_raw += len(rows)
            for row in rows:
                m = _to_match_dict(row, league, season, registry, registry_keys)
                fdco_matches.append(m)

    # Count how many got real (positive) IDs
    matched = sum(
        1 for m in fdco_matches
        if m["homeTeam"]["id"] > 0 and m["awayTeam"]["id"] > 0
    )
    print(f"    [fdco] {len(fdco_matches)} partidos históricos añadidos "
          f"({matched} con ID real, {len(fdco_matches)-matched} con ID sintético).")

    # Prepend older data, then sort by date to be safe
    combined = fdco_matches + fd_matches
    combined.sort(key=lambda m: m.get("utcDate", ""))
    return combined


# ---------------------------------------------------------------------------
# Odds enrichment (attach market odds to existing matches, e.g. 2023+ fd.org)
# ---------------------------------------------------------------------------

def _name_sim(a: str, b: str) -> float:
    """Token-aware name similarity in [0,1]; 1.0 when one token set ⊆ the other."""
    if a == b:
        return 1.0
    ta, tb = set(a.split()), set(b.split())
    if ta and tb and (ta <= tb or tb <= ta):
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _fdco_season_for(date_str: str) -> int | None:
    """Map YYYY-MM-DD to its fdco season year (Aug–Jul span: month>=7 → year)."""
    try:
        y, mth = int(date_str[:4]), int(date_str[5:7])
    except (ValueError, TypeError):
        return None
    return y if mth >= 7 else y - 1


def enrich_with_odds(
    matches: list[dict],
    league_filter: str | None = None,
) -> list[dict]:
    """
    Attach market odds (_bk_h/_bk_d/_bk_a + _bk_source) to matches that lack them
    by matching against football-data.co.uk CSVs for the seasons they span.

    Unlike augment_historical (which *adds* 2020-2022 fdco rows as new matches),
    this *enriches* the matches already present — e.g. 2023+ from football-data.org,
    which carries no odds — so the backtest can compute value-bet ROI at market odds.

    Matching key: (date ±1 day, final score) with team-name similarity as the
    tiebreak. Score+date is language-agnostic and sidesteps the name-format gap
    between the two sources (full names vs abbreviations, accents) that defeats
    pure fuzzy matching for PD/FL1. Matches already carrying _bk_* are skipped.
    Prefers Pinnacle (sharpest line), falls back to B365.

    Mutates and returns `matches`; prints a coverage summary.
    """
    from collections import defaultdict
    from datetime import datetime, timedelta

    leagues = (
        [league_filter] if league_filter and league_filter in _FDCO_LEAGUE_CODES
        else list(_FDCO_LEAGUE_CODES)
    )

    # Which (league, season) pairs do the odds-less matches span?
    need: dict[str, set[int]] = defaultdict(set)
    for m in matches:
        if m.get("_bk_h"):
            continue
        lg = m.get("_league_code")
        if lg not in leagues:
            continue
        season = _fdco_season_for(m.get("utcDate", "")[:10])
        if season is not None:
            need[lg].add(season)
    if not need:
        return matches

    # Per-league date index of fdco rows that carry odds
    index: dict[str, dict[str, list[dict]]] = {}
    for lg, seasons in need.items():
        by_date: dict[str, list[dict]] = defaultdict(list)
        for season in sorted(seasons):
            for r in fetch_season(lg, season):
                if r.get("psh") or r.get("b365_h"):
                    by_date[r["match_date"]].append(r)
        index[lg] = by_date

    matched = eligible = 0
    for m in matches:
        if m.get("_bk_h"):
            continue
        lg = m.get("_league_code")
        if lg not in index:
            continue
        sc = (m.get("score") or {}).get("fullTime") or {}
        hg, ag = sc.get("home"), sc.get("away")
        d = m.get("utcDate", "")[:10]
        if hg is None or ag is None or not d:
            continue
        eligible += 1
        h = _norm(m.get("homeTeam", {}).get("name", ""))
        a = _norm(m.get("awayTeam", {}).get("name", ""))
        try:
            base = datetime.strptime(d, "%Y-%m-%d")
        except ValueError:
            continue

        cands = []
        for off in (0, -1, 1):
            dd = (base + timedelta(days=off)).strftime("%Y-%m-%d")
            for r in index[lg].get(dd, []):
                if r["home_goals"] == hg and r["away_goals"] == ag:
                    cands.append(r)
        if not cands:
            continue

        best = max(cands, key=lambda r: _name_sim(h, _norm(r["home_name"]))
                                       + _name_sim(a, _norm(r["away_name"])))
        # plausibility floor — never accept a same-score row whose names disagree
        if (_name_sim(h, _norm(best["home_name"])) < 0.4
                and _name_sim(a, _norm(best["away_name"])) < 0.4):
            continue

        if best.get("psh"):
            m["_bk_h"], m["_bk_d"], m["_bk_a"] = best["psh"], best["psd"], best["psa"]
            m["_bk_source"] = "Pinnacle"
        else:
            m["_bk_h"], m["_bk_d"], m["_bk_a"] = best["b365_h"], best["b365_d"], best["b365_a"]
            m["_bk_source"] = "B365"
        matched += 1

    if eligible:
        print(f"    [fdco] Cuotas de mercado añadidas a {matched}/{eligible} "
              f"partidos ({matched / eligible * 100:.0f}%).")
    return matches
