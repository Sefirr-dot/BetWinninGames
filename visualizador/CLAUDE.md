# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A single-file static frontend app (`index.html`) that displays football match predictions from TXT files located in the `PREDICCIONES/` folder. No build system, no dependencies, no server required — open directly in a browser.

## Architecture

The entire application lives in `index.html` with three sections:

### 1. Data layer

`ALL_MATCHES` array at the top of the `<script>` block. Each object represents one match parsed from the TXT prediction files. Fields: `date`, `home/away/homeShort/awayShort`, `league` (BL1/SA/FL1/PD), `time`, `stars` (1–5), `prob1/probX/prob2`, `over25/over25yn`, `btts/bttsyn`, `corners/cornersOver`, `goalsHome/goalsAway`, `eloConf` (float, e.g. `50.8`), `formHome/formAway` (string: `+++` / `++` / `+` / `-` / `--`), `bestScore/bestScoreProb`, `fairOdds`, `h2h` (string summary, e.g. `"Bayern 5V 0E 0D → domina"` or `null`). Fields are `null` when not available (matches without detailed pick analysis).

Two companion arrays:
- `DATES` — ordered list of date strings, e.g. `["2026-03-06","2026-03-07",...]`
- `DATE_LABELS` — map of date string to display label, e.g. `{"2026-03-06":"Jue 06/03",...}`

Both must be updated when adding a new day alongside `ALL_MATCHES`.

### 2. State

Three variables: `activeDate` (date string, `"ALL"`, or `"BEST"`), `activeLeague` (league code or `"ALL"`), `sortMode` (read from `<select>`). Sort options: `stars` (default), `prob`, `time`.

### 3. Rendering

Pure DOM manipulation via `render()` which dispatches between the normal card view and the Best Bets view:

- **Normal view**: calls `buildDayNav()`, `buildLeagueChips()`, `buildSummary()`, and generates card HTML via `renderCard(m)`.
- **Best Bets view** (`activeDate === "BEST"`): hides normal controls/grid, renders `renderBestBetsView()` which shows a suggested parlay box + ranked bet lists per category.

## Best Bets Logic

`calcBestBets()` — scans `ALL_MATCHES` and emits bet objects for matches with `stars >= 2`:
- `victoria`: win probability ≥ 55% for the favored side
- `over25`: `over25yn === true` and `over25 >= 58`
- `btts`: `bttsyn === true` and `btts >= 58`

Each bet has a `score = stars² × prob / 100` used for ranking.

`getSuggestedParlay()` — takes up to 4 legs from `calcBestBets()` filtered to `stars >= 3`, sorted by raw probability, one leg per match (deduped by `date|home` key). Computes `combinedProb` and `combinedOdds` (product of `fairOdds`, only when all legs have it).

`BET_TYPE_CONFIG` maps `victoria / over25 / btts` to display label and color values.

## Data Source TXT Format

Files in `PREDICCIONES/` follow the naming `predicciones_YYYY-MM-DD.txt`. Each file has four sections in order:

1. **RESUMEN EJECUTIVO** — header line with total match and high-confidence pick counts. Not parsed into data; informational only.
2. **Detailed picks** — one block per pick under "PICKS DE ALTA CONFIANZA (★★★★★)" and "PICKS DE CONFIANZA MEDIA (★★★☆☆)". Each block contains: resultado + prob + fairOdds, over2.5 yn + %, btts yn + %, corners over/under + expected count, expected goals home/away, eloConf %, form strings, bestScore + prob, and optionally an H2H line. Some picks include a `! ATENCION:` warning line when H2H history contradicts the model — skip this line when parsing.
3. **VALUE BETS** — matches where the model edge exceeds a threshold. Each entry has: match name, league, bet direction (apuesta), edge %, cuota (odds), kelly %, modelo %, implícita %. These fields (`edge`, `cuota`, `kelly`, `modeloProb`, `implicitaProb`) are not currently stored in `ALL_MATCHES` but the section must be skipped cleanly when parsing.
4. **TODOS LOS PARTIDOS** — summary table with all matches as a compact row: `#  Partido  Liga  1%  X%  2%  O2.5%  BTTS%  ★`

When adding new prediction files, parse both sections: use the detailed section for `corners`, `goalsHome/Away`, `eloConf`, `formHome/Away`, `bestScore`, `fairOdds`, `h2h`; use the summary table for all matches including those without detailed analysis (set detail fields to `null`).

Note: the star rating in the summary table is the authoritative value per match. A match appearing under the ★★★★★ section may show ★★★★☆ in the table — trust the table.

## Adding New Data

Append entries to `ALL_MATCHES`, add the date to `DATES`, and add the label to `DATE_LABELS`. The `bestDate` crown is computed automatically from `dayScore()` (sum of `25 × maxWinProb` for all ★★★★★ picks per day).

## Styling Conventions

- CSS variables defined in `:root` — use them for all colors, never hardcode hex values inline.
- Star tier colors and labels: 5=`--accent-green` (Alta Confianza), 4=`--accent-teal` (Confianza Alta-Media), 3=`--accent-yellow` (Confianza Media), 2=`--accent-orange` (Baja Confianza), 1=`--accent-gray` (Mínima). The 4-star tier appears in the summary table and needs a distinct color variable.
- League badge colors: BL1=blue (`--bl1`), SA=purple (`--sa`), FL1=cyan (`--fl1`), PD=red (`--pd`) — classes `badge-BL1`, `badge-SA`, `badge-FL1`, `badge-PD`.
- Font: `Barlow Condensed` for headings/numbers, `Barlow` for body text (loaded from Google Fonts).
