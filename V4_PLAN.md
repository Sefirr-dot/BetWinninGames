# BetWinninGames v4.0 — Plan de Desarrollo

Generado: 2026-03-08
Estado actual: v3.x completado (backtest multi-liga, AI Advisor, combinadas múltiples, meta-learner con early stopping)

---

## Estado actual (v3.x)

El sistema predice correctamente la dirección general pero tiene puntos ciegos críticos:
- El modelo de corners **nunca ha sido validado** (coeficientes hardcodeados, display-only)
- Las métricas del tracker son solo globales (imposible saber si PL funciona mejor que FL1)
- El valor real de los picks de value bet no se trackea en retrospectiva
- BTTS no separa tasas por venue (en casa vs fuera)
- El confidence solo mide varianza en el outcome ganador, no en la distribución completa

---

## Fase 1 — Diagnóstico real (PRIORIDAD MÁXIMA)

> "Si no lo mides, no lo puedes mejorar"

### 1.1 Tracker con desglose por liga, mercado y estrellas

**Archivo:** `tracker.py` → `compute_metrics()`
**Problema:** Devuelve un solo dict global. No sabes si PL tiene 65% accuracy o 52%.

**Cambios:**
```
compute_metrics() → métricas globales + breakdown:
  per_league:  { PL: {accuracy, roi, brier, n}, PD: {...}, BL1: {...}, FL1: {...} }
  per_stars:   { 3: {accuracy, roi, n}, 4: {...}, 5: {...} }
  per_market:  { 1x2: {accuracy, roi}, over25: {accuracy}, btts: {accuracy} }
```

**Cambios en `tracker_data.js` + visualizador:**
- Tabla de ROI por liga en la vista TRACK
- Gráfico de accuracy por estrellas (¿son los 5★ realmente más fiables?)
- Sin esto, las Fases 3.2 (Kelly por liga) y 4.2 (ROI en visualizador) no son posibles

---

### 1.2 Backtest por mercado con calibración separada

**Archivo:** `backtest.py` → `compute_metrics()`
**Problema:** Solo calibra 1X2. Over 2.5 y BTTS nunca han tenido calibración propia.

**Cambios:**
```python
compute_metrics() añade:
  calibration_over25: { "0.4-0.5": {n, avg_prob, actual_rate}, "0.5-0.6": {...}, ... }
  calibration_btts:   { ... }
  per_league_over25:  { PL: {accuracy, n}, PD: {...}, ... }
  per_league_btts:    { PL: {accuracy, n}, ... }
```

---

### 1.3 Validación del modelo de corners (CRÍTICO)

**Archivo:** `algorithms/corners.py`
**Problema:** Coeficientes hardcodeados de research papers, nunca backtestado. Nadie sabe si funciona.

**Acción inmediata:**
```python
# backtest.py — añadir tracking de corners si disponible en fdco CSV
"actual_corners": match.get("_total_corners")
# compute_metrics() añade:
corners_mae:      float   # mean absolute error (objetivo < 1.8)
corners_accuracy: float   # % partidos donde over/under corners fue correcto
```

Si MAE > 2.5 corners → rediseñar modelo en Fase 2.1.
Los datos de corners están en los CSVs de football-data.co.uk (columnas HC, AC).

---

### 1.4 Hindsight value tracking

**Archivo:** `tracker.py`
**Problema:** Una vez resuelto un pick, no sabemos si el "value bet" era real en retrospectiva.

**Cambios:**
```python
# Para cada pick resuelto con market_odds:
hindsight_edge = (1.0 if ganó else 0.0) - (1.0 / market_odds)
# ¿El modelo estaba bien cuando decía edge positivo?

# En TRACKER_METRICS añadir:
avg_hindsight_edge_by_league: { PL: float, PD: float, ... }
avg_hindsight_edge_by_stars:  { 3: float, 4: float, 5: float }
```

---

## Fase 2 — Modelos mejorados

### 2.1 Modelo de corners específico por equipo

**Archivo:** `algorithms/corners.py`
**Problema:** Solo usa `total_xG` como feature. Man City genera 11 corners/partido, Burnley 4.

**Nuevo modelo:**
```
Features:
  - team_corner_rate_home / team_corner_rate_away  (media histórica ponderada)
  - opponent_corners_conceded_rate                 (cuántos concede el rival)
  - total_xG                                       (ya existe)
  - home_advantage_corners                         (+1.2 corners de media en casa)

Implementación:
  - Extraer corners de fdco CSVs (columnas HC=home corners, AC=away corners)
  - Guardar en understat_xg.db o nueva tabla corners_history
  - Regresión simple (scipy.stats.linregress) calibrada contra datos reales
  - Tracking MAE en backtest (objetivo < 1.8)
```

---

### 2.2 BTTS con tasas específicas por venue

**Archivo:** `algorithms/btts.py` → `_scoring_rate()` y `_conceding_rate()`
**Problema:** Calcula tasas sin separar partidos en casa de partidos fuera.

**Cambio quirúrgico:**
```python
def _scoring_rate(team_id, matches, as_home=True, min_matches=5):
    venue_matches = [m for m in matches if _played_at_home(m, team_id) == as_home]
    if len(venue_matches) < min_matches:
        return _scoring_rate(team_id, matches, as_home=None)  # fallback global
    # Decay temporal sobre venue_matches
    return weighted_rate
```

Impacto esperado: +2-3% accuracy BTTS en ligas defensivas (FL1, BL1).

---

### 2.3 Confidence mejorado en ensemble

**Archivo:** `algorithms/ensemble.py` → `_confidence()`
**Problema:** Solo mide varianza en el outcome ganador. No detecta desacuerdo en draw/away probs.

**Cambio:**
```python
def _confidence_v2(dc, elo, form, h2h):
    # 1. Medir varianza en los 3 outcomes, no solo el ganador
    var_home = variance([d["prob_home"] for d in models])
    var_draw = variance([d["prob_draw"] for d in models])
    var_away = variance([d["prob_away"] for d in models])
    total_variance = var_home + var_draw + var_away

    # 2. Penalizar si los modelos predicen ganadores DIFERENTES
    winners = [max(d, key=lambda k: d[k]) for d in models]
    consensus_rate = Counter(winners).most_common(1)[0][1] / len(winners)

    # 3. Threshold mínimo: confidence < 0.65 → forzar stars = max(stars-1, 1)
```

---

### 2.4 Dos ligas nuevas: Serie A + Eredivisie

**Archivo:** `config.py`
**Cambio mínimo:**
```python
LEAGUES = {
    "PL":  2021, "PD": 2014, "BL1": 2002, "FL1": 2015,
    "SA":  2019,  # Serie A (Italia) — Understat: Serie_A
    "DED": 2003,  # Eredivisie (Países Bajos) — Understat: Eredivisie
}
```

**Ajustes adicionales en config.py:**
```python
ELO_HOME_BONUS_BY_LEAGUE["SA"]  = 105   # Italia: alta ventaja local
ELO_HOME_BONUS_BY_LEAGUE["DED"] = 85    # Eredivisie: moderada
DRAW_RATE_BY_LEAGUE["SA"]       = 0.275 # Serie A: muchos empates
DRAW_RATE_BY_LEAGUE["DED"]      = 0.245
DC_XI_BY_LEAGUE["SA"]           = 0.006
DC_XI_BY_LEAGUE["DED"]          = 0.006
VALUE_BET_EDGE_THRESHOLD_BY_LEAGUE["SA"]  = 0.08
VALUE_BET_EDGE_THRESHOLD_BY_LEAGUE["DED"] = 0.09
UNDERSTAT_LEAGUES["SA"]         = "Serie_A"
UNDERSTAT_LEAGUES["DED"]        = "Eredivisie"
```

Más ligas = más picks = meta-learner llega a 200 picks reales antes.

---

## Fase 3 — Inteligencia de mercado

### 3.1 Tracking de movimiento de cuotas

**Archivos:** `odds_fetcher.py`, `value_detector.py`, `reporter.py`
**Idea:** Cuotas que se acortan mucho (2.40→1.95) indican sharp money. Si el modelo también recomienda ese pick → señal muy fuerte.

**Implementación:**
```python
# odds_fetcher.py — guardar timestamp en CSV
home_team,away_team,odds_1,odds_x,odds_2,fetched_at

# Nueva tabla SQLite: odds_history (en cache/football_data.db)
(match_id, fetched_at, odds_1, odds_x, odds_2, league, match_date)

# value_detector.py — nueva señal
if prev_odds and current_odds:
    movement = prev_odds / current_odds  # > 1.10 = sharp money ese outcome
    if movement > 1.10 and our_model_agrees:
        edge_bonus = 0.02  # refuerza el edge
```

Añadir `oddsMovement` al schema de `ALL_MATCHES` en predictions.js.

---

### 3.2 Kelly sizing dinámico por liga

**Archivo:** `algorithms/value_detector.py`
**Problema:** Kelly igual para todas las ligas. Si PL tiene ROI +12% y FL1 tiene ROI -3%, deberían tener Kelly diferente.

**Implementación:**
```python
# value_detector.py — cargar desde tracker metrics
def _load_league_kelly_multipliers() -> dict:
    # Lee cache/tracker_metrics.json (generado por tracker.py)
    # Calcula multiplier = max(0.3, min(1.5, 1.0 + league_roi))
    return {"PL": 1.2, "PD": 0.9, "BL1": 0.5, "FL1": 0.8}

KELLY_MULTIPLIER = _load_league_kelly_multipliers()
```

**Requiere:** Fase 1.1 (per-league ROI en tracker) completada primero.

---

### 3.3 Match context tags

**Archivo:** nuevo `match_context.py`
**Idea:** Ciertos contextos tienen ROI sistemáticamente mejor (top-6 clash, equipos en racha, etc.)

```python
def classify(home_id, away_id, league, standings, form, h2h) -> list[str]:
    tags = []
    if both_top6(standings):        tags.append("top6_clash")
    if elo_diff < 50:               tags.append("even_match")
    if home_win_streak >= 5:        tags.append("home_in_form")
    if h2h_dominant(h2h):          tags.append("h2h_clear_favorite")
    if relegation_battle(standings): tags.append("relegation_6ptr")
    return tags

# db_picks.py — guardar tags en nueva columna match_tags TEXT
# tracker.py  — compute_metrics() añade ROI por tag
```

---

## Fase 4 — Automatización y UX

### 4.1 Scheduler automático (Windows Task Scheduler)

**Archivo nuevo:** `run_weekend.bat`
```batch
@echo off
cd C:\Users\MSI\Desktop\REPOS\BetWinninGames
python main.py >> logs\main_%date:/=-%_.log 2>&1
```

Configurar en Task Scheduler:
- Viernes 10:00 → `python main.py`
- Sábado 09:00 → `python main.py`
- Lunes 10:00 → `python tracker.py` (resolver resultados del fin de semana)

---

### 4.2 ROI por liga en el visualizador

**Archivo:** `visualizador/index.html` → vista TRACK
**Requiere:** Fase 1.1 completada (per-league en TRACKER_METRICS)

Tabla en vista TRACK:
```
Liga    Picks   Accuracy   ROI      Brier   Value Bets
PL       142    64.1%     +8.3%    0.401      23 (ROI +11.2%)
PD       118    61.0%     +4.2%    0.418      18 (ROI +6.7%)
BL1       97    58.2%     -1.1%    0.441       8 (ROI +0.8%)
FL1      103    60.7%     +2.8%    0.423      15 (ROI +4.0%)
```

---

### 4.3 Persistencia de estado en el visualizador

**Archivo:** `visualizador/index.html`
**Cambio mínimo:**
```javascript
// Guardar estado al cambiar
function saveState() {
    localStorage.setItem('bwg_state', JSON.stringify({
        activeDate, activeLeague, sortMode
    }));
}
// Restaurar al cargar
const saved = JSON.parse(localStorage.getItem('bwg_state') || '{}');
activeDate   = saved.activeDate   || dates[0] || 'ALL';
activeLeague = saved.activeLeague || 'ALL';
sortMode     = saved.sortMode     || 'stars';
```

---

## Orden de ejecución recomendado

| # | Fase | Tarea | Impacto | Esfuerzo | Archivos principales |
|---|---|---|---|---|---|
| 1 | 1.1 | Tracker per-liga/mercado/estrellas | 🔴 Crítico | Medio | `tracker.py`, `index.html` |
| 2 | 1.3 | Validar corners (¿MAE aceptable?) | 🔴 Crítico | Bajo | `backtest.py`, `corners.py` |
| 3 | 1.2 | Calibración Over/BTTS en backtest | 🟠 Alto | Bajo | `backtest.py` |
| 4 | 1.4 | Hindsight value tracking | 🟠 Alto | Bajo | `tracker.py` |
| 5 | 2.1 | Corners con features de equipo | 🟠 Alto | Medio | `corners.py`, `fdco_fetcher.py` |
| 6 | 2.4 | Serie A + Eredivisie | 🟡 Medio | Muy bajo | `config.py` |
| 7 | 2.2 | BTTS venue-specific | 🟡 Medio | Bajo | `btts.py` |
| 8 | 2.3 | Confidence full-distribution | 🟡 Medio | Bajo | `ensemble.py` |
| 9 | 3.2 | Kelly dinámico por liga | 🟠 Alto | Bajo | `value_detector.py` |
| 10 | 3.1 | Movimiento de cuotas | 🟡 Medio | Medio | `odds_fetcher.py`, `value_detector.py` |
| 11 | 3.3 | Match context tags | 🟡 Medio | Medio | nuevo `match_context.py` |
| 12 | 4.1 | Scheduler automático | 🟡 Medio | Muy bajo | nuevo `run_weekend.bat` |
| 13 | 4.2 | ROI por liga en visualizador | 🟡 Medio | Bajo | `index.html` |
| 14 | 4.3 | localStorage estado | 🟢 Bajo | Muy bajo | `index.html` |

---

## Criterios de éxito v4.0

| Métrica | Actual (estimado) | Objetivo v4.0 |
|---|---|---|
| Accuracy 1X2 global | ~60% | ≥ 63% |
| Accuracy Over 2.5 | ~57% | ≥ 61% |
| Accuracy BTTS | ~58% | ≥ 62% |
| Corners MAE | desconocido | < 1.8 |
| ROI picks 4-5★ (cuotas justas) | +8% est. | ≥ +10% |
| Value bet ROI (cuotas reales) | variable | ≥ +5% sostenido |
| Picks reales en DB para meta-learner | 38 | ≥ 200 |

---

## Notas técnicas para la implementación

- **Empezar siempre por el backtest** antes de cambiar un modelo — validar que la métrica mejora.
- **Los corners de fdco** están en columnas `HC` (home corners) y `AC` (away corners) de los CSVs. `fdco_fetcher.py` ya los lee pero los descarta — solo hay que persistirlos.
- **Serie A en Understat** usa el slug `Serie_A` (con mayúscula). Eredivisie no está en Understat — usar solo datos de fdco y football-data.org.
- **El meta-learner se activa con 200+ picks reales** (source='live'). Con 6 ligas y picks semanales, se llegaría en ~6-8 semanas.
- **No tocar `meta_learner.pkl`** hasta tener ≥200 picks reales resueltos. El ensemble clásico es más robusto con pocos datos.
