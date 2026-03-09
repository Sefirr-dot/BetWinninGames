# BetWinninGames — Hoja de Ruta de Mejoras

Objetivo: maximizar ROI real, no solo precisión de modelo.
Referencia backtest: FL1 2023-2024 → Accuracy 53.1% | Value Bets ROI +28.3% (60.2% acc, n=520)

---

## FASE 1 — Correcciones críticas (alto impacto, bajo esfuerzo)

### 1.1 [x] Corregir `_kelly_score` en ensemble.py
**Problema**: usa `prob * 0.90` como cuota implícita siempre, sin importar si hay cuotas reales.
Esto contamina el `profitability_score` y por tanto las estrellas y el parlay.
**Fix**: cuando hay `market_odds` en el contexto, usar la cuota real del outcome ganador.
Cuando no hay cuotas, usar estimación por liga (en lugar de 0.9 genérico).
**Archivos**: `algorithms/ensemble.py` (`_kelly_score`, `predict_match`)

### 1.2 [x] Añadir detección de valor en Over 2.5 y BTTS
**Problema**: `value_detector.find_edges()` solo detecta valor en 1X2.
El backtest muestra: O2.5 accuracy 55.2%, BTTS accuracy 52.8%. Con cuotas de mercado
reales hay margen de valor explotable en estos mercados.
**Fix**: en `find_edges()`, si el CSV tiene columnas `odds_o25`/`odds_btts`, comparar
contra `dc_over25` y `btts_prob`. Si no, añadir `VALUE_BET_O25_THRESHOLD` en config.
**Archivos**: `algorithms/value_detector.py`, `config.py`, `odds_fetcher.py`

### 1.3 [x] Corregir sesgo de calibración en picks de alta confianza
**Problema detectado en backtest**:
- Prob predicha 60-70%: real 69.3% → modelo subestima +4.7%
- Prob predicha 70%+: real 82.9% → modelo subestima +7.8% (¡significativo!)
El modelo es demasiado conservador en picks fuertes: un pick que dice 75% realmente gana 83%.
**Fix**: aplicar corrección post-calibración en ensemble.py para probs > 0.60.
Fórmula simple: `p_corr = p + alpha * max(0, p - 0.55) * (1 - p)` con alpha≈0.25.
Alternativamente, bajar `HIGH_CONFIDENCE_THRESHOLD` de 0.62 → 0.58.
**Archivos**: `algorithms/ensemble.py`, `config.py`

### 1.4 [x] Filtro de cuotas mínimas en value bets
**Problema**: cuotas < 1.40 son tan eficientes que el "edge" del modelo es probablemente ruido.
Detectar edge de 5% en una cuota de 1.20 (implied 83%) requiere exactitud del 88%, imposible.
**Fix**: añadir `VALUE_BET_MIN_ODDS = 1.40` en config. Filtrar en `find_edges()`.
**Archivos**: `algorithms/value_detector.py`, `config.py`

---

## FASE 2 — Mejoras de modelo (impacto alto, esfuerzo medio)

### 2.1 [x] Feature de tendencia de forma (momentum)
**Problema**: `form.py` calcula nivel medio de forma, no tendencia.
Un equipo con últimos 3 resultados: W-W-W vs uno con W-D-L pueden tener el mismo
promedio ponderado pero el primero está "caliente".
**Fix**: añadir `form_trend` = diferencia entre forma de últimos 3 vs anteriores 3-8.
Exportar como feature al ensemble y al meta_learner.
**Archivos**: `algorithms/form.py`, `algorithms/ensemble.py`, `algorithms/meta_learner.py`

### 2.2 [x] Detector especializado de empates
**Problema**: los empates son el resultado más difícil de predecir (draw rate ~24-27%).
El nudge actual del 25% hacia `DRAW_RATE_BY_LEAGUE` es genérico.
**Fix**: crear `draw_score` compuesto por:
- Distancia Elo < 100 puntos → ambos equipos muy igualados
- λ + μ < 2.5 → partido esperado de pocos goles (más probable empate)
- H2H draw rate > 35% cuando disponible
- Ambos equipos con forma "-" o "+" (ni muy buena ni muy mala)
Cuando `draw_score` es alto, amplificar el nudge al 40% en lugar del 25%.
**Archivos**: `algorithms/ensemble.py`

### 2.3 [ ] Filtro de cuotas de mercado obsoletas
**Problema**: si el CSV de cuotas tiene más de 6 horas antes del partido, las cuotas
pueden haber movido significativamente (lesiones, noticias). El blending con cuotas
viejas puede dañar el modelo.
**Fix**: leer timestamp del CSV y reducir `MARKET_BLEND_WEIGHT` si el partido es
en < 2 horas pero las cuotas tienen > 4 horas. Loggear warning.
**Archivos**: `odds_fetcher.py`, `algorithms/ensemble.py`

### 2.4 [x] Score de confianza basado en varianza (no en votos)
**Problema**: `_confidence()` cuenta cuántos modelos votan por el mismo ganador.
Pero DC al 65% + Elo al 64% + Form al 63% debería ser más confiable que
DC al 70% + Elo al 55% + Form al 40% (aunque ambos den mayoría para Home).
**Fix**: calcular varianza de las probabilidades del outcome ganador entre modelos.
`confidence = 1 - std(probs_del_ganador) / mean(probs_del_ganador)` normalizado.
**Archivos**: `algorithms/ensemble.py` (`_confidence`)

### 2.5 [ ] Selección de parlay por valor esperado, no solo estrellas
**Problema**: `getSuggestedParlay()` toma los 4 picks con mayor `stars >= 3`.
Debería priorizar picks donde hay valor real (edge positivo con cuotas de mercado).
**Fix**: cuando hay `valueBets` en el pick, priorizar esas piernas en el parlay.
Añadir `combinedEV` al parlay = producto de (1 + edge_i) en lugar de solo odds.
**Archivos**: `visualizador/index.html`

---

## FASE 3 — Datos y features avanzadas (impacto alto, esfuerzo alto)

### 3.1 [ ] Añadir Serie A (SA) como liga completa
**Observado**: las predicciones incluyen partidos SA (Juventus, Atalanta) pero
`LEAGUES` en config.py no tiene SA. Investigar origen y unificar.
**Fix**: añadir `"SA": 2019` a LEAGUES, añadir a UNDERSTAT_LEAGUES, DC_XI_BY_LEAGUE,
ELO_HOME_BONUS_BY_LEAGUE, DRAW_RATE_BY_LEAGUE, VALUE_BET_EDGE_THRESHOLD_BY_LEAGUE.
**Archivos**: `config.py`

### 3.2 [ ] Tracking de movimiento de línea de cuotas
**Concepto**: la diferencia entre cuotas de apertura y cierre es la señal más potente
de "dinero inteligente" (sharp money). Si la cuota de Home baja de 2.50 → 1.90, es que
el mercado sabe algo.
**Fix**: guardar cuotas en timestamped DB. Al predecir, comparar cuota actual vs
apertura. Añadir `line_movement` como feature al meta_learner.
**Archivos**: nuevo `line_tracker.py`, `config.py`, `algorithms/meta_learner.py`

### 3.3 [ ] Seed automático del meta_learner vía backtest ampliado
**Problema**: el meta_learner necesita 200+ picks resueltos para activarse.
Con partidos reales, tardará semanas. El backtest puede sembrar miles de picks.
**Fix**: ejecutar `backtest.py --seed-db` para todas las ligas y temporadas 2021-2024.
Esto activa calibrator, weight_optimizer, y meta_learner de inmediato.
**Comando**: `python backtest.py --league PL --seasons 2021 2022 2023 2024 --seed-db`
(repetir para PD, BL1, FL1)

---

## FASE 4 — Estrategia de apuestas (impacto directo en ganancias)

### 4.1 [ ] Bankroll management dinámico
**Concepto**: en lugar de Kelly fijo al 25%, implementar Kelly fraccionado adaptativo:
- Full Kelly solo para picks ≥ 5★ con edge ≥ 15%
- Half Kelly para picks 4★ con edge 10-15%
- Quarter Kelly para picks 3★ con edge 5-10%
Limitar exposición total por jornada a 10% del bankroll.
**Archivos**: `algorithms/value_detector.py` (nuevo campo `kelly_fraction_adjusted`)

### 4.2 [ ] Alertas de valor extremo (early value)
**Concepto**: las mejores cuotas están disponibles 72-48h antes del partido.
El modelo puede detectar si las cuotas actuales son "early money" vs "closing line".
**Fix**: añadir threshold `EARLY_VALUE_EDGE = 0.15` para alertas de máximo valor.
Integrar en Telegram notification con flag especial.
**Archivos**: `telegram_notifier.py`, `config.py`

---

## Orden de implementación recomendado

1. Fase 1.4 (filtro odds mínimas) — 30 min — no hay riesgo
2. Fase 1.1 (fix Kelly score) — 1h — impacta directamente las estrellas
3. Fase 1.3 (corrección calibración alta confianza) — 1h — más 5★ reales
4. Fase 1.2 (O2.5 + BTTS en value detector) — 2h — +40% oportunidades de valor
5. Fase 2.4 (confianza por varianza) — 1h — mejor separación picks buenos/malos
6. Fase 2.2 (draw detector) — 2h — tackles el punto débil histórico
7. Fase 3.3 (seed meta_learner) — 30 min ejecución — activa el modelo más potente
8. Fase 3.1 (añadir SA) — 1h — más mercado
9. Fase 2.1 (form momentum) — 3h — feature de alta calidad

---

## Revisión final

- [ ] Backtest completo post-mejoras con todas las ligas
- [ ] Comparar ROI value bets antes/después
- [ ] Verificar que probs suman 1.0000 en todos los paths
- [ ] Verificar calibración mejorada en buckets 0.6-0.7 y 0.7+
