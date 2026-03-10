# BetWinninGames v6.0 — Betting Autopilot

## Objetivo

Pasar de "herramienta de datos" a "asistente de apuestas".
El usuario abre la app, ve exactamente qué apostar, cuánto y dónde. Un botón. Listo.

---

## Fases

### Fase 1 — Perfil de usuario + Motor de selección (backend ligero, frontend puro)

**Duración estimada: 1 sesión**

#### 1a. Perfil de usuario (`bwg_profile` en localStorage)

```json
{
  "riskLevel":       "balanced",     // "conservative" | "balanced" | "aggressive"
  "maxBetsPerDay":   3,              // 1-5
  "leagues":         ["PL","PD","BL1","FL1"],
  "markets":         ["1x2","over25","btts"],
  "minOdds":         1.50,
  "minStars":        5,              // override global (default: 5)
  "maxExposurePct":  10              // max % bankroll at risk simultaneously
}
```

Setup wizard en primera visita al autopilot (3 preguntas simples, no un formulario).

#### 1b. Motor de selección (`selectAutopilotBets()`)

Algoritmo puro JS, sin servidor:

```
1. Filtra ALL_MATCHES:
   - stars >= profile.minStars
   - league en profile.leagues
   - tiene valueBets con edge positivo
   - odds >= profile.minOdds
   - mercado en profile.markets

2. Puntúa cada pick:
   score = edge × kelly × league_roi_multiplier
   (league_roi_multiplier de TRACKER_METRICS.per_league)

3. Ordena por score DESC, deduplica por partido

4. Limita a profile.maxBetsPerDay picks

5. Calcula stakes:
   - Balanced: Kelly estándar
   - Conservative: Kelly × 0.5
   - Aggressive: Kelly × 1.5
   - Cap: ningún pick > maxExposurePct/2 del bankroll

6. Detección de correlación:
   - Si 2+ picks en la misma liga el mismo día → reduce stakes 20%
   - Si 3+ → reduce 35%

7. Calcula EV en euros por pick:
   EV = stake × edge (en decimal)
```

#### 1c. Vista AUTOPILOT (nueva tab en sidebar)

```
┌──────────────────────────────────────────────────────────┐
│  🤖 Autopilot  [perfil: Balanced ▾]                      │
├──────────────────────────────────────────────────────────┤
│  Este fin de semana · 3 apuestas seleccionadas           │
│  Exposición: €67 (6.7% bankroll) · EV total: +€8.50     │
├──────────────────────────────────────────────────────────┤
│  1. PD ★★★★★  Barcelona vs Atletico                     │
│     Local @1.85 · Stake €32 · EV +€3.84                 │
│     Edge 12.0% · Kelly 6.4%                             │
├──────────────────────────────────────────────────────────┤
│  2. PL ★★★★★  Arsenal vs Chelsea                        │
│     Over 2.5 @1.92 · Stake €25 · EV +€3.00             │
│     Edge 12.0% · Kelly 5.0%                             │
├──────────────────────────────────────────────────────────┤
│  3. PD ★★★★★  Real Madrid vs Sevilla                    │
│     Local @1.72 · Stake €10 · EV +€1.20                │
│     Edge 8.0% · Kelly 2.0%                              │
├──────────────────────────────────────────────────────────┤
│  [✅ Añadir todo al slip]   [🔧 Ajustar perfil]          │
└──────────────────────────────────────────────────────────┘
```

"Añadir todo al slip" → carga los picks en `bwg_slip` con stakes calculados → navega al SLIP.

---

### Fase 2 — Comparativa de casas de apuestas

**Duración estimada: 1 sesión**

#### 2a. Backend: odds por casa (Python)

Modificar `odds_fetcher._fetch_sport()` para guardar también los precios individuales por bookmaker (no solo el máximo). Añadir una nueva llamada con los bookmakers más comunes del mercado EU:

```python
# Bookmakers a comparar (todos disponibles en the-odds-api EU)
_COMPARE_BOOKMAKERS = [
    "bet365", "unibet", "bwin", "betfair_ex_eu",
    "williamhill", "pinnacle", "betway"
]
```

Nuevo CSV: `odds/bk_YYYY-MM-DD.csv` con columna por bookmaker:
```
home_team,away_team,bet365_1,bet365_x,bet365_2,unibet_1,...
```

Coste: 1 API call adicional por liga por run = 4 calls más = 12 calls total/run.
Bien dentro de los 500/mes.

#### 2b. Frontend: "Mejor casa para este pick"

En la vista AUTOPILOT y en los value bets:
- Mostrar qué bookmaker ofrece mejor precio para el outcome recomendado
- "Bet365 @1.92 · Unibet @1.89 · Media @1.85"
- Badge verde en el mejor precio

En el match modal:
- Tabla comparativa de precios por casa

#### 2c. `get_best_bookmaker(home, away, outcome, date)` → `{bk_name, odds, extra_edge}`

Función nueva en `value_detector.py` o en `odds_fetcher.py`.

---

### Fase 3 — Expected Value en euros + display improvements

**Duración estimada: media sesión**

#### 3a. EV en euros en todas las vistas

Actualmente: "Edge +12.0%"
v6: "Edge +12.0% · EV +€4.80 (con €40 stake)"

Fórmula: `EV_eur = stake × edge`

Mostrar en:
- AUTOPILOT (ya incluido en Fase 1)
- VALUE view
- SLIP activo (calcular con el stake introducido)
- Match modal en sección value bets

#### 3b. EV total del fin de semana

En el header del AUTOPILOT y del SLIP:
"EV esperado este fin de semana: +€12.30"

Actualiza en tiempo real cuando el usuario cambia stakes.

#### 3c. ROI histórico del usuario (en SLIP Historial)

Añadir al stats bar:
- "EV apostado total: €X" (sum of EV at time of bet)
- "EV realizado: €Y" (actual P&L)
- "EV ratio: Z%" — ¿estás capturando el edge que el modelo predice?

---

### Fase 4 — Ollama conversacional

**Duración estimada: 1 sesión**

#### 4a. Chat widget en el AUTOPILOT

Un cuadro de texto debajo del plan de apuestas:

```
┌──────────────────────────────────────────────────┐
│  💬 Pregunta al modelo                           │
│  ┌────────────────────────────────────────────┐  │
│  │ ¿Debería apostar el Barça hoy?            │  │
│  └────────────────────────────────────────────┘  │
│  [Preguntar]                                      │
│                                                   │
│  > Sí. 5★, edge 12%, Kelly sugiere €32 de €1000. │
│    PD es tu mejor liga (+15.9% ROI histórico).   │
│    La única duda: partido sin Lewandowski (lesión)│
│    según noticias de hoy.                         │
└──────────────────────────────────────────────────┘
```

#### 4b. Prompts pre-construidos (botones de acceso rápido)

- "¿Qué apuesto hoy?" → genera el plan autopilot explicado en lenguaje natural
- "¿Es segura mi combinada?" → analiza el slip actual con correlaciones y riesgo
- "¿Cuánto debería apostar en total este fin de semana?" → bankroll management advice
- "¿En qué liga confías más?" → análisis de ROI histórico por liga

#### 4c. Contexto completo que se envía a Ollama

```
- Picks disponibles hoy (liga, prob, edge, kelly, stars)
- Bankroll actual y ROI histórico del usuario
- Slip actual si hay picks ya añadidos
- Últimos 5 resultados del usuario (ganó/perdió)
- Noticias del partido si disponibles (de ai_advisor)
- Métricas del modelo por liga (TRACKER_METRICS)
```

Ollama responde en el idioma del navegador (ya detectado por `LANG`).

---

### Fase 5 — Autopilot Telegram

**Duración estimada: media sesión**

El viernes/sábado cuando `main.py` corre, además de los 5 mensajes actuales, enviar un mensaje nuevo tipo "Autopilot digest":

```
🤖 AUTOPILOT — Fin de semana

📋 PLAN ÓPTIMO (perfil: Balanced)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. PD ★★★★★ Barcelona vs Atletico
   Local @1.85 · Edge 12% · Kelly 6.4%
   EV esperado: +€3.84 (con €32)

2. PL ★★★★★ Arsenal vs Chelsea
   Over 2.5 @1.92 · Edge 12% · Kelly 5%
   EV esperado: +€3.00 (con €25)

💰 Exposición total: €57 (5.7% bankroll)
📈 EV total esperado: +€6.84
🏆 Mejor liga: PD (+15.9% ROI histórico)
```

Requiere que el perfil de usuario sea configurable desde `config.py` (para el bot de Telegram, que no tiene acceso al localStorage del navegador).

---

## Arquitectura de cambios

### Nuevos ficheros
- `algorithms/autopilot.py` — motor de selección, correlación, EV calculation
- `visualizador/data/bookmakers.js` — precios por casa (generado por odds_fetcher)

### Ficheros modificados
- `config.py` — perfil Telegram del autopilot (`AUTOPILOT_RISK_LEVEL`, `AUTOPILOT_MAX_BETS`)
- `odds_fetcher.py` — nueva función `fetch_bookmaker_comparison()`
- `reporter.py` — incluir `bestBookmaker` y `evEuros` en value bets
- `telegram_notifier.py` — mensaje 6: Autopilot digest
- `visualizador/index.html` — nueva vista AUTOPILOT, chat Ollama, EV display, bookmaker comparison

### localStorage nuevo
- `bwg_profile` — preferencias del autopilot (persistente)

---

## Orden de implementación recomendado

| Prioridad | Fase | Por qué |
|---|---|---|
| 1 | Fase 1 (perfil + motor) | Es el núcleo. Todo lo demás depende de esto. |
| 2 | Fase 3a (EV en euros) | Pequeño cambio, gran impacto en claridad |
| 3 | Fase 2 (comparativa casas) | Requiere cambio en odds_fetcher, mayor impacto práctico |
| 4 | Fase 4 (Ollama chat) | El diferencial wow — pero necesita Fase 1 para tener contexto |
| 5 | Fase 3b/3c (EV histórico) | Métricas adicionales, no bloqueante |
| 6 | Fase 5 (Telegram autopilot) | Guinda del pastel — requiere todas las anteriores |

---

## Decisiones pendientes antes de empezar

1. **¿Bookmaker comparison usa calls adicionales de the-odds-api?**
   Coste: +4 calls/run → 12 total → ~48/mes (bien dentro de 500).
   ¿Aceptable?

2. **¿El perfil del autopilot es solo frontend (localStorage) o también en config.py?**
   Para Telegram necesita estar en config.py. Para el navegador, localStorage.
   Propuesta: ambos, con localStorage como override.

3. **¿Qué bookmakers incluir en la comparativa?**
   Propuesta inicial: Bet365, Unibet, bwin, William Hill, Betfair Exchange, Pinnacle.
   El usuario puede tener cuenta en unos y no en otros — ¿filtramos?

4. **¿El chat de Ollama es siempre visible o un panel desplegable?**
   Propuesta: panel desplegable bajo el plan autopilot. No ocupa espacio si no se usa.

---

## Métricas de éxito de v6

- Tiempo hasta primera apuesta registrada: debería bajar de ~5 minutos a < 1 minuto
- Picks aceptados del autopilot vs modificados por el usuario: tracking en `bwg_history`
- EV ratio: P&L real / EV esperado — ¿está el usuario capturando el edge del modelo?
