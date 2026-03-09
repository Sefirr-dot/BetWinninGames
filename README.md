# BetWinninGames

Motor de predicción de fútbol basado en estadística avanzada. Combina múltiples modelos cuantitativos para predecir resultados de partidos de las principales ligas europeas y detectar value bets contra cuotas de mercado.

## Modelos

| Modelo | Peso | Descripción |
|---|---|---|
| Dixon-Coles | 45% | Poisson bivariado MLE + corrección Sarmanov para resultados bajos |
| Elo | 20% | Ratings dinámicos con ventaja local por liga |
| Form | 20% | Decay exponencial con ajuste por xG y SoS |
| BTTS | 5% | Poisson exacto con tasas por venue |
| Corners | 5% | Proxy basado en goles esperados |
| H2H | 5% | Head-to-head con decay temporal anual |

Además incluye calibrador de Platt, optimizador de pesos y meta-learner XGBoost (se activa con ≥200 picks reales resueltos).

## Ligas soportadas

- Premier League (PL)
- La Liga (PD)
- Bundesliga (BL1)
- Ligue 1 (FL1)

## Instalación

```bash
git clone https://github.com/Sefirr-dot/BetWinninGames.git
cd BetWinninGames
pip install -r requirements.txt
cp config.example.py config.py
# Editar config.py con tus API keys
```

### API keys necesarias

- **football-data.org** (gratis) → `API_KEY`
- **the-odds-api.com** (gratis, 500 req/mes) → `ODDS_API_KEY`
- **Telegram** (opcional) → `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`
- **Ollama local** (opcional, para AI Advisor) → `ollama serve`

## Uso

```bash
# Predicciones del fin de semana (auto-detecta Vie-Lun)
python main.py

# Fecha específica
python main.py --date 2026-03-07

# Filtrar por liga
python main.py --league PL   # PL · PD · BL1 · FL1

# Forzar re-fetch ignorando caché
python main.py --no-cache

# Backtest walk-forward
python backtest.py --league PL --seasons 2023 2024
python backtest.py --league ALL --seasons 2023 2024

# Actualizar tracker con resultados reales
python tracker.py
```

Tras ejecutar, abre `visualizador/index.html` directamente en el navegador (no necesita servidor).

## Visualizador

App estática single-file. Vistas disponibles:

- **Por fecha** — todos los partidos del fin de semana con probabilidades y estrellas
- **Best Bets** — parlays sugeridos (Doble Segura / Triple Media / Cuádruple / EV+)
- **Value Bets** — picks con edge positivo contra cuotas de mercado
- **Tracker** — histórico de picks resueltos con accuracy y ROI
- **Backtest** — curva de bankroll y métricas por fold

## Estructura

```
├── main.py               # Entrada principal
├── config.example.py     # Plantilla de configuración (copiar a config.py)
├── algorithms/
│   ├── dixon_coles.py
│   ├── elo.py
│   ├── form.py
│   ├── btts.py
│   ├── corners.py
│   ├── h2h.py
│   ├── ensemble.py
│   ├── value_detector.py
│   ├── calibrator.py
│   ├── weight_optimizer.py
│   └── meta_learner.py
├── backtest.py
├── tracker.py
├── fetcher.py            # football-data.org API client
├── odds_fetcher.py       # the-odds-api.com client
├── fdco_fetcher.py       # football-data.co.uk (temporadas 2020-2022)
├── understat_fetcher.py  # xG desde Understat
├── ai_advisor.py         # Enriquecimiento con Ollama (lesiones, alineaciones)
├── odds/                 # CSVs de cuotas (YYYY-MM-DD.csv)
├── cache/                # SQLite: partidos, xG, picks históricos
└── visualizador/
    ├── index.html
    └── data/             # JS generados por main.py / tracker.py / backtest.py
```

## Flujo de datos

1. Fetch histórico desde football-data.org + football-data.co.uk (2020–2025)
2. Enriquecimiento con xG desde Understat
3. Ajuste de fatiga y cuotas de mercado
4. Predicción por ensemble → ranking por estrellas
5. AI Advisor (Ollama) para picks ≥ 3★ — noticias de lesiones/alineaciones
6. Detección de value bets (edge vs cuota implícita)
7. Persistencia en SQLite → tracker de resultados → calibración automática
