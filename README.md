# Backtest: 4H-EMA200 + Fib 0,5–0,7 + 1H-Reaktionskerze

## Setup

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
```

## Start

```bash
# Schnelltest mit Yahoo-Daten (nur ~2 Jahre 1H, Qualität mittel)
.venv/bin/python backtest.py --source yf --pairs EURUSD GBPUSD USDJPY AUDUSD EURJPY --plot-trades 20

# Alte Richtungsregel (Trendlinie) zum Vergleich
.venv/bin/python backtest.py --source yf --pairs EURUSD GBPUSD --direction trendline

# Eigene Daten: data/<PAAR>.csv (1H oder kleiner, wird auf 1H verdichtet)
.venv/bin/python backtest.py --pairs EURUSD GBPUSD --data-dir data --csv-tz UTC --news news.csv

# MT5-Export (Serverzeit NY+7, Spalten <DATE> <TIME> <OPEN> ...)
.venv/bin/python backtest.py --pairs EURUSD --csv-tz server

# Dukascopy-Export ("Gmt time" 01.01.2024 00:00:00.000)
.venv/bin/python backtest.py --pairs EURUSD --csv-tz UTC --dayfirst
```

Ausgabe in `results/`: `report.txt`, `trades.csv`, `equity.png`, `config.txt`, optional `trades/*.png`
(4H-Chart je Trade mit Trendlinie, Berührungen, Fib-Zone, Einstieg/SL/TP – zum Prüfen der Erkennung).

`--from 2025-01-01 --to 2026-01-01` begrenzt den Handelszeitraum (Indikatoren laufen vorher warm).

### News-Datei

CSV mit den Spalten `datetime,currency,impact` (oder `date,time,currency,impact`), z. B.

```
datetime,currency,impact
2026-10-02 12:30,USD,High
```

Zeitzone über `--news-tz` (Standard UTC). Gezählt wird nur `impact` = High. Ohne Datei ist der
News-Filter **aus** (steht im Report). Historische Kalender gibt es z. B. als ForexFactory-Exporte.

## Umsetzung der Regeln

| Regel | Umsetzung im Code |
|---|---|
| **1. Richtung (4H)** – Standard `--direction ema` | Letzte geschlossene 4H-Kerze schließt über dem EMA200 (4H) → nur Long, darunter → nur Short. Die 1H-Reaktionskerze darf nicht auf der falschen Seite des EMA schließen. Optional `--ema-slope-bars N`: EMA muss zusätzlich über N 4H-Kerzen steigen (Long) / fallen (Short). Einschwingphase: 400 4H-Kerzen (~2,5 Monate) ohne Trades. |
| **1. Richtung (4H)** – alternativ `--direction trendline` | Trendlinie durch zwei 4H-Pivots (Long: steigende Tiefs, Short: fallende Hochs) innerhalb der letzten `tl_lookback` = 180 4H-Kerzen. **Intakt** = keine 4H-Schlusskurs jenseits der Linie (Toleranz 0,25 × ATR). **Berührung** = Pivot innerhalb der Toleranz; Pivots näher als 6 4H-Kerzen zählen als eine Berührung. Mind. 3 Berührungen. Die Reaktionskerze darf nicht jenseits der Linie schließen. |
| **2. Location (4H)** | Letzte Bewegung = letztes bestätigtes 4H-Swing-Hoch (Long) und das tiefste Tief seit dem vorherigen Swing-Hoch. Ungültig, wenn danach ein neues Hoch oder ein Bruch des Ursprungs kam. Zone = 0,5–0,7 Retracement. |
| **3. Trigger (1H)** | Reaktionskerze schließt in der Zone, bullisch (Long) / bärisch (Short). StochRSI-%K (14,14,3) in der Reaktionskerze oder der Kerze davor < 20 / > 80. Einstieg zur **Eröffnung der nächsten 1H-Kerze**. |
| **4. Stop** | Hinter dem Extrem der Reaktionskerze + `stop_buffer_pips` (1 Pip); bei Short zusätzlich + Spread, da der Stop auf Ask auslöst. |
| **4. Management** | Bracket SL/TP. Schließt eine 4H-Kerze (nach Einstieg) bei ≥ +1,5R, wandert der Stop einmalig auf den Einstiegskurs. Sonst nichts. |
| **5. Ziel** | Letztes 4H-Swing-Hoch/-Tief (= Extrem der Fib-Bewegung), verfeinert auf den höchsten/tiefsten **1H-Kerzenkörper** innerhalb dieser 4H-Kerze. CRV < 2 → kein Trade. |
| **6. Größe** | Risiko = 1 % bzw. 0,5 % vom Kontostand; Ergebnis = R × Risikobetrag. Lots werden zur Info ausgegeben (nur wenn Kontowährung = Basis- oder Quotewährung). |
| **7. Rahmen** | Einstieg 07:00–19:00 Uhr Berlin (Einstiegszeit = Eröffnung der Folgekerze). News: keine High-Impact-News von Basis-/Quotewährung in den 30 min vor dem Einstieg. Max. 1 offene Position und 1 Trade-Tag pro Kalendertag (DE) über **alle Paare** – ein Tag mit Einstieg *oder* Schließung ist verbraucht. Positionen laufen übers Wochenende; ein Gap über SL/TP wird zum Eröffnungskurs gefüllt. |
| **8. Drawdown-Regel** | DD vom höchsten Kontostand (geschlossene Trades) ≥ 3 % → 0,5 %. Zurück auf 1 % erst bei neuem Hoch. |
| **9. Bewertung** | Report enthält Zwischenbilanz (Trade 1–30) und Urteil (Trade 1–50) nach den festgelegten Schwellen, plus Auswertung nach Long/Short, Paar und Einstiegsstunde. Notbremse: DD ≥ 6 % → 14 Tage keine neuen Einstiege (wieder scharf, sobald DD < 6 %). `--no-brake` schaltet das ab. |

### Annahmen, die du kennen solltest

- **Trendlinie, Swings, „letzte Bewegung"** sind diskretionär. Die Werte oben sind eine
  Interpretation; das Ergebnis reagiert spürbar darauf (`--swing-len`, `--tl-pivot-len`,
  `--tl-tol-atr`, `--tl-touch-gap`). Mit `--plot-trades` prüfen, ob die erkannten Linien deinen
  eigenen entsprechen.
- **4H-Kerzen** sind wie bei MT5-Brokern auf den NY-Close ausgerichtet (`--h4-anchor ny`).
- **Kosten**: Daten werden als Bid behandelt, Spread `--spread-pips` (Standard 1,0, pro Paar über
  `--spreads GBPJPY=2.0`), Kommission `--commission-pips` (Standard 0,6 Round-Turn ≈ 3 $/Lot/Seite).
- **Intrabar**: Treffen SL und TP dieselbe 1H-Kerze, wird konservativ der SL gewertet (im Report gezählt).
- **Mehrere Signale gleichzeitig**: das erste Paar in der `--pairs`-Reihenfolge gewinnt.
- **Lot-Rundung** wird ignoriert (Ergebnis exakt 1 % / 0,5 % pro 1R).
- **Sehr enge Stops** (wenige Pips) sind nach den Regeln erlaubt, Kosten fressen dort aber viel R.
  `--min-stop-pips` filtert sie testweise heraus (nicht Teil der Strategie).
