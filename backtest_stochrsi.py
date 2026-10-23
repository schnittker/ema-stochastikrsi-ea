#!/usr/bin/env python3
"""
Backtest für EMA_StochRSI_Scanner.mq5 (1H, mehrere Paare gleichzeitig).

Regeln wie im EA:
  Long : EMA50 > EMA100, Close > EMA50, %K kreuzt %D nach oben; %K/%D der Vorkerze und %D der
         Signalkerze < 25
  Short: spiegelbildlich (EMA50 < EMA100, Close < EMA50, Kreuz nach unten, Zone > 75)
  Signal auf geschlossener Kerze, Einstieg zur Eröffnung der nächsten Kerze (Long Ask, Short Bid)
  SL   : tiefstes Tief / höchstes Hoch der letzten N Kerzen -/+ Spread -/+ Puffer
  TP   : CRV x SL-Abstand; kein weiteres Management
  Risiko % vom Kontostand (geschlossene Trades), 1 Position pro Paar, max. X gleichzeitig

Annahmen: Kursdaten = Bid, Ask = Bid + fester Spread je Paar. Treffen SL und TP in derselben
Kerze, zählt konservativ SL. Öffnet eine Kerze hinter SL/TP (Gap), wird zum Eröffnungskurs
geschlossen. Mindestlot/Lotschritt werden nicht simuliert (Risiko exakt in %).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", module="yfinance")

from backtest import BERLIN, clean_h1, load_csv, load_yf, pip_size

# Standard-Watchlist aus ~/private/tradingjournal (WatchlistService), ohne XAUUSD
DEFAULT_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "EURJPY",
    "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY", "EURCAD", "EURCHF", "EURNZD",
    "EURAUD", "GBPCHF", "GBPNZD", "GBPCAD", "GBPAUD", "AUDNZD", "AUDCAD", "CADCHF",
]

# Typische Retail-Spreads in Pips (Durchschnitt, nicht Minimum)
DEFAULT_SPREADS = {
    "EURUSD": 0.8, "GBPUSD": 1.2, "USDJPY": 1.0, "USDCHF": 1.4, "AUDUSD": 1.0, "NZDUSD": 1.5,
    "USDCAD": 1.5, "EURJPY": 1.6, "GBPJPY": 2.5, "EURGBP": 1.3, "AUDJPY": 1.8, "EURAUD": 2.2,
    "GBPAUD": 2.8, "CADJPY": 2.0, "NZDJPY": 2.0, "CHFJPY": 2.2, "EURCAD": 2.2, "EURCHF": 1.6,
    "EURNZD": 3.0, "GBPCHF": 2.5, "GBPNZD": 3.5, "GBPCAD": 3.0, "AUDNZD": 2.0, "AUDCAD": 1.8,
    "CADCHF": 2.0,
}


@dataclass
class Config:
    start_balance: float = 100_000.0
    risk_pct: float = 1.0
    rr: float = 2.0
    ema_fast: int = 50
    ema_slow: int = 100
    price_filter: bool = True
    rsi_len: int = 14
    stoch_len: int = 14
    k_smooth: int = 3
    d_smooth: int = 3
    oversold: float = 25.0
    overbought: float = 75.0
    lookback: int = 5
    buffer_pips: float = 0.0
    add_spread: bool = True
    max_trades: int = 5
    spread_pips: float = 1.5            # für Paare ohne Eintrag in DEFAULT_SPREADS / --spreads
    spreads: dict = field(default_factory=dict)
    commission_pips: float = 0.0        # Round-Turn in Pips
    warmup_bars: int = 300              # Kerzen ohne Trades, bis die Indikatoren eingeschwungen sind


# --------------------------------------------------------------------------------------
# Indikatoren
# --------------------------------------------------------------------------------------

def ema(x: pd.Series, n: int) -> np.ndarray:
    return x.ewm(span=n, adjust=False).mean().to_numpy()


def stoch_rsi(close: pd.Series, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Wie TradingView / EA: RSI (Wilder), Stochastik darauf, %K = SMA(k), %D = SMA(%K, d)."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / cfg.rsi_len, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / cfg.rsi_len, adjust=False).mean()
    rsi = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(100.0)
    lo, hi = rsi.rolling(cfg.stoch_len).min(), rsi.rolling(cfg.stoch_len).max()
    rng = hi - lo
    raw = ((rsi - lo) / rng.where(rng > 0) * 100).where(rng > 0, 50.0)
    k = raw.rolling(cfg.k_smooth).mean()
    d = k.rolling(cfg.d_smooth).mean()
    return k.to_numpy(), d.to_numpy()


class Market:
    def __init__(self, pair: str, h1: pd.DataFrame, cfg: Config):
        self.pair = pair
        self.pip = pip_size(pair)
        self.spread = cfg.spreads.get(pair, DEFAULT_SPREADS.get(pair, cfg.spread_pips)) * self.pip
        self.t = h1.index
        self.pos = {t: i for i, t in enumerate(self.t)}
        self.o, self.h, self.l, self.c = (h1[x].to_numpy() for x in ("open", "high", "low", "close"))

        ef, es = ema(h1["close"], cfg.ema_fast), ema(h1["close"], cfg.ema_slow)
        k, d = stoch_rsi(h1["close"], cfg)
        k1, d1 = k, d
        k2, d2 = np.r_[np.nan, k[:-1]], np.r_[np.nan, d[:-1]]
        with np.errstate(invalid="ignore"):
            up = (ef > es) & ((not cfg.price_filter) | (self.c > ef))
            dn = (ef < es) & ((not cfg.price_filter) | (self.c < ef))
            cross_up = (k2 <= d2) & (k1 > d1) & (k2 < cfg.oversold) & (d2 < cfg.oversold) & (d1 < cfg.oversold)
            cross_dn = (k2 >= d2) & (k1 < d1) & (k2 > cfg.overbought) & (d2 > cfg.overbought) & (d1 > cfg.overbought)
        self.sig = np.where(up & cross_up, 1, np.where(dn & cross_dn, -1, 0))
        self.sig[: cfg.warmup_bars] = 0
        self.swing_lo = h1["low"].rolling(cfg.lookback).min().to_numpy()
        self.swing_hi = h1["high"].rolling(cfg.lookback).max().to_numpy()
        self.ema_f, self.ema_s, self.k, self.d = ef, es, k, d


# --------------------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------------------

@dataclass
class Position:
    pair: str
    side: int
    signal_time: pd.Timestamp
    entry_idx: int
    entry: float
    sl: float
    tp: float
    risk_dist: float
    risk_money: float


class Backtest:
    def __init__(self, markets: list[Market], cfg: Config, start=None, end=None):
        self.m, self.cfg, self.start, self.end = markets, cfg, start, end
        self.balance = cfg.start_balance
        self.open: dict[str, Position] = {}
        self.trades: list[dict] = []
        self.skipped: dict[str, int] = {}
        self.ambiguous = 0

    def skip(self, why: str):
        self.skipped[why] = self.skipped.get(why, 0) + 1

    def try_open(self, mk: Market, i: int):
        s = i - 1
        side = int(mk.sig[s])
        if side == 0:
            return
        if mk.pair in self.open:
            return self.skip("bereits Position auf dem Paar")
        if len(self.open) >= self.cfg.max_trades:
            return self.skip(f"max. {self.cfg.max_trades} Trades offen")

        buf = self.cfg.buffer_pips * mk.pip + (mk.spread if self.cfg.add_spread else 0.0)
        bid = mk.o[i]
        ask = bid + mk.spread
        if side > 0:
            entry, sl = ask, mk.swing_lo[s] - buf
            if not bid > sl:
                return self.skip("SL auf falscher Seite (Gap)")
            tp = entry + (entry - sl) * self.cfg.rr
        else:
            entry, sl = bid, mk.swing_hi[s] + buf
            if not ask < sl:
                return self.skip("SL auf falscher Seite (Gap)")
            tp = entry - (sl - entry) * self.cfg.rr
        risk = abs(entry - sl)
        self.open[mk.pair] = Position(mk.pair, side, mk.t[s], i, entry, sl, tp, risk,
                                      self.balance * self.cfg.risk_pct / 100)

    def manage(self, mk: Market, p: Position, i: int) -> bool:
        sp = mk.spread if p.side < 0 else 0.0   # Short schließt zum Ask
        o, h, l = mk.o[i] + sp, mk.h[i] + sp, mk.l[i] + sp
        px = kind = None
        if p.side > 0:
            if i > p.entry_idx and o <= p.sl:
                px, kind = o, "SL"
            elif i > p.entry_idx and o >= p.tp:
                px, kind = o, "TP"
            else:
                hit_sl, hit_tp = l <= p.sl, h >= p.tp
                if hit_sl and hit_tp:
                    self.ambiguous += 1
                if hit_sl:
                    px, kind = p.sl, "SL"
                elif hit_tp:
                    px, kind = p.tp, "TP"
        else:
            if i > p.entry_idx and o >= p.sl:
                px, kind = o, "SL"
            elif i > p.entry_idx and o <= p.tp:
                px, kind = o, "TP"
            else:
                hit_sl, hit_tp = h >= p.sl, l <= p.tp
                if hit_sl and hit_tp:
                    self.ambiguous += 1
                if hit_sl:
                    px, kind = p.sl, "SL"
                elif hit_tp:
                    px, kind = p.tp, "TP"
        if kind is None:
            return False
        self.close(mk, p, i, px, kind)
        return True

    def close(self, mk: Market, p: Position, i: int, px: float, kind: str):
        r = (px - p.entry) * p.side / p.risk_dist - self.cfg.commission_pips * mk.pip / p.risk_dist
        pnl = r * p.risk_money
        self.balance += pnl
        self.trades.append({
            "pair": p.pair, "side": "long" if p.side > 0 else "short",
            "signal_time_de": p.signal_time.tz_convert(BERLIN),
            "entry_time_de": mk.t[p.entry_idx].tz_convert(BERLIN),
            "exit_time_de": mk.t[i].tz_convert(BERLIN),
            "entry": p.entry, "sl": p.sl, "tp": p.tp, "exit": px, "result": kind,
            "sl_pips": p.risk_dist / mk.pip, "r": r, "risk_money": p.risk_money, "pnl": pnl,
            "balance": self.balance, "bars_held": i - p.entry_idx + 1,
        })
        del self.open[p.pair]

    def run(self):
        timeline = sorted(set().union(*[set(mk.t) for mk in self.m]))
        if self.start is not None:
            timeline = [t for t in timeline if t >= self.start]
        if self.end is not None:
            timeline = [t for t in timeline if t < self.end]
        by_pair = {mk.pair: mk for mk in self.m}

        for T in timeline:
            # 1) Einstiege zur Eröffnung von T (Signal = gerade geschlossene Kerze)
            for mk in self.m:
                i = mk.pos.get(T)
                if i is not None and i >= 1:
                    self.try_open(mk, i)
            # 2) Kerze T für alle offenen Positionen auswerten (inkl. Einstiegskerze)
            for pair, p in list(self.open.items()):
                mk = by_pair[pair]
                i = mk.pos.get(T)
                if i is not None and i >= p.entry_idx:
                    self.manage(mk, p, i)

        # Am Datenende noch offene Positionen zum letzten Schlusskurs bewerten
        for pair, p in list(self.open.items()):
            mk = by_pair[pair]
            last = len(mk.t) - 1
            px = mk.c[last] + (mk.spread if p.side < 0 else 0.0)
            self.close(mk, p, last, px, "OFFEN")
        self.trades.sort(key=lambda x: x["exit_time_de"])
        bal = self.cfg.start_balance
        for tr in self.trades:          # Kontostand in Ausstiegsreihenfolge neu aufbauen
            bal += tr["pnl"]
            tr["balance"] = bal
        return self


# --------------------------------------------------------------------------------------
# Auswertung
# --------------------------------------------------------------------------------------

def stats(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {"trades": 0}
    gp = df.loc[df.r > 0, "r"].sum()     # in R, damit Zinseszins/Kontostand nicht verzerrt
    gl = -df.loc[df.r < 0, "r"].sum()
    tp = int((df.result == "TP").sum())
    return {
        "trades": n, "TP": tp, "SL": int((df.result == "SL").sum()),
        "trefferquote_%": 100 * tp / n,
        "profit_factor": gp / gl if gl > 0 else float("inf"),
        "avg_R": df.r.mean(), "sum_R": df.r.sum(),
    }


def breakdown(df: pd.DataFrame, col: str) -> str:
    rows = {k: stats(g) for k, g in df.groupby(col)}
    t = pd.DataFrame(rows).T[["trades", "TP", "SL", "trefferquote_%", "profit_factor", "avg_R", "sum_R"]]
    return t.astype(float).round(2).to_string()


def plot_equity(df: pd.DataFrame, cfg: Config, path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ink, muted, surface, series = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6"
    x = [df.entry_time_de.iloc[0]] + list(df.exit_time_de)
    bal = np.r_[cfg.start_balance, df.balance.to_numpy()]
    peak = np.maximum.accumulate(bal)
    dd = (bal - peak) / peak * 100

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.3]}, facecolor=surface)
    for ax in (a1, a2):
        ax.set_facecolor(surface)
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c9c8c3")
        ax.tick_params(colors=muted, labelsize=9)
    a1.step(x, bal, where="post", color=series, linewidth=2)
    a1.axhline(cfg.start_balance, color=muted, linewidth=1, linestyle=(0, (4, 3)))
    a1.set_title("Kontostand (geschlossene Trades)", loc="left", color=ink, fontsize=12)
    a1.annotate(f"{bal[-1]:,.0f}", (x[-1], bal[-1]), xytext=(6, 0), textcoords="offset points",
                color=ink, fontsize=9, va="center")
    a2.fill_between(x, dd, 0, step="post", color=series, alpha=0.25, linewidth=0)
    a2.step(x, dd, where="post", color=series, linewidth=1.5)
    a2.set_title("Drawdown vom Hoch in %", loc="left", color=ink, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def report(bt: Backtest, cfg: Config, out: Path) -> str:
    df = pd.DataFrame(bt.trades)
    t0 = bt.start or min(m.t[cfg.warmup_bars] for m in bt.m if len(m.t) > cfg.warmup_bars)
    t1 = bt.end or max(m.t[-1] for m in bt.m)
    lines = ["=" * 78, "BACKTEST EMA 50/100 + Stochastik RSI (1H)", "=" * 78,
             f"Paare: {', '.join(m.pair for m in bt.m)}",
             f"Zeitraum: {t0.tz_convert(BERLIN):%d.%m.%Y} – {t1.tz_convert(BERLIN):%d.%m.%Y}",
             f"Startkapital: {cfg.start_balance:,.0f}   Risiko {cfg.risk_pct} %   CRV 1:{cfg.rr:g}   "
             f"Swing-Lookback {cfg.lookback}   max. {cfg.max_trades} Trades",
             f"Preis-EMA-Filter: {'an' if cfg.price_filter else 'aus'}   "
             f"Zonen {cfg.oversold:g}/{cfg.overbought:g}   Kommission {cfg.commission_pips} Pips", ""]
    if df.empty:
        lines.append("Keine Trades.")
    else:
        s = stats(df)
        bal = np.r_[cfg.start_balance, df.balance.to_numpy()]
        peak = np.maximum.accumulate(bal)
        max_dd = ((peak - bal) / peak * 100).max()
        years = (t1 - t0).days / 365.25
        be = 100 / (1 + cfg.rr)
        lines += [
            f"Trades: {s['trades']}  (TP {s['TP']} / SL {s['SL']} / offen {(df.result == 'OFFEN').sum()})"
            f"   ≈ {s['trades'] / max(years, 1e-9):.0f} pro Jahr",
            f"Trefferquote: {s['trefferquote_%']:.1f} %   (Break-even bei 1:{cfg.rr:g} ≈ {be:.1f} % inkl. Kosten etwas mehr)",
            f"Profit Factor: {s['profit_factor']:.2f}",
            f"Ø R pro Trade: {s['avg_R']:+.3f}   Summe R: {s['sum_R']:+.1f}",
            f"Endkapital: {bt.balance:,.0f}  ({(bt.balance / cfg.start_balance - 1) * 100:+.1f} %)",
            f"Max. Drawdown vom Hoch (geschlossen): {max_dd:.1f} %",
            f"Längste Verlustserie: {longest_streak(df.r.to_numpy() < 0)}",
            f"Ø SL-Abstand: {df.sl_pips.mean():.1f} Pips   Ø Haltedauer: {df.bars_held.mean():.1f} Kerzen",
            f"SL und TP in derselben Kerze (als SL gewertet): {bt.ambiguous}",
            "", "-- Long/Short --", breakdown(df, "side"),
            "", "-- Paare --", breakdown(df, "pair"),
        ]
        df["jahr"] = df.entry_time_de.map(lambda x: x.year)
        lines += ["", "-- Jahr --", breakdown(df, "jahr")]
        df.to_csv(out / "trades.csv", index=False)
        plot_equity(df, cfg, out / "equity.png")
    if bt.skipped:
        lines += ["", "-- Signale ohne Trade --"]
        lines += [f"  {k:<40} {v}" for k, v in sorted(bt.skipped.items(), key=lambda kv: -kv[1])]
    text = "\n".join(lines)
    (out / "report.txt").write_text(text, encoding="utf-8")
    return text


def longest_streak(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    ap.add_argument("--source", choices=["csv", "yf"], default="yf")
    ap.add_argument("--data-dir", type=Path, default=Path("data"), help="CSV-Dateien <PAAR>.csv")
    ap.add_argument("--csv-tz", default="UTC", help="UTC, Europe/Berlin, ... oder 'server' (MT5, NY+7)")
    ap.add_argument("--dayfirst", action="store_true")
    ap.add_argument("--from", dest="date_from", help="Startdatum JJJJ-MM-TT")
    ap.add_argument("--to", dest="date_to")
    ap.add_argument("--out", type=Path, default=Path("results_stochrsi"))
    ap.add_argument("--balance", type=float, default=Config.start_balance)
    ap.add_argument("--risk", type=float, default=Config.risk_pct)
    ap.add_argument("--rr", type=float, default=Config.rr)
    ap.add_argument("--lookback", type=int, default=Config.lookback)
    ap.add_argument("--max-trades", type=int, default=Config.max_trades)
    ap.add_argument("--oversold", type=float, default=Config.oversold)
    ap.add_argument("--overbought", type=float, default=Config.overbought)
    ap.add_argument("--no-price-filter", action="store_true", help="Close-vs-EMA50-Filter abschalten")
    ap.add_argument("--buffer-pips", type=float, default=Config.buffer_pips)
    ap.add_argument("--no-spread-in-sl", action="store_true")
    ap.add_argument("--spread-pips", type=float, default=Config.spread_pips)
    ap.add_argument("--spreads", default="", help="z. B. GBPJPY=2.0,EURUSD=0.6")
    ap.add_argument("--commission-pips", type=float, default=Config.commission_pips)
    a = ap.parse_args(argv)

    cfg = Config(
        start_balance=a.balance, risk_pct=a.risk, rr=a.rr, lookback=a.lookback, max_trades=a.max_trades,
        oversold=a.oversold, overbought=a.overbought, price_filter=not a.no_price_filter,
        buffer_pips=a.buffer_pips, add_spread=not a.no_spread_in_sl, spread_pips=a.spread_pips,
        spreads={k.upper(): float(v) for k, v in (s.split("=") for s in a.spreads.split(",") if s)},
        commission_pips=a.commission_pips,
    )

    markets = []
    for pair in [x.upper() for arg in a.pairs for x in arg.replace(",", " ").split()]:
        try:
            if a.source == "yf":
                h1 = load_yf(pair)
            else:
                f = a.data_dir / f"{pair}.csv"
                if not f.exists():
                    sys.exit(f"Datei fehlt: {f}")
                h1 = load_csv(f, a.csv_tz, a.dayfirst)
        except ValueError as e:
            print(f"{pair}: übersprungen ({e})", file=sys.stderr)
            continue
        h1 = clean_h1(h1)
        print(f"{pair}: {len(h1)} 1H-Kerzen  {h1.index[0]} – {h1.index[-1]}", file=sys.stderr)
        markets.append(Market(pair, h1, cfg))
    if not markets:
        sys.exit("Keine Daten geladen.")

    start = pd.Timestamp(a.date_from, tz=BERLIN).tz_convert("UTC") if a.date_from else None
    end = pd.Timestamp(a.date_to, tz=BERLIN).tz_convert("UTC") if a.date_to else None
    a.out.mkdir(parents=True, exist_ok=True)
    bt = Backtest(markets, cfg, start, end).run()
    print(report(bt, cfg, a.out))
    (a.out / "config.txt").write_text("\n".join(f"{k} = {v}" for k, v in asdict(cfg).items()), encoding="utf-8")


if __name__ == "__main__":
    main()
