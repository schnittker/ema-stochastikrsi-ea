#!/usr/bin/env python3
"""
Backtest: 4H-EMA200 + Fib 0,5–0,7 (Location) + 1H-Reaktionskerze/StochRSI (Trigger).

Ablauf pro Kandidat (siehe README.md für alle Interpretationen):
  1. Richtung  (4H): 4H-Schluss über EMA200 -> Long, darunter -> Short
                     (alternativ --direction trendline: intakte Trendlinie mit >= 3 Berührungen)
  2. Location  (4H): Fib-Zone 0,5–0,7 der letzten Bewegung in Trendrichtung
  3. Trigger   (1H): Reaktionskerze schließt in der Zone (Long bullisch / Short bärisch),
                     StochRSI in Reaktionskerze oder Kerze davor < 20 / > 80,
                     Einstieg zur Eröffnung der nächsten 1H-Kerze
  4. Stop hinter dem Extrem der Reaktionskerze, TP am letzten 4H-Swing (1H-Körper), CRV >= 1:2
     Break-even einmalig, wenn eine 4H-Kerze bei >= +1,5R schließt
  5. Rahmen: 07–19 Uhr DE, News-Filter, max. 1 Trade/Tag, max. 1 Position, Drawdown-Regel
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field, asdict
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

BERLIN = "Europe/Berlin"


# --------------------------------------------------------------------------------------
# Konfiguration
# --------------------------------------------------------------------------------------

@dataclass
class Config:
    # Konto / Risiko
    start_balance: float = 100_000.0
    account_ccy: str = "USD"
    risk_pct: float = 1.0               # Standard-Risiko pro Trade
    risk_pct_dd: float = 0.5            # Risiko bei aktiver Drawdown-Regel
    dd_reduce_pct: float = 3.0          # ab diesem DD vom Hoch -> risk_pct_dd
    dd_brake_pct: float = 6.0           # Notbremse
    brake_pause_days: int = 14
    use_brake: bool = True
    ftmo_max_loss_pct: float = 10.0

    # Setup
    min_crv: float = 2.0
    fib_lo: float = 0.5
    fib_hi: float = 0.7
    stoch_rsi_len: int = 14
    stoch_len: int = 14
    stoch_k: int = 3
    stoch_long_below: float = 20.0
    stoch_short_above: float = 80.0
    direction: str = "ema"              # 'ema' (EMA200 4H) oder 'trendline'
    ema_len: int = 200
    ema_slope_bars: int = 0             # >0: EMA muss zusätzlich über N 4H-Kerzen steigen/fallen
    swing_len: int = 3                  # Fraktal-Länge der 4H-Swings (Fib-Bewegung + TP)
    tl_pivot_len: int = 2               # Fraktal-Länge der Trendlinien-Berührungen
    tl_min_touches: int = 3
    tl_min_touch_gap: int = 6           # Mindestabstand zwischen gezählten Berührungen (4H-Kerzen)
    tl_lookback: int = 180              # 4H-Kerzen (~30 Handelstage)
    tl_tol_atr: float = 0.25            # Berührungs-/Bruch-Toleranz in ATR(4H)
    atr_len: int = 14
    be_trigger_r: float = 1.5

    # Rahmen
    session_start: str = "07:00"
    session_end: str = "19:00"
    news_window_min: int = 30

    # Ausführung / Kosten
    stop_buffer_pips: float = 1.0       # Abstand hinter dem Kerzen-Extrem
    min_stop_pips: float = 0.0          # 0 = aus (nicht Teil der Regeln, nur zum Testen)
    spread_pips: float = 1.0
    spreads: dict = field(default_factory=dict)   # Overrides pro Paar
    commission_pips: float = 0.6        # Round-Turn, in Pips umgerechnet
    h4_anchor: str = "ny"               # 'ny' = 4H-Kerzen wie MT5-Broker (NY-Close), 'utc'


def pip_size(pair: str) -> float:
    p = pair.upper()
    if p.startswith("XAU"):
        return 0.1
    if p.startswith("XAG"):
        return 0.01
    if "JPY" in p:
        return 0.01
    return 0.0001


# --------------------------------------------------------------------------------------
# Daten
# --------------------------------------------------------------------------------------

def _to_utc_index(ts: pd.Series, tz: str) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(ts)
    if idx.tz is not None:
        return idx.tz_convert("UTC")
    if tz == "server":
        # MT5-Serverzeit mit NY-Close-Ausrichtung = New York + 7h
        ny = (idx - pd.Timedelta(hours=7)).tz_localize(
            "America/New_York", ambiguous=False, nonexistent="shift_forward")
        return ny.tz_convert("UTC")
    return idx.tz_localize(tz, ambiguous=False, nonexistent="shift_forward").tz_convert("UTC")


def load_csv(path: Path, tz: str, dayfirst: bool) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=None, engine="python")
    raw.columns = [c.strip().strip("<>").lower() for c in raw.columns]
    if "date" in raw.columns and "time" in raw.columns:
        ts = pd.to_datetime(raw["date"].astype(str) + " " + raw["time"].astype(str),
                            format="mixed", dayfirst=dayfirst)
    else:
        tcol = next((c for c in ("datetime", "time", "timestamp", "gmt time", "local time", "date")
                     if c in raw.columns), None)
        if tcol is None:
            raise ValueError(f"{path}: keine Zeitspalte gefunden ({list(raw.columns)})")
        ts = pd.to_datetime(raw[tcol], format="mixed", dayfirst=dayfirst)
    df = pd.DataFrame({c: raw[c].astype(float).to_numpy() for c in ("open", "high", "low", "close")},
                      index=_to_utc_index(ts, tz))
    return df


def load_yf(pair: str) -> pd.DataFrame:
    import yfinance as yf
    d = yf.download(f"{pair}=X", period="730d", interval="1h", progress=False, auto_adjust=False)
    if d.empty:
        raise ValueError(f"yfinance: keine Daten für {pair}")
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.rename(columns=str.lower)[["open", "high", "low", "close"]]
    d.index = d.index.tz_convert("UTC")
    return d


def clean_h1(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="last")].sort_index().dropna()
    df = df[(df["high"] >= df["low"]) & (df["high"] > 0)]
    step = df.index.to_series().diff().median()
    if step < pd.Timedelta(hours=1):          # z. B. M1/M5 -> auf 1H verdichten
        df = df.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    df.index = df.index.floor("1h")
    df = df[~df.index.duplicated(keep="first")]
    return df


def load_news(path: Path, tz: str) -> dict[str, np.ndarray]:
    """CSV mit Spalten datetime (oder date+time), currency, impact. Gibt {Währung: sortierte UTC-Zeiten}."""
    raw = pd.read_csv(path, sep=None, engine="python")
    raw.columns = [c.strip().lower() for c in raw.columns]
    if "datetime" in raw.columns:
        ts = pd.to_datetime(raw["datetime"], format="mixed")
    else:
        ts = pd.to_datetime(raw["date"].astype(str) + " " + raw["time"].astype(str), format="mixed")
    raw["utc"] = _to_utc_index(ts, tz)
    imp = raw["impact"].astype(str).str.strip().str.lower()
    high = raw[imp.isin(["high", "h", "3", "red", "high impact expected"])]
    out = {}
    for ccy, g in high.groupby(high["currency"].astype(str).str.upper().str.strip()):
        out[ccy] = np.sort(g["utc"].to_numpy(dtype="datetime64[ns]"))
    return out


# --------------------------------------------------------------------------------------
# Indikatoren
# --------------------------------------------------------------------------------------

def stoch_rsi_k(close: pd.Series, rsi_len: int, stoch_len: int, k: int) -> np.ndarray:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / rsi_len, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / rsi_len, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    rsi = rsi.fillna(100.0)
    lo, hi = rsi.rolling(stoch_len).min(), rsi.rolling(stoch_len).max()
    st = ((rsi - lo) / (hi - lo).replace(0, np.nan) * 100).ffill()
    return st.rolling(k).mean().to_numpy()


def atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int) -> np.ndarray:
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def pivots(x: np.ndarray, n: int, high: bool) -> np.ndarray:
    """Indizes von Fraktal-Hochs/-Tiefs (n Kerzen links strikt, n Kerzen rechts nicht übertroffen)."""
    out = []
    for i in range(n, len(x) - n):
        left, right = x[i - n:i], x[i + 1:i + n + 1]
        if high and x[i] > left.max() and x[i] >= right.max():
            out.append(i)
        elif not high and x[i] < left.min() and x[i] <= right.min():
            out.append(i)
    return np.array(out, dtype=int)


# --------------------------------------------------------------------------------------
# Marktdaten pro Paar (1H + abgeleitete 4H)
# --------------------------------------------------------------------------------------

class Market:
    def __init__(self, pair: str, h1: pd.DataFrame, cfg: Config):
        self.pair = pair.upper()
        self.cfg = cfg
        self.base, self.quote = self.pair[:3], self.pair[3:6]
        self.pip = pip_size(self.pair)
        self.spread = cfg.spreads.get(self.pair, cfg.spread_pips) * self.pip

        self.t = h1.index                                  # 1H-Eröffnungszeiten (UTC)
        self.o, self.h, self.l, self.c = (h1[k].to_numpy() for k in ("open", "high", "low", "close"))
        self.k = stoch_rsi_k(h1["close"], cfg.stoch_rsi_len, cfg.stoch_len, cfg.stoch_k)
        self.pos = {ts: i for i, ts in enumerate(self.t)}

        # 4H-Kerzen gruppieren
        if cfg.h4_anchor == "ny":
            local = self.t.tz_convert("America/New_York").tz_localize(None) + pd.Timedelta(hours=7)
        else:
            local = self.t.tz_localize(None)
        key = local.floor("4h")
        codes, _ = pd.factorize(key)                      # monoton, da sortiert
        self.h4_id = codes
        n4 = codes.max() + 1
        first = np.full(n4, -1)
        last = np.full(n4, -1)
        for i, g in enumerate(codes):
            if first[g] < 0:
                first[g] = i
            last[g] = i
        self.h4_first, self.h4_last = first, last
        self.H4o = self.o[first]
        self.H4h = np.maximum.reduceat(self.h, first)
        self.H4l = np.minimum.reduceat(self.l, first)
        self.H4c = self.c[last]
        self.h4_end = (self.t[last] + pd.Timedelta(hours=1)).to_numpy(dtype="datetime64[ns]")
        self.atr4 = atr(self.H4h, self.H4l, self.H4c, cfg.atr_len)
        self.ema4 = pd.Series(self.H4c).ewm(span=cfg.ema_len, adjust=False).mean().to_numpy()

        self.sw_hi = pivots(self.H4h, cfg.swing_len, True)
        self.sw_lo = pivots(self.H4l, cfg.swing_len, False)
        self.tl_hi = pivots(self.H4h, cfg.tl_pivot_len, True)
        self.tl_lo = pivots(self.H4l, cfg.tl_pivot_len, False)
        self._tl_cache: dict = {}
        self._fib_cache: dict = {}

    # ---- 4H-Zustand zum Zeitpunkt T: Index der letzten geschlossenen 4H-Kerze
    def last_closed_h4(self, t: np.datetime64) -> int:
        return int(np.searchsorted(self.h4_end, t, side="right")) - 1

    # ---- 1. Richtung: Trendlinie
    def trendline(self, c: int, side: str):
        key = (c, side)
        if key in self._tl_cache:
            return self._tl_cache[key]
        cfg = self.cfg
        n = cfg.tl_pivot_len
        piv_all = self.tl_lo if side == "long" else self.tl_hi
        price = self.H4l if side == "long" else self.H4h
        piv = piv_all[(piv_all >= c - cfg.tl_lookback) & (piv_all + n <= c)]
        tol = cfg.tl_tol_atr * self.atr4[c]
        best = None
        for a in range(len(piv) - 1):
            i = piv[a]
            yi = price[i]
            for b in range(a + 1, len(piv)):
                j = piv[b]
                yj = price[j]
                if j - i < cfg.tl_min_touch_gap:
                    continue
                if (side == "long" and yj <= yi) or (side == "short" and yj >= yi):
                    continue
                slope = (yj - yi) / (j - i)
                line = yi + slope * (np.arange(i, c + 1) - i)
                closes = self.H4c[i:c + 1]
                broken = (closes < line - tol).any() if side == "long" else (closes > line + tol).any()
                if broken:
                    continue
                pv = piv[a:]
                near = pv[np.abs(price[pv] - (yi + slope * (pv - i))) <= tol]
                touches = [near[0]]
                for x in near[1:]:                      # Pivot-Cluster zählen als eine Berührung
                    if x - touches[-1] >= cfg.tl_min_touch_gap:
                        touches.append(x)
                touches = np.array(touches)
                if len(touches) < cfg.tl_min_touches:
                    continue
                score = (touches[-1], len(touches))     # jüngste Berührung, dann meiste Berührungen
                if best is None or score > best["score"]:
                    best = {"i": int(i), "yi": float(yi), "slope": float(slope), "tol": float(tol),
                            "touches": [int(x) for x in touches], "score": score}
        self._tl_cache[key] = best
        return best

    # ---- 2. Location: Fib-Zone der letzten Bewegung in Trendrichtung
    def fib_leg(self, c: int, side: str):
        key = (c, side)
        if key in self._fib_cache:
            return self._fib_cache[key]
        n = self.cfg.swing_len
        res = None
        if side == "long":
            ph = self.sw_hi[self.sw_hi + n <= c]
            if len(ph):
                H = ph[-1]
                prev = ph[-2] if len(ph) > 1 else max(0, H - self.cfg.tl_lookback)
                if H - prev > 1:
                    L = prev + 1 + int(np.argmin(self.H4l[prev + 1:H]))
                    hi, lo = self.H4h[H], self.H4l[L]
                    no_new_ext = self.H4h[H + 1:c + 1].max(initial=-np.inf) <= hi
                    intact = self.H4l[H + 1:c + 1].min(initial=np.inf) > lo
                    if hi > lo and no_new_ext and intact:
                        leg = hi - lo
                        res = {"ext_idx": int(H), "origin_idx": int(L), "ext": hi, "origin": lo,
                               "zone_lo": hi - self.cfg.fib_hi * leg, "zone_hi": hi - self.cfg.fib_lo * leg}
        else:
            pl = self.sw_lo[self.sw_lo + n <= c]
            if len(pl):
                Lw = pl[-1]
                prev = pl[-2] if len(pl) > 1 else max(0, Lw - self.cfg.tl_lookback)
                if Lw - prev > 1:
                    Hh = prev + 1 + int(np.argmax(self.H4h[prev + 1:Lw]))
                    lo, hi = self.H4l[Lw], self.H4h[Hh]
                    no_new_ext = self.H4l[Lw + 1:c + 1].min(initial=np.inf) >= lo
                    intact = self.H4h[Lw + 1:c + 1].max(initial=-np.inf) < hi
                    if hi > lo and no_new_ext and intact:
                        leg = hi - lo
                        res = {"ext_idx": int(Lw), "origin_idx": int(Hh), "ext": lo, "origin": hi,
                               "zone_lo": lo + self.cfg.fib_lo * leg, "zone_hi": lo + self.cfg.fib_hi * leg}
        self._fib_cache[key] = res
        return res

    # ---- 5. Ziel: letzter 4H-Swing, verfeinert auf 1H-Kerzenkörper
    def take_profit(self, ext_idx: int, side: str) -> float:
        a, b = self.h4_first[ext_idx], self.h4_last[ext_idx] + 1
        if side == "long":
            return float(np.maximum(self.o[a:b], self.c[a:b]).max())
        return float(np.minimum(self.o[a:b], self.c[a:b]).min())


# --------------------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------------------

@dataclass
class Position:
    pair: str
    side: str
    entry_idx: int
    entry_time: pd.Timestamp
    entry: float
    sl: float
    tp: float
    risk_dist: float
    risk_pct: float
    risk_money: float
    lots: float
    crv: float
    signal_time: pd.Timestamp
    tl_touches: int
    ema: float
    sl_initial: float = 0.0
    tl: dict | None = None
    fib: dict | None = None
    be_done: bool = False
    be_time: pd.Timestamp | None = None
    ambiguous: bool = False


class Backtest:
    def __init__(self, markets: list[Market], cfg: Config, news: dict | None,
                 start: pd.Timestamp | None, end: pd.Timestamp | None):
        self.m = markets
        self.cfg = cfg
        self.news = news
        self.start, self.end = start, end
        self.s_start = dtime.fromisoformat(cfg.session_start)
        self.s_end = dtime.fromisoformat(cfg.session_end)

        self.balance = cfg.start_balance
        self.peak = cfg.start_balance
        self.reduced = False
        self.brake_armed = True
        self.pause_until: pd.Timestamp | None = None
        self.pauses: list[tuple] = []
        self.blocked_days: set = set()
        self.pos: Position | None = None
        self.trades: list[dict] = []
        self.closed: list[Position] = []
        self.rejects: dict[str, int] = {}

    # ---------------------------------------------------------------- Hilfen
    def _reject(self, reason: str):
        self.rejects[reason] = self.rejects.get(reason, 0) + 1
        return None

    def _risk_pct(self) -> float:
        return self.cfg.risk_pct_dd if self.reduced else self.cfg.risk_pct

    def _lots(self, mk: Market, dist: float, price: float, money: float) -> float:
        units = None
        if self.cfg.account_ccy == mk.quote:
            units = money / dist
        elif self.cfg.account_ccy == mk.base:
            units = money * price / dist
        return round(units / 100_000, 2) if units else float("nan")

    def _news_blocked(self, mk: Market, t: pd.Timestamp) -> bool:
        if not self.news:
            return False
        t64 = t.to_datetime64()
        lo = t64 - np.timedelta64(self.cfg.news_window_min, "m")
        for ccy in (mk.base, mk.quote):
            arr = self.news.get(ccy)
            if arr is not None and len(arr):
                j = np.searchsorted(arr, lo, side="left")
                if j < len(arr) and arr[j] <= t64:
                    return True
        return False

    # ---------------------------------------------------------------- Signal
    def check_signal(self, mk: Market, r: int) -> Position | None:
        """Reaktionskerze r ist gerade geschlossen; Einstieg zur Eröffnung von r+1."""
        cfg = self.cfg
        if r < 1 or r + 1 >= len(mk.t):
            return None
        o, h, l, c = mk.o[r], mk.h[r], mk.l[r], mk.c[r]
        side = "long" if c > o else "short" if c < o else None
        if side is None:
            return None

        # StochRSI: Reaktionskerze oder die Kerze davor
        k0, k1 = mk.k[r], mk.k[r - 1]
        if side == "long" and not (k0 < cfg.stoch_long_below or k1 < cfg.stoch_long_below):
            return None
        if side == "short" and not (k0 > cfg.stoch_short_above or k1 > cfg.stoch_short_above):
            return None

        entry_t = mk.t[r + 1]
        cidx = mk.last_closed_h4(entry_t.to_datetime64())
        # EMA braucht ~2x Länge zum Einschwingen (Startgewicht < 2 %)
        if cidx < (2 * cfg.ema_len if cfg.direction == "ema" else max(cfg.tl_lookback // 3, 30)):
            return self._reject("zu wenig 4H-Historie")

        # 2. Location – Kerze schließt in der Fib-Zone
        fib = mk.fib_leg(cidx, side)
        if fib is None:
            return self._reject("keine gültige 4H-Bewegung")
        if not (fib["zone_lo"] <= c <= fib["zone_hi"]):
            return self._reject("Schluss nicht in Fib 0,5–0,7")

        # 1. Richtung
        tl, ema = None, float(mk.ema4[cidx])
        if cfg.direction == "ema":
            # 4H-Schluss auf der richtigen Seite des EMA200, Reaktionskerze ebenfalls
            above = mk.H4c[cidx] > ema
            if (side == "long") != above:
                return self._reject("4H-Schluss auf falscher Seite des EMA200")
            if (side == "long" and c < ema) or (side == "short" and c > ema):
                return self._reject("Reaktionskerze jenseits des EMA200")
            n = cfg.ema_slope_bars
            if n > 0:
                slope = ema - mk.ema4[cidx - n]
                if (side == "long" and slope <= 0) or (side == "short" and slope >= 0):
                    return self._reject("EMA200-Steigung passt nicht")
        else:
            # Trendlinie intakt, >= 3 Berührungen, Preis auf der richtigen Seite
            tl = mk.trendline(cidx, side)
            if tl is None:
                return self._reject("keine intakte Trendlinie (>=3 Berührungen)")
            line_now = tl["yi"] + tl["slope"] * (mk.h4_id[r] - tl["i"])
            if (side == "long" and c < line_now - tl["tol"]) or (side == "short" and c > line_now + tl["tol"]):
                return self._reject("Reaktionskerze jenseits der Trendlinie")

        # Rahmen
        if self.pos is not None:
            return self._reject("Position offen")
        local = entry_t.tz_convert(BERLIN)
        if not (self.s_start <= local.time() <= self.s_end):
            return self._reject("außerhalb 07–19 Uhr")
        if local.date() in self.blocked_days:
            return self._reject("Trade-Tag bereits verbraucht")
        if self.pause_until is not None and entry_t < self.pause_until:
            return self._reject("Notbremse-Pause")
        if self._news_blocked(mk, entry_t):
            return self._reject("High-Impact-News < 30 min")

        # 4. Stop / Einstieg (Daten = Bid; Long kauft Ask, Short-Stop wird auf Ask ausgelöst)
        buf = cfg.stop_buffer_pips * mk.pip
        if side == "long":
            entry = mk.o[r + 1] + mk.spread
            sl = l - buf
            risk = entry - sl
        else:
            entry = mk.o[r + 1]
            sl = h + buf + mk.spread
            risk = sl - entry
        if risk <= 0:
            return self._reject("Eröffnung jenseits des Stops")
        if risk < cfg.min_stop_pips * mk.pip:
            return self._reject("Stop kleiner als --min-stop-pips")

        # 5. Ziel + CRV
        tp = mk.take_profit(fib["ext_idx"], side)
        reward = (tp - entry) if side == "long" else (entry - tp)
        if reward <= 0:
            return self._reject("TP liegt nicht vor dem Einstieg")
        crv = reward / risk
        if crv < cfg.min_crv:
            return self._reject("CRV < 1:2")

        # 6. Positionsgröße
        rp = self._risk_pct()
        money = self.balance * rp / 100
        return Position(mk.pair, side, r + 1, entry_t, entry, sl, tp, risk, rp, money,
                        self._lots(mk, risk, entry, money), crv, mk.t[r], len(tl["touches"]) if tl else 0, ema, sl, tl, fib)

    # ---------------------------------------------------------------- Trade-Management
    def manage_bar(self, mk: Market, i: int) -> bool:
        """Verarbeitet 1H-Kerze i für die offene Position. True, wenn geschlossen."""
        p = self.pos
        if p.side == "long":
            o, h, l = mk.o[i], mk.h[i], mk.l[i]                         # Bid
            hit_sl, hit_tp = l <= p.sl, h >= p.tp
            if hit_sl and o <= p.sl:
                return self._close(mk, i, "SL" if not p.be_done else "BE", o)
            if hit_tp and o >= p.tp:
                return self._close(mk, i, "TP", o)
        else:
            o, h, l = mk.o[i] + mk.spread, mk.h[i] + mk.spread, mk.l[i] + mk.spread   # Ask
            hit_sl, hit_tp = h >= p.sl, l <= p.tp
            if hit_sl and o >= p.sl:
                return self._close(mk, i, "SL" if not p.be_done else "BE", o)
            if hit_tp and o <= p.tp:
                return self._close(mk, i, "TP", o)
        if hit_sl and hit_tp:
            p.ambiguous = True                                           # konservativ: Stop zuerst
        if hit_sl:
            return self._close(mk, i, "BE" if p.be_done else "SL", p.sl)
        if hit_tp:
            return self._close(mk, i, "TP", p.tp)

        # Break-even: 4H-Kerze schließt bei >= +1,5R (einmalig)
        if not p.be_done and i == mk.h4_last[mk.h4_id[i]]:
            close_px = mk.c[i] if p.side == "long" else mk.c[i] + mk.spread
            move = (close_px - p.entry) if p.side == "long" else (p.entry - close_px)
            if move >= self.cfg.be_trigger_r * p.risk_dist:
                p.sl = p.entry
                p.be_done = True
                p.be_time = mk.h4_end[mk.h4_id[i]]
        return False

    def _close(self, mk: Market, i: int, kind: str, px: float) -> bool:
        p = self.pos
        cfg = self.cfg
        exit_t = mk.t[i] + pd.Timedelta(hours=1)   # spätester Zeitpunkt innerhalb der Kerze
        gross = (px - p.entry) if p.side == "long" else (p.entry - px)
        net = gross - cfg.commission_pips * mk.pip
        r_mult = net / p.risk_dist
        pnl = r_mult * p.risk_money
        bal_before = self.balance
        self.balance += pnl
        dd_before = (self.peak - bal_before) / self.peak * 100

        if self.balance >= self.peak:
            self.peak = self.balance
            self.reduced = False
        dd = (self.peak - self.balance) / self.peak * 100
        if dd >= cfg.dd_reduce_pct:
            self.reduced = True
        if cfg.use_brake:
            if dd < cfg.dd_brake_pct:
                self.brake_armed = True
            elif self.brake_armed:
                self.pause_until = exit_t + pd.Timedelta(days=cfg.brake_pause_days)
                self.brake_armed = False
                self.pauses.append((exit_t, self.pause_until, dd))

        self.blocked_days.add(mk.t[i].tz_convert(BERLIN).date())
        self.trades.append({
            "pair": p.pair, "side": p.side,
            "signal_time_de": p.signal_time.tz_convert(BERLIN),
            "entry_time_de": p.entry_time.tz_convert(BERLIN),
            "exit_time_de": mk.t[i].tz_convert(BERLIN),
            "entry": p.entry, "sl_initial": p.sl_initial, "tp": p.tp,
            "exit": px, "result": kind, "r": r_mult, "crv_planned": p.crv,
            "risk_pct": p.risk_pct, "risk_money": p.risk_money, "lots": p.lots,
            "pnl": pnl, "balance": self.balance, "dd_pct_before": dd_before, "dd_pct_after": dd,
            "be_moved": p.be_done, "be_time_de": pd.Timestamp(p.be_time).tz_localize("UTC").tz_convert(BERLIN)
            if p.be_time is not None else None,
            "tl_touches": p.tl_touches, "ema200_4h": p.ema, "ambiguous_bar": p.ambiguous,
            "bars_held": i - p.entry_idx + 1,
        })
        self.closed.append(p)
        self.pos = None
        return True

    # ---------------------------------------------------------------- Hauptschleife
    def run(self):
        timeline = sorted(set().union(*[set(mk.t) for mk in self.m]))
        if self.start is not None:
            timeline = [t for t in timeline if t >= self.start]
        if self.end is not None:
            timeline = [t for t in timeline if t < self.end]
        by_pair = {mk.pair: mk for mk in self.m}

        for T in timeline:
            # offene Position managen
            if self.pos is not None:
                mk = by_pair[self.pos.pair]
                i = mk.pos.get(T)
                if i is not None and i > self.pos.entry_idx:
                    self.manage_bar(mk, i)
                if self.pos is not None:
                    continue
            # neue Einstiege: Reaktionskerze = die gerade geschlossene Kerze
            for mk in self.m:
                i = mk.pos.get(T)
                if i is None or i < 1:
                    continue
                p = self.check_signal(mk, i - 1)
                if p is None:
                    continue
                self.pos = p
                self.blocked_days.add(T.tz_convert(BERLIN).date())
                self.manage_bar(mk, i)          # Einstiegskerze selbst kann bereits SL/TP treffen
                break
        return self


# --------------------------------------------------------------------------------------
# Auswertung
# --------------------------------------------------------------------------------------

def stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"trades": 0}
    gp = df.loc[df.pnl > 0, "pnl"].sum()
    gl = -df.loc[df.pnl < 0, "pnl"].sum()
    n = len(df)
    tp = (df.result == "TP").sum()
    sl = (df.result == "SL").sum()
    be = (df.result == "BE").sum()
    return {
        "trades": n, "TP": int(tp), "SL": int(sl), "BE": int(be),
        "trefferquote_%": 100 * tp / n,
        "trefferquote_ohne_BE_%": 100 * tp / (tp + sl) if tp + sl else float("nan"),
        "profit_factor": gp / gl if gl > 0 else float("inf"),
        "avg_R": df.r.mean(), "sum_R": df.r.sum(), "pnl": df.pnl.sum(),
    }


def verdict(df: pd.DataFrame) -> list[str]:
    out = []
    if len(df) >= 30:
        s = stats(df.iloc[:30])
        bad = s["trefferquote_%"] < 33 or s["profit_factor"] < 1.2
        out.append(f"Zwischenbilanz nach 30 Trades: Trefferquote {s['trefferquote_%']:.1f} %, "
                   f"PF {s['profit_factor']:.2f} -> "
                   + ("ÜBERPRÜFEN (Long/Short, Paare, Uhrzeit), nicht verwerfen" if bad else "unverändert weiter"))
    else:
        out.append(f"Zwischenbilanz (30 Trades) noch nicht erreicht ({len(df)} Trades)")
    if len(df) >= 50:
        s = stats(df.iloc[:50])
        pf = s["profit_factor"]
        v = ("funktioniert in dieser Form nicht -> genau eine Regel anpassen, neu zählen" if pf < 1.2 else
             "knapp profitabel -> weitermachen" if pf <= 1.5 else "bestätigt")
        out.append(f"Urteil nach 50 Trades: PF {pf:.2f} -> {v}")
    else:
        out.append(f"Urteil (50 Trades) noch nicht erreicht ({len(df)} Trades)")
    return out


def breakdown(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = {k: stats(g) for k, g in df.groupby(col)}
    return pd.DataFrame(rows).T[["trades", "TP", "SL", "BE", "trefferquote_%", "profit_factor", "sum_R"]]


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
    a1.set_title("Kontostand (geschlossene Trades)", loc="left", color=ink, fontsize=12)
    a1.annotate(f"{bal[-1]:,.0f}", (x[-1], bal[-1]), xytext=(6, 0), textcoords="offset points",
                color=ink, fontsize=9, va="center")
    a2.fill_between(x, dd, 0, step="post", color=series, alpha=0.25, linewidth=0)
    a2.step(x, dd, where="post", color=series, linewidth=1.5)
    for lvl, lab in ((-cfg.dd_reduce_pct, "0,5 %-Risiko"), (-cfg.dd_brake_pct, "Notbremse")):
        a2.axhline(lvl, color=muted, linewidth=1, linestyle=(0, (4, 3)))
        a2.text(x[-1], lvl, f"{lab} ({lvl:.0f} %) ", color=muted, fontsize=8, va="bottom", ha="right")
    a2.set_title("Drawdown vom Hoch in %", loc="left", color=ink, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_trade(mk: Market, p: Position, tr: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    up, down, tl_col, zone_col = "#2a78d6", "#52514e", "#eb6834", "#eda100"
    e4 = mk.h4_id[p.entry_idx]
    x4 = mk.h4_id[mk.pos[tr["exit_time_de"].tz_convert("UTC")]]
    a = max(0, min(p.tl["i"] if p.tl else e4 - 60, p.fib["origin_idx"]) - 5)
    b = min(len(mk.H4c) - 1, max(x4, e4) + 6)
    xs = np.arange(a, b + 1)

    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor=surface)
    ax.set_facecolor(surface)
    o, h, l, c = mk.H4o[xs], mk.H4h[xs], mk.H4l[xs], mk.H4c[xs]
    col = np.where(c >= o, up, down)
    ax.vlines(xs, l, h, color=col, linewidth=1)
    ax.bar(xs, np.maximum(abs(c - o), mk.pip * 0.3), bottom=np.minimum(o, c), color=col, width=0.6)

    if p.tl:
        line_x = np.arange(p.tl["i"], b + 1)
        ax.plot(line_x, p.tl["yi"] + p.tl["slope"] * (line_x - p.tl["i"]), color=tl_col, linewidth=2)
        tp_pts = np.array(p.tl["touches"])
        tprice = mk.H4l if p.side == "long" else mk.H4h
        ax.scatter(tp_pts, tprice[tp_pts], s=70, facecolor="none", edgecolor=tl_col, linewidth=2, zorder=5)
        ax.text(line_x[-1], p.tl["yi"] + p.tl["slope"] * (line_x[-1] - p.tl["i"]),
                f" Trendlinie ({len(tp_pts)} Berührungen)", color=ink, fontsize=8, va="center")
    else:
        ax.plot(xs, mk.ema4[xs], color=tl_col, linewidth=2)
        ax.text(xs[-1], mk.ema4[xs[-1]], " EMA200 (4H)", color=ink, fontsize=8, va="center")

    ax.fill_between([p.fib["ext_idx"], b + 0.5], p.fib["zone_lo"], p.fib["zone_hi"], color=zone_col,
                    alpha=0.2, linewidth=0)
    ax.plot([p.fib["origin_idx"], p.fib["ext_idx"]], [p.fib["origin"], p.fib["ext"]], color=muted,
            linewidth=1, linestyle=(0, (4, 3)))
    ax.text(p.fib["ext_idx"], p.fib["zone_hi"], " Fib 0,5–0,7", color=ink, fontsize=8, va="bottom")

    for y, lab in ((p.entry, "Einstieg"), (tr["sl_initial"], "SL"), (p.tp, "TP")):
        ax.hlines(y, e4 - 0.5, x4 + 0.5, color=ink if lab == "Einstieg" else muted, linewidth=1.2,
                  linestyle="-" if lab == "Einstieg" else (0, (2, 2)))
        ax.text(x4 + 0.6, y, f" {lab} {y:.5g}", color=ink, fontsize=8, va="center")
    ax.scatter([x4], [tr["exit"]], marker="x", s=60, color=ink, zorder=6)

    ticks = xs[:: max(1, len(xs) // 10)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([mk.t[mk.h4_first[i]].tz_convert(BERLIN).strftime("%d.%m %H:%M") for i in ticks],
                       fontsize=8, color=muted)
    ax.tick_params(colors=muted, labelsize=8)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"{p.pair} {p.side.upper()}  Einstieg {tr['entry_time_de']:%d.%m.%Y %H:%M}  ->  "
                 f"{tr['result']} {tr['r']:+.2f}R  (CRV {p.crv:.1f})", loc="left", color=ink, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def report(bt: Backtest, cfg: Config, out: Path, news_active: bool) -> str:
    df = pd.DataFrame(bt.trades)
    lines = ["=" * 78, "BACKTEST-ERGEBNIS", "=" * 78,
             f"Paare: {', '.join(m.pair for m in bt.m)}",
             "Richtung: " + (f"EMA{cfg.ema_len} 4H" + (f", Steigung über {cfg.ema_slope_bars} Kerzen"
                                                       if cfg.ema_slope_bars else "")
                             if cfg.direction == "ema" else "Trendlinie 4H (>= 3 Berührungen)"),
             f"Zeitraum: {(bt.start or bt.m[0].t[0]).tz_convert(BERLIN):%d.%m.%Y} – "
             f"{(bt.end or bt.m[0].t[-1]).tz_convert(BERLIN):%d.%m.%Y}",
             f"News-Filter: {'aktiv' if news_active else 'NICHT aktiv (keine --news Datei)'}",
             f"Startkapital: {cfg.start_balance:,.0f} {cfg.account_ccy}", ""]
    if df.empty:
        lines.append("Keine Trades.")
    else:
        s = stats(df)
        bal = np.r_[cfg.start_balance, df.balance.to_numpy()]
        peak = np.maximum.accumulate(bal)
        max_dd = ((peak - bal) / peak * 100).max()
        min_bal = bal.min()
        lines += [
            f"Trades: {s['trades']}  (TP {s['TP']} / SL {s['SL']} / BE {s['BE']})",
            f"Trefferquote (TP/alle): {s['trefferquote_%']:.1f} %   (TP/(TP+SL): {s['trefferquote_ohne_BE_%']:.1f} %)"
            f"   Break-even-Quote bei 1:2: 33,3 %",
            f"Profit Factor: {s['profit_factor']:.2f}",
            f"Ø R pro Trade: {s['avg_R']:+.2f}   Summe R: {s['sum_R']:+.1f}",
            f"Endkapital: {bt.balance:,.0f}  ({(bt.balance / cfg.start_balance - 1) * 100:+.1f} %)",
            f"Max. Drawdown vom Hoch (geschlossen): {max_dd:.1f} %",
            f"FTMO-Limit {cfg.ftmo_max_loss_pct:.0f} %: tiefster Kontostand {min_bal:,.0f} -> "
            + ("VERLETZT" if min_bal <= cfg.start_balance * (1 - cfg.ftmo_max_loss_pct / 100) else "eingehalten"),
            f"Trades mit 0,5 % Risiko (DD-Regel): {(df.risk_pct < cfg.risk_pct).sum()}",
            f"Trades mit BE-Verschiebung: {df.be_moved.sum()}",
            f"Kerzen mit SL+TP in derselben 1H-Kerze (konservativ als SL gewertet): {df.ambiguous_bar.sum()}",
            f"Notbremse-Pausen: {len(bt.pauses)}"
            + "".join(f"\n   ab {a.tz_convert(BERLIN):%d.%m.%Y} bis {b.tz_convert(BERLIN):%d.%m.%Y} (DD {d:.1f} %)"
                      for a, b, d in bt.pauses),
            "", "-- Bewertung (Abschnitt 9) --", *verdict(df), "",
            "-- Long/Short --", breakdown(df, "side").round(2).to_string(), "",
            "-- Paare --", breakdown(df, "pair").round(2).to_string(), "",
        ]
        df["entry_hour"] = df.entry_time_de.map(lambda x: x.hour)
        lines += ["-- Einstiegsstunde (DE) --", breakdown(df, "entry_hour").round(2).to_string(), ""]
        df.to_csv(out / "trades.csv", index=False)
        plot_equity(df, cfg, out / "equity.png")
    lines += ["-- Verworfene Kandidaten (erster nicht erfüllter Punkt) --"]
    lines += [f"  {k:<45} {v}" for k, v in sorted(bt.rejects.items(), key=lambda kv: -kv[1])]
    text = "\n".join(lines)
    (out / "report.txt").write_text(text, encoding="utf-8")
    return text


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="+", default=["EURUSD"], help="z. B. EURUSD GBPUSD USDJPY")
    ap.add_argument("--source", choices=["csv", "yf"], default="csv")
    ap.add_argument("--data-dir", type=Path, default=Path("data"), help="CSV-Dateien <PAAR>.csv")
    ap.add_argument("--csv-tz", default="UTC",
                    help="Zeitzone der CSV-Zeitstempel: UTC, Europe/Berlin, ... oder 'server' (MT5, NY+7)")
    ap.add_argument("--dayfirst", action="store_true", help="Datumsformat TT.MM.JJJJ")
    ap.add_argument("--news", type=Path, help="CSV mit datetime,currency,impact")
    ap.add_argument("--news-tz", default="UTC")
    ap.add_argument("--from", dest="date_from", help="Startdatum JJJJ-MM-TT (Indikatoren laufen vorher warm)")
    ap.add_argument("--to", dest="date_to")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--balance", type=float, default=Config.start_balance)
    ap.add_argument("--account-ccy", default=Config.account_ccy)
    ap.add_argument("--spread-pips", type=float, default=Config.spread_pips)
    ap.add_argument("--spreads", default="", help="z. B. GBPJPY=2.0,EURUSD=0.6")
    ap.add_argument("--commission-pips", type=float, default=Config.commission_pips)
    ap.add_argument("--stop-buffer-pips", type=float, default=Config.stop_buffer_pips)
    ap.add_argument("--min-stop-pips", type=float, default=Config.min_stop_pips)
    ap.add_argument("--direction", choices=["ema", "trendline"], default=Config.direction)
    ap.add_argument("--ema-len", type=int, default=Config.ema_len)
    ap.add_argument("--ema-slope-bars", type=int, default=Config.ema_slope_bars,
                    help="0 = aus; sonst muss der EMA über N 4H-Kerzen steigen (Long) / fallen (Short)")
    ap.add_argument("--swing-len", type=int, default=Config.swing_len)
    ap.add_argument("--tl-pivot-len", type=int, default=Config.tl_pivot_len)
    ap.add_argument("--tl-tol-atr", type=float, default=Config.tl_tol_atr)
    ap.add_argument("--tl-lookback", type=int, default=Config.tl_lookback)
    ap.add_argument("--tl-touch-gap", type=int, default=Config.tl_min_touch_gap)
    ap.add_argument("--h4-anchor", choices=["ny", "utc"], default=Config.h4_anchor)
    ap.add_argument("--no-brake", action="store_true", help="Notbremse-Pause nicht simulieren")
    ap.add_argument("--plot-trades", type=int, default=0, help="Charts der ersten N Trades nach <out>/trades/")
    a = ap.parse_args(argv)

    cfg = Config(
        start_balance=a.balance, account_ccy=a.account_ccy.upper(), spread_pips=a.spread_pips,
        spreads={k.upper(): float(v) for k, v in (s.split("=") for s in a.spreads.split(",") if s)},
        commission_pips=a.commission_pips, stop_buffer_pips=a.stop_buffer_pips, min_stop_pips=a.min_stop_pips,
        direction=a.direction, ema_len=a.ema_len, ema_slope_bars=a.ema_slope_bars,
        swing_len=a.swing_len, tl_pivot_len=a.tl_pivot_len, tl_tol_atr=a.tl_tol_atr,
        tl_lookback=a.tl_lookback, tl_min_touch_gap=a.tl_touch_gap, h4_anchor=a.h4_anchor, use_brake=not a.no_brake,
    )

    markets = []
    pairs = [x.upper() for arg in a.pairs for x in arg.replace(",", " ").split()]
    for pair in pairs:
        if a.source == "yf":
            h1 = load_yf(pair)
        else:
            f = a.data_dir / f"{pair}.csv"
            if not f.exists():
                sys.exit(f"Datei fehlt: {f}")
            h1 = load_csv(f, a.csv_tz, a.dayfirst)
        h1 = clean_h1(h1)
        print(f"{pair}: {len(h1)} 1H-Kerzen  {h1.index[0]} – {h1.index[-1]}", file=sys.stderr)
        markets.append(Market(pair, h1, cfg))

    news = load_news(a.news, a.news_tz) if a.news else None
    tz_start = pd.Timestamp(a.date_from, tz=BERLIN).tz_convert("UTC") if a.date_from else None
    tz_end = pd.Timestamp(a.date_to, tz=BERLIN).tz_convert("UTC") if a.date_to else None

    a.out.mkdir(parents=True, exist_ok=True)
    bt = Backtest(markets, cfg, news, tz_start, tz_end).run()
    print(report(bt, cfg, a.out, news is not None))
    if a.plot_trades:
        (a.out / "trades").mkdir(exist_ok=True)
        by_pair = {m.pair: m for m in markets}
        for n, (p, tr) in enumerate(list(zip(bt.closed, bt.trades))[: a.plot_trades], 1):
            plot_trade(by_pair[p.pair], p, tr, a.out / "trades" / f"{n:03d}_{p.pair}_{p.side}.png")
    (a.out / "config.txt").write_text("\n".join(f"{k} = {v}" for k, v in asdict(cfg).items()), encoding="utf-8")


if __name__ == "__main__":
    main()
