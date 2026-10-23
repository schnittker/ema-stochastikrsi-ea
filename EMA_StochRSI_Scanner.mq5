//+------------------------------------------------------------------+
//|                                         EMA_StochRSI_Scanner.mq5 |
//|  Multi-Symbol-Scanner: EMA 50/100 Trendfilter + Stochastik RSI   |
//|                                                                  |
//|  Long : EMA50 > EMA100, Close > EMA50, %K kreuzt %D nach oben,   |
//|         Kreuzung unterhalb 25                                    |
//|  Short: EMA50 < EMA100, Close < EMA50, %K kreuzt %D nach unten,  |
//|         Kreuzung oberhalb 75                                     |
//|  SL   : tiefstes Tief / höchstes Hoch der letzten N Kerzen       |
//|         +/- Spread + Puffer, TP = CRV x Risiko                   |
//+------------------------------------------------------------------+
#property copyright "Markus Schnittker"
#property version   "1.00"
#property description "Scannt alle Symbole der Market Watch (H1): EMA 50/100 + Stochastik-RSI-Kreuz, CRV 1:2"

#include <Trade\Trade.mqh>

//--- Strategie
input group "Strategie"
input ENUM_TIMEFRAMES InpTimeframe      = PERIOD_H1; // Zeitrahmen
input int             InpEmaFast        = 50;        // EMA schnell
input int             InpEmaSlow        = 100;       // EMA langsam
input bool            InpPriceEmaFilter = true;      // Long nur Close > EMA schnell, Short nur Close < EMA schnell
input int             InpRsiPeriod      = 14;        // Stoch RSI: RSI-Länge
input int             InpStochPeriod    = 14;        // Stoch RSI: Stochastik-Länge
input int             InpKSmooth        = 3;         // Stoch RSI: %K-Glättung
input int             InpDSmooth        = 3;         // Stoch RSI: %D-Glättung
input double          InpOversold       = 25.0;      // Long-Zone (Kreuzung unterhalb)
input double          InpOverbought     = 75.0;      // Short-Zone (Kreuzung oberhalb)

//--- Risiko & Trade
input group "Risiko & Trade"
input double          InpRiskPercent    = 1.0;       // Risiko pro Trade in % vom Kontostand
input double          InpRewardRisk     = 2.0;       // CRV (TP = x * SL-Abstand)
input int             InpSwingLookback  = 5;         // SL: Swing-Lookback (geschlossene Kerzen)
input int             InpSlBufferPoints = 0;         // SL: zusätzlicher Puffer in Punkten
input bool            InpAddSpreadToSL  = true;      // SL: aktuellen Spread als Puffer addieren
input int             InpMaxTotalTrades = 5;         // Max. offene Trades gesamt (1 pro Symbol)

//--- Allgemein
input group "Allgemein"
input string          InpSymbols          = "";      // Symbolliste (leer = alle aus Market Watch), z.B. "EURUSD,GBPUSD"
input int             InpMaxEntryDelayMin = 15;      // Signal verfällt, wenn Kerze älter als x Minuten
input ulong           InpMagic            = 50100;   // Magic Number
input int             InpSlippagePoints   = 10;      // Max. Slippage in Punkten
input string          InpComment          = "EMA_StochRSI"; // Order-Kommentar
input int             InpTimerSeconds     = 2;       // Scan-Intervall in Sekunden

//--- Zustand je Symbol
struct SymbolState
  {
   string            name;
   int               hEmaFast;
   int               hEmaSlow;
   int               hRsi;
   datetime          lastBar;   // zuletzt ausgewertete Kerze (Open-Zeit der Kerze 0)
  };

SymbolState g_syms[];
string      g_ignored[];        // Symbole, die nicht handelbar sind / keine Indikatoren liefern
CTrade      g_trade;
ulong       g_lastRefreshMs = 0;

//+------------------------------------------------------------------+
int OnInit()
  {
   if(InpEmaFast < 1 || InpEmaSlow < 1 || InpRsiPeriod < 1 || InpStochPeriod < 1 ||
      InpKSmooth < 1 || InpDSmooth < 1 || InpSwingLookback < 1 || InpRewardRisk <= 0 ||
      InpRiskPercent <= 0 || InpMaxTotalTrades < 1 || InpTimerSeconds < 1)
     {
      Print("Ungültige Eingabeparameter.");
      return INIT_PARAMETERS_INCORRECT;
     }

   g_trade.SetExpertMagicNumber(InpMagic);
   g_trade.SetDeviationInPoints(InpSlippagePoints);
   g_trade.SetAsyncMode(false);

   RefreshSymbols();
   PrintFormat("EMA_StochRSI_Scanner gestartet: %d Symbole auf %s", ArraySize(g_syms), EnumToString(InpTimeframe));

   EventSetTimer(InpTimerSeconds);
   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
   for(int i = 0; i < ArraySize(g_syms); i++)
      ReleaseState(g_syms[i]);
   ArrayResize(g_syms, 0);
   Comment("");
  }

//+------------------------------------------------------------------+
void OnTimer()
  {
   // Market Watch kann sich zur Laufzeit ändern -> Liste jede Minute abgleichen
   if(GetTickCount64() - g_lastRefreshMs >= 60000)
      RefreshSymbols();
   Scan();
  }

//+------------------------------------------------------------------+
void OnTick()
  {
   Scan();
  }

//+------------------------------------------------------------------+
//| Symbolverwaltung                                                 |
//+------------------------------------------------------------------+
bool InList(const string name, const string &list[])
  {
   for(int i = 0; i < ArraySize(list); i++)
      if(list[i] == name)
         return true;
   return false;
  }

void AppendString(string &list[], const string value)
  {
   int n = ArraySize(list);
   ArrayResize(list, n + 1);
   list[n] = value;
  }

int FindState(const string name)
  {
   for(int i = 0; i < ArraySize(g_syms); i++)
      if(g_syms[i].name == name)
         return i;
   return -1;
  }

void GetWantedSymbols(string &out[])
  {
   ArrayResize(out, 0);
   if(StringLen(InpSymbols) > 0)
     {
      string parts[];
      int n = StringSplit(InpSymbols, ',', parts);
      for(int i = 0; i < n; i++)
        {
         string s = parts[i];
         StringTrimLeft(s);
         StringTrimRight(s);
         if(s == "" || InList(s, out))
            continue;
         if(!SymbolSelect(s, true))
           {
            if(!InList(s, g_ignored))
              {
               PrintFormat("Symbol %s nicht gefunden - wird ignoriert.", s);
               AppendString(g_ignored, s);
              }
            continue;
           }
         AppendString(out, s);
        }
     }
   else
     {
      int total = SymbolsTotal(true);
      for(int i = 0; i < total; i++)
         AppendString(out, SymbolName(i, true));
     }
  }

bool AddSymbol(const string name)
  {
   if(SymbolInfoInteger(name, SYMBOL_CUSTOM) ||
      SymbolInfoInteger(name, SYMBOL_TRADE_MODE) == SYMBOL_TRADE_MODE_DISABLED)
      return false;

   int hFast = iMA(name, InpTimeframe, InpEmaFast, 0, MODE_EMA, PRICE_CLOSE);
   int hSlow = iMA(name, InpTimeframe, InpEmaSlow, 0, MODE_EMA, PRICE_CLOSE);
   int hRsi  = iRSI(name, InpTimeframe, InpRsiPeriod, PRICE_CLOSE);
   if(hFast == INVALID_HANDLE || hSlow == INVALID_HANDLE || hRsi == INVALID_HANDLE)
     {
      if(hFast != INVALID_HANDLE) IndicatorRelease(hFast);
      if(hSlow != INVALID_HANDLE) IndicatorRelease(hSlow);
      if(hRsi  != INVALID_HANDLE) IndicatorRelease(hRsi);
      PrintFormat("%s: Indikatoren konnten nicht erstellt werden (Fehler %d).", name, GetLastError());
      return false;
     }

   int n = ArraySize(g_syms);
   ArrayResize(g_syms, n + 1);
   g_syms[n].name     = name;
   g_syms[n].hEmaFast = hFast;
   g_syms[n].hEmaSlow = hSlow;
   g_syms[n].hRsi     = hRsi;
   g_syms[n].lastBar  = 0;
   return true;
  }

void ReleaseState(SymbolState &s)
  {
   if(s.hEmaFast != INVALID_HANDLE) IndicatorRelease(s.hEmaFast);
   if(s.hEmaSlow != INVALID_HANDLE) IndicatorRelease(s.hEmaSlow);
   if(s.hRsi     != INVALID_HANDLE) IndicatorRelease(s.hRsi);
   s.hEmaFast = s.hEmaSlow = s.hRsi = INVALID_HANDLE;
  }

void RefreshSymbols()
  {
   g_lastRefreshMs = GetTickCount64();

   string wanted[];
   GetWantedSymbols(wanted);

   // Nicht mehr gewünschte Symbole entfernen
   for(int i = ArraySize(g_syms) - 1; i >= 0; i--)
     {
      if(InList(g_syms[i].name, wanted))
         continue;
      ReleaseState(g_syms[i]);
      int last = ArraySize(g_syms) - 1;
      for(int j = i; j < last; j++)
         g_syms[j] = g_syms[j + 1];
      ArrayResize(g_syms, last);
     }

   // Neue Symbole hinzufügen
   for(int i = 0; i < ArraySize(wanted); i++)
     {
      if(FindState(wanted[i]) >= 0 || InList(wanted[i], g_ignored))
         continue;
      if(!AddSymbol(wanted[i]))
         AppendString(g_ignored, wanted[i]);
     }
  }

//+------------------------------------------------------------------+
//| Scan                                                             |
//+------------------------------------------------------------------+
void Scan()
  {
   for(int i = 0; i < ArraySize(g_syms); i++)
      ProcessSymbol(g_syms[i]);

   if(!MQLInfoInteger(MQL_TESTER) || MQLInfoInteger(MQL_VISUAL_MODE))
      Comment(StringFormat("EMA/StochRSI Scanner (%s)\nSymbole: %d\nOffene Trades: %d / %d",
                           EnumToString(InpTimeframe), ArraySize(g_syms),
                           CountPositions(), InpMaxTotalTrades));
  }

void ProcessSymbol(SymbolState &s)
  {
   datetime bar0 = iTime(s.name, InpTimeframe, 0);
   if(bar0 == 0 || bar0 <= s.lastBar)
      return;

   // Neue Kerze erst spät erkannt (Terminal offline, keine Ticks) -> Signal verfallen lassen
   if(TimeCurrent() - bar0 > (long)InpMaxEntryDelayMin * 60)
     {
      s.lastBar = bar0;
      return;
     }

   int signal = 0;
   if(!GetSignal(s, signal))
      return;                   // Daten noch nicht bereit -> beim nächsten Durchlauf erneut

   s.lastBar = bar0;
   if(signal != 0)
      TryOpen(s.name, signal, bar0);
  }

//+------------------------------------------------------------------+
//| Signal auf der zuletzt geschlossenen Kerze (Shift 1)             |
//+------------------------------------------------------------------+
bool GetSignal(const SymbolState &s, int &signal)
  {
   signal = 0;
   if(iBars(s.name, InpTimeframe) < InpEmaSlow + InpRsiPeriod + InpStochPeriod + 10)
      return false;

   double emaF[], emaS[], close[];
   if(CopyBuffer(s.hEmaFast, 0, 1, 1, emaF) != 1) return false;
   if(CopyBuffer(s.hEmaSlow, 0, 1, 1, emaS) != 1) return false;
   if(CopyClose(s.name, InpTimeframe, 1, 1, close) != 1) return false;

   double k1, d1, k2, d2;
   if(!CalcStochRsi(s.hRsi, k1, d1, k2, d2))
      return false;

   bool trendUp = emaF[0] > emaS[0] && (!InpPriceEmaFilter || close[0] > emaF[0]);
   bool trendDn = emaF[0] < emaS[0] && (!InpPriceEmaFilter || close[0] < emaF[0]);

   // Kreuzung zwischen Kerze 2 und Kerze 1; vor der Kreuzung lagen beide Linien in der Zone,
   // %D liegt auch auf der Signalkerze noch in der Zone (%K darf sie gerade verlassen)
   bool crossUp = k2 <= d2 && k1 > d1 &&
                  k2 < InpOversold && d2 < InpOversold && d1 < InpOversold;
   bool crossDn = k2 >= d2 && k1 < d1 &&
                  k2 > InpOverbought && d2 > InpOverbought && d1 > InpOverbought;

   if(trendUp && crossUp)
      signal = 1;
   else if(trendDn && crossDn)
      signal = -1;

   if(signal != 0)
      PrintFormat("%s %s-Signal: EMA%d=%.5f EMA%d=%.5f | %%K %.1f->%.1f  %%D %.1f->%.1f",
                  s.name, signal > 0 ? "LONG" : "SHORT", InpEmaFast, emaF[0], InpEmaSlow, emaS[0],
                  k2, k1, d2, d1);
   return true;
  }

// Stochastik RSI wie TradingView: %K = SMA(Stoch(RSI), kSmooth), %D = SMA(%K, dSmooth)
// Liefert %K/%D für Kerze 1 (k1/d1) und Kerze 2 (k2/d2)
bool CalcStochRsi(const int hRsi, double &k1, double &d1, double &k2, double &d2)
  {
   int need = InpStochPeriod + InpKSmooth + InpDSmooth;
   double rsi[];
   ArraySetAsSeries(rsi, true);
   if(CopyBuffer(hRsi, 0, 1, need, rsi) != need)
      return false;             // rsi[0] = Kerze 1

   int nK   = InpDSmooth + 1;   // %K für Index 0..dSmooth
   int nRaw = nK + InpKSmooth - 1;
   double raw[], k[];
   ArrayResize(raw, nRaw);
   ArrayResize(k, nK);

   for(int i = 0; i < nRaw; i++)
     {
      double lo = rsi[i], hi = rsi[i];
      for(int j = 1; j < InpStochPeriod; j++)
        {
         lo = MathMin(lo, rsi[i + j]);
         hi = MathMax(hi, rsi[i + j]);
        }
      raw[i] = (hi - lo) > 0.0 ? (rsi[i] - lo) / (hi - lo) * 100.0 : 50.0;
     }

   for(int i = 0; i < nK; i++)
     {
      double sum = 0.0;
      for(int j = 0; j < InpKSmooth; j++)
         sum += raw[i + j];
      k[i] = sum / InpKSmooth;
     }

   double s1 = 0.0, s2 = 0.0;
   for(int j = 0; j < InpDSmooth; j++)
     {
      s1 += k[j];
      s2 += k[j + 1];
     }

   k1 = k[0];
   k2 = k[1];
   d1 = s1 / InpDSmooth;
   d2 = s2 / InpDSmooth;
   return true;
  }

//+------------------------------------------------------------------+
//| Positionen                                                       |
//+------------------------------------------------------------------+
int CountPositions()
  {
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(PositionGetSymbol(i) == "")
         continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) == InpMagic)
         count++;
     }
   return count;
  }

bool HasPositionOnSymbol(const string sym)
  {
   // Netting-Konto: jede Position auf dem Symbol würde verrechnet -> blockieren
   if(AccountInfoInteger(ACCOUNT_MARGIN_MODE) != ACCOUNT_MARGIN_MODE_RETAIL_HEDGING)
      return PositionSelect(sym);

   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(PositionGetSymbol(i) != sym)
         continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) == InpMagic)
         return true;
     }
   return false;
  }

// Schutz bei EA-Neustart: auf dieser Kerze wurde bereits ein Trade eröffnet (evtl. schon geschlossen)
bool AlreadyTradedBar(const string sym, const datetime barTime)
  {
   if(!HistorySelect(barTime, TimeCurrent() + 60))
      return false;
   for(int i = HistoryDealsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0)
         continue;
      if(HistoryDealGetString(ticket, DEAL_SYMBOL) == sym &&
         (ulong)HistoryDealGetInteger(ticket, DEAL_MAGIC) == InpMagic &&
         HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_IN)
         return true;
     }
   return false;
  }

//+------------------------------------------------------------------+
//| Order                                                            |
//+------------------------------------------------------------------+
double NormalizePrice(const string sym, double price)
  {
   double tickSize = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_SIZE);
   if(tickSize > 0.0)
      price = MathRound(price / tickSize) * tickSize;
   return NormalizeDouble(price, (int)SymbolInfoInteger(sym, SYMBOL_DIGITS));
  }

int VolumeDigits(double step)
  {
   int digits = 0;
   while(digits < 8 && MathAbs(step - MathRound(step)) > 1e-9)
     {
      step *= 10.0;
      digits++;
     }
   return digits;
  }

double CalcLots(const string sym, const int signal, const double entry, const double sl)
  {
   double riskMoney = AccountInfoDouble(ACCOUNT_BALANCE) * InpRiskPercent / 100.0;
   ENUM_ORDER_TYPE type = signal > 0 ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;

   double loss = 0.0;
   if(!OrderCalcProfit(type, sym, 1.0, entry, sl, loss) || loss >= 0.0)
     {
      double tickSize  = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_SIZE);
      double tickValue = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE_LOSS);
      if(tickSize <= 0.0 || tickValue <= 0.0)
         return 0.0;
      loss = -MathAbs(entry - sl) / tickSize * tickValue;
     }
   double lossPerLot = MathAbs(loss);
   if(lossPerLot <= 0.0)
      return 0.0;

   double step   = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);
   double minLot = SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(sym, SYMBOL_VOLUME_MAX);
   if(step <= 0.0)
      step = minLot;

   double lots = MathFloor(riskMoney / lossPerLot / step + 1e-9) * step;
   if(lots < minLot)
     {
      PrintFormat("%s: Lot %.4f unter Mindestlot %.2f (SL zu weit für %.2f%% Risiko) - kein Trade.",
                  sym, lots, minLot, InpRiskPercent);
      return 0.0;
     }
   lots = NormalizeDouble(MathMin(lots, maxLot), VolumeDigits(step));

   double margin = 0.0;
   if(OrderCalcMargin(type, sym, lots, entry, margin) && margin > AccountInfoDouble(ACCOUNT_MARGIN_FREE))
     {
      PrintFormat("%s: nicht genug freie Margin für %.2f Lot - kein Trade.", sym, lots);
      return 0.0;
     }
   return lots;
  }

void TryOpen(const string sym, const int signal, const datetime barTime)
  {
   string dir = signal > 0 ? "LONG" : "SHORT";

   if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED) || !MQLInfoInteger(MQL_TRADE_ALLOWED))
     {
      PrintFormat("%s %s: Algo-Trading nicht erlaubt - kein Trade.", sym, dir);
      return;
     }
   if(CountPositions() >= InpMaxTotalTrades)
     {
      PrintFormat("%s %s: max. Anzahl offener Trades (%d) erreicht.", sym, dir, InpMaxTotalTrades);
      return;
     }
   if(HasPositionOnSymbol(sym))
     {
      PrintFormat("%s %s: es besteht bereits eine Position auf dem Symbol.", sym, dir);
      return;
     }
   if(AlreadyTradedBar(sym, barTime))
      return;

   long mode = SymbolInfoInteger(sym, SYMBOL_TRADE_MODE);
   if(mode == SYMBOL_TRADE_MODE_DISABLED || mode == SYMBOL_TRADE_MODE_CLOSEONLY ||
      (signal > 0 && mode == SYMBOL_TRADE_MODE_SHORTONLY) ||
      (signal < 0 && mode == SYMBOL_TRADE_MODE_LONGONLY))
     {
      PrintFormat("%s %s: Richtung auf diesem Symbol nicht handelbar.", sym, dir);
      return;
     }

   MqlTick tick;
   if(!SymbolInfoTick(sym, tick) || tick.ask <= 0.0 || tick.bid <= 0.0)
      return;

   double point  = SymbolInfoDouble(sym, SYMBOL_POINT);
   double buffer = InpSlBufferPoints * point + (InpAddSpreadToSL ? tick.ask - tick.bid : 0.0);

   double entry, sl, tp;
   if(signal > 0)
     {
      double lows[];
      if(CopyLow(sym, InpTimeframe, 1, InpSwingLookback, lows) != InpSwingLookback)
         return;
      entry = tick.ask;
      sl    = NormalizePrice(sym, lows[ArrayMinimum(lows)] - buffer);
      tp    = NormalizePrice(sym, entry + (entry - sl) * InpRewardRisk);
     }
   else
     {
      double highs[];
      if(CopyHigh(sym, InpTimeframe, 1, InpSwingLookback, highs) != InpSwingLookback)
         return;
      entry = tick.bid;
      sl    = NormalizePrice(sym, highs[ArrayMaximum(highs)] + buffer);
      tp    = NormalizePrice(sym, entry - (sl - entry) * InpRewardRisk);
     }

   // SL muss auf der richtigen Seite liegen und den Stops-Level des Brokers einhalten
   double minDist = (double)SymbolInfoInteger(sym, SYMBOL_TRADE_STOPS_LEVEL) * point;
   bool stopsOk = signal > 0 ? (tick.bid - sl > minDist && tp - tick.bid > minDist)
                             : (sl - tick.ask > minDist && tick.ask - tp > minDist);
   if(!stopsOk)
     {
      PrintFormat("%s %s: SL/TP ungültig (Entry %.5f SL %.5f TP %.5f) - kein Trade.", sym, dir, entry, sl, tp);
      return;
     }

   double lots = CalcLots(sym, signal, entry, sl);
   if(lots <= 0.0)
      return;

   g_trade.SetTypeFillingBySymbol(sym);
   bool ok = signal > 0 ? g_trade.Buy(lots, sym, 0.0, sl, tp, InpComment)
                        : g_trade.Sell(lots, sym, 0.0, sl, tp, InpComment);

   if(ok && (g_trade.ResultRetcode() == TRADE_RETCODE_DONE || g_trade.ResultRetcode() == TRADE_RETCODE_PLACED))
      PrintFormat("%s %s eröffnet: %.2f Lot @ %.5f  SL %.5f  TP %.5f", sym, dir, lots,
                  g_trade.ResultPrice(), sl, tp);
   else
      PrintFormat("%s %s fehlgeschlagen: %d %s", sym, dir, g_trade.ResultRetcode(),
                  g_trade.ResultRetcodeDescription());
  }
//+------------------------------------------------------------------+
