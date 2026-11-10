#!/usr/bin/env python3
"""
daily_pick.py - سهم واحد يومياً من السوق الأمريكي + اختبار تاريخي

الفكرة:
  عند إغلاق كل يوم يُحسب لكل سهم تقييم من:
    1) التدفق (Flow): الحجم النسبي (حجم اليوم / متوسط 20 يوماً) مضروباً في
       موضع الإغلاق داخل مدى اليوم (إغلاق قرب القمة = تدفق شراء).
    2) نسبة الارتفاع: عائد 5 أيام وعائد 20 يوماً (الزخم).
    3) الأخبار/التوقعات (اختياري): ملف CSV فيه date,ticker,score من -1 إلى +1.
  السهم الأعلى تقييماً يُشترى عند افتتاح اليوم التالي ويُباع عند إغلاقه.
  لا يُستخدم أي سعر من المستقبل في القرار.

البيانات:
  مجلد فيه ملف CSV لكل سهم، اسم الملف هو الرمز (مثل NVDA.csv)،
  والأعمدة: Date,Open,High,Low,Close,Volume

أمثلة:
  python3 daily_pick.py backtest --data data/ --year 2024 --capital-sar 100000
  python3 daily_pick.py pick --data data/
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

SAR_PER_USD = 3.75


# ---------- تحميل البيانات ----------
def load_data(folder):
    frames = {}
    for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        ticker = os.path.splitext(os.path.basename(path))[0].upper().replace(".US", "")
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        need = {"date", "open", "high", "low", "close", "volume"}
        if not need.issubset(df.columns):
            print(f"تخطي {path}: أعمدة ناقصة {need - set(df.columns)}", file=sys.stderr)
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").drop_duplicates("date").set_index("date")
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        df = df[(df["close"] > 0) & (df["open"] > 0)]
        frames[ticker] = df
    if not frames:
        sys.exit("لا توجد ملفات CSV صالحة في المجلد")
    return frames


def load_news(path):
    if not path:
        return None
    n = pd.read_csv(path)
    n.columns = [c.strip().lower() for c in n.columns]
    n["date"] = pd.to_datetime(n["date"])
    n["ticker"] = n["ticker"].str.upper()
    return n.pivot_table(index="date", columns="ticker", values="score", aggfunc="mean")


# ---------- التقييم ----------
def zscore_rows(df):
    """تطبيع أفقي: مقارنة الأسهم ببعضها في اليوم نفسه."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def compute_scores(frames, news=None, w_flow=0.4, w_mom=0.4, w_news=0.2):
    close = pd.DataFrame({t: f["close"] for t, f in frames.items()})
    high = pd.DataFrame({t: f["high"] for t, f in frames.items()})
    low = pd.DataFrame({t: f["low"] for t, f in frames.items()})
    vol = pd.DataFrame({t: f["volume"] for t, f in frames.items()})

    rel_vol = vol / vol.rolling(20).mean()
    rng = (high - low).replace(0, np.nan)
    close_pos = ((close - low) / rng * 2 - 1).fillna(0)      # -1 قاع اليوم، +1 قمته
    flow = rel_vol * close_pos

    mom = 0.5 * close.pct_change(5) + 0.5 * close.pct_change(20)
    trend_ok = close > close.rolling(20).mean()              # فلتر: فوق متوسط 20 يوماً

    score = w_flow * zscore_rows(flow) + w_mom * zscore_rows(mom)
    if news is not None:
        nz = news.reindex(index=close.index, columns=close.columns).fillna(0)
        score = score + w_news * nz
    score = score.where(trend_ok)
    return score


# ---------- الاختبار التاريخي ----------
def backtest(frames, year, capital_usd, cost_pct=0.05, stop_pct=None,
             min_score=0.0, news=None, compound=False):
    score = compute_scores(frames, news)
    opn = pd.DataFrame({t: f["open"] for t, f in frames.items()})
    low = pd.DataFrame({t: f["low"] for t, f in frames.items()})
    close = pd.DataFrame({t: f["close"] for t, f in frames.items()})

    dates = score.index
    equity = capital_usd
    rows = []
    for i in range(len(dates) - 1):
        sig_day, trade_day = dates[i], dates[i + 1]
        if trade_day.year != year:
            continue
        s = score.loc[sig_day].dropna()
        if s.empty or s.max() < min_score:
            rows.append(dict(date=trade_day, ticker="", pnl=0.0, ret=0.0, equity=equity))
            continue
        t = s.idxmax()
        o, c, l = opn.at[trade_day, t], close.at[trade_day, t], low.at[trade_day, t]
        if np.isnan(o) or np.isnan(c):
            rows.append(dict(date=trade_day, ticker="", pnl=0.0, ret=0.0, equity=equity))
            continue
        exit_px = c
        if stop_pct is not None and l <= o * (1 - stop_pct / 100):
            exit_px = o * (1 - stop_pct / 100)
        ret = exit_px / o - 1 - 2 * cost_pct / 100           # تكلفة دخول + خروج
        base = equity if compound else capital_usd
        pnl = base * ret
        equity += pnl
        rows.append(dict(date=trade_day, ticker=t, pnl=pnl, ret=ret, equity=equity))
    return pd.DataFrame(rows)


def report(tr, capital_usd, target):
    if tr.empty:
        print("لا توجد أيام تداول في هذه السنة ضمن البيانات")
        return
    traded = tr[tr["ticker"] != ""]
    total = tr["pnl"].sum()
    peak = tr["equity"].cummax()
    dd = (tr["equity"] - peak).min()
    print("=" * 52)
    print(f"رأس المال:              ${capital_usd:,.0f}")
    print(f"أيام السوق:             {len(tr)}")
    print(f"أيام فيها صفقة:         {len(traded)}")
    print(f"إجمالي الربح/الخسارة:   ${total:,.0f}  ({total / capital_usd * 100:.1f}%)")
    if len(traded):
        print(f"متوسط اليوم (صفقات):    ${traded['pnl'].mean():,.0f}")
        print(f"نسبة الأيام الرابحة:    {(traded['pnl'] > 0).mean() * 100:.1f}%")
        print(f"أفضل يوم / أسوأ يوم:    ${traded['pnl'].max():,.0f} / ${traded['pnl'].min():,.0f}")
    print(f"أقصى تراجع:             ${dd:,.0f}")
    hit = (tr["pnl"] >= target).sum()
    print(f"أيام ربحها >= ${target:,.0f}:   {hit} من {len(tr)} ({hit / len(tr) * 100:.1f}%)")
    print(f"أيام خاسرة:             {(tr['pnl'] < 0).sum()}")
    print("-" * 52)
    m = tr.set_index("date")["pnl"].resample("ME").sum()
    for d, v in m.items():
        print(f"  {d:%Y-%m}:  ${v:>9,.0f}")
    print("=" * 52)


def pick_today(frames, news=None, top=5):
    score = compute_scores(frames, news)
    last = score.index[-1]
    s = score.loc[last].dropna().sort_values(ascending=False)
    print(f"بناء على إغلاق {last:%Y-%m-%d} (الشراء عند افتتاح الجلسة التالية):")
    if s.empty:
        print("  لا يوجد سهم يجتاز الفلتر اليوم - لا صفقة")
        return
    for k, (t, v) in enumerate(s.head(top).items(), 1):
        mark = "  <== سهم اليوم" if k == 1 else ""
        print(f"  {k}. {t:<6} التقييم {v:+.2f}{mark}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest")
    b.add_argument("--data", required=True)
    b.add_argument("--year", type=int, default=2024)
    b.add_argument("--capital-sar", type=float, default=100000)
    b.add_argument("--target", type=float, default=300, help="هدف الربح اليومي بالدولار")
    b.add_argument("--cost", type=float, default=0.05, help="تكلفة كل جهة %%")
    b.add_argument("--stop", type=float, default=None, help="وقف خسارة %% من سعر الافتتاح")
    b.add_argument("--min-score", type=float, default=0.0)
    b.add_argument("--news", default=None)
    b.add_argument("--compound", action="store_true")
    b.add_argument("--out", default="trades.csv")
    k = sub.add_parser("pick")
    k.add_argument("--data", required=True)
    k.add_argument("--news", default=None)
    a = p.parse_args()

    frames = load_data(a.data)
    news = load_news(a.news)
    if a.cmd == "pick":
        pick_today(frames, news)
        return
    cap = a.capital_sar / SAR_PER_USD
    tr = backtest(frames, a.year, cap, a.cost, a.stop, a.min_score, news, a.compound)
    report(tr, cap, a.target)
    tr.to_csv(a.out, index=False)
    print(f"تفاصيل الصفقات: {a.out}")


if __name__ == "__main__":
    main()
