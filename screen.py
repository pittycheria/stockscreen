#!/usr/bin/env python3
"""
S&P 500 undervalued-stock screen -> static dashboard (index.html).

Runs standalone on a GitHub Actions runner. No API keys.

Pipeline:
  1. Pull current S&P 500 constituents (GitHub 'datasets' mirror of the index).
  2. Pull fundamentals + analyst data for every member via yfinance (threaded),
     plus 13 months of daily prices in one bulk download for momentum.
  3. Score each name on five pillars (sector-relative where it matters):
       Value     35%  winsorized z-scores WITHIN GICS SECTOR on forward
                      earnings yield, EBITDA/EV yield, FCF yield
                      (financials: forward + trailing earnings yield)
       Momentum  20%  12-1 month price momentum, z-scored across the index
       Quality   20%  continuous sector-relative z on ROA, margins,
                      leverage, liquidity (no binary cliff gates)
       Rev. DCF  15%  the 5-year FCF growth the current price IMPLIES at a
                      CAPM-ish discount rate — low implied growth = cheap
                      (skipped for financials / REITs; cash burners floored)
       Upside    10%  consensus price target vs price (weakest signal,
                      deliberately the smallest weight)
     Risk flags deduct 4 points each. Bottom-decile momentum is flagged
     ("Weak momentum") so cheap-and-falling names are marked as knives.
  4. Persist a dated snapshot to data/snapshots/ — the accruing paper track
     record is computed from aged snapshots and shown on the page.
  5. Diff against the previous index.html's embedded state blob.
  6. Write index.html.

Design notes (v2, quant review):
  - Sector-relative scoring removes the structural telecom/banks tilt that
    raw cross-index percentile ranks produce.
  - Trailing P/E and P/B were dropped from the value set: collinear with the
    survivors, and P/B is broken by buyback-driven negative equity.
  - The forward DCF was replaced by a reverse DCF: instead of asserting a
    growth rate and "valuing" the stock, we solve for the growth the market
    is pricing in and score names on how little they are asked to deliver.
  - Analyst target LEVELS are upward-biased and anti-momentum, so their
    weight fell from 20% to 10%; a Hold consensus is no longer a flag
    (sentiment is already priced into the pillar).
"""

import concurrent.futures as cf
import csv
import glob
import io
import json
import math
import os
import re
import sys
import time
import urllib.request
from datetime import datetime
from statistics import mean, pstdev
from zoneinfo import ZoneInfo

import yfinance as yf

CONSTITUENTS_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies"
    "/main/data/constituents.csv"
)
FIN_SECTORS = {"Financials", "Real Estate"}
SECTOR_ABBR = {
    "Communication Services": "Comm. Svcs",
    "Information Technology": "Info Tech",
    "Consumer Discretionary": "Cons. Disc.",
    "Consumer Staples": "Cons. Staples",
    "Health Care": "Health Care",
    "Financials": "Financials",
    "Industrials": "Industrials",
    "Real Estate": "Real Estate",
    "Utilities": "Utilities",
    "Materials": "Materials",
    "Energy": "Energy",
}
CENTRAL = ZoneInfo("America/Chicago")
SHOW_ROWS = 40          # rows rendered in the table
RATINGS_N = 5           # analyst actions listed per stock
MIN_VALUE_METRICS = 2   # value z-components required to rank
MAX_UPSIDE = 1.50       # above this, treat the analyst target as a data error
DOWNSIDE_FLOOR = -0.20  # upside scale runs from this to +0.40
WINSOR = 3.0            # z-scores clipped to +/- this many sigmas
MIN_SECTOR_POOL = 8     # sector z needs this many peers, else universe pool
MOM_SKIP = 21           # skip the most recent month (12-1 convention)
MOM_LOOKBACK = 252      # ~12 trading months
TERMINAL_G = 0.025      # reverse-DCF terminal growth
IMPLIED_G_LO = 0.12     # implied growth >= this scores 0 (expensive)
IMPLIED_G_SPAN = 0.17   # score = (0.12 - g) / 0.17, so g = -5% scores 100
WEIGHTS = {"value": .35, "momentum": .20, "quality": .20,
           "rdcf": .15, "upside": .10}
SNAP_DIR = os.path.join("data", "snapshots")
MAX_WORKERS = 8
OUT = "index.html"


# --------------------------------------------------------------------------
# 1. universe
# --------------------------------------------------------------------------
def constituents():
    req = urllib.request.Request(
        CONSTITUENTS_URL, headers={"User-Agent": "sp500-screen/1.0"}
    )
    txt = urllib.request.urlopen(req, timeout=60).read().decode()
    out = {}
    for r in csv.DictReader(io.StringIO(txt)):
        sym = r["Symbol"].strip().replace(".", "-")   # BRK.B -> BRK-B for Yahoo
        out[sym] = (r["Security"].strip(), r["GICS Sector"].strip())
    return out


# --------------------------------------------------------------------------
# 2. fundamentals
# --------------------------------------------------------------------------
def _pct(v, already_pct=False):
    """Normalize a ratio to a percentage number."""
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if already_pct else v * 100


def fetch_one(sym, attempts=3):
    for i in range(attempts):
        try:
            info = yf.Ticker(sym).info or {}
            if not info.get("marketCap") and not info.get("currentPrice"):
                raise ValueError("empty info")

            price = info.get("currentPrice") or info.get("regularMarketPrice")
            mc = info.get("marketCap")
            fcf = info.get("freeCashflow")

            # Yahoo reports debtToEquity as a percentage (124.0 == 1.24x). A
            # NEGATIVE value means negative book equity, which makes the ratio
            # meaningless — record that fact instead of a nonsense number.
            de = info.get("debtToEquity")
            neg_book = de is not None and de < 0
            de = round(de / 100, 2) if (de is not None and de >= 0) else None

            # Yahoo returns "none" for uncovered names; that must not become the
            # literal string "None" in the consensus column.
            raw_rec = (info.get("recommendationKey") or "").strip().lower()
            rec = {"strong_buy": "Strong Buy", "buy": "Buy", "hold": "Hold",
                   "underperform": "Sell", "sell": "Sell",
                   "strong_sell": "Strong Sell"}.get(raw_rec)

            return {
                "ticker": sym,
                "price": price,
                "market_cap_b": (mc / 1e9) if mc else None,
                "pe": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "pb": info.get("priceToBook"),
                "ev_ebitda": info.get("enterpriseToEbitda"),
                "fcf_ttm_b": (fcf / 1e9) if fcf else None,
                "fcf_yield_pct": (fcf / mc * 100) if (fcf and mc) else None,
                "roe_pct": _pct(info.get("returnOnEquity")),
                "roa_pct": _pct(info.get("returnOnAssets")),
                "gross_margin_pct": _pct(info.get("grossMargins")),
                "debt_equity": de,
                "current_ratio": info.get("currentRatio"),
                "rev_growth_pct": _pct(info.get("revenueGrowth")),
                "net_margin_pct": _pct(info.get("profitMargins")),
                "eps": info.get("trailingEps"),
                "neg_book_hint": neg_book,
                "payout_pct": _pct(info.get("payoutRatio")),
                "target": info.get("targetMeanPrice"),
                "consensus": rec,
                "analyst_count": info.get("numberOfAnalystOpinions"),
                "beta": info.get("beta"),
            }
        except Exception:
            if i == attempts - 1:
                return None
            time.sleep(1.5 * (i + 1))
    return None


ACTION_WORDS = {"up": "Upgrade", "down": "Downgrade", "main": "Maintained",
                "init": "Initiated", "reit": "Reiterated"}


def fetch_ratings(sym, limit=RATINGS_N, attempts=2):
    """Recent individual analyst actions: firm, new grade, date.

    yfinance exposes these as a DataFrame indexed by grade date with
    Firm / ToGrade / FromGrade / Action columns. Absent or malformed data
    is not fatal — the name simply shows no analyst detail.
    """
    for i in range(attempts):
        try:
            df = yf.Ticker(sym).upgrades_downgrades
            if df is None or len(df) == 0:
                return []
            out = []
            # take more than we need, then filter — trimming first would drop
            # a good row for every blank-firm row inside the window
            for idx, row in df.sort_index(ascending=False).head(limit * 3).iterrows():
                if len(out) >= limit:
                    break
                try:
                    date = idx.strftime("%Y-%m-%d")
                except Exception:
                    date = str(idx)[:10]
                firm = str(row.get("Firm") or "").strip()
                if not firm:
                    continue
                act = str(row.get("Action") or "").strip().lower()
                out.append({
                    "firm": firm,
                    "to": str(row.get("ToGrade") or "").strip() or None,
                    "from": str(row.get("FromGrade") or "").strip() or None,
                    "action": ACTION_WORDS.get(act, act.title() or None),
                    "date": date,
                })
            return out
        except Exception:
            if i == attempts - 1:
                return []
            time.sleep(1.0)
    return []


def attach_ratings(rows):
    """Pull analyst action history for the names we actually display."""
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch_ratings, r["ticker"]): r for r in rows}
        for f in cf.as_completed(futs):
            futs[f]["ratings"] = f.result() or []
    return rows


def fetch_all(symbols):
    rows, failed = [], []
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch_one, s): s for s in symbols}
        for n, f in enumerate(cf.as_completed(futs), 1):
            sym = futs[f]
            r = f.result()
            (rows if r else failed).append(r if r else sym)
            if n % 50 == 0:
                print(f"  fetched {n}/{len(symbols)}", flush=True)
    return rows, failed


def fetch_momentum(symbols):
    """12-1 month price momentum from one bulk daily-price download.

    Returns {symbol: pct} for names with enough history; anything missing
    (recent index adds, download hiccups) simply has no momentum score and
    the composite renormalizes without the pillar. A total failure returns
    {} and the page notes that momentum was unavailable this run.
    """
    try:
        px = yf.download(symbols, period="400d", interval="1d",
                         auto_adjust=True, progress=False, threads=True)["Close"]
    except Exception as e:
        print(f"  momentum download failed: {e}", file=sys.stderr, flush=True)
        return {}
    out = {}
    for s in symbols:
        try:
            ser = px[s].dropna() if hasattr(px, "columns") else px.dropna()
        except Exception:
            continue
        if len(ser) < MOM_LOOKBACK - 20:      # tolerate a few missing days
            continue
        try:
            p_then = float(ser.iloc[-min(MOM_LOOKBACK, len(ser))])
            p_skip = float(ser.iloc[-MOM_SKIP])
            if p_then > 0:
                out[s] = (p_skip / p_then - 1) * 100
        except Exception:
            continue
    return out


# --------------------------------------------------------------------------
# 3. scoring
# --------------------------------------------------------------------------
def winsorized_z(pool, v, w=WINSOR):
    """z of v against pool (non-None values), clipped to +/- w sigmas."""
    xs = [x for x in pool if x is not None]
    if v is None or len(xs) < 4:
        return None
    sd = pstdev(xs)
    if sd == 0:
        return 0.0
    return max(-w, min(w, (v - mean(xs)) / sd))


def z_to_100(z):
    """Map a z-score to 0-100 via the normal CDF (keeps display continuity)."""
    if z is None:
        return None
    return round(50 * (1 + math.erf(z / math.sqrt(2))), 1)


def implied_growth(r):
    """Reverse DCF: the constant 5-year FCF growth rate that makes
    PV(FCF, disc) equal today's market cap (terminal growth 2.5%).

    Returns (g, applicable). Financials/REITs: (None, False) — pillar dropped
    and renormalized. Cash burners: (None, True) — pillar scored at the floor,
    because dropping it would hand a burner the average of its other pillars.
    """
    if r["sector"] in FIN_SECTORS:
        return None, False
    fcf, mc = r.get("fcf_ttm_b"), r.get("market_cap_b")
    if not mc:
        return None, False
    if not fcf or fcf <= 0:
        return None, True
    beta = r.get("beta") if r.get("beta") is not None else 1.0
    disc = min(max(0.043 + 0.045 * beta, 0.075), 0.12)

    def pv(g):
        f, s = fcf, 0.0
        for yr in range(1, 6):
            f *= 1 + g
            s += f / (1 + disc) ** yr
        s += f * (1 + TERMINAL_G) / (disc - TERMINAL_G) / (1 + disc) ** 5
        return s

    lo, hi = -0.90, 1.00           # pv is monotonically increasing in g
    if pv(lo) >= mc:
        return lo, True
    if pv(hi) <= mc:
        return hi, True
    for _ in range(60):
        mid = (lo + hi) / 2
        if pv(mid) < mc:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, True


def rdcf_score_from_g(g):
    """Low implied growth = cheap. g = -5% -> 100, g = +12% -> 0, linear."""
    return round(min(max((IMPLIED_G_LO - g) / IMPLIED_G_SPAN, 0), 1) * 100, 1)


def flags(r, weak_mom_cutoff):
    """Short labels — the table explains them once in a legend below it.

    'Unprofitable' is a factual claim published on a public page, so it needs
    corroboration: a missing trailing P/E alone can mean a recent spin-off or
    a partial Yahoo response, not a loss. We require negative earnings or a
    negative margin, and say 'No earnings data' when we simply don't know.

    A Hold consensus is deliberately NOT a flag any more — analyst sentiment
    already lives in the (down-weighted) upside pillar, and flagging it was
    double-charging. Outright Sell calls are rare enough to stay newsworthy.
    """
    fl = []
    eps, margin, pe = r.get("eps"), r.get("net_margin_pct"), r.get("pe")
    loss = (eps is not None and eps < 0) or (margin is not None and margin < 0)
    if loss:
        fl.append("Unprofitable")
    elif pe is None and eps is None and margin is None:
        fl.append("No earnings data")
    if (r.get("payout_pct") or 0) > 90:
        fl.append("Payout >90%")
    de = r.get("debt_equity")
    if de is not None and de > 2.5 and r["sector"] not in FIN_SECTORS:
        fl.append(f"Leverage {de:.1f}x")
    if r.get("consensus") in ("Sell", "Strong Sell"):
        fl.append(f"{r['consensus']} consensus")
    if (r.get("analyst_count") or 0) < 5:
        fl.append("Thin coverage")
    if r.get("neg_equity"):
        fl.append("Negative equity")
    mom = r.get("mom_12_1_pct")
    if mom is not None and weak_mom_cutoff is not None and mom <= weak_mom_cutoff:
        fl.append("Weak momentum")
    if r.get("sparse"):
        fl.append("Sparse data")
    if r.get("target_suspect"):
        fl.append("Target looks stale")
    return fl


def score(rows):
    # -- derived per-name metrics (yields, so higher = better everywhere) ----
    for r in rows:
        pb = r.get("pb")
        r["neg_equity"] = bool(r.get("neg_book_hint")) or (pb is not None and pb < 0)
        fpe, tpe, ee = r.get("forward_pe"), r.get("pe"), r.get("ev_ebitda")
        r["_fey"] = 100 / fpe if (fpe and fpe > 0) else None   # fwd earnings yield
        r["_tey"] = 100 / tpe if (tpe and tpe > 0) else None   # trailing earnings yield
        r["_eby"] = 100 / ee if (ee and ee > 0) else None      # EBITDA/EV yield
        r["_fcfy"] = r.get("fcf_yield_pct")                    # may be negative: real info
        r["_negde"] = -r["debt_equity"] if r.get("debt_equity") is not None else None

    # -- sector pools with universe fallback ---------------------------------
    VAL_KEYS = ("_fey", "_tey", "_eby", "_fcfy")
    QUAL_KEYS = ("roa_pct", "net_margin_pct", "gross_margin_pct",
                 "_negde", "current_ratio")
    sectors = {}
    for r in rows:
        sectors.setdefault(r["sector"], []).append(r)
    uni_pool = {k: [r.get(k) for r in rows] for k in VAL_KEYS + QUAL_KEYS}
    sec_pool = {s: {k: [r.get(k) for r in members] for k in VAL_KEYS + QUAL_KEYS}
                for s, members in sectors.items()}

    def rel_z(r, key):
        pool = sec_pool[r["sector"]][key]
        if sum(1 for x in pool if x is not None) < MIN_SECTOR_POOL:
            pool = uni_pool[key]
        return winsorized_z(pool, r.get(key))

    # -- momentum pool (cross-index, not sector-relative) --------------------
    moms = sorted(r["mom_12_1_pct"] for r in rows
                  if r.get("mom_12_1_pct") is not None)
    weak_cutoff = moms[max(0, len(moms) // 10 - 1)] if len(moms) >= 20 else None

    for r in rows:
        # Value: sector-relative z on the yield set (financials use earnings
        # yields only — EV/EBITDA and FCF are not meaningful for banks).
        keys = ("_fey", "_tey") if r["sector"] in FIN_SECTORS \
            else ("_fey", "_eby", "_fcfy")
        zs = [z for z in (rel_z(r, k) for k in keys) if z is not None]
        r["n_value_metrics"] = len(zs)
        r["value_score"] = z_to_100(mean(zs)) if len(zs) >= MIN_VALUE_METRICS else None

        # Momentum: z across the whole index.
        mz = winsorized_z(moms, r.get("mom_12_1_pct"))
        r["momentum_score"] = z_to_100(mz)

        # Reverse DCF.
        g, applicable = implied_growth(r)
        r["implied_g_pct"] = round(g * 100, 1) if g is not None else None
        if not applicable:
            r["rdcf_score"] = None
        elif g is None:                       # cash burner: floor, not skip
            r["rdcf_score"] = 0.0
        else:
            r["rdcf_score"] = rdcf_score_from_g(g)

        # Upside runs from -20% (score 0) to +40% (score 100), so a stock
        # trading ABOVE its consensus target is genuinely penalised rather
        # than tying with one that merely has no upside.
        up = (r["target"] / r["price"] - 1) if (r.get("target") and r.get("price")) else None
        r["target_suspect"] = up is not None and up > MAX_UPSIDE
        if r["target_suspect"]:
            up = None                      # stale/unadjusted target: don't score it
        r["upside_pct"] = round(up * 100, 1) if up is not None else None
        r["upside_score"] = None if up is None else round(
            min(max((up - DOWNSIDE_FLOOR) / 0.60, 0), 1) * 100, 1)

        # Quality: continuous sector-relative z — no cliff gates. ROA instead
        # of ROE so buyback-driven negative equity is scored, not excluded.
        qkeys = ("roa_pct", "net_margin_pct") if r["sector"] in FIN_SECTORS \
            else QUAL_KEYS
        qz = [z for z in (rel_z(r, k) for k in qkeys) if z is not None]
        r["n_quality_checks"] = len(qz)
        r["quality_score"] = z_to_100(mean(qz)) if len(qz) >= 2 else None

        r["sparse"] = (r["n_value_metrics"] < MIN_VALUE_METRICS) \
            or (r["n_quality_checks"] < 2)
        r["flags"] = flags(r, weak_cutoff)

        parts = [(r["value_score"], WEIGHTS["value"]),
                 (r["momentum_score"], WEIGHTS["momentum"]),
                 (r["quality_score"], WEIGHTS["quality"]),
                 (r["rdcf_score"], WEIGHTS["rdcf"]),
                 (r["upside_score"], WEIGHTS["upside"])]
        parts = [(s, w) for s, w in parts if s is not None]
        wsum = sum(w for _, w in parts)
        comp = (sum(s * w for s, w in parts) / wsum if wsum else 0) - 4 * len(r["flags"])
        r["composite"] = round(max(comp, 0), 1)

    # A name must carry enough evidence to be ranked at all. Without this a
    # stock with one cheap-looking ratio and nothing else scores near 100 on
    # the strength of the single fact we happen to know about it.
    ranked = [r for r in rows
              if r.get("price")
              and r["value_score"] is not None
              and r["quality_score"] is not None]
    dropped = len(rows) - len(ranked)
    ranked.sort(key=lambda r: -r["composite"])
    return ranked, dropped


# --------------------------------------------------------------------------
# 4. snapshots & paper track record
# --------------------------------------------------------------------------
def write_snapshot(rows, today_iso):
    os.makedirs(SNAP_DIR, exist_ok=True)
    path = os.path.join(SNAP_DIR, f"{today_iso}.json")
    payload = [{"ticker": r["ticker"], "rank": i + 1,
                "composite": r["composite"], "price": r.get("price"),
                "sector": r["sector"]}
               for i, r in enumerate(rows)]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"date": today_iso, "rows": payload}, fh)
    return path


def track_record(rows, today):
    """Compare each aged snapshot's top-40 basket against the equal-weight
    screened universe, marked to today's prices. Buckets: ~1m (21-45 days),
    ~3m (63-120 days). Returns list of (label, n_snaps, basket, universe)."""
    px = {r["ticker"]: r["price"] for r in rows if r.get("price")}
    buckets = {"≈1 month": (21, 45, []), "≈3 months": (63, 120, [])}
    for path in glob.glob(os.path.join(SNAP_DIR, "*.json")):
        try:
            snap = json.load(open(path, encoding="utf-8"))
            d = datetime.strptime(snap["date"], "%Y-%m-%d").date()
        except Exception:
            continue
        age = (today - d).days
        rets = [(s["ticker"], px[s["ticker"]] / s["price"] - 1)
                for s in snap.get("rows", [])
                if s.get("price") and px.get(s["ticker"])]
        if len(rets) < 100:
            continue
        top = [ret for i, (t, ret) in enumerate(rets) if i < SHOW_ROWS]
        for label, (lo, hi, acc) in buckets.items():
            if lo <= age <= hi:
                acc.append((mean(top) * 100, mean(x for _, x in rets) * 100))
    out = []
    for label, (lo, hi, acc) in buckets.items():
        if acc:
            out.append((label, len(acc),
                        mean(a for a, _ in acc), mean(b for _, b in acc)))
    return out


# --------------------------------------------------------------------------
# 5. previous state
# --------------------------------------------------------------------------
def previous_state(path=OUT):
    try:
        html = open(path, encoding="utf-8").read()
    except OSError:
        return None
    m = re.search(
        r'<script type="application/json" id="screen-state">(.*?)</script>',
        html, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------
# 6. render
# --------------------------------------------------------------------------
def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def signed(v, suf="%", nd=1):
    if v is None:
        return "—"
    return f"{'+' if v >= 0 else '−'}{abs(v):.{nd}f}{suf}"


def num(v, pre="", suf="", nd=2):
    if v is None:
        return "—"
    return f"{pre}{v:,.{nd}f}{suf}"


def nice_date(iso):
    """'2026-08-04' -> 'Aug 4, 2026'."""
    if not iso:
        return "—"
    try:
        return datetime.strptime(iso[:10], "%Y-%m-%d").strftime("%b %-d, %Y")
    except ValueError:
        return iso


def latest_rating(r):
    rs = r.get("ratings") or []
    return rs[0] if rs else None


def rating_tooltip(r):
    """Full recent history, shown on hover over the analyst cell."""
    rs = r.get("ratings") or []
    if not rs:
        return ""
    lines = []
    for a in rs:
        move = (f"{a['from']} → {a['to']}" if a.get("from") and a.get("to")
                and a["from"] != a["to"] else (a.get("to") or ""))
        lines.append(f"{nice_date(a['date'])} · {a['firm']}: "
                     f"{a.get('action') or ''} {move}".strip())
    return esc("\n".join(lines))


def changes_block(rows, prev):
    if not prev or not prev.get("rows"):
        return ("<section class='card'><h2>What changed</h2><p class='quiet'>"
                "First edition of the v2 methodology — new entrants, rank moves, "
                "rating changes and price moves will appear here from the next "
                "refresh onward.</p></section>")

    p = {r["ticker"]: r for r in prev["rows"]}
    shown = rows[:SHOW_ROWS]
    cur = {r["ticker"]: dict(r, rank=i + 1) for i, r in enumerate(shown)}
    when = (prev.get("generated_at") or "")[:10]
    items = []
    if prev.get("methodology") != "v2":
        items.append("Methodology upgraded to v2 (sector-relative value, momentum "
                     "pillar, reverse DCF, continuous quality) — scores and ranks "
                     "are NOT comparable with the previous edition, so today's "
                     "moves reflect the new lens, not new information.")

    entered = [t for t in cur if t not in p]
    left = [t for t in p if t not in cur]
    if entered:
        items.append("Entered the top {}: ".format(SHOW_ROWS) + ", ".join(
            f"{t} at #{cur[t]['rank']} (score {cur[t]['composite']})"
            for t in sorted(entered, key=lambda x: cur[x]["rank"])))
    if left:
        items.append("Dropped out: " + ", ".join(sorted(left)))

    movers = []
    for t, c in cur.items():
        if t in p:
            dr = p[t]["rank"] - c["rank"]
            ds = round(c["composite"] - (p[t].get("composite") or 0), 1)
            if abs(dr) >= 2 or abs(ds) >= 3:
                if dr == 0:
                    msg = (f"{t}: score {signed(ds, '', 1)} to {c['composite']} "
                           f"(rank unchanged at #{c['rank']})")
                else:
                    word = "▲ up" if dr > 0 else "▼ down"
                    msg = (f"{t}: {word} {abs(dr)} place{'s' if abs(dr) != 1 else ''} "
                           f"to #{c['rank']} (score {signed(ds, '', 1)} to {c['composite']})")
                movers.append((abs(dr), abs(ds), msg))
    for _, _, m in sorted(movers, key=lambda x: (-x[0], -x[1]))[:8]:
        items.append(m)

    for t, c in cur.items():
        if t in p and p[t].get("consensus") and c.get("consensus") \
                and p[t]["consensus"] != c["consensus"]:
            items.append(f"{t}: analyst consensus {p[t]['consensus']} → {c['consensus']}")
    # fresh individual analyst actions since the last refresh
    new_calls = []
    for t, c in cur.items():
        a = latest_rating(c)          # current rows carry 'ratings'; state carries 'latest_rating'
        if not a:
            continue
        old = (p.get(t) or {}).get("latest_rating") or {}
        if (a.get("date"), a.get("firm"), a.get("to")) != \
           (old.get("date"), old.get("firm"), old.get("to")):
            phrase = {
                "Upgrade": ("upgraded", "to"),
                "Downgrade": ("downgraded", "to"),
                "Initiated": ("initiated coverage of", "at"),
                "Reiterated": ("reiterated", "at"),
                "Maintained": ("maintained", "at"),
            }.get(a.get("action"), ("rated", ""))
            verb, prep = phrase
            grade = f" {prep} {a['to']}".rstrip() if a.get("to") else ""
            new_calls.append((a.get("date") or "",
                              f"{a['firm']} {verb} {t}{grade} "
                              f"({nice_date(a.get('date'))})"))
    for _, m in sorted(new_calls, reverse=True)[:6]:
        items.append(m)

    price_moves = []
    for t, c in cur.items():
        if t in p and p[t].get("price") and c.get("price"):
            mv = (c["price"] / p[t]["price"] - 1) * 100
            if abs(mv) >= 3:
                price_moves.append((abs(mv), f"{t}: price {'▲ up' if mv > 0 else '▼ down'} "
                                             f"{abs(mv):.1f}% to ${c['price']:,.2f}"))
    for _, m in sorted(price_moves, reverse=True)[:6]:
        items.append(m)

    if not items:
        body = ("<p class='quiet'>No material changes since the previous refresh — "
                "same names in the same order, scores within a few points.</p>")
    else:
        body = "<ul>" + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ul>"
    return f"<section class='card'><h2>What changed since {esc(when)}</h2>{body}</section>"


def explainer_block():
    """Collapsible plain-language methodology explainer. Pure HTML
    <details>/<summary> — no JavaScript, so it cannot break a static page."""
    w = {k: f"{v:.0%}" for k, v in WEIGHTS.items()}
    return f"""<details class="card explainer">
<summary>How the score is calculated — plain-language explainer</summary>
<div class="exp">
<p>Every stock gets a score from 0 to 100. It is a weighted blend of five
"pillars," each of which also runs 0–100. Think of each pillar as answering one
question:</p>
<table class="mini exptbl">
<thead><tr><th>Pillar</th><th>Weight</th><th>The question it answers</th></tr></thead>
<tbody>
<tr><td><strong>Value</strong></td><td class="num">{w['value']}</td>
<td>Is this stock cheap <em>compared to companies in its own sector</em>? We look at
three "yields" — earnings expected next year, cash operating profits (EBITDA), and free
cash flow, each divided by what you pay. Higher yield = more business per dollar.
A bank is only ever compared with banks, a software company with software companies —
otherwise the list would just fill up with industries that always look cheap.</td></tr>
<tr><td><strong>Momentum</strong></td><td class="num">{w['momentum']}</td>
<td>Is the market warming up to the stock or giving up on it? We measure the share-price
move over the past 12 months, skipping the most recent month (a standard convention).
This matters in a value screen because the classic mistake is buying something cheap
that keeps getting cheaper — the "falling knife."</td></tr>
<tr><td><strong>Quality</strong></td><td class="num">{w['quality']}</td>
<td>Is this a good business or a cheap-for-a-reason business? Profitability (return on
assets, profit margins), balance-sheet strength (debt load, ability to pay near-term
bills), all measured against sector peers on a sliding scale — no arbitrary pass/fail
lines.</td></tr>
<tr><td><strong>Reverse DCF</strong></td><td class="num">{w['rdcf']}</td>
<td>How much growth is already baked into the price? Instead of guessing what a company
is "worth," we solve the opposite problem: what yearly free-cash-flow growth over the
next five years would justify today's price? If the answer is "shrink 5% a year," the
bar is on the floor — that scores well. If the answer is "grow 12% a year," a lot has
to go right — that scores poorly. (Not applicable to banks and REITs, whose cash flows
don't work this way.)</td></tr>
<tr><td><strong>Analyst upside</strong></td><td class="num">{w['upside']}</td>
<td>What do Wall Street analysts think it's worth? The gap between the average analyst
price target and today's price. This gets the <em>smallest</em> weight on purpose:
targets are known to be optimistic on average and slow to change.</td></tr>
</tbody></table>
<p><strong>Reading a pillar score.</strong> Pillar scores are percentile-like: 50 means
"typical for its comparison group," 84 means roughly one standard deviation better than
typical, 16 one worse. Extreme outliers are capped so a single weird data point can't
dominate.</p>
<p><strong>Worked example.</strong> Say a stock scores Value 80, Momentum 55, Quality 70,
Reverse&nbsp;DCF 90, Upside 40. The blend is
80×{w['value']} + 55×{w['momentum']} + 70×{w['quality']} + 90×{w['rdcf']} + 40×{w['upside']}
= <strong>72.5</strong>. If it also carries one risk flag (say, leverage), 4 points come
off: final score <strong>68.5</strong>. When a pillar can't be computed (a bank has no
reverse DCF), the remaining pillars are re-weighted proportionally rather than the stock
being punished for it.</p>
<p><strong>Risk flags</strong> are simple warnings — heavy debt, a dividend eating nearly
all earnings, negative earnings, bottom-decile momentum, very few analysts covering, and
so on. Each one subtracts 4 points and is spelled out in the Flags column, so you can see
<em>why</em> a score was docked, not just that it was.</p>
<p><strong>What the score is not.</strong> It is not a price target, a prediction, or
investment advice. It is a ranking device: a high score means "cheap versus its sector,
decent business, market not fleeing, low expectations bar" — which is a good place to
<em>start research</em>, not to end it. Stocks can be cheap for excellent reasons the
data can't see (lawsuits, disruption, management). The paper-track-record box above
tracks how the screen's own picks have actually done, so the methodology is graded in
public.</p>
</div></details>"""


def sector_block(rows, universe_counts):
    """Explicit disclosure of the top-40 sector tilt vs the index."""
    shown = rows[:SHOW_ROWS]
    counts = {}
    for r in shown:
        counts[r["sector"]] = counts.get(r["sector"], 0) + 1
    total_uni = sum(universe_counts.values()) or 1
    cells = []
    for sec, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        share = n / len(shown) * 100
        idx_share = universe_counts.get(sec, 0) / total_uni * 100
        tilt = share - idx_share
        cells.append(f"<tr><td>{esc(SECTOR_ABBR.get(sec, sec))}</td>"
                     f"<td class='num'>{n}</td>"
                     f"<td class='num'>{share:.0f}%</td>"
                     f"<td class='num'>{idx_share:.0f}%</td>"
                     f"<td class='num'>{signed(tilt, '%', 0)}</td></tr>")
    return f"""<section class="card"><h2>Where the top {SHOW_ROWS} concentrates</h2>
<p class="quiet">Scoring is sector-relative, but cheapness still clusters — this table makes
the residual bet explicit instead of hiding it inside the ranking.</p>
<div class="tblwrap"><table class="mini">
<thead><tr><th>Sector</th><th># in top {SHOW_ROWS}</th><th>Share</th>
<th>Index share</th><th>Tilt</th></tr></thead>
<tbody>{''.join(cells)}</tbody></table></div></section>"""


def track_block(perf, n_snaps):
    if not perf:
        return (f"<section class='card'><h2>Paper track record</h2><p class='quiet'>"
                f"Accruing — {n_snaps} daily snapshot(s) stored so far. The first "
                f"1-month readout (top-{SHOW_ROWS} basket vs the equal-weight screened "
                f"universe, marked to current prices) appears once a snapshot is at "
                f"least 21 days old. This is the screen grading its own homework; "
                f"judge the methodology by this section, not by the scores.</p></section>")
    rows = "".join(
        f"<tr><td>{esc(label)}</td><td class='num'>{n}</td>"
        f"<td class='num'>{signed(b)}</td><td class='num'>{signed(u)}</td>"
        f"<td class='num'>{signed(b - u)}</td></tr>"
        for label, n, b, u in perf)
    return f"""<section class="card"><h2>Paper track record</h2>
<p class="quiet">Average forward return of each aged snapshot's top-{SHOW_ROWS} basket vs the
equal-weight screened universe, marked to current prices. Short windows are noise —
read nothing into this until months of history exist.</p>
<div class="tblwrap"><table class="mini">
<thead><tr><th>Horizon</th><th>Snapshots</th><th>Top-{SHOW_ROWS} basket</th>
<th>Universe</th><th>Excess</th></tr></thead>
<tbody>{rows}</tbody></table></div></section>"""


def analyst_cells(r):
    """Three aligned cells listing this stock's last few analyst actions.

    Each cell holds one single-line div per action, so the firm, its grade and
    the date line up row-for-row across the three columns. Long firm names are
    ellipsized rather than wrapped — wrapping would break that alignment — with
    the full name on hover.
    """
    rs = (r.get("ratings") or [])[:RATINGS_N]
    if not rs:
        return ("<td class='firm'>—</td><td class='grade'>—</td>"
                "<td class='num rdate'>—</td>")

    firms, grades, dates = [], [], []
    for a in rs:
        firms.append(f"<div class='ln' title=\"{esc(a['firm'])}\">"
                     f"{esc(a['firm'])}</div>")
        g = esc(a.get("to") or "—")
        act = a.get("action")
        if act in ("Upgrade", "Downgrade"):
            g += f" <span class='act'>({act.lower()})</span>"
        grades.append(f"<div class='ln'>{g}</div>")
        dates.append(f"<div class='ln'>{nice_date(a.get('date'))}</div>")

    return (f"<td class='firm'>{''.join(firms)}</td>"
            f"<td class='grade'>{''.join(grades)}</td>"
            f"<td class='num rdate'>{''.join(dates)}</td>")


def render(rows, prev, screened, failed, dropped=0, universe_counts=None,
           perf=None, n_snaps=0, mom_ok=True):
    now = datetime.now(CENTRAL)
    stamp = now.strftime("%A, %B %-d, %Y at %-I:%M %p Central")
    shown = rows[:SHOW_ROWS]

    state = {
        "generated_at": now.isoformat(),
        "methodology": "v2",
        "rows": [{"ticker": r["ticker"], "rank": i + 1, "composite": r["composite"],
                  "price": r.get("price"), "target": r.get("target"),
                  "consensus": r.get("consensus"), "upside_pct": r.get("upside_pct"),
                  "latest_rating": latest_rating(r)}
                 for i, r in enumerate(shown)],
    }

    n = len(shown)
    showing = (f"Showing the top {n} of {screened} screened."
               if screened > n else f"Showing all {screened} screened.")
    top_moms = [r["mom_12_1_pct"] for r in shown if r.get("mom_12_1_pct") is not None]
    top_moms.sort()
    med_mom = top_moms[len(top_moms) // 2] if top_moms else None
    tiles = f"""
<div class="tiles">
  <div class="tile"><div class="tv">{esc(shown[0]['ticker'])}</div>
    <div class="tl">Top ranked · score {shown[0]['composite']}</div></div>
  <div class="tile"><div class="tv">{screened}</div>
    <div class="tl">Index members ranked</div></div>
  <div class="tile"><div class="tv">{signed(med_mom, nd=1) if med_mom is not None else '—'}</div>
    <div class="tl">Median 12-1 momentum (top {n})</div></div>
  <div class="tile"><div class="tv">{sum(1 for r in shown if not r['flags'])}</div>
    <div class="tl">Ranked names with no flags</div></div>
</div>"""

    trs = []
    for i, r in enumerate(shown, 1):
        g = r.get("implied_g_pct")
        if r.get("rdcf_score") is None:
            gtxt = "n/a"
        elif g is None:
            gtxt = "cash burn"
        else:
            gtxt = signed(g, nd=0)
        link = (f"<a href='https://stockanalysis.com/stocks/{r['ticker'].lower()}/'"
                f" target='_blank' rel='noopener'>{esc(r['ticker'])}</a>")
        qual = ("—" if r.get("quality_score") is None
                else f"{r['quality_score']:.0f}")
        trs.append(
            f"<tr><td class='num'>{i}</td><td>{link}</td>"
            f"<td class='co'>{esc(r['name'])}</td>"
            f"<td class='sec'>{esc(SECTOR_ABBR.get(r['sector'], r['sector']))}</td>"
            f"<td class='num'>{num(r.get('price'), '$')}</td>"
            f"<td class='num'>{num(r.get('forward_pe'), nd=1)}</td>"
            f"<td class='num'>{num(r.get('fcf_yield_pct'), suf='%', nd=1)}</td>"
            f"<td class='num'>{signed(r.get('mom_12_1_pct'), nd=0)}</td>"
            f"<td class='num'>{esc(r.get('consensus') or '—')}</td>"
            f"<td class='num'>{num(r.get('target'), '$')}</td>"
            f"<td class='num'>{signed(r.get('upside_pct'))}</td>"
            f"<td class='num'>{gtxt}</td>"
            f"<td class='num'>{qual}</td>"
            f"<td class='num score'>{r['composite']}</td>"
            f"{analyst_cells(r)}"
            f"<td class='flags'>{esc('; '.join(r['flags']) or '—')}</td></tr>")

    miss = (f" {len(failed)} member(s) returned no usable data." if failed else "")
    if dropped:
        miss += (f" {dropped} more were excluded from the ranking for having too "
                 f"few value or quality inputs to judge.")
    if not mom_ok:
        miss += (" Momentum data was unavailable this run — the pillar was "
                 "renormalized away and no Weak-momentum flags were applied.")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>S&amp;P 500 Undervalued Stock Screen</title>
<style>
:root {{ color-scheme: light dark; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  background:#f9f9f7; color:#0b0b0b; }}
.wrap {{ max-width:1500px; margin:0 auto; padding:28px 20px 48px; }}
h1 {{ font-size:22px; margin:0 0 4px; }}
h2 {{ font-size:15px; margin:0 0 10px; }}
.sub {{ color:#52514e; font-size:13px; margin:0; line-height:1.5; }}
.stamp {{ display:inline-block; font-size:13px; font-weight:600; padding:5px 12px;
  border:1px solid rgba(11,11,11,.10); border-radius:999px; background:#fcfcfb;
  margin:12px 0 18px; }}
.card {{ background:#fcfcfb; border:1px solid rgba(11,11,11,.10); border-radius:10px;
  padding:16px 18px; margin:0 0 18px; }}
.card ul {{ margin:6px 0 2px; padding-left:20px; }}
.card li {{ font-size:13.5px; margin:4px 0; }}
details.explainer {{ padding:0; }}
details.explainer summary {{ cursor:pointer; font-size:14px; font-weight:650;
  padding:14px 18px; list-style:none; }}
details.explainer summary::before {{ content:"▸ "; color:#898781; }}
details.explainer[open] summary::before {{ content:"▾ "; }}
details.explainer summary::-webkit-details-marker {{ display:none; }}
details.explainer summary:hover {{ background:#f0efec; border-radius:10px; }}
details.explainer[open] summary:hover {{ border-radius:10px 10px 0 0; }}
.exp {{ padding:2px 18px 16px; }}
.exp p {{ font-size:13.5px; line-height:1.6; margin:10px 0; }}
.exptbl td {{ white-space:normal; line-height:1.55; }}
.exptbl td:first-child {{ white-space:nowrap; }}
.exptbl {{ margin:6px 0; }}
.quiet {{ color:#52514e; font-size:13.5px; margin:4px 0; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
  gap:12px; margin:0 0 18px; }}
.tile {{ background:#fcfcfb; border:1px solid rgba(11,11,11,.10);
  border-radius:10px; padding:14px 16px; }}
.tv {{ font-size:24px; font-weight:650; }}
.tl {{ font-size:12px; color:#52514e; margin-top:2px; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
@media (max-width:900px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
.grid2 .card {{ margin:0 0 18px; }}
.tblwrap {{ overflow-x:auto; }}
table {{ width:100%; border-collapse:collapse; font-size:12.5px; }}
table.mini {{ width:auto; min-width:340px; }}
table.mini td, table.mini th {{ padding-right:22px; }}
th {{ text-align:left; font-size:10.5px; text-transform:uppercase; letter-spacing:.03em;
  color:#898781; font-weight:600; padding:8px 6px; border-bottom:1px solid #c3c2b7;
  white-space:nowrap; }}
td {{ padding:7px 6px; border-bottom:1px solid #e1e0d9; vertical-align:top; }}
td.num {{ font-variant-numeric: tabular-nums; white-space:nowrap; }}
td.score {{ font-weight:700; }}
td.co, td.sec {{ color:#52514e; }}
td.sec {{ white-space:nowrap; }}
td.flags {{ color:#52514e; font-size:11.5px; min-width:105px; max-width:150px;
  white-space:normal; overflow-wrap:break-word; }}
td.co {{ max-width:145px; }}
td.firm {{ width:130px; max-width:130px; }}
td.grade, td.rdate {{ white-space:nowrap; }}
/* one action per line; identical line boxes keep the three columns aligned */
.ln {{ white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
  line-height:1.65; }}
.ln + .ln {{ border-top:1px dotted rgba(11,11,11,.07); }}
td.firm .ln {{ cursor:help; }}
.act {{ color:#898781; }}
.legend {{ color:#52514e; font-size:12px; margin:12px 2px 0; line-height:1.6; }}
tbody tr:hover {{ background:#f0efec; }}
a {{ color:#2a78d6; text-decoration:none; }}
a:hover {{ text-decoration:underline; }}
footer {{ color:#52514e; font-size:12px; line-height:1.6; margin-top:22px; }}
@media (prefers-color-scheme: dark) {{
  body {{ background:#0d0d0d; color:#fff; }}
  .card, .tile, .stamp {{ background:#1a1a19; border-color:rgba(255,255,255,.10); }}
  th {{ color:#898781; border-color:#2c2c2a; }}
  td {{ border-color:#2c2c2a; }}
  .sub, .quiet, .tl, td.co, td.sec, td.flags, footer {{ color:#c3c2b7; }}
  .ln + .ln {{ border-color:rgba(255,255,255,.08); }}
  tbody tr:hover {{ background:#222220; }}
  details.explainer summary:hover {{ background:#222220; }}
  a {{ color:#3987e5; }}
}}
</style></head><body><div class="wrap">
<h1>S&amp;P 500 Undervalued Stock Screen</h1>
<p class="sub">Every current index member scored on five pillars — <strong>value 35%</strong>
(forward earnings yield, EBITDA/EV yield and FCF yield, winsorized z-scores <em>within GICS
sector</em>, so a bank is judged against banks, not against software) ·
<strong>momentum 20%</strong> (12-1 month price momentum across the index — cheapness without
a falling knife) · <strong>quality 20%</strong> (continuous sector-relative scoring of ROA,
margins, leverage and liquidity — no pass/fail cliffs) · <strong>reverse DCF 15%</strong>
(the 5-year FCF growth today's price implies at a beta-scaled discount rate; the less a stock
needs to deliver, the better it scores; not applicable to financials/REITs) ·
<strong>analyst upside 10%</strong> (deliberately the smallest weight — target levels are
upward-biased and lag prices). Each risk flag deducts 4 points; bottom-decile momentum is
itself a flag. {showing}</p>
<div class="stamp">Last refreshed: {stamp}</div>
{tiles}
{explainer_block()}
{changes_block(rows, prev)}
<div class="grid2">
{sector_block(rows, universe_counts or {})}
{track_block(perf, n_snaps)}
</div>
<section class="card"><h2>Ranked candidates</h2><div class="tblwrap"><table>
<thead><tr><th>#</th><th>Ticker</th><th>Company</th><th>Sector</th><th>Price</th>
<th>Fwd P/E</th><th>FCF Yld</th><th>12-1 Mom</th><th>Consensus</th><th>Target</th>
<th>Upside</th><th>Implied 5y gr.</th><th>Qual.</th><th>Score</th>
<th>Analyst (last {RATINGS_N})</th><th>Rating</th><th>Rated</th><th>Flags</th></tr></thead>
<tbody>
{chr(10).join(trs)}
</tbody></table></div>
<p class="legend"><strong>Implied 5y gr.</strong> — the reverse-DCF output: the annual FCF
growth over five years needed to justify today's price. Negative means the market is pricing
FCF decline; "cash burn" means trailing FCF is negative and the pillar is floored.<br>
<strong>Flags</strong> (each deducts 4 points): <em>Sell consensus</em> — the street is
outright negative · <em>Payout &gt;90%</em> — dividend consumes nearly all earnings ·
<em>Leverage</em> — debt/equity above 2.5x · <em>Unprofitable</em> — negative trailing
earnings or margin · <em>No earnings data</em> — profitability could not be determined ·
<em>Negative equity</em> — book-value metrics unreliable (usually buyback-driven; quality is
scored on ROA so these names are ranked, not excluded) · <em>Weak momentum</em> —
bottom-decile 12-1 momentum, the classic value-trap marker · <em>Thin coverage</em> — fewer
than five analysts · <em>Sparse data</em> — thin inputs, treat the score as low confidence ·
<em>Target looks stale</em> — implied upside above {MAX_UPSIDE:.0%}, usually an unrevised
target, so the upside pillar was not scored.<br>
<strong>Analyst / Rating / Rated</strong> — the last {RATINGS_N} individual rating actions on
record, newest first: the covering firm, the grade it assigned, and the date. These are
individual firms' calls, not the consensus — the Consensus column is the aggregate view, and
the two often disagree. Coverage depth varies, so some names list fewer than {RATINGS_N}.</p></section>
<footer>Data: Yahoo Finance via yfinance; index membership from the public
s-and-p-500-companies dataset.{miss} Prices reflect the latest available close.
The reverse DCF discounts trailing FCF at a beta-scaled rate capped at 7.5–12% and ignores
net debt — read implied growth on heavily levered names with that in mind. Analyst consensus
and targets are third-party estimates. Sector-relative scoring means a stock is cheap
<em>versus its sector</em>; the concentration table above shows the residual sector bet.
This page is a quantitative screen to guide further research, not investment advice;
screened stocks can be cheap for good reasons. Rebuilds daily at 5:00 AM Central.</footer>
</div>
<script type="application/json" id="screen-state">{json.dumps(state)}</script>
</body></html>"""


# --------------------------------------------------------------------------
def main():
    print("Fetching S&P 500 constituents…", flush=True)
    cons = constituents()
    print(f"  {len(cons)} members", flush=True)

    print("Fetching fundamentals (this takes a few minutes)…", flush=True)
    raw, failed = fetch_all(list(cons))
    print(f"  ok={len(raw)} failed={len(failed)}", flush=True)
    if len(raw) < 100:
        print("FATAL: too few tickers returned data — refusing to publish a bad page.",
              file=sys.stderr)
        sys.exit(1)

    for r in raw:
        r["name"], r["sector"] = cons[r["ticker"]]

    print("Fetching 13 months of prices for momentum…", flush=True)
    moms = fetch_momentum([r["ticker"] for r in raw])
    mom_ok = len(moms) >= 100
    print(f"  momentum for {len(moms)}/{len(raw)} names", flush=True)
    for r in raw:
        r["mom_12_1_pct"] = moms.get(r["ticker"])

    rows, dropped = score(raw)
    print(f"Scored {len(rows)} (dropped {dropped} for insufficient data). Top 5: "
          + ", ".join(f"{r['ticker']} {r['composite']}" for r in rows[:5]), flush=True)
    if not rows:
        print("FATAL: nothing ranked.", file=sys.stderr)
        sys.exit(1)

    today = datetime.now(CENTRAL).date()
    snap_path = write_snapshot(rows, today.isoformat())
    n_snaps = len(glob.glob(os.path.join(SNAP_DIR, "*.json")))
    perf = track_record(rows, today)
    print(f"Snapshot {snap_path} written ({n_snaps} total); "
          f"track-record buckets with data: {len(perf)}", flush=True)

    print(f"Fetching analyst rating history for the top {SHOW_ROWS}…", flush=True)
    attach_ratings(rows[:SHOW_ROWS])
    got = sum(1 for r in rows[:SHOW_ROWS] if r.get("ratings"))
    print(f"  analyst history for {got}/{min(SHOW_ROWS, len(rows))} names", flush=True)

    universe_counts = {}
    for r in rows:
        universe_counts[r["sector"]] = universe_counts.get(r["sector"], 0) + 1

    prev = previous_state()
    html = render(rows, prev, len(rows), failed, dropped,
                  universe_counts=universe_counts, perf=perf,
                  n_snaps=n_snaps, mom_ok=mom_ok)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Wrote {OUT} ({len(html):,} bytes)", flush=True)


if __name__ == "__main__":
    main()
