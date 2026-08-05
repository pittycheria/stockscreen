#!/usr/bin/env python3
"""
S&P 500 undervalued-stock screen -> static dashboard (index.html).

Runs standalone on a GitHub Actions runner. No API keys.

Pipeline:
  1. Pull current S&P 500 constituents (GitHub 'datasets' mirror of the index).
  2. Pull fundamentals + analyst data for every member via yfinance (threaded).
  3. Score each name:
       Value    35%  percentile rank on fwd P/E, P/E, P/B, EV/EBITDA, FCF yield
       DCF      25%  two-stage FCF model (skipped for Financials / Real Estate)
       Upside   20%  consensus price target vs price
       Quality  20%  Graham-style gates (missing inputs are skipped, not failed)
     Risk flags deduct 4 points each.
  4. Diff against the previous index.html's embedded state blob.
  5. Write index.html.

Scoring logic is the audited version: arithmetic independently verified,
missing data skipped rather than failed, negative FCF yield ranked worst,
negative-equity names excluded from book metrics and flagged.
"""

import concurrent.futures as cf
import csv
import io
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime
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

            # yfinance quirks: debtToEquity arrives as a percentage (124.0 = 1.24x);
            # margins/growth/ROE arrive as decimals; dividendYield varies by version.
            de = info.get("debtToEquity")
            de = round(de / 100, 2) if de is not None else None

            dy = info.get("dividendYield")
            if dy is not None:
                dy = float(dy)
                dy = dy * 100 if dy < 1 else dy       # tolerate both encodings

            rec = (info.get("recommendationKey") or "").replace("_", " ").title()
            rec = {"Strong Buy": "Strong Buy", "Buy": "Buy", "Hold": "Hold",
                   "Underperform": "Sell", "Sell": "Sell",
                   "Strong Sell": "Strong Sell"}.get(rec) or (rec or None)

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
                "roic_pct": None,                     # not exposed by yfinance
                "debt_equity": de,
                "current_ratio": info.get("currentRatio"),
                "rev_growth_pct": _pct(info.get("revenueGrowth")),
                "net_margin_pct": _pct(info.get("profitMargins")),
                "div_yield_pct": dy,
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
            for idx, row in df.sort_index(ascending=False).head(limit).iterrows():
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


# --------------------------------------------------------------------------
# 3. scoring  (audited logic)
# --------------------------------------------------------------------------
def pct_rank(pool, v, lower_better):
    xs = sorted(x for x in pool if x is not None and x > 0)
    if v is None or v <= 0 or not xs:
        return None
    p = sum(1 for x in xs if x < v) / len(xs) * 100
    return 100 - p if lower_better else p


def dcf_discount(r):
    if r["sector"] in FIN_SECTORS:
        return None
    fcf, mc = r.get("fcf_ttm_b"), r.get("market_cap_b")
    if not fcf or not mc or fcf <= 0:
        return None
    de = r.get("debt_equity")
    beta = r.get("beta") if r.get("beta") is not None else 1.0
    disc = min(max(0.043 + 0.045 * beta, 0.075), 0.12)
    roic = r.get("roic_pct")
    g1 = 0.08 if (roic and roic >= 20) else 0.05 if (roic and roic >= 10) else 0.03
    if r.get("rev_growth_pct") is not None:
        g1 = min(max(r["rev_growth_pct"] / 100, 0.02), 0.12)
    gT = 0.025
    pv, f = 0.0, fcf
    for yr in range(1, 6):
        f *= 1 + g1
        pv += f / (1 + disc) ** yr
    pv += f * (1 + gT) / (disc - gT) / (1 + disc) ** 5
    d = pv / mc - 1
    if de is not None and de > 2:          # model ignores net debt; haircut levered names
        d *= 0.5
    return min(d, 1.5)


def quality(r):
    fin = r["sector"] in FIN_SECTORS
    checks = []
    roe = r.get("roe_pct")
    if not r.get("neg_equity") and roe is not None:
        checks.append(roe >= (8 if fin else 10))
    if not fin:
        if r.get("roic_pct") is not None:
            checks.append(r["roic_pct"] >= 8)
        if r.get("debt_equity") is not None:
            checks.append(r["debt_equity"] <= 1.5)
        if r.get("current_ratio") is not None:
            checks.append(r["current_ratio"] >= 1.0)
    if r.get("net_margin_pct") is not None:
        checks.append(r["net_margin_pct"] >= (15 if fin else 8))
    checks.append(r.get("pe") is not None)
    return sum(checks) / len(checks) * 100 if checks else 0.0


def flags(r):
    """Short labels — the table explains them once in a legend below it."""
    fl = []
    if r.get("pe") is None:
        fl.append("Unprofitable")
    if (r.get("payout_pct") or 0) > 90:
        fl.append("Payout >90%")
    de = r.get("debt_equity")
    if de is not None and de > 2.5 and r["sector"] not in FIN_SECTORS:
        fl.append(f"Leverage {de:.1f}x")
    if r.get("consensus") in ("Hold", "Sell", "Strong Sell"):
        fl.append(f"{r['consensus']} consensus")
    if (r.get("analyst_count") or 0) < 5:
        fl.append("Thin coverage")
    if r.get("neg_equity"):
        fl.append("Negative equity")
    return fl


def score(rows):
    for r in rows:
        # defensive: tolerate a D/E that arrives in percent form (124.0 = 1.24x)
        de = r.get("debt_equity")
        if de is not None and de > 50:
            r["debt_equity"] = round(de / 100, 2)
        # negative/near-zero book equity makes P/B and ROE meaningless
        r["neg_equity"] = (r.get("pb") or 0) > 100
        if r["neg_equity"]:
            r["pb"] = None
    pool = {k: [r.get(k) for r in rows]
            for k in ("forward_pe", "pe", "pb", "ev_ebitda", "fcf_yield_pct")}
    for r in rows:
        ranks = [
            pct_rank(pool["forward_pe"], r.get("forward_pe"), True),
            pct_rank(pool["pe"], r.get("pe"), True),
            pct_rank(pool["pb"], r.get("pb"), True),
            pct_rank(pool["ev_ebitda"], r.get("ev_ebitda"), True),
            0.0 if (r.get("fcf_yield_pct") is not None and r["fcf_yield_pct"] <= 0)
            else pct_rank(pool["fcf_yield_pct"], r.get("fcf_yield_pct"), False),
        ]
        av = [x for x in ranks if x is not None]
        r["value_score"] = round(sum(av) / len(av), 1) if av else None

        d = dcf_discount(r)
        r["dcf_discount_pct"] = round(d * 100, 1) if d is not None else None
        r["dcf_score"] = None if d is None else round(
            min(max((d + 0.20) / 0.70, 0), 1) * 100, 1)

        up = (r["target"] / r["price"] - 1) if (r.get("target") and r.get("price")) else None
        r["upside_pct"] = round(up * 100, 1) if up is not None else None
        r["upside_score"] = None if up is None else round(
            min(max(up / 0.40, 0), 1) * 100, 1)

        r["quality_score"] = round(quality(r), 1)
        r["flags"] = flags(r)

        parts = [(r["value_score"], .35), (r["dcf_score"], .25),
                 (r["upside_score"], .20), (r["quality_score"], .20)]
        parts = [(s, w) for s, w in parts if s is not None]
        wsum = sum(w for _, w in parts)
        comp = (sum(s * w for s, w in parts) / wsum if wsum else 0) - 4 * len(r["flags"])
        r["composite"] = round(max(comp, 0), 1)

    # a name needs enough signal to be ranked at all
    rows = [r for r in rows if r["value_score"] is not None and r.get("price")]
    rows.sort(key=lambda r: -r["composite"])
    return rows


# --------------------------------------------------------------------------
# 4. previous state
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
# 5. render
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
                "First edition — new entrants, rank moves, rating changes and price "
                "moves will appear here from the next refresh onward.</p></section>")

    p = {r["ticker"]: r for r in prev["rows"]}
    shown = rows[:SHOW_ROWS]
    cur = {r["ticker"]: dict(r, rank=i + 1) for i, r in enumerate(shown)}
    when = (prev.get("generated_at") or "")[:10]
    items = []

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
                word = "▲ up" if dr > 0 else ("▼ down" if dr < 0 else "→ flat")
                movers.append((abs(dr), abs(ds),
                               f"{t}: {word} {abs(dr)} place{'s' if abs(dr) != 1 else ''} "
                               f"to #{c['rank']} (score {signed(ds, '', 1)} to {c['composite']})"))
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


def render(rows, prev, screened, failed):
    now = datetime.now(CENTRAL)
    stamp = now.strftime("%A, %B %-d, %Y at %-I:%M %p Central")
    shown = rows[:SHOW_ROWS]

    state = {
        "generated_at": now.isoformat(),
        "rows": [{"ticker": r["ticker"], "rank": i + 1, "composite": r["composite"],
                  "price": r.get("price"), "target": r.get("target"),
                  "consensus": r.get("consensus"), "upside_pct": r.get("upside_pct"),
                  "latest_rating": latest_rating(r)}
                 for i, r in enumerate(shown)],
    }

    n = len(shown)
    scope = f"top {n}" if screened > n else "all"
    showing = (f"Showing the top {n} of {screened} screened."
               if screened > n else f"Showing all {screened} screened.")
    ups = [r["upside_pct"] for r in shown if r["upside_pct"] is not None]
    tiles = f"""
<div class="tiles">
  <div class="tile"><div class="tv">{esc(shown[0]['ticker'])}</div>
    <div class="tl">Top ranked · score {shown[0]['composite']}</div></div>
  <div class="tile"><div class="tv">{screened}</div>
    <div class="tl">Index members screened</div></div>
  <div class="tile"><div class="tv">{(sum(ups)/len(ups) if ups else 0):+.1f}%</div>
    <div class="tl">Avg. upside to target ({scope})</div></div>
  <div class="tile"><div class="tv">{sum(1 for r in shown if not r['flags'])}</div>
    <div class="tl">Ranked names with no flags</div></div>
</div>"""

    trs = []
    for i, r in enumerate(shown, 1):
        d = r["dcf_discount_pct"]
        dcf = "n/a" if d is None else (">100%" if d > 100 else signed(d, nd=0))
        link = (f"<a href='https://stockanalysis.com/stocks/{r['ticker'].lower()}/'"
                f" target='_blank' rel='noopener'>{esc(r['ticker'])}</a>")
        trs.append(
            f"<tr><td class='num'>{i}</td><td>{link}</td>"
            f"<td class='co'>{esc(r['name'])}</td>"
            f"<td class='sec'>{esc(SECTOR_ABBR.get(r['sector'], r['sector']))}</td>"
            f"<td class='num'>{num(r.get('price'), '$')}</td>"
            f"<td class='num'>{num(r.get('forward_pe'), nd=1)}</td>"
            f"<td class='num'>{num(r.get('fcf_yield_pct'), suf='%', nd=1)}</td>"
            f"<td class='num'>{esc(r.get('consensus') or '—')}</td>"
            f"<td class='num'>{num(r.get('target'), '$')}</td>"
            f"<td class='num'>{signed(r.get('upside_pct'))}</td>"
            f"<td class='num'>{dcf}</td>"
            f"<td class='num'>{r['quality_score']:.0f}</td>"
            f"<td class='num score'>{r['composite']}</td>"
            f"{analyst_cells(r)}"
            f"<td class='flags'>{esc('; '.join(r['flags']) or '—')}</td></tr>")

    miss = (f" {len(failed)} member(s) skipped for missing data."
            if failed else "")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>S&amp;P 500 Undervalued Stock Screen</title>
<style>
:root {{ color-scheme: light dark; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  background:#f9f9f7; color:#0b0b0b; }}
.wrap {{ max-width:1460px; margin:0 auto; padding:28px 20px 48px; }}
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
.quiet {{ color:#52514e; font-size:13.5px; margin:4px 0; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
  gap:12px; margin:0 0 18px; }}
.tile {{ background:#fcfcfb; border:1px solid rgba(11,11,11,.10);
  border-radius:10px; padding:14px 16px; }}
.tv {{ font-size:24px; font-weight:650; }}
.tl {{ font-size:12px; color:#52514e; margin-top:2px; }}
.tblwrap {{ overflow-x:auto; }}
table {{ width:100%; border-collapse:collapse; font-size:12.5px; }}
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
  a {{ color:#3987e5; }}
}}
</style></head><body><div class="wrap">
<h1>S&amp;P 500 Undervalued Stock Screen</h1>
<p class="sub">Every current index member scored on four pillars — relative value 35%
(fwd P/E, P/E, P/B, EV/EBITDA, FCF yield, percentile-ranked across the index) ·
DCF fair value 25% (two-stage free-cash-flow model, skipped for financials and REITs) ·
analyst upside 20% · quality gates 20% (ROE, leverage, liquidity, margins, profitability).
Each risk flag deducts 4 points. {showing}</p>
<div class="stamp">Last refreshed: {stamp}</div>
{tiles}
{changes_block(rows, prev)}
<section class="card"><h2>Ranked candidates</h2><div class="tblwrap"><table>
<thead><tr><th>#</th><th>Ticker</th><th>Company</th><th>Sector</th><th>Price</th>
<th>Fwd P/E</th><th>FCF Yld</th><th>Consensus</th><th>Target</th><th>Upside</th>
<th>DCF</th><th>Qual.</th><th>Score</th>
<th>Analyst (last {RATINGS_N})</th><th>Rating</th><th>Rated</th><th>Flags</th></tr></thead>
<tbody>
{chr(10).join(trs)}
</tbody></table></div>
<p class="legend"><strong>Flags</strong> (each deducts 4 points): <em>Hold/Sell consensus</em> —
the street is not constructive · <em>Payout &gt;90%</em> — dividend consumes nearly all
earnings · <em>Leverage</em> — debt/equity above 2.5x · <em>Unprofitable</em> — no trailing
earnings · <em>Negative equity</em> — book-value metrics unreliable · <em>Thin coverage</em> —
fewer than five analysts.<br>
<strong>Analyst / Rating / Rated</strong> — the last {RATINGS_N} individual rating actions on
record, newest first: the covering firm, the grade it assigned, and the date. These are
individual firms' calls, not the consensus — the Consensus column is the aggregate view, and
the two often disagree. Coverage depth varies, so some names list fewer than {RATINGS_N}.</p></section>
<footer>Data: Yahoo Finance via yfinance; index membership from the public
s-and-p-500-companies dataset.{miss} Prices reflect the latest available close.
The DCF is a deliberately simple two-stage model that ignores net debt — treat
"&gt;100%" discounts on heavily levered names as "statistically very cheap," not as
price targets. Analyst consensus and targets are third-party estimates, not forecasts
of actual returns. This page is a quantitative screen to guide further research, not
investment advice; screened stocks can be cheap for good reasons. Rebuilds daily at
5:00 AM Central.</footer>
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

    rows = score(raw)
    print(f"Scored {len(rows)}. Top 5: "
          + ", ".join(f"{r['ticker']} {r['composite']}" for r in rows[:5]), flush=True)

    print(f"Fetching analyst rating history for the top {SHOW_ROWS}…", flush=True)
    attach_ratings(rows[:SHOW_ROWS])
    got = sum(1 for r in rows[:SHOW_ROWS] if r.get("ratings"))
    print(f"  analyst history for {got}/{min(SHOW_ROWS, len(rows))} names", flush=True)

    prev = previous_state()
    html = render(rows, prev, len(rows), failed)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Wrote {OUT} ({len(html):,} bytes)", flush=True)


if __name__ == "__main__":
    main()
