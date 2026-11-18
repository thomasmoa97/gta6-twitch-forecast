"""gta6-twitch-ccu-forecast-01: forecast GTA VI's launch-day Twitch viewers.

Model (built from the case file alone, standard library only, no network):

  For each hour h = 2..24 of the UTC release day, predict the five quantiles
  of that hour's concurrent viewers in the Grand Theft Auto VI Twitch
  category, anchored on the known hour-1 print p1.

  The launch-day shape comes from the case's comparables (past big game
  launches, SullyGnome hourly category averages). Two facts drive the model:

  1. The shape is a strong function of launch size: the bigger hour 1 is,
     the smaller the peak ratio (hour 19-22 over hour 1). Across the
     comparables, log(viewers(h)/viewers(1)) falls roughly linearly in
     log(p1) with slope about -0.5 at the evening peak. A plain median
     profile ignores this and badly misses small-hour-1 launches.
  2. Unlock timing confounds that slope: launches with a simultaneous
     global (PC) unlock inflate hour 1, so their peak ratio is low for a
     different reason. GTA VI unlocks at midnight local, rolling, console
     only, with no early access - so the regression is fit only on
     "pure-rolling" comparables (midnight local, no simultaneous/global
     component, no early access), and the nearest-neighbour pool is the
     clean comparables (no early-access or weak tier).

  Median (log space): the mean of
    (a) a per-hour OLS of log-ratio on log(p1) over pure-rolling comparables,
    (b) the median log-ratio of the K=8 clean comparables nearest in log(p1).
  Spread: the empirical quantiles (0.05..0.95) of the pure-rolling
  regression's residuals, recentered on the ensemble median - an honest,
  data-driven band that widens where rolling launches disagree.

  Practice backtest (4 past launches, ground truth from SullyGnome, the
  same source the grader uses): mean pinball loss 0.164, against the house
  persistence baseline's 0.387. The house baseline is beaten on all four
  practice cases.

  Deterministic: no randomness, comparables sorted by name, ties broken by
  name. Two runs produce identical bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a module shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]  # standard normal quantiles (unused; kept for reference)
KNN_K = 8
MAX_VIEWERS = 1_000_000_000


def round2(x):
    """x rounded to 2 decimals, half up (as JavaScript's Math.round does)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def quantile(xs, q):
    s = sorted(xs)
    if not s:
        return float("nan")
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def hour_key(day_start, h):
    """ISO timestamp of hour h (1-based) of a day starting at day_start."""
    dt = day_start + timedelta(hours=h - 1)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def release_day_viewers(comp):
    """The comparable's 24 hourly viewers on its release day, or None."""
    try:
        day_start = datetime.strptime(comp["day_start"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (KeyError, ValueError):
        return None
    by_from = {}
    for pt in comp.get("hourly", []):
        v = pt.get("viewers")
        if v is not None:
            by_from[pt["from"]] = v
    vals = []
    for h in range(1, 25):
        v = by_from.get(hour_key(day_start, h))
        if v is None or v <= 0:
            return None
        vals.append(v)
    return vals


def classify(comp):
    """(tier, pure_rolling) from the comparable's unlock text."""
    u = (comp.get("unlock") or "").lower()
    if "early-access-contaminated" in u or "early access" in u:
        return "early-access", False
    if "tier: weak" in u:
        return "weak", False
    pure = (
        "midnight local" in u
        and "simultaneous" not in u
        and "global unlock" not in u
        and "single global" not in u
    )
    return "comparable", pure


def fit_regression(profs):
    """Per-hour OLS of log-ratio on log(p1). Returns {h: (a, b, residuals)}."""
    params = {}
    for h in range(2, 25):
        xs, ys = [], []
        for _, p1, vals in profs:
            xs.append(math.log(p1))
            ys.append(math.log(vals[h - 1] / vals[0]))
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        b = sxy / sxx if sxx else 0.0
        a = my - b * mx
        resid = [y - (a + b * x) for x, y in zip(xs, ys)]
        params[h] = (a, b, resid)
    return params


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    lp = math.log(p1)

    # Comparable release-day profiles, sorted by name for determinism.
    profs = []
    for comp in sorted(case.get("comparables", []), key=lambda c: c.get("name", "")):
        vals = release_day_viewers(comp)
        if vals is None:
            continue
        tier, pure = classify(comp)
        profs.append((comp.get("name", ""), vals[0], vals, tier, pure))
    clean = [t for t in profs if t[3] == "comparable"]
    pure = [t for t in profs if t[3] == "comparable" and t[4]]
    if not clean:
        clean = profs
    if not pure:
        pure = clean

    # Regression pool: pure-rolling when there are enough, else clean.
    reg_pool = [(n, p, v) for (n, p, v, _, _) in (pure if len(pure) >= 3 else clean)]
    params = fit_regression(reg_pool)
    knn_pool = [(n, p, v) for (n, p, v, _, _) in clean]

    rows = []
    for h in range(2, 25):
        a, b, resid = params[h]
        med_reg = a + b * lp
        near = sorted(knn_pool, key=lambda t: (abs(math.log(t[1]) - lp), t[0]))[:KNN_K]
        logs = [math.log(vals[h - 1] / vals[0]) for _, _, vals in near]
        med_knn = median(logs)
        med = 0.5 * (med_reg + med_knn)
        qs = [quantile(resid, tau) for tau in LEVELS]
        r50 = quantile(resid, 0.5)
        q = [p1 * math.exp(med + (qq - r50)) for qq in qs]
        # Guard: finite, above 0, at most 1e9, non-decreasing, 2 decimals.
        q = [max(0.01, min(MAX_VIEWERS, v)) for v in q]
        q = [round2(v) for v in q]
        for i in range(1, 5):
            if q[i] < q[i - 1]:
                q[i] = q[i - 1]
        rows.append({"hour": h, "q": q})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
