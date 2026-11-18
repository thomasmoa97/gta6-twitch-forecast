# GTA VI Twitch CCU Forecast

[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/thomasmoa97/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/thomasmoa97/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

A program that forecasts how many people watch Grand Theft Auto VI live on Twitch in each hour of its release day, 2026-11-19 (UTC), after the first hour's print arrives. For each of hours 2 to 24 it writes five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of that hour's concurrent viewers. Python 3.13, standard library only, no network, deterministic.

## How it works

The launch-day shape comes from the case's own comparables: 42 past big game launches' hourly Twitch category curves (SullyGnome). Two facts drive the model:

1. **The shape depends on launch size.** The bigger hour 1 is, the smaller the evening peak ratio (hours 19-22 over hour 1). Across the comparables, log(viewers(h)/viewers(1)) falls roughly linearly in log(hour-1 viewers), with slope about -0.5 at the peak. A plain median profile ignores this and badly misses small-hour-1 launches.
2. **Unlock timing confounds that slope.** Launches with a simultaneous global (PC) unlock inflate hour 1, so their peak ratio is low for a different reason. GTA VI unlocks at midnight local, rolling, console only, with no early access, so the regression is fit only on "pure-rolling" comparables (midnight local, no simultaneous/global component, no early access), and the nearest-neighbour pool is the clean comparables (no early-access or weak tier).

The median (in log space) is the mean of (a) a per-hour OLS of log-ratio on log(hour-1 viewers) over pure-rolling comparables, and (b) the median log-ratio of the 8 clean comparables nearest in log(hour-1 viewers). The spread is the empirical quantiles of the pure-rolling regression's residuals, recentered on the ensemble median: an honest, data-driven band that widens where rolling launches disagree.

## Practice backtest

Ground truth for the four practice launches (Marvel's Wolverine, Red Dead Redemption 2, God of War Ragnarök, Marvel's Spider-Man 2) from SullyGnome, the same source the grader uses. Mean pinball loss on the log scale:

| model | p00 | p01 | p02 | p03 | mean |
|---|---|---|---|---|---|
| house persistence baseline | 0.744 | 0.287 | 0.215 | 0.303 | **0.387** |
| this entry | 0.175 | 0.138 | 0.128 | 0.214 | **0.164** |

It beats the house baseline on all four practice cases.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory and writes `<id>.out` for each: `{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}` for hours 2 to 24. Runs in about 0.2 s for four cases, deterministic (two runs produce identical bytes).
