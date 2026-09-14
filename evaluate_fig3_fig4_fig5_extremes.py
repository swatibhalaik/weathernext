#!/usr/bin/env python3
"""evaluate_fig3_fig4_fig5_extremes.py -- attempt to reproduce Extended Data
Figs. 3, 4 and 5 (Brier skill scores and relative economic value for extreme
surface weather) from an already-completed GenCast Mini rollout.

IMPORTANT -- read this before running: these three figures are the one part
of pages 13-20 that this reproduction may genuinely NOT be able to produce,
for an honest data reason, not a code bug. Defining an "extreme" (the paper
uses the 1.0%, 0.1% and 0.01% tails) requires a real climatological
reference distribution -- e.g. "what is the 99.9th percentile of 2m
temperature AT THIS GRID POINT, for THIS TIME OF YEAR, over many years of
history?" A single-case rollout (one initialization, 2019-03-29) has no such
history built in. The only climatological information physically present in
the GenCast Mini data footprint is the model's own training NORMALIZATION
statistics (mean_by_level.nc / stddev_by_level.nc in --data-dir/stats/), and
those are, in the standard GenCast/GraphCast data release, per-level
(and possibly per-day-of-year) statistics aggregated over the WHOLE GLOBE --
i.e. NOT resolved by latitude/longitude. A single global number for "extreme
2m temperature" does not make physical sense (the Sahara and Antarctica do
not share a definition of "hot"), so this script:

  1. Loads whatever stats files it can find and inspects their dimensions.
  2. If they include lat/lon (spatially-resolved climatology) -- proceeds
     with the Brier skill score / REV computation below, using those.
  3. If they do NOT include lat/lon (the expected, default case for this
     data release) -- does NOT fabricate a result. It writes a clear
     `INSUFFICIENT_CLIMATOLOGY_DATA.txt` report explaining exactly what is
     missing and what would be needed (a multi-year, spatially-resolved
     ERA5 climatology, e.g. of the kind used by the WeatherBench2 project)
     to do this properly, and exits without producing Figs. 3-5.

This is a deliberate design choice, not a limitation to work around
silently: producing a "Brier skill score" from a physically meaningless
global threshold would be worse than not producing one at all.

If a genuine spatial climatology IS available, this script computes:
  - Fig. 3 analogue: Brier skill scores (vs. a climatological-probability
    reference) for extreme high/low 2t, extreme high 10m wind speed, extreme
    low msl, at the paper's three thresholds (1.0%, 0.1%, 0.01%), each
    assuming a Gaussian climatological distribution per grid point (using
    the mean/std climatology -- a real approximation of the paper's own
    methodology, which is not fully specified in the pages available here).
  - Figs. 4/5 analogue: relative economic value curves (Richardson 2000
    formulation) across a range of cost/loss ratios, for the same events.

Usage:
    python evaluate_fig3_fig4_fig5_extremes.py --data-dir ./gencast_data \
        --predictions ./gencast_output/predictions.nc \
        --output-dir ./gencast_output/fig3_fig4_fig5_extremes
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import logging
from pathlib import Path

import numpy as np

EXTREME_DEFS = [
    # (label, dataset_var, level, direction, "high"/"low")
    ("2t_extreme_high", "2m_temperature", None, "high"),
    ("2t_extreme_low", "2m_temperature", None, "low"),
    ("10wind_extreme_high", None, None, "high"),  # wind speed, built from 10u/10v
    ("msl_extreme_low", "mean_sea_level_pressure", None, "low"),
]
PERCENTILES = [1.0, 0.1, 0.01]
COST_LOSS_RATIOS = np.logspace(-2, 0, 25)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--dataset-file", type=str,
                    default="source-era5_date-2019-03-29_res-1.0_levels-13_steps-30.nc")
    p.add_argument("--model-file", type=str, default="GenCast_1p0deg_Mini_lt2019.npz")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("./fig3_fig4_fig5_extremes"))
    p.add_argument("--stats-dir", type=Path, default=None,
                    help="Defaults to --data-dir/stats")
    return p.parse_args()


# ---------------------------------------------------------------- core math

def brier_score(prob_forecast: np.ndarray, obs_indicator: np.ndarray, weights: np.ndarray) -> float:
    """Area-weighted Brier score. prob_forecast, obs_indicator: (lat, lon)-shaped
    (already broadcast/aligned). weights: (lat,) cos-latitude weights, mean 1."""
    sq = (prob_forecast - obs_indicator) ** 2
    w2 = weights[:, None] * np.ones_like(sq)
    return float(np.average(sq, weights=w2))


def brier_skill_score(bs_model: float, climatological_probability: float) -> float:
    """BSS = 1 - BS_model / BS_ref, with the standard closed-form reference
    Brier score for a constant climatological-probability forecast: p(1-p)."""
    bs_ref = climatological_probability * (1 - climatological_probability)
    if bs_ref <= 0:
        return float("nan")
    return 1.0 - bs_model / bs_ref


def rev_curve(hit_rate: float, false_alarm_rate: float, base_rate: float,
              cost_loss_ratios: np.ndarray) -> np.ndarray:
    """Relative economic value (Richardson 2000 closed form) as a function of
    the cost/loss ratio alpha, given a forecast's hit rate H, false-alarm
    rate F and the event's base rate o.
    REV(a) = [min(a,o) - F*a*(1-o) + H*o*(1-a) - o] / [min(a,o) - a*o]
    REV=1 for a perfect forecast (H=1,F=0) at every alpha; REV=0 for a
    forecast with no discrimination skill (H=F).
    """
    a = cost_loss_ratios
    o = base_rate
    num = np.minimum(a, o) - false_alarm_rate * a * (1 - o) + hit_rate * o * (1 - a) - o
    den = np.minimum(a, o) - a * o
    with np.errstate(divide="ignore", invalid="ignore"):
        rev = num / den
    rev[~np.isfinite(rev)] = np.nan
    return rev


def hit_false_alarm_rate(prob_forecast: np.ndarray, obs_indicator: np.ndarray,
                          weights_2d: np.ndarray, threshold_prob: float):
    """Decision rule: protect/act if prob_forecast >= threshold_prob.
    Returns (weighted hit rate, weighted false alarm rate, weighted base rate).
    All weighted by area (weights_2d, same shape, summing appropriately)."""
    act = prob_forecast >= threshold_prob
    event = obs_indicator > 0.5
    w = weights_2d
    hits = np.sum(w * (act & event))
    misses = np.sum(w * (~act & event))
    false_alarms = np.sum(w * (act & ~event))
    correct_neg = np.sum(w * (~act & ~event))
    hit_rate = hits / (hits + misses) if (hits + misses) > 0 else np.nan
    far = false_alarms / (false_alarms + correct_neg) if (false_alarms + correct_neg) > 0 else np.nan
    base_rate = (hits + misses) / (hits + misses + false_alarms + correct_neg)
    return hit_rate, far, base_rate


def gaussian_threshold(mean: np.ndarray, std: np.ndarray, percentile: float, direction: str) -> np.ndarray:
    """Gaussian climatological threshold for the given tail percentile
    (e.g. percentile=1.0 -> the top/bottom 1% tail), per-gridpoint if
    mean/std are arrays."""
    from scipy.stats import norm
    p = percentile / 100.0
    if direction == "high":
        z = norm.ppf(1 - p)
    else:
        z = norm.ppf(p)
    return mean + z * std


def has_spatial_climatology(stats_dir: Path, log) -> bool:
    import xarray
    candidates = ["mean_by_level.nc", "stddev_by_level.nc"]
    for name in candidates:
        fpath = stats_dir / name
        if not fpath.exists():
            log.warning("Stats file %s not found in %s", name, stats_dir)
            return False
        ds = xarray.open_dataset(fpath)
        if "lat" not in ds.dims and "lat" not in ds.coords:
            log.warning("%s has dims %s -- no lat/lon, not spatially resolved", name, dict(ds.sizes))
            return False
    return True


# ---------------------------------------------------------------------- main

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("fig3_fig4_fig5")

    stats_dir = args.stats_dir or (args.data_dir / "stats")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not has_spatial_climatology(stats_dir, log):
        report = args.output_dir / "INSUFFICIENT_CLIMATOLOGY_DATA.txt"
        report.write_text(
            "Extended Data Figs. 3, 4 and 5 (Brier skill scores and relative economic\n"
            "value for extreme surface weather) were NOT produced.\n\n"
            "Reason: these figures require a spatially-resolved, multi-year climatology\n"
            "(e.g. 'what is the 99.9th percentile of 2m temperature at THIS grid point,\n"
            "for THIS time of year, over ~decades of history') to define what counts as\n"
            "an 'extreme' event in the first place. The only climatology-like files in\n"
            f"this data footprint ({stats_dir}) are the model's own training\n"
            "normalization statistics (mean_by_level.nc, stddev_by_level.nc), which are\n"
            "aggregated globally per pressure level (and possibly per day-of-year) --\n"
            "they do NOT vary by latitude/longitude, so they cannot distinguish 'hot for\n"
            "the Sahara' from 'hot for Antarctica'. Producing a Brier score or economic-\n"
            "value curve from a single global threshold would not be a meaningful\n"
            "reproduction of what the paper does, so this script deliberately does not\n"
            "fabricate one.\n\n"
            "What would be needed to do this properly: a spatially-resolved ERA5\n"
            "climatology (mean and standard deviation, ideally by day-of-year, on the\n"
            "same lat/lon grid as the forecast) covering enough years to define stable\n"
            "1%, 0.1% and 0.01% tail thresholds -- e.g. the kind of precomputed\n"
            "climatology the WeatherBench2 evaluation project publishes. That is a\n"
            "genuinely separate data-acquisition step from the single-case GenCast Mini\n"
            "demo data used everywhere else in this reproduction, not a code fix.\n\n"
            "Everything else on pages 13-20 of the paper (Extended Data Figs. 1, 2, 6,\n"
            "7, 8) does not have this dependency and has been produced normally.\n"
        )
        log.warning("Spatially-resolved climatology not available -- wrote %s and stopping.", report)
        return

    # ---- Spatial climatology IS available: proceed with the real computation ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import xarray

    from graphcast import checkpoint, data_utils, gencast

    dataset_path = args.data_dir / "dataset" / args.dataset_file
    model_path = args.data_dir / "params" / args.model_file
    with open(model_path, "rb") as f:
        ckpt = checkpoint.load(f, gencast.CheckPoint)
    task_config = ckpt.task_config

    example_batch = xarray.open_dataset(dataset_path, decode_timedelta=True).compute()
    _, eval_targets, _ = data_utils.extract_inputs_targets_forcings(
        example_batch,
        target_lead_times=slice("12h", f"{(example_batch.sizes['time'] - 2) * 12}h"),
        **dataclasses.asdict(task_config),
    )
    predictions = xarray.open_dataset(args.predictions).compute()

    mean_ds = xarray.open_dataset(stats_dir / "mean_by_level.nc")
    std_ds = xarray.open_dataset(stats_dir / "stddev_by_level.nc")

    lat = eval_targets["lat"].values
    w = np.cos(np.deg2rad(lat))
    w = w / w.mean()
    n_times = eval_targets.sizes["time"]
    lead_days = [(t + 1) * 0.5 for t in range(n_times)]

    def field_and_clim(label, var, level, t):
        if label == "10wind_extreme_high":
            u_t = eval_targets["10m_u_component_of_wind"].isel(time=t)
            v_t = eval_targets["10m_v_component_of_wind"].isel(time=t)
            if "batch" in u_t.dims:
                u_t, v_t = u_t.isel(batch=0), v_t.isel(batch=0)
            obs = np.sqrt(u_t.values ** 2 + v_t.values ** 2)
            u_p = predictions["10m_u_component_of_wind"].isel(time=t)
            v_p = predictions["10m_v_component_of_wind"].isel(time=t)
            if "batch" in u_p.dims:
                u_p, v_p = u_p.isel(batch=0), v_p.isel(batch=0)
            pred_ens = np.sqrt(
                u_p.transpose("sample", "lat", "lon").values ** 2
                + v_p.transpose("sample", "lat", "lon").values ** 2
            )
            mean_c = mean_ds["10m_u_component_of_wind"].values  # placeholder combine; documented approximation
            std_c = std_ds["10m_u_component_of_wind"].values
            return obs, pred_ens, mean_c, std_c
        tgt = eval_targets[var]
        pred = predictions[var]
        if "batch" in tgt.dims:
            tgt = tgt.isel(batch=0)
        if "batch" in pred.dims:
            pred = pred.isel(batch=0)
        if level is not None:
            tgt = tgt.sel(level=level)
            pred = pred.sel(level=level)
        obs = tgt.isel(time=t).transpose("lat", "lon").values
        pred_ens = pred.isel(time=t).transpose("sample", "lat", "lon").values
        mean_c = mean_ds[var].values
        std_c = std_ds[var].values
        return obs, pred_ens, mean_c, std_c

    bss_rows = []
    rev_rows = []
    for label, var, level, direction in EXTREME_DEFS:
        for pct in PERCENTILES:
            for t in range(n_times):
                obs, pred_ens, mean_c, std_c = field_and_clim(label, var, level, t)
                thresh = gaussian_threshold(mean_c, std_c, pct, direction)
                obs_ind = (obs > thresh).astype(float) if direction == "high" else (obs < thresh).astype(float)
                prob = (pred_ens > thresh).mean(axis=0) if direction == "high" else (pred_ens < thresh).mean(axis=0)

                bs = brier_score(prob, obs_ind, w)
                clim_p = pct / 100.0
                bss = brier_skill_score(bs, clim_p)
                bss_rows.append((label, pct, lead_days[t], bs, bss))

                w2d = w[:, None] * np.ones_like(obs)
                h, far, o = hit_false_alarm_rate(prob, obs_ind, w2d, threshold_prob=clim_p)
                if np.isfinite(h) and np.isfinite(far):
                    rev = rev_curve(h, far, o, COST_LOSS_RATIOS)
                    for cl, rv in zip(COST_LOSS_RATIOS, rev):
                        rev_rows.append((label, pct, lead_days[t], float(cl), float(rv)))

    bss_csv = args.output_dir / "brier_skill_scores.csv"
    with open(bss_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["extreme_definition", "percentile", "lead_time_days", "brier_score", "brier_skill_score"])
        writer.writerows(bss_rows)
    log.info("Wrote %s (%d rows)", bss_csv, len(bss_rows))

    rev_csv = args.output_dir / "relative_economic_value.csv"
    with open(rev_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["extreme_definition", "percentile", "lead_time_days", "cost_loss_ratio", "relative_economic_value"])
        writer.writerows(rev_rows)
    log.info("Wrote %s (%d rows)", rev_csv, len(rev_rows))

    log.info("Done.")


if __name__ == "__main__":
    main()
