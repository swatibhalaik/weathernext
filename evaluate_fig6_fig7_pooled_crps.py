#!/usr/bin/env python3
"""evaluate_fig6_fig7_pooled_crps.py -- reproduce the structure of Extended
Data Figs. 6 and 7 (average-pooled and max-pooled CRPS scorecards) from an
already-completed GenCast Mini rollout.

The paper's Figs. 6/7 (p.18-19) show, for wind_speed/geopotential/temperature/
specific_humidity (x13 levels) plus 2t/msl/10m wind_speed/total_precipitation_12hr,
a scorecard of "% GenCast better than ENS" CRPS at 5 spatial pooling scales
(241, 481, 962, 1922, 3828 km), using average-pooling (Fig.6) and max-pooling
(Fig.7) of the forecast/truth fields before scoring. Pooling matters because a
forecast can be excellent at capturing *where roughly* a feature will be
without matching it point-for-point -- pooling rewards that coarser kind of
skill and is especially relevant for granular fields like precipitation.

What this script does differently from the paper, and why (stated on every
plot, not just here):
  1. No ECMWF ENS forecast is available in this reproduction -- as with the
     rest of this evaluation suite, we substitute a PERSISTENCE baseline
     (forecast = the initial ERA5 analysis, held constant, then pooled the
     same way) and report "% GenCast Mini better than persistence".
  2. GenCast Mini here runs at 1-degree resolution (the paper's GenCast runs
     at 0.25 degree). We approximate each physical pooling scale as the
     nearest whole number of 1-degree grid cells (241 km/~111 km per degree
     at the equator =~2 cells, 481 km=~4, 962 km=~9, 1922 km=~17, 3828 km=~34).
     At 1 degree resolution the smallest scale (241 km, ~2 cells) barely
     differs from the unpooled grid -- this is a real resolution-mismatch
     limitation of Mini, not a bug, and is labelled on the plots.
  3. Pooling here is simple non-overlapping block pooling directly on the
     lat/lon grid (trimming any leftover cells that don't fill a whole
     block), not necessarily the exact windowing scheme the paper used
     (which is not fully specified in the main text/methods excerpt
     available here).

Usage:
    python evaluate_fig6_fig7_pooled_crps.py --data-dir ./gencast_data \
        --predictions ./gencast_output/predictions.nc \
        --output-dir ./gencast_output/fig6_fig7_pooled_crps
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import logging
from pathlib import Path

import numpy as np

PRESSURE_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)

# (label_km, approx block size in 1-degree grid cells)
SCALES = [
    (241, 2),
    (481, 4),
    (962, 9),
    (1922, 17),
    (3828, 34),
]

ATMOS_FAMILIES = ["wind_speed", "geopotential", "temperature", "specific_humidity"]
SURFACE_ROWS = ["2t", "msl", "10wind_speed", "total_precipitation_12hr"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--dataset-file", type=str,
                    default="source-era5_date-2019-03-29_res-1.0_levels-13_steps-30.nc")
    p.add_argument("--model-file", type=str, default="GenCast_1p0deg_Mini_lt2019.npz")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("./fig6_fig7_pooled_crps"))
    return p.parse_args()


def fair_crps(ensemble: np.ndarray, obs: np.ndarray, axis: int = 0) -> np.ndarray:
    """Fair (bias-corrected) ensemble CRPS, Gneiting & Raftery (2007). Same
    implementation as evaluate_predictions.py / evaluate_full_scorecard.py,
    unit-tested there against the closed-form Gaussian CRPS."""
    M = ensemble.shape[axis]
    ens = np.moveaxis(ensemble, axis, 0)
    term1 = np.mean(np.abs(ens - obs[None, ...]), axis=0)
    diff = np.abs(ens[:, None, ...] - ens[None, :, ...])
    term2 = diff.sum(axis=(0, 1)) / (M * (M - 1))
    return term1 - 0.5 * term2


def block_pool(field: np.ndarray, block: int, mode: str = "mean") -> np.ndarray:
    """Non-overlapping block pooling over the LAST TWO axes (lat, lon).
    field: (..., lat, lon). Leftover rows/cols that don't fill a whole block
    are trimmed (dropped), not padded, to keep every pooled cell a genuine
    full-size block.
    """
    if block <= 1:
        return field
    *lead, lat, lon = field.shape
    lat_t = (lat // block) * block
    lon_t = (lon // block) * block
    trimmed = field[..., :lat_t, :lon_t]
    new_shape = tuple(lead) + (lat_t // block, block, lon_t // block, block)
    reshaped = trimmed.reshape(new_shape)
    if mode == "mean":
        return reshaped.mean(axis=(-3, -1))
    elif mode == "max":
        return reshaped.max(axis=(-3, -1))
    else:
        raise ValueError(f"unknown pooling mode {mode!r}")


def pooled_lat_weights(lat: np.ndarray, block: int) -> np.ndarray:
    lat_t = (lat.shape[0] // block) * block
    lat_trim = lat[:lat_t].reshape(-1, block).mean(axis=1)
    w = np.cos(np.deg2rad(lat_trim))
    return w / w.mean()


def crps_deterministic(forecast: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """CRPS of a deterministic (single-valued) forecast reduces to the
    absolute error -- standard result (Gneiting & Raftery 2007, sec. 4.1)."""
    return np.abs(forecast - obs)


def get_field(ds, var, level, batch0=True):
    da = ds[var]
    if "batch" in da.dims and batch0:
        da = da.isel(batch=0)
    if level is not None and "level" in da.coords:
        da = da.sel(level=level)
    return da


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("fig6_fig7")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import xarray

    from graphcast import checkpoint, data_utils, gencast

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = args.data_dir / "dataset" / args.dataset_file
    model_path = args.data_dir / "params" / args.model_file

    with open(model_path, "rb") as f:
        ckpt = checkpoint.load(f, gencast.CheckPoint)
    task_config = ckpt.task_config

    example_batch = xarray.open_dataset(dataset_path, decode_timedelta=True).compute()
    eval_inputs, eval_targets, _ = data_utils.extract_inputs_targets_forcings(
        example_batch,
        target_lead_times=slice("12h", f"{(example_batch.sizes['time'] - 2) * 12}h"),
        **dataclasses.asdict(task_config),
    )
    predictions = xarray.open_dataset(args.predictions).compute()
    n_times = eval_targets.sizes["time"]
    lead_days = [(t + 1) * 0.5 for t in range(n_times)]
    lat = eval_targets["lat"].values

    # (family_key, required_var_to_check, is_wind_speed, level_for_wind_vars)
    # level_for_wind_vars distinguishes the atmospheric wind_speed family
    # (u/v_component_of_wind @ 13 levels) from the surface "10wind_speed" row
    # (10m_u/v_component_of_wind, no level).
    VAR_MAP = {
        "geopotential": "geopotential", "temperature": "temperature",
        "specific_humidity": "specific_humidity",
        "2t": "2m_temperature", "msl": "mean_sea_level_pressure",
        "total_precipitation_12hr": "total_precipitation_12hr",
    }

    def get_family_field(ds, family, level, time_idx):
        """Returns a numpy array: (sample, lat, lon) if ds has a 'sample'
        dim (forecast ensemble), else plain (lat, lon) (target/persistence)."""
        if family in ("wind_speed", "10wind_speed"):
            if family == "wind_speed":
                u_var, v_var = "u_component_of_wind", "v_component_of_wind"
            else:
                u_var, v_var = "10m_u_component_of_wind", "10m_v_component_of_wind"
            u = get_field(ds, u_var, level).isel(time=time_idx)
            v = get_field(ds, v_var, level).isel(time=time_idx)
            if "sample" in u.dims:
                u = u.transpose("sample", "lat", "lon").values
                v = v.transpose("sample", "lat", "lon").values
            else:
                u = u.transpose("lat", "lon").values
                v = v.transpose("lat", "lon").values
            return np.sqrt(u ** 2 + v ** 2)
        var = VAR_MAP[family]
        da = get_field(ds, var, level).isel(time=time_idx)
        if "sample" in da.dims:
            return da.transpose("sample", "lat", "lon").values
        return da.transpose("lat", "lon").values

    def required_var_for(family):
        if family == "wind_speed":
            return "u_component_of_wind"
        if family == "10wind_speed":
            return "10m_u_component_of_wind"
        return VAR_MAP[family]

    families_and_levels = [(fam, lvl) for fam in ATMOS_FAMILIES for lvl in PRESSURE_LEVELS]
    families_and_levels += [(fam, None) for fam in SURFACE_ROWS]

    rows = []  # (family, level, scale_km, block, pooling_mode, lead_days, crps_gencast, crps_persistence, skill_pct)

    for mode in ("mean", "max"):
        log.info("=== Pooling mode: %s (%s) ===", mode, "Fig.6 average-pooled" if mode == "mean" else "Fig.7 max-pooled")
        for family, level in families_and_levels:
            required_var = required_var_for(family)
            if required_var not in eval_targets or required_var not in predictions:
                log.warning("Skipping %s (level %s) -- variable not found", family, level)
                continue
            for scale_km, block in SCALES:
                w_pooled = pooled_lat_weights(lat, block)
                for t in range(n_times):
                    tgt_field = get_family_field(eval_targets, family, level, t)
                    pred_field = get_family_field(predictions, family, level, t)
                    pers_field = get_family_field(eval_inputs, family, level, -1)

                    tgt_pooled = block_pool(tgt_field, block, mode)
                    pred_pooled = block_pool(pred_field, block, mode)  # (sample, lat_p, lon_p)
                    pers_pooled = block_pool(pers_field, block, mode)

                    crps_field = fair_crps(pred_pooled, tgt_pooled, axis=0)
                    crps_g = float(np.average(crps_field.mean(axis=-1), weights=w_pooled))

                    crps_pers_field = crps_deterministic(pers_pooled, tgt_pooled)
                    crps_p = float(np.average(crps_pers_field.mean(axis=-1), weights=w_pooled))

                    skill = 100.0 * (crps_p - crps_g) / crps_p if crps_p > 0 else float("nan")
                    rows.append((family, level, scale_km, block, mode, lead_days[t], crps_g, crps_p, skill))
            log.info("  done: %s%s", family, f"@{level}hPa" if level else "")

    csv_path = args.output_dir / "pooled_crps_data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["family", "level_hpa", "scale_km", "block_cells", "pooling_mode",
                          "lead_time_days", "crps_gencast", "crps_persistence", "skill_pct_vs_persistence"])
        writer.writerows(rows)
    log.info("Wrote %s (%d rows)", csv_path, len(rows))

    # Composite figure per pooling mode, mimicking the paper's panel layout:
    # rows = 4 atmospheric families (level x lead-time heatmap) + 4 surface rows
    # (single-row heatmap), columns = 5 spatial scales.
    for mode, mode_label, fig_name in (("mean", "Average-pooled (cf. Extended Data Fig. 6)", "scorecard_avgpool.png"),
                                        ("max", "Max-pooled (cf. Extended Data Fig. 7)", "scorecard_maxpool.png")):
        fig, axes = plt.subplots(len(ATMOS_FAMILIES) + len(SURFACE_ROWS), len(SCALES),
                                  figsize=(3.2 * len(SCALES), 2.2 * (len(ATMOS_FAMILIES) + len(SURFACE_ROWS))))
        vmax = 30.0
        for ri, family in enumerate(ATMOS_FAMILIES):
            for ci, (scale_km, block) in enumerate(SCALES):
                sub = [r for r in rows if r[0] == family and r[2] == scale_km and r[4] == mode]
                grid = np.full((len(PRESSURE_LEVELS), n_times), np.nan)
                for fam, lvl, skm, blk, md, ld, cg, cp, sk in sub:
                    li = PRESSURE_LEVELS.index(lvl)
                    ti = int(round(ld / 0.5)) - 1
                    grid[li, ti] = sk
                ax = axes[ri][ci]
                im = ax.imshow(grid, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax,
                                extent=[0, lead_days[-1], len(PRESSURE_LEVELS) - 0.5, -0.5])
                if ri == 0:
                    ax.set_title(f"{scale_km} km")
                if ci == 0:
                    ax.set_ylabel(f"{family}\nlevel idx")
                else:
                    ax.set_yticks([])
        for si, srow in enumerate(SURFACE_ROWS):
            ri = len(ATMOS_FAMILIES) + si
            for ci, (scale_km, block) in enumerate(SCALES):
                sub = [r for r in rows if r[0] == srow and r[2] == scale_km and r[4] == mode]
                sub = sorted(sub, key=lambda r: r[5])
                vec = np.full((1, n_times), np.nan)
                for fam, lvl, skm, blk, md, ld, cg, cp, sk in sub:
                    ti = int(round(ld / 0.5)) - 1
                    vec[0, ti] = sk
                ax = axes[ri][ci]
                ax.imshow(vec, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax,
                           extent=[0, lead_days[-1], 0.5, -0.5])
                ax.set_yticks([])
                if ci == 0:
                    ax.set_ylabel(srow)
                if ri == len(ATMOS_FAMILIES) + len(SURFACE_ROWS) - 1:
                    ax.set_xlabel("Lead time (days)")
        fig.suptitle(
            f"GenCast Mini pooled-CRPS scorecard -- {mode_label}\n"
            "% better than PERSISTENCE (not ENS -- unavailable in this reproduction). "
            "1-degree grid; scales are approximate block sizes, not exact km windows.",
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out = args.output_dir / fig_name
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Wrote %s", out)

    log.info("Done.")


if __name__ == "__main__":
    main()
