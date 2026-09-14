#!/usr/bin/env python3
"""evaluate_full_scorecard.py -- extend the paper-comparable evaluation from the
original 10 headline variables to (almost) everything GenCast Mini predicts.

Context: predictions.nc already contains a full 84-channel forecast (6 surface
variables + 6 atmospheric variables x 13 pressure levels) -- the model predicts
all of this in every run, we just only *scored* 10 of them so far. This script
re-uses that same predictions.nc (no GPU, no re-run of the rollout) and scores
everything else, in two tiers:

  Tier 1 (69 variable/level combinations): exactly the set the paper's own
  Extended Data Fig. 1 scorecard covers -- 2m_temperature, mean_sea_level_pressure,
  10m_u_component_of_wind, 10m_v_component_of_wind (surface), and temperature,
  geopotential, u_component_of_wind, v_component_of_wind, specific_humidity at
  all 13 pressure levels. The paper's own scorecard shows "% GenCast better than
  ENS" -- we don't have ECMWF's ENS forecasts for this case, so we substitute a
  persistence baseline (forecast = the initial condition held constant, a
  standard meteorological baseline) and report "% GenCast better than
  persistence" instead. This is clearly a different (weaker) baseline than ENS,
  labelled as such everywhere it appears -- not a claim of reproducing the
  paper's actual ENS-relative numbers.

  Tier 2 (15 variable/level combinations): the rest of what the model predicts
  but which the paper's headline scorecard (Fig. 1) does not itself visualize --
  vertical_velocity at all 13 levels, plus total_precipitation_12hr and
  sea_surface_temperature (both of which the paper *does* define as prediction
  targets with their own loss weights in Methods, and *does* evaluate elsewhere
  via pooled-CRPS scorecards at multiple spatial scales in Extended Data Figs.
  6-7 and Supplementary Figs. B13-B14 -- but this script applies the same
  simple point-wise RMSE/CRPS used throughout this reproduction, not that
  specific pooled methodology). Scored with plain RMSE + CRPS vs. lead time,
  same as the original 10 variables, just without a paper figure to visually
  pair against.

Usage:
    python evaluate_full_scorecard.py --data-dir ./gencast_data \
        --predictions ./gencast_output/predictions.nc \
        --output-dir ./gencast_output/full_scorecard
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import logging
from pathlib import Path

import numpy as np

PRESSURE_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)

# Tier 1: exactly what the paper's Extended Data Fig. 1 scorecard covers.
TIER1_SURFACE_VARS = [
    ("2t", "2m_temperature"),
    ("msl", "mean_sea_level_pressure"),
    ("10u", "10m_u_component_of_wind"),
    ("10v", "10m_v_component_of_wind"),
]
TIER1_ATMOSPHERIC_FAMILIES = [
    ("temperature", "temperature"),
    ("geopotential", "geopotential"),
    ("u_component_of_wind", "u_component_of_wind"),
    ("v_component_of_wind", "v_component_of_wind"),
    ("specific_humidity", "specific_humidity"),
]

# Tier 2: everything else the model predicts, not in the paper's headline figure.
TIER2_ATMOSPHERIC_FAMILIES = [
    ("vertical_velocity", "vertical_velocity"),
]
TIER2_SURFACE_VARS = [
    ("total_precipitation_12hr", "total_precipitation_12hr"),
    ("sea_surface_temperature", "sea_surface_temperature"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--dataset-file", type=str,
                    default="source-era5_date-2019-03-29_res-1.0_levels-13_steps-30.nc")
    p.add_argument("--model-file", type=str, default="GenCast_1p0deg_Mini_lt2019.npz")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("./full_scorecard"))
    return p.parse_args()


def fair_crps(ensemble: np.ndarray, obs: np.ndarray, axis: int = 0) -> np.ndarray:
    """Fair (bias-corrected) ensemble CRPS, Gneiting & Raftery (2007). Same
    implementation as evaluate_predictions.py, unit-tested there against the
    closed-form Gaussian CRPS."""
    M = ensemble.shape[axis]
    ens = np.moveaxis(ensemble, axis, 0)
    term1 = np.mean(np.abs(ens - obs[None, ...]), axis=0)
    diff = np.abs(ens[:, None, ...] - ens[None, :, ...])
    term2 = diff.sum(axis=(0, 1)) / (M * (M - 1))
    return term1 - 0.5 * term2


def lat_weights(lat: np.ndarray) -> np.ndarray:
    w = np.cos(np.deg2rad(lat))
    return w / w.mean()


def weighted_rmse(a, b, w_da):
    return float(np.sqrt(float((((a - b) ** 2) * w_da).mean())))


def get_field(ds, var, level, time_idx=None, batch0=True):
    da = ds[var]
    if "batch" in da.dims and batch0:
        da = da.isel(batch=0)
    if level is not None and "level" in da.coords:
        da = da.sel(level=level)
    if time_idx is not None:
        da = da.isel(time=time_idx)
    return da


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("full_scorecard")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors
    import matplotlib.pyplot as plt
    import xarray

    from graphcast import checkpoint, data_utils, gencast

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = args.data_dir / "dataset" / args.dataset_file
    model_path = args.data_dir / "params" / args.model_file

    log.info("Loading task_config from checkpoint %s", model_path)
    with open(model_path, "rb") as f:
        ckpt = checkpoint.load(f, gencast.CheckPoint)
    task_config = ckpt.task_config

    log.info("Loading example dataset from %s", dataset_path)
    example_batch = xarray.open_dataset(dataset_path, decode_timedelta=True).compute()

    eval_inputs, eval_targets, _ = data_utils.extract_inputs_targets_forcings(
        example_batch,
        target_lead_times=slice("12h", f"{(example_batch.sizes['time'] - 2) * 12}h"),
        **dataclasses.asdict(task_config),
    )

    log.info("Loading predictions from %s", args.predictions)
    predictions = xarray.open_dataset(args.predictions).compute()
    n_members = predictions.sizes.get("sample", None)

    lat = eval_targets["lat"].values
    w = lat_weights(lat)
    w_da = xarray.DataArray(w, dims=["lat"], coords={"lat": eval_targets["lat"]})

    n_times = eval_targets.sizes["time"]
    lead_days = [t * 0.5 for t in range(n_times)]

    # ---------------- TIER 1: Fig.1-matching scorecard vs. persistence ----------------
    log.info("=== Tier 1: %d variable/level combinations (paper's Fig.1 set) ===",
              len(TIER1_SURFACE_VARS) + len(TIER1_ATMOSPHERIC_FAMILIES) * len(PRESSURE_LEVELS))

    scorecard_rows = []  # (family, level_or_None, lead_days, rmse_gencast, rmse_persistence, skill_pct)

    def score_one(var, level, family_label):
        if var not in eval_targets or var not in predictions:
            log.warning("Variable %s not found -- skipping", var)
            return
        target_da = get_field(eval_targets, var, level)
        pred_da = get_field(predictions, var, level)
        persistence_val = get_field(eval_inputs, var, level, time_idx=-1)  # last input frame = t=0 analysis
        for t in range(n_times):
            tgt_t = target_da.isel(time=t)
            pred_mean_t = pred_da.isel(time=t).mean(dim="sample")
            rmse_g = weighted_rmse(pred_mean_t, tgt_t, w_da)
            rmse_p = weighted_rmse(persistence_val, tgt_t, w_da)
            skill = 100.0 * (rmse_p - rmse_g) / rmse_p if rmse_p > 0 else float("nan")
            scorecard_rows.append((family_label, level, lead_days[t], rmse_g, rmse_p, skill))

    for label, var in TIER1_SURFACE_VARS:
        log.info("Tier 1 surface: %s", label)
        score_one(var, None, label)

    for label, var in TIER1_ATMOSPHERIC_FAMILIES:
        log.info("Tier 1 atmospheric family: %s (13 levels)", label)
        for lvl in PRESSURE_LEVELS:
            score_one(var, lvl, label)

    # Write raw scorecard data
    csv_path = args.output_dir / "scorecard_data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["family", "level_hpa", "lead_time_days", "rmse_gencast", "rmse_persistence", "skill_pct_vs_persistence"])
        writer.writerows(scorecard_rows)
    log.info("Wrote %s (%d rows)", csv_path, len(scorecard_rows))

    # ---- Composite scorecard: ONE figure, laid out exactly like the paper's own
    # Extended Data Fig. 1 (surface strip + u/v wind on the left, temperature/
    # geopotential/specific humidity on the right, one SHARED colorbar), so this
    # is visually comparable to the paper panel-for-panel rather than six
    # separately-scaled images that can't be compared to each other at a glance.
    #
    # Colorbar: SEQUENTIAL, not diverging. The paper uses a +/-20% diverging
    # scale because its GenCast-vs-ENS comparison genuinely has both signs in
    # meaningful quantity. Our GenCast-vs-persistence skill here is ENTIRELY
    # positive (persistence is a much weaker baseline) -- checked directly
    # against this run's own data (min/max over all 2,070 rows), not assumed.
    # A symmetric +/-X% diverging scale on all-positive data wastes its whole
    # negative half and compresses the real 5-92% spread into a fraction of
    # the remaining half, which is why the six separate per-family panels
    # read as "solid color, can't see any difference." A single-hue sequential
    # map spanning the actual data range gives that spread the full color
    # range instead.
    all_skill = np.array([r[5] for r in scorecard_rows if np.isfinite(r[5])])
    if all_skill.size:
        shared_vmin = float(np.floor(all_skill.min() / 5.0) * 5.0)
        shared_vmax = float(np.ceil(all_skill.max() / 5.0) * 5.0)
    else:
        shared_vmin, shared_vmax = 0.0, 20.0
    composite_cmap = matplotlib.colormaps["Blues"].copy()
    composite_cmap.set_under("#b2182b")  # in case a future re-run does have a genuine negative cell
    composite_norm = matplotlib.colors.Normalize(vmin=shared_vmin, vmax=shared_vmax)
    log.info("Composite scorecard shared colorbar: sequential %.0f%%-%.0f%% "
             "(this run's own min/max skill, all positive -- not a +/-X%% diverging scale)",
             shared_vmin, shared_vmax)

    def _surface_grid():
        labels = [lbl for lbl, _ in TIER1_SURFACE_VARS]
        g = np.full((len(labels), n_times), np.nan)
        for fam, level, ld, rg, rp, skill in scorecard_rows:
            if fam in labels:
                g[labels.index(fam), int(round(ld / 0.5))] = skill
        return g, labels

    def _atmos_grid(fam_name):
        g = np.full((len(PRESSURE_LEVELS), n_times), np.nan)
        for fam, level, ld, rg, rp, skill in scorecard_rows:
            if fam == fam_name and level is not None:
                g[PRESSURE_LEVELS.index(level), int(round(ld / 0.5))] = skill
        return g

    SURFACE_DISPLAY_NAMES = {"2t": "2m temperature", "msl": "mean sea level pressure",
                              "10u": "10m u wind", "10v": "10m v wind"}
    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25, left=0.11, right=0.87, top=0.90, bottom=0.08)
    xticks = [t for t in (1, 3, 5, 7, 10, 15) if t <= lead_days[-1]]
    panel_axes = []
    im = None

    ax_surf = fig.add_subplot(gs[0, 0])
    grid, labels = _surface_grid()
    im = ax_surf.imshow(grid, aspect="auto", cmap=composite_cmap, norm=composite_norm,
                         extent=[0, lead_days[-1], len(labels) - 0.5, -0.5], interpolation="nearest")
    ax_surf.set_yticks(range(len(labels)))
    ax_surf.set_yticklabels([SURFACE_DISPLAY_NAMES.get(l, l) for l in labels], fontsize=8)
    ax_surf.set_xticks(xticks)
    ax_surf.tick_params(labelbottom=False)
    panel_axes.append(ax_surf)

    left_families = [("u_component_of_wind", "u component of wind"), ("v_component_of_wind", "v component of wind")]
    right_families = [("temperature", "temperature"), ("geopotential", "geopotential"), ("specific_humidity", "specific humidity")]
    for i, (fam, title) in enumerate(left_families):
        ax = fig.add_subplot(gs[i + 1, 0])
        ax.imshow(_atmos_grid(fam), aspect="auto", cmap=composite_cmap, norm=composite_norm,
                  extent=[0, lead_days[-1], len(PRESSURE_LEVELS) - 0.5, -0.5], interpolation="nearest")
        ax.set_yticks(range(len(PRESSURE_LEVELS))); ax.set_yticklabels(PRESSURE_LEVELS, fontsize=8)
        ax.set_ylabel("Pressure (hPa)", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(xticks)
        if i == len(left_families) - 1:
            ax.set_xlabel("Lead time (days)", fontsize=9)
        else:
            ax.tick_params(labelbottom=False)
        panel_axes.append(ax)
    for i, (fam, title) in enumerate(right_families):
        ax = fig.add_subplot(gs[i, 1])
        ax.imshow(_atmos_grid(fam), aspect="auto", cmap=composite_cmap, norm=composite_norm,
                  extent=[0, lead_days[-1], len(PRESSURE_LEVELS) - 0.5, -0.5], interpolation="nearest")
        ax.set_yticks(range(len(PRESSURE_LEVELS))); ax.set_yticklabels(PRESSURE_LEVELS, fontsize=8)
        ax.set_ylabel("Pressure (hPa)", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(xticks)
        if i == len(right_families) - 1:
            ax.set_xlabel("Lead time (days)", fontsize=9)
        else:
            ax.tick_params(labelbottom=False)
        panel_axes.append(ax)

    fig.suptitle("GenCast Mini vs PERSISTENCE RMSE scorecard (not ENS -- unavailable in this reproduction)\n"
                 f"Sequential colorbar {shared_vmin:.0f}%-{shared_vmax:.0f}% (this run's actual skill range, all positive), "
                 "shared across every panel",
                 fontsize=11)
    cbar = fig.colorbar(im, ax=panel_axes, fraction=0.035, pad=0.03, extend="min")
    cbar.set_label("% GenCast Mini better than persistence")
    out = args.output_dir / "scorecard_composite.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out)

    # Heatmaps: one per atmospheric family, pressure level (y) x lead time (x) -> skill %
    # (kept alongside the composite above for anyone who wants a larger, single-
    # variable, independently-scaled zoom on one family)
    for label, _ in TIER1_ATMOSPHERIC_FAMILIES:
        rows = [r for r in scorecard_rows if r[0] == label and r[1] is not None]
        grid = np.full((len(PRESSURE_LEVELS), n_times), np.nan)
        for family, level, ld, rg, rp, skill in rows:
            li = PRESSURE_LEVELS.index(level)
            ti = int(round(ld / 0.5))
            grid[li, ti] = skill
        finite_grid = grid[np.isfinite(grid)]
        if finite_grid.size:
            panel_vmin = float(np.floor(np.percentile(finite_grid, 2) / 5.0) * 5.0)
            panel_vmax = float(np.ceil(np.percentile(finite_grid, 98) / 5.0) * 5.0)
            if panel_vmax <= panel_vmin:
                panel_vmax = panel_vmin + 5.0
        else:
            panel_vmin, panel_vmax = 0.0, 100.0
        panel_cmap = matplotlib.colormaps["Blues"].copy()
        panel_cmap.set_under("#b2182b")
        panel_cmap.set_over("#08306b")
        panel_norm = matplotlib.colors.Normalize(vmin=panel_vmin, vmax=panel_vmax)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        im = ax.imshow(grid, aspect="auto", cmap=panel_cmap, norm=panel_norm,
                        extent=[0, lead_days[-1], len(PRESSURE_LEVELS) - 0.5, -0.5], interpolation="nearest")
        ax.set_yticks(range(len(PRESSURE_LEVELS)))
        ax.set_yticklabels(PRESSURE_LEVELS)
        ax.set_ylabel("Pressure (hPa)")
        ax.set_xlabel("Lead time (days)")
        ax.set_title(f"{label} -- % GenCast Mini better than PERSISTENCE (not ENS)")
        cbar = fig.colorbar(im, ax=ax, extend="both")
        cbar.set_label("% better than persistence (sequential scale, not paper's diverging ENS scale)")
        fig.tight_layout()
        out = args.output_dir / f"scorecard_{label}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        log.info("Wrote %s", out)

    # Surface panel: one small-multiple figure, skill vs lead time, 4 lines
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, _ in TIER1_SURFACE_VARS:
        rows = sorted([r for r in scorecard_rows if r[0] == label], key=lambda r: r[2])
        if not rows:
            continue
        ax.plot([r[2] for r in rows], [r[5] for r in rows], marker="o", label=label)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("% better than persistence")
    ax.set_title("Surface variables -- % GenCast Mini better than PERSISTENCE (not ENS)")
    ax.legend()
    fig.tight_layout()
    out = args.output_dir / "scorecard_surface.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out)

    # ---------------- TIER 2: everything else, plain RMSE + CRPS ----------------
    log.info("=== Tier 2: %d additional variable/level combinations (paper-defined, not in Fig.1) ===",
              len(TIER2_SURFACE_VARS) + len(TIER2_ATMOSPHERIC_FAMILIES) * len(PRESSURE_LEVELS))

    def compute_rmse_crps_curve(var, level):
        target_da = get_field(eval_targets, var, level)
        pred_da = get_field(predictions, var, level)
        rows = []
        for t in range(n_times):
            tgt_t = target_da.isel(time=t)
            pred_t = pred_da.isel(time=t)
            pred_mean_t = pred_t.mean(dim="sample")
            rmse = weighted_rmse(pred_mean_t, tgt_t, w_da)
            ens_vals = pred_t.transpose("sample", "lat", "lon").values
            obs_vals = tgt_t.transpose("lat", "lon").values
            crps_field = fair_crps(ens_vals, obs_vals, axis=0)
            crps = float(np.average(crps_field.mean(axis=1), weights=w))
            rows.append((t, rmse, crps))
        return rows

    tier2_summary = []
    for label, var in TIER2_SURFACE_VARS:
        if var not in eval_targets or var not in predictions:
            log.warning("Tier 2 variable %s not found -- skipping", var)
            continue
        log.info("Tier 2 surface: %s", label)
        rows = compute_rmse_crps_curve(var, None)
        csvp = args.output_dir / f"extra_{label}.csv"
        with open(csvp, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_index", "lead_time_days", "rmse", "crps"])
            writer.writerows([(t, t * 0.5, r, c) for t, r, c in rows])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(lead_days, [r[1] for r in rows], marker="o")
        axes[0].set_title(f"{label} -- RMSE (ensemble mean)")
        axes[0].set_xlabel("Lead time (days)")
        axes[1].plot(lead_days, [r[2] for r in rows], marker="o", color="darkorange")
        axes[1].set_title(f"{label} -- CRPS")
        axes[1].set_xlabel("Lead time (days)")
        fig.suptitle(f"GenCast Mini -- not in paper's Fig.1/Fig.8 headline figures ({n_members}-member ensemble)")
        fig.tight_layout()
        fig.savefig(args.output_dir / f"extra_{label}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        tier2_summary.append((label, None, rows[0][1], rows[-1][1], rows[0][2], rows[-1][2]))

    for label, var in TIER2_ATMOSPHERIC_FAMILIES:
        if var not in eval_targets or var not in predictions:
            log.warning("Tier 2 family %s not found -- skipping", var)
            continue
        log.info("Tier 2 atmospheric family: %s (13 levels, RMSE heatmap only)", label)
        grid_rmse = np.full((len(PRESSURE_LEVELS), n_times), np.nan)
        for li, lvl in enumerate(PRESSURE_LEVELS):
            rows = compute_rmse_crps_curve(var, lvl)
            for t, rmse, crps in rows:
                grid_rmse[li, t] = rmse
            tier2_summary.append((f"{label}@{lvl}hPa", lvl, rows[0][1], rows[-1][1], rows[0][2], rows[-1][2]))
        fig, ax = plt.subplots(figsize=(9, 4.5))
        im = ax.imshow(grid_rmse, aspect="auto", cmap="viridis",
                        extent=[0, lead_days[-1], len(PRESSURE_LEVELS) - 0.5, -0.5], interpolation="nearest")
        ax.set_yticks(range(len(PRESSURE_LEVELS)))
        ax.set_yticklabels(PRESSURE_LEVELS)
        ax.set_ylabel("Pressure (hPa)")
        ax.set_xlabel("Lead time (days)")
        ax.set_title(f"{label} -- RMSE (no paper baseline available for this variable)")
        fig.colorbar(im, ax=ax, label="RMSE (native units)")
        fig.tight_layout()
        fig.savefig(args.output_dir / f"extra_{label}_rmse_by_level.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    summary_path = args.output_dir / "tier2_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variable", "level_hpa", "rmse_first_leadtime", "rmse_last_leadtime", "crps_first_leadtime", "crps_last_leadtime"])
        writer.writerows(tier2_summary)
    log.info("Wrote %s", summary_path)

    log.info("Done. Tier 1 scorecard + Tier 2 extras written to %s", args.output_dir)


if __name__ == "__main__":
    main()
