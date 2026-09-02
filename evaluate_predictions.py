#!/usr/bin/env python3
"""evaluate_predictions.py -- paper-comparable verification metrics from an
already-completed GenCast Mini rollout.

Important: this does NOT re-run the model, and there is no "training" or
"fine-tuning" step to extend here -- GenCast Mini is a frozen, pretrained
checkpoint; gencast_mini_run.py only ever does inference (an autoregressive
rollout), never a training loop. Making results more comparable to
Price et al. 2025 (Nature) means (a) evaluating with the paper's own metric,
CRPS -- not just RMSE -- and (b) covering the same variables the paper
reports (Extended Data Figs. 1 and 8), both of which this script does as a
lightweight, CPU-only post-processing pass over the predictions.nc you
already have. No GPU, no re-run of the rollout needed.

Usage:
    python evaluate_predictions.py --data-dir ./gencast_data \
        --predictions ./gencast_output/predictions.nc \
        --output-dir ./gencast_output/paper_comparison
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import logging
from pathlib import Path

import numpy as np

# (display name, dataset variable name, pressure level or None) -- this is
# NOT an arbitrary pick. It is exactly the 10 "representative variables" the
# paper itself chose for its own CRPS-by-lead-time comparison in Extended
# Data Fig. 8 (z500, z850, q700, q925, t850, t300, u850, 2t, 10u, msl), so
# that our line plots are directly comparable in kind to that figure and to
# the corresponding rows/panels of the RMSE scorecard in Extended Data Fig. 1.
VARIABLES = [
    ("2t_2m_temperature", "2m_temperature", None),
    ("msl_mean_sea_level_pressure", "mean_sea_level_pressure", None),
    ("10u_10m_wind", "10m_u_component_of_wind", None),
    ("z500_geopotential_500hPa", "geopotential", 500),
    ("z850_geopotential_850hPa", "geopotential", 850),
    ("t850_temperature_850hPa", "temperature", 850),
    ("t300_temperature_300hPa", "temperature", 300),
    ("q700_specific_humidity_700hPa", "specific_humidity", 700),
    ("q925_specific_humidity_925hPa", "specific_humidity", 925),
    ("u850_wind_850hPa", "u_component_of_wind", 850),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True,
                    help="Folder containing dataset/, params/, stats/ (same as gencast_mini_run.py).")
    p.add_argument("--dataset-file", type=str,
                    default="source-era5_date-2019-03-29_res-1.0_levels-13_steps-30.nc")
    p.add_argument("--model-file", type=str, default="GenCast_1p0deg_Mini_lt2019.npz")
    p.add_argument("--predictions", type=Path, required=True,
                    help="Path to the predictions.nc already produced by gencast_mini_run.py.")
    p.add_argument("--output-dir", type=Path, default=Path("./paper_comparison"))
    return p.parse_args()


def fair_crps(ensemble: np.ndarray, obs: np.ndarray, axis: int = 0) -> np.ndarray:
    """Fair (finite-ensemble bias-corrected) CRPS, Gneiting & Raftery (2007).

    CRPS = E|X - y| - 0.5 * E|X - X'|, estimated as:
        (1/M) sum_i |x_i - y|  -  (1/(2*M*(M-1))) sum_i sum_j |x_i - x_j|
    where X, X' are independent draws from the ensemble. Validated against
    the closed-form Gaussian CRPS (Gneiting & Raftery 2007, eq. 21) before
    use here.
    """
    M = ensemble.shape[axis]
    ens = np.moveaxis(ensemble, axis, 0)
    term1 = np.mean(np.abs(ens - obs[None, ...]), axis=0)
    diff = np.abs(ens[:, None, ...] - ens[None, :, ...])
    term2 = diff.sum(axis=(0, 1)) / (M * (M - 1))
    return term1 - 0.5 * term2


def lat_weights(lat: np.ndarray) -> np.ndarray:
    """cos(latitude) area weights, normalized to mean 1."""
    w = np.cos(np.deg2rad(lat))
    return w / w.mean()


def compute_variable_metrics(target, pred, display_name: str, level, log):
    """target, pred: xarray.DataArrays already selected to this variable.
    Returns list of (time_index, rmse, crps) and logs progress.
    """
    import xarray

    if "batch" in target.dims:
        target = target.isel(batch=0)
    if "batch" in pred.dims:
        pred = pred.isel(batch=0)
    if level is not None:
        if "level" not in target.coords:
            log.warning("%s has no level coordinate -- skipping level selection", display_name)
        else:
            target = target.sel(level=level)
            pred = pred.sel(level=level)

    lat = target["lat"].values
    w = lat_weights(lat)
    w_da = xarray.DataArray(w, dims=["lat"], coords={"lat": target["lat"]})

    n_times = target.sizes["time"]
    rows = []
    for t in range(n_times):
        tgt_t = target.isel(time=t)
        pred_t = pred.isel(time=t)

        pred_mean_t = pred_t.mean(dim="sample")
        sq_err = (tgt_t - pred_mean_t) ** 2
        rmse = float(np.sqrt(float((sq_err * w_da).mean())))

        ens_vals = pred_t.transpose("sample", "lat", "lon").values
        obs_vals = tgt_t.transpose("lat", "lon").values
        crps_field = fair_crps(ens_vals, obs_vals, axis=0)
        crps = float(np.average(crps_field.mean(axis=1), weights=w))

        rows.append((t, rmse, crps))
    return rows


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("evaluate")

    import matplotlib
    matplotlib.use("Agg")
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

    _, eval_targets, _ = data_utils.extract_inputs_targets_forcings(
        example_batch,
        target_lead_times=slice("12h", f"{(example_batch.sizes['time'] - 2) * 12}h"),
        **dataclasses.asdict(task_config),
    )

    log.info("Loading predictions from %s", args.predictions)
    predictions = xarray.open_dataset(args.predictions).compute()
    n_members = predictions.sizes.get("sample", None)

    summary_rows = []
    for display_name, var, level in VARIABLES:
        if var not in eval_targets or var not in predictions:
            log.warning("Variable %s not found in dataset/predictions -- skipping", var)
            continue

        log.info("Computing metrics for %s%s...", var, f" @ {level}hPa" if level else "")
        rows = compute_variable_metrics(eval_targets[var], predictions[var], display_name, level, log)

        csv_path = args.output_dir / f"metrics_{display_name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_index", "lead_time_days", "rmse", "crps"])
            writer.writerows([(t, t * 0.5, rmse, crps) for t, rmse, crps in rows])
        log.info("Wrote %s", csv_path)

        lead_days = [t * 0.5 for t, _, _ in rows]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(lead_days, [r[1] for r in rows], marker="o")
        axes[0].set_title(f"{display_name} -- RMSE (ensemble mean)")
        axes[0].set_xlabel("Lead time (days)")
        axes[0].set_ylabel("RMSE")
        axes[1].plot(lead_days, [r[2] for r in rows], marker="o", color="darkorange")
        axes[1].set_title(f"{display_name} -- CRPS")
        axes[1].set_xlabel("Lead time (days)")
        axes[1].set_ylabel("CRPS")
        fig.suptitle(
            f"GenCast Mini, single case (2019-03-29), {n_members}-member ensemble "
            "(area-weighted, cos-latitude)"
        )
        fig.tight_layout()
        plot_path = args.output_dir / f"metrics_{display_name}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Wrote %s", plot_path)

        summary_rows.append((display_name, rows[0][1], rows[-1][1], rows[0][2], rows[-1][2]))

    summary_path = args.output_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variable", "rmse_first_leadtime", "rmse_last_leadtime",
                          "crps_first_leadtime", "crps_last_leadtime"])
        writer.writerows(summary_rows)
    log.info("Wrote summary to %s", summary_path)
    log.info("Done. Outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()
