#!/usr/bin/env python3
"""evaluate_fig2_rank_histograms.py -- reproduce the shape of Extended Data
Fig. 2 (rank histograms / Talagrand diagrams) from an already-completed
GenCast Mini rollout.

Extended Data Fig. 2 (p.14 of the paper) shows rank histograms for 5
variables (z500, t850, q700, 2t, 10u) at 5 lead times (12h, 3d, 5d, 10d,
15d), comparing ENS, GenCast-Perturbed and GenCast. A rank histogram checks
ensemble calibration: for every grid point, rank the true observation among
the M ensemble members (giving an integer 0..M), pool that rank over many
grid points/cases, and histogram it. A perfectly calibrated ensemble gives a
flat histogram (every rank equally likely); a U-shape means the ensemble is
under-dispersive (too narrow -- truth falls outside the ensemble spread too
often); skew means systematic bias.

What this script can and cannot reproduce:
  - CAN: the same diagnostic, computed the same way, on our own GenCast Mini
    ensemble vs. the ERA5 target already used throughout this reproduction.
    No ENS, no GenCast-Perturbed -- only "GenCast" (our rollout) is plotted.
  - CANNOT fully match resolution: the paper's ensembles have M=50 members
    (51 possible ranks); GenCast Mini here has M=6 (only 7 possible ranks).
    A 7-bin histogram is intrinsically coarser/noisier-looking than a
    51-bin one and is not directly comparable in smoothness -- only in
    gross shape (flat vs. U-shaped vs. skewed). This is stated on every plot
    and must be stated wherever this figure is used.

Usage:
    python evaluate_fig2_rank_histograms.py --data-dir ./gencast_data \
        --predictions ./gencast_output/predictions.nc \
        --output-dir ./gencast_output/fig2_rank_histograms
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import logging
from pathlib import Path

import numpy as np

# (display name, dataset variable, level or None) -- exactly the 5 variables
# the paper's own Extended Data Fig. 2 uses: z500, t850, q700, 2t, 10u.
RANK_VARS = [
    ("z500", "geopotential", 500),
    ("t850", "temperature", 850),
    ("q700", "specific_humidity", 700),
    ("2t", "2m_temperature", None),
    ("10u", "10m_u_component_of_wind", None),
]

# (label, lead_time_days) -- the paper's own 5 lead times for this figure.
LEAD_TIMES = [
    ("12 hours", 0.5),
    ("3 days", 3.0),
    ("5 days", 5.0),
    ("10 days", 10.0),
    ("15 days", 15.0),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--dataset-file", type=str,
                    default="source-era5_date-2019-03-29_res-1.0_levels-13_steps-30.nc")
    p.add_argument("--model-file", type=str, default="GenCast_1p0deg_Mini_lt2019.npz")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("./fig2_rank_histograms"))
    return p.parse_args()


def rank_of_truth(ensemble: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Pooled-rank (Talagrand) computation.

    ensemble: (M, ...) forecast ensemble members.
    obs: (...) matching truth field, same trailing shape as ensemble.
    Returns an integer array (same trailing shape) giving, at every point,
    the rank of `obs` among the M+1 values {ensemble members, obs} --
    i.e. an integer in [0, M]. Ties are broken by argsort's stable order
    (adequate for continuous physical fields where exact float ties are
    vanishingly rare).
    """
    combined = np.concatenate([ensemble, obs[None, ...]], axis=0)  # (M+1, ...)
    order = np.argsort(combined, axis=0)
    ranks = np.argsort(order, axis=0)  # rank of each entry within combined, 0..M
    return ranks[-1]  # rank assigned to the obs row we appended last


def histogram_from_ranks(ranks: np.ndarray, M: int):
    """ranks: any-shape integer array with values in [0, M].
    Returns (bin_index 0..M, counts, frequency) where frequency is
    normalized so a perfectly flat/uniform histogram reads as 1.0.
    """
    counts = np.bincount(ranks.ravel(), minlength=M + 1)[: M + 1].astype(float)
    total = counts.sum()
    expected_per_bin = total / (M + 1) if total > 0 else 1.0
    frequency = counts / expected_per_bin
    return np.arange(M + 1), counts, frequency


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("fig2")

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
    M = predictions.sizes.get("sample", None)
    if M is None:
        raise SystemExit("predictions.nc has no 'sample' (ensemble) dimension -- cannot compute rank histograms.")
    log.info("Ensemble size M=%d (paper uses M=50 for this figure -- see caveat in file docstring)", M)

    n_times = eval_targets.sizes["time"]

    def time_index_for(lead_days: float) -> int:
        # eval_targets index 0 corresponds to the first target lead time (12h = 0.5 days),
        # stepping by 12h (0.5 days) thereafter, so lead_days = (idx + 1) * 0.5.
        idx = round(lead_days / 0.5) - 1
        return max(0, min(idx, n_times - 1))

    all_rows = []  # (variable, lead_label, lead_days, rank, count, frequency)
    fig, axes = plt.subplots(len(RANK_VARS), len(LEAD_TIMES), figsize=(4 * len(LEAD_TIMES), 3 * len(RANK_VARS)),
                              squeeze=False)

    for vi, (label, var, level) in enumerate(RANK_VARS):
        if var not in eval_targets or var not in predictions:
            log.warning("Variable %s not found -- skipping row", var)
            continue
        for li, (lead_label, lead_days) in enumerate(LEAD_TIMES):
            t = time_index_for(lead_days)
            tgt = eval_targets[var]
            pred = predictions[var]
            if "batch" in tgt.dims:
                tgt = tgt.isel(batch=0)
            if "batch" in pred.dims:
                pred = pred.isel(batch=0)
            if level is not None:
                tgt = tgt.sel(level=level)
                pred = pred.sel(level=level)
            tgt_t = tgt.isel(time=t).transpose("lat", "lon").values
            pred_t = pred.isel(time=t).transpose("sample", "lat", "lon").values

            ranks = rank_of_truth(pred_t, tgt_t)
            bins, counts, freq = histogram_from_ranks(ranks, M)

            for b, c, f in zip(bins, counts, freq):
                all_rows.append((label, lead_label, lead_days, int(b), int(c), float(f)))

            ax = axes[vi][li]
            x_norm = bins / M
            ax.plot(x_norm, freq, marker="o", color="tab:blue", label="GenCast Mini (M=6)")
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
            if vi == 0:
                ax.set_title(lead_label)
            if li == 0:
                ax.set_ylabel(f"{label}\nFrequency")
            if vi == len(RANK_VARS) - 1:
                ax.set_xlabel("Rank (normalized)")
            ax.set_ylim(0, max(2.2, float(np.nanmax(freq)) * 1.1))

    fig.suptitle(
        "GenCast Mini rank histograms (own reproduction) -- cf. Extended Data Fig. 2.\n"
        "M=6 ensemble members here vs. the paper's M=50 -- only 7 possible ranks, "
        "so bins are coarser/noisier by construction, not necessarily worse calibration. "
        "ENS and GenCast-Perturbed are not available in this reproduction and are omitted.",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_png = args.output_dir / "fig2_rank_histograms.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_png)

    csv_path = args.output_dir / "fig2_rank_histogram_data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variable", "lead_time_label", "lead_time_days", "rank", "count", "frequency_normalized"])
        writer.writerows(all_rows)
    log.info("Wrote %s (%d rows)", csv_path, len(all_rows))
    log.info("Done.")


if __name__ == "__main__":
    main()
