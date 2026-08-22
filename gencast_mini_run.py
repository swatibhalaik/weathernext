#!/usr/bin/env python3
"""GenCast Mini — standalone script version (for running on a GPU server, not a notebook).

Reproduces the same pipeline as DeepMind's `gencast_mini_demo.ipynb`, and the
Kaggle-adapted version of it, but as a plain CLI script: load a GenCast Mini
checkpoint, load an ERA5 example batch, run an autoregressive ensemble rollout,
and write the predictions + a few sanity-check plots to disk.

Paper: Price et al. (2025), "Probabilistic weather forecasting with machine
learning", Nature 637. https://doi.org/10.1038/s41586-024-08252-9

--------------------------------------------------------------------------
SETUP (run once on the server)
--------------------------------------------------------------------------
# 1. A fresh virtualenv/conda env is recommended.
python3 -m venv gencast-env
source gencast-env/bin/activate

# 2. GPU-enabled JAX. This server has an H200 (Hopper, same CUDA family as
#    A100/H100) -> the cuda12 wheel is correct.
pip install -U "jax[cuda12]"

# 3. Model code, pinned to the same commit the Kaggle notebooks use, so the
#    `from graphcast import ...` imports below are guaranteed to match (the
#    upstream repo was renamed/restructured to google-deepmind/weathernext
#    in Aug 2026 -- pinning avoids that breaking these imports).
pip install "git+https://github.com/google-deepmind/weathernext.git@08cf73625c9d12bd9aaa038868bcb2fe488f2a22"

# 4. Everything else this script needs.
pip install xarray netCDF4 h5netcdf matplotlib numpy dm-haiku

--------------------------------------------------------------------------
DATA
--------------------------------------------------------------------------
Copy your `gencast` data folder onto this server (same one used for Kaggle):
    <data-dir>/dataset/source-era5_date-2019-03-29_res-1.0_levels-13_steps-30.nc
    <data-dir>/params/GenCast_1p0deg_Mini_lt2019.npz
    <data-dir>/stats/diffs_stddev_by_level.nc
    <data-dir>/stats/mean_by_level.nc
    <data-dir>/stats/stddev_by_level.nc
    <data-dir>/stats/min_by_level.nc

--------------------------------------------------------------------------
RUN
--------------------------------------------------------------------------
python gencast_mini_run.py --data-dir /path/to/gencast --output-dir ./out

For a long run over SSH, use nohup / tmux / screen so it survives a dropped
connection, e.g.:
    tmux new -s gencast
    python gencast_mini_run.py --data-dir /path/to/gencast --output-dir ./out
    # Ctrl-b d to detach, `tmux attach -t gencast` to come back.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run GenCast Mini inference (autoregressive ensemble rollout) "
        "from the command line.",
    )
    p.add_argument(
        "--data-dir", type=Path, required=True,
        help="Folder containing dataset/, params/, stats/ subfolders.",
    )
    p.add_argument(
        "--dataset-file", type=str,
        default="source-era5_date-2019-03-29_res-1.0_levels-13_steps-30.nc",
        help="Filename under <data-dir>/dataset/.",
    )
    p.add_argument(
        "--model-file", type=str,
        default="GenCast_1p0deg_Mini_lt2019.npz",
        help="Filename under <data-dir>/params/.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("./gencast_output"),
        help="Where to write predictions.nc and sanity-check PNGs.",
    )
    p.add_argument(
        "--num-ensemble-members", type=int, default=8,
        help="Ensemble size. Should be a multiple of the number of local JAX "
        "devices. GenCast Mini's memory footprint is small, so this is mostly "
        "limited by wall-clock time, not GPU memory, even on a single GPU.",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Base PRNG seed (each ensemble member folds in its index).",
    )
    p.add_argument(
        "--plot-variable", type=str, default="2m_temperature",
        help="Variable to render in the sanity-check PNGs.",
    )
    p.add_argument(
        "--plot-level", type=int, default=500,
        help="Pressure level (hPa) to use if --plot-variable has a level dimension.",
    )
    p.add_argument(
        "--no-plots", action="store_true",
        help="Skip writing PNG sanity-check plots (still writes predictions.nc).",
    )
    p.add_argument(
        "--demo-training-step", action="store_true",
        help="Also run one loss + gradient computation (mirrors the notebook's "
        "optional 'Train the model' section). Off by default.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("gencast")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Imports that need JAX/graphcast already installed -----------------
    # (imported here, not at module top, so --help works even before setup)
    import haiku as hk
    import jax
    import matplotlib
    matplotlib.use("Agg")  # headless server, no display
    import matplotlib.pyplot as plt
    import xarray

    from graphcast import autoregressive  # noqa: F401  (imported for parity/completeness)
    from graphcast import checkpoint
    from graphcast import data_utils
    from graphcast import denoiser
    from graphcast import gencast
    from graphcast import nan_cleaning
    from graphcast import normalization
    from graphcast import rollout
    from graphcast import xarray_jax
    from graphcast import xarray_tree

    log.info("JAX version: %s", jax.__version__)
    log.info("JAX devices: %s", jax.devices())

    data_dir: Path = args.data_dir
    dataset_path = data_dir / "dataset" / args.dataset_file
    model_path = data_dir / "params" / args.model_file
    stats_dir = data_dir / "stats"

    required_files = [
        dataset_path,
        model_path,
        stats_dir / "diffs_stddev_by_level.nc",
        stats_dir / "mean_by_level.nc",
        stats_dir / "stddev_by_level.nc",
        stats_dir / "min_by_level.nc",
    ]
    missing = [f for f in required_files if not f.exists()]
    if missing:
        log.error("Missing required files:\n%s", "\n".join(map(str, missing)))
        sys.exit(1)

    # ---- Load checkpoint -----------------------------------------------------
    log.info("Loading checkpoint from %s", model_path)
    with open(model_path, "rb") as f:
        ckpt = checkpoint.load(f, gencast.CheckPoint)

    params = ckpt.params
    state: dict = {}
    task_config = ckpt.task_config
    sampler_config = ckpt.sampler_config
    noise_config = ckpt.noise_config
    noise_encoder_config = ckpt.noise_encoder_config
    denoiser_architecture_config = ckpt.denoiser_architecture_config
    log.info("Model description: %s", ckpt.description)

    # ---- Load example data ----------------------------------------------------
    log.info("Loading example dataset from %s", dataset_path)
    example_batch = xarray.open_dataset(dataset_path).compute()
    assert example_batch.dims["time"] >= 3, "Need >=2 input steps and >=1 target step."
    log.info("Dataset dimensions: %s", dict(example_batch.sizes))

    train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
        example_batch, target_lead_times=slice("12h", "12h"),
        **dataclasses.asdict(task_config),
    )
    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        example_batch,
        target_lead_times=slice("12h", f"{(example_batch.dims['time'] - 2) * 12}h"),
        **dataclasses.asdict(task_config),
    )
    log.info("Eval inputs:   %s", eval_inputs.dims.mapping)
    log.info("Eval targets:  %s", eval_targets.dims.mapping)
    log.info("Eval forcings: %s", eval_forcings.dims.mapping)

    # ---- Load normalization stats ----------------------------------------------
    log.info("Loading normalization stats from %s", stats_dir)
    diffs_stddev_by_level = xarray.load_dataset(stats_dir / "diffs_stddev_by_level.nc").compute()
    mean_by_level = xarray.load_dataset(stats_dir / "mean_by_level.nc").compute()
    stddev_by_level = xarray.load_dataset(stats_dir / "stddev_by_level.nc").compute()
    min_by_level = xarray.load_dataset(stats_dir / "min_by_level.nc").compute()

    # ---- Build jitted functions -------------------------------------------------
    def construct_wrapped_gencast():
        predictor = gencast.GenCast(
            sampler_config=sampler_config,
            task_config=task_config,
            denoiser_architecture_config=denoiser_architecture_config,
            noise_config=noise_config,
            noise_encoder_config=noise_encoder_config,
        )
        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=diffs_stddev_by_level,
            mean_by_level=mean_by_level,
            stddev_by_level=stddev_by_level,
        )
        predictor = nan_cleaning.NaNCleaner(
            predictor=predictor,
            reintroduce_nans=True,
            fill_value=min_by_level,
            var_to_clean="sea_surface_temperature",
        )
        return predictor

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        predictor = construct_wrapped_gencast()
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    @hk.transform_with_state
    def loss_fn(inputs, targets, forcings):
        predictor = construct_wrapped_gencast()
        loss, diagnostics = predictor.loss(inputs, targets, forcings)
        return xarray_tree.map_structure(
            lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
            (loss, diagnostics),
        )

    def grads_fn(params, state, inputs, targets, forcings):
        def _aux(params, state, i, t, f):
            (loss, diagnostics), next_state = loss_fn.apply(
                params, state, jax.random.PRNGKey(0), i, t, f
            )
            return loss, (diagnostics, next_state)

        (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
            _aux, has_aux=True
        )(params, state, inputs, targets, forcings)
        return loss, diagnostics, next_state, grads

    if params is None:
        log.info("No checkpoint params -- initializing random weights.")
        init_jitted = jax.jit(loss_fn.init)
        params, state = init_jitted(
            rng=jax.random.PRNGKey(0),
            inputs=train_inputs,
            targets=train_targets,
            forcings=train_forcings,
        )

    loss_fn_jitted = jax.jit(
        lambda rng, i, t, f: loss_fn.apply(params, state, rng, i, t, f)[0]
    )
    grads_fn_jitted = jax.jit(grads_fn)
    run_forward_jitted = jax.jit(
        lambda rng, i, t, f: run_forward.apply(params, state, rng, i, t, f)[0]
    )
    run_forward_pmap = xarray_jax.pmap(run_forward_jitted, dim="sample")

    # ---- Autoregressive rollout --------------------------------------------------
    n_devices = len(jax.local_devices())
    log.info("Local devices: %d", n_devices)
    if args.num_ensemble_members % max(n_devices, 1) != 0:
        log.warning(
            "num_ensemble_members (%d) is not a multiple of the device count (%d); "
            "GenCast's pmapped rollout expects it to be.",
            args.num_ensemble_members, n_devices,
        )

    rng = jax.random.PRNGKey(args.seed)
    rngs = np.stack(
        [jax.random.fold_in(rng, i) for i in range(args.num_ensemble_members)], axis=0
    )

    log.info(
        "Starting autoregressive rollout: %d ensemble member(s)...",
        args.num_ensemble_members,
    )
    t0 = time.time()
    chunks = []
    for chunk in rollout.chunked_prediction_generator_multiple_runs(
        predictor_fn=run_forward_pmap,
        rngs=rngs,
        inputs=eval_inputs,
        targets_template=eval_targets * np.nan,
        forcings=eval_forcings,
        num_steps_per_chunk=1,
        num_samples=args.num_ensemble_members,
        pmap_devices=jax.local_devices(),
    ):
        chunks.append(chunk)
    predictions = xarray.combine_by_coords(chunks)
    log.info("Rollout finished in %.1f s", time.time() - t0)

    predictions_path = args.output_dir / "predictions.nc"
    predictions.to_netcdf(predictions_path)
    log.info("Wrote predictions to %s", predictions_path)

    # ---- Optional sanity-check plots ------------------------------------------------
    if not args.no_plots:
        _write_sanity_plots(
            args, predictions, eval_targets, args.plot_variable, args.plot_level,
        )

    # ---- Optional training-step demo -------------------------------------------------
    if args.demo_training_step:
        log.info("Running one demo loss + gradient computation...")
        loss, diagnostics = loss_fn_jitted(
            jax.random.PRNGKey(0), train_inputs, train_targets, train_forcings
        )
        log.info("Loss: %s", float(loss))
        loss, diagnostics, next_state, grads = grads_fn_jitted(
            params=params, state=state,
            inputs=train_inputs, targets=train_targets, forcings=train_forcings,
        )
        mean_grad = np.mean(
            jax.tree_util.tree_flatten(
                jax.tree_util.tree_map(lambda x: np.abs(x).mean(), grads)
            )[0]
        )
        log.info("Loss: %.4f, Mean |grad|: %.6f", loss, mean_grad)

    log.info("Done. Outputs in %s", args.output_dir)


def _write_sanity_plots(args, predictions, eval_targets, variable: str, level: int) -> None:
    import matplotlib.pyplot as plt

    def select(data, name, lvl=None):
        d = data[name]
        if "batch" in d.dims:
            d = d.isel(batch=0)
        if lvl is not None and "level" in d.coords:
            d = d.sel(level=lvl)
        return d

    fig_title = variable
    has_level = "level" in predictions[variable].coords
    if has_level:
        fig_title += f" at {level} hPa"

    target = select(eval_targets, variable, level if has_level else None)
    pred_mean = select(predictions.mean(dim=["sample"]), variable, level if has_level else None)

    last_step = target.sizes.get("time", 1) - 1
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, title in (
        (axes[0], target, "Target"),
        (axes[1], pred_mean, "Ensemble mean prediction"),
    ):
        im = ax.imshow(
            data.isel(time=last_step, missing_dims="ignore"),
            origin="lower", cmap="viridis",
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.03)
    fig.suptitle(f"{fig_title} — final lead time")
    out_path = args.output_dir / f"sanity_check_{variable}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.getLogger("gencast").info("Wrote sanity-check plot to %s", out_path)


if __name__ == "__main__":
    main()
