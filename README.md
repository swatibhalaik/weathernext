# GenCast Mini — reproduction (PhD paper implementation #1)

Paper: Price, I., Sanchez-Gonzalez, A., Alet, F., Andersson, T.R., El-Kadi, A.,
Masters, D., Ewalds, T., Stott, J., Mohamed, S., Battaglia, P., et al. (2025).
*Probabilistic weather forecasting with machine learning.* Nature, 637(8044).
https://doi.org/10.1038/s41586-024-08252-9 (preprint: arXiv:2312.15796).

Model code: [google-deepmind/weathernext](https://github.com/google-deepmind/weathernext)
(formerly `google-deepmind/graphcast`), pinned at commit
`08cf73625c9d12bd9aaa038868bcb2fe488f2a22` — the last commit before the repo was
restructured into `weathernext/utils` + `weathernext1_gen`/`weathernext1_graph` +
`weathernext2`. Pinning keeps `from graphcast import ...` imports working.

## Files

- `gencast_mini_run.py` — standalone CLI script (no notebook needed). Setup and
  usage instructions are in its header comment. This is the primary path, meant
  for a GPU server reached over SSH.
- `requirements.txt` — pip dependencies (jax GPU build + model code install
  commands are commented at the top; everything else installs normally).
- `gencast_mini_demo.ipynb` — same pipeline as a notebook, adapted to run on
  Kaggle (backup path if the GPU server isn't available).
- `explore_gencast_data.ipynb` — inspects the ERA5 example file (dimensions,
  variables, quick plots). Works on Kaggle or locally.

## Data

Not included in this repo (too large for git). Get it directly from the public
GCS bucket `dm_graphcast` — either onto a Kaggle Dataset, or straight onto a GPU
server with internet access:

```bash
mkdir -p gencast_data/{dataset,params,stats}
cd gencast_data

wget -O dataset/source-era5_date-2019-03-29_res-1.0_levels-13_steps-30.nc \
  "https://storage.googleapis.com/dm_graphcast/gencast/dataset/source-era5_date-2019-03-29_res-1.0_levels-13_steps-30.nc"

wget -O params/GenCast_1p0deg_Mini_lt2019.npz \
  "https://storage.googleapis.com/dm_graphcast/gencast/params/GenCast%201p0deg%20Mini%20%3C2019.npz"

wget -O stats/diffs_stddev_by_level.nc "https://storage.googleapis.com/dm_graphcast/gencast/stats/diffs_stddev_by_level.nc"
wget -O stats/mean_by_level.nc        "https://storage.googleapis.com/dm_graphcast/gencast/stats/mean_by_level.nc"
wget -O stats/stddev_by_level.nc      "https://storage.googleapis.com/dm_graphcast/gencast/stats/stddev_by_level.nc"
wget -O stats/min_by_level.nc         "https://storage.googleapis.com/dm_graphcast/gencast/stats/min_by_level.nc"
```

(~900 MB total: ~669 MB dataset + ~230 MB model params + a few KB of stats.)
Filenames/sizes verified directly against the bucket's listing on 2026-08-22.

## Run (GPU server)

```bash
git clone <this-repo-url>
cd <this-repo>
python3 -m venv gencast-env && source gencast-env/bin/activate
pip install -U "jax[cuda12]"
pip install "git+https://github.com/google-deepmind/weathernext.git@08cf73625c9d12bd9aaa038868bcb2fe488f2a22"
pip install -r requirements.txt
python gencast_mini_run.py --data-dir ./gencast_data --output-dir ./gencast_output
```

Outputs: `gencast_output/predictions.nc` (full ensemble rollout) and a sanity-check
PNG. See `gencast_mini_run.py --help` for options (ensemble size, variable/level
to plot, optional training-step demo).

## Run (Kaggle, backup)

Upload `gencast_data/` as a Kaggle Dataset, attach it to `gencast_mini_demo.ipynb`,
set `KAGGLE_DATASET_SLUG` in the setup cell, Accelerator → GPU, Internet → On, run
top to bottom.
