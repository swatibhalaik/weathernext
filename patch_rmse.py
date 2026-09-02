import re

path = "gencast_mini_run.py"
with open(path) as f:
    content = f.read()

old = '''    fig.suptitle(f"{fig_title} — final lead time")
    out_path = args.output_dir / f"sanity_check_{variable}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.getLogger("gencast").info("Wrote sanity-check plot to %s", out_path)'''

new = '''    fig.suptitle(f"{fig_title} — final lead time")
    out_path = args.output_dir / f"sanity_check_{variable}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.getLogger("gencast").info("Wrote sanity-check plot to %s", out_path)

    # ---- Error map + RMSE-by-lead-time (does this actually show forecast skill,
    # or does it just look like climatology because the color scale spans 220-310K?)
    import numpy as np
    import csv

    log2 = logging.getLogger("gencast")

    error_final = target.isel(time=last_step, missing_dims="ignore") - pred_mean.isel(
        time=last_step, missing_dims="ignore"
    )
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    im2 = ax2.imshow(error_final, origin="lower", cmap="RdBu_r", vmin=-10, vmax=10)
    ax2.set_title(f"{fig_title} — Target minus prediction (final lead time)")
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.colorbar(im2, ax=ax2, fraction=0.03)
    err_path = args.output_dir / f"error_map_{variable}.png"
    fig2.savefig(err_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    log2.info("Wrote error map to %s", err_path)

    n_times = target.sizes.get("time", 1)
    rmse_rows = []
    for t in range(n_times):
        diff = (
            target.isel(time=t, missing_dims="ignore")
            - pred_mean.isel(time=t, missing_dims="ignore")
        )
        rmse = float(np.sqrt((diff ** 2).mean()))
        rmse_rows.append((t, rmse))

    csv_path = args.output_dir / f"rmse_by_lead_time_{variable}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_index", "rmse"])
        writer.writerows(rmse_rows)
    log2.info("Wrote per-lead-time RMSE to %s", csv_path)
    log2.info(
        "RMSE at first lead time: %.3f, at final lead time: %.3f (should generally increase "
        "with lead time -- flat near-zero across all lead times would suggest a leakage bug, "
        "not genuine skill).",
        rmse_rows[0][1], rmse_rows[-1][1],
    )

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.plot([r[0] for r in rmse_rows], [r[1] for r in rmse_rows], marker="o")
    ax3.set_xlabel("Lead time index (12h steps)")
    ax3.set_ylabel("RMSE")
    ax3.set_title(f"{fig_title} — RMSE by lead time")
    rmse_plot_path = args.output_dir / f"rmse_by_lead_time_{variable}.png"
    fig3.savefig(rmse_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    log2.info("Wrote RMSE-by-lead-time plot to %s", rmse_plot_path)'''

if old not in content:
    raise SystemExit("Anchor text not found -- aborting, don't want to corrupt the file.")

content = content.replace(old, new, 1)
with open(path, "w") as f:
    f.write(content)

print("Patched successfully.")
