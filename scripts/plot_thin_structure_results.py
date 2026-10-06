"""Figures and a flat table from the results of the thin-structure study. Reads
report/thin_structure_results.json and computes nothing new: every plotted value and interval is
taken from that file as written by scripts/thin_structure_study.py.

    python scripts/plot_thin_structure_results.py

Outputs: report/thin_structure_response_surface.csv (all 96 rows), and the figures
report/thin_structure_{width_response,edge_response,geometry,heatmaps}.png. Within a figure every
panel shares one axis range, and the colour scale of the maps is symmetric about zero.
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "report/thin_structure_results.json")
ROLES = (("control", "Control networks"), ("intervention", "Intervention networks"))
ROLE_COLOUR = {"control": "#2a78d6", "intervention": "#eb6834"}
ORDERED = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b")       # one hue, light to dark, for ordered levels
MARKERS = ("o", "s", "^", "D", "v")
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
DIVERGING = LinearSegmentedColormap.from_list("behind_ahead", ["#eb6834", "#f0efec", "#2a78d6"])
ADVANTAGE = "PSNR minus Tikhonov [dB]"


def _style(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def load():
    with open(RESULTS) as f:
        results = json.load(f)
    cells = {(c["geometry"], c["core_width_px"], c["edge_sigma_px"], c["sensors"]): c for c in results["cells"]}
    return results, cells


def write_csv(results, path):
    columns = ["geometry", "core_width_px", "edge_sigma_px", "sensors", "n_images", "fwhm_px", "tikhonov_psnr", "control_psnr",
               "intervention_psnr", "control_psnr_network_sd", "intervention_psnr_network_sd", "control_minus_tikhonov",
               "control_minus_tikhonov_lo", "control_minus_tikhonov_hi", "intervention_minus_tikhonov",
               "intervention_minus_tikhonov_lo", "intervention_minus_tikhonov_hi", "intervention_minus_control",
               "intervention_minus_control_lo", "intervention_minus_control_hi", "tikhonov_ssim", "control_ssim", "intervention_ssim"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(columns)
        for c in results["cells"]:
            ctl, itv = c["control"], c["intervention"]
            writer.writerow([c["geometry"], c["core_width_px"], c["edge_sigma_px"], c["sensors"], c["n_images"], f"{c['profile']['fwhm']:.4f}"]
                            + [f"{v:.4f}" for v in (c["tikhonov"]["psnr"][0], ctl["psnr"][0], itv["psnr"][0], ctl["psnr_network_sd"],
                                                    itv["psnr_network_sd"], *ctl["psnr_minus_tikhonov"], *itv["psnr_minus_tikhonov"],
                                                    *c["intervention_minus_control"], c["tikhonov"]["ssim"][0], ctl["ssim"][0], itv["ssim"][0])])


def response(results, cells, path, x_name):
    """Advantage against one factor with one line per level of the other, straight lines only."""
    p = results["config"]["phantom"]
    W, S, sensors = p["core_widths_px"], p["edge_sigmas_px"], results["config"]["evaluation"]["sensor_counts"]
    xs, levels = (W, S) if x_name == "width" else (S, W)
    fig, axes = plt.subplots(len(sensors), 2, figsize=(10, 7.4), sharex=True, sharey=True)
    for i, k in enumerate(sensors):
        for j, (role, title) in enumerate(ROLES):
            ax = axes[i, j]
            _style(ax)
            ax.axhline(0, color=INK, lw=0.9)
            for level, colour, marker in zip(levels, ORDERED, MARKERS):
                pts = [(x, cells[("straight", *((x, level) if x_name == "width" else (level, x)), k)][role]["psnr_minus_tikhonov"])
                       for x in xs if ("straight", *((x, level) if x_name == "width" else (level, x)), k) in cells]
                x, v = [q[0] for q in pts], np.array([q[1] for q in pts])
                ax.errorbar(x, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], color=colour, marker=marker, ms=5, lw=2, capsize=2,
                            label=f"{'edge' if x_name == 'width' else 'core'} width {level:g} px")
            ax.set_title(f"{title}, {k} sensors", fontsize=10, color=INK)
            if i == len(sensors) - 1:
                ax.set_xlabel("core width w [px]" if x_name == "width" else "edge width s [px]", color=INK)
            if j == 0:
                ax.set_ylabel(ADVANTAGE, color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=9)
    what = "core width at fixed edge width" if x_name == "width" else "edge width at fixed core width"
    fig.suptitle(f"Advantage over Tikhonov against {what}: straight lines, 14 dB SNR, 50 images per point, 95 % interval",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def geometry(results, cells, path):
    sensors = results["config"]["evaluation"]["sensor_counts"]
    profiles = sorted({(w, s) for _, w, s, _ in cells})
    values = [cells[(g, w, s, k)][r]["psnr_minus_tikhonov"][0] for g in ("straight", "curved") for w, s in profiles for k in sensors for r, _ in ROLES]
    lo, hi = np.floor(min(values)) - 1, np.ceil(max(values)) + 1
    fig, axes = plt.subplots(1, len(sensors), figsize=(10, 5.6), sharex=True, sharey=True)
    for ax, k in zip(axes, sensors):
        _style(ax)
        ax.fill_between([lo, hi], [lo - 1, hi - 1], [lo + 1, hi + 1], color=GRID, lw=0, label="within 1 dB")
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, label="equal")
        for role, title in ROLES:
            ax.scatter([cells[("straight", w, s, k)][role]["psnr_minus_tikhonov"][0] for w, s in profiles],
                       [cells[("curved", w, s, k)][role]["psnr_minus_tikhonov"][0] for w, s in profiles],
                       s=34, color=ROLE_COLOUR[role], edgecolor="#fcfcfb", lw=0.8, marker="o" if role == "control" else "s",
                       label=title, zorder=3)
        ax.set_xlim(lo, hi), ax.set_ylim(lo, hi), ax.set_aspect("equal")
        ax.set_title(f"{k} sensors", fontsize=10, color=INK)
        ax.set_xlabel(f"straight lines: {ADVANTAGE}", color=INK)
    axes[0].set_ylabel(f"curved lines: {ADVANTAGE}", color=INK)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Straight against curved lines at the same profile: one point per (core width, edge width), 14 dB SNR", fontsize=10, color=INK)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.11, top=0.9, wspace=0.08)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def heatmaps(results, cells, path):
    p = results["config"]["phantom"]
    W, S, sensors = p["core_widths_px"], p["edge_sigmas_px"], results["config"]["evaluation"]["sensor_counts"]
    rows = [(k, g) for k in sensors for g in p["geometries"]]
    lim = max(abs(c[r]["psnr_minus_tikhonov"][0]) for c in cells.values() for r, _ in ROLES)
    fig, axes = plt.subplots(len(rows), 2, figsize=(8.6, 3.0 * len(rows)), squeeze=False)
    for i, (k, g) in enumerate(rows):
        for j, (role, title) in enumerate(ROLES):
            grid = np.full((len(W), len(S)), np.nan)
            for a, w in enumerate(W):
                for b, s in enumerate(S):
                    if (g, w, s, k) in cells:
                        grid[a, b] = cells[(g, w, s, k)][role]["psnr_minus_tikhonov"][0]
            ax = axes[i, j]
            image = ax.imshow(grid, origin="lower", cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
            for a in range(len(W)):
                for b in range(len(S)):
                    if np.isfinite(grid[a, b]):
                        ax.text(b, a, f"{grid[a, b]:+.1f}", ha="center", va="center", fontsize=8, color=INK)
            ax.set_xticks(range(len(S)), [f"{s:g}" for s in S])
            ax.set_yticks(range(len(W)), [f"{w:g}" for w in W])
            ax.tick_params(colors=MUTED, labelsize=8, length=0)
            for side in ax.spines.values():
                side.set_visible(False)
            ax.set_title(f"{title}: {g}, {k} sensors", fontsize=9, color=INK)
            if i == len(rows) - 1:
                ax.set_xlabel("edge width s [px]", color=INK)
            if j == 0:
                ax.set_ylabel("core width w [px]", color=INK)
    fig.suptitle(f"{ADVANTAGE} at 14 dB SNR (orange: behind Tikhonov, blue: ahead)", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 0.86, 0.975))
    fig.colorbar(image, cax=fig.add_axes((0.88, 0.25, 0.018, 0.5)), label=ADVANTAGE)
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    results, cells = load()
    out = os.path.join(ROOT, "report")
    write_csv(results, os.path.join(out, "thin_structure_response_surface.csv"))
    response(results, cells, os.path.join(out, "thin_structure_width_response.png"), "width")
    response(results, cells, os.path.join(out, "thin_structure_edge_response.png"), "edge")
    geometry(results, cells, os.path.join(out, "thin_structure_geometry.png"))
    heatmaps(results, cells, os.path.join(out, "thin_structure_heatmaps.png"))
    print("wrote the response-surface table and four figures to report/")
