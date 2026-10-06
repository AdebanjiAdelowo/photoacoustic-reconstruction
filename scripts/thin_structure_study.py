"""Thin-structure diagnostic, phase 1: the existing networks on a width x edge-profile x geometry test set.

After the edge-sharpness intervention the mixed-family networks still fall behind Tikhonov on the
thinnest vessel-like lines with 64 sensors. This script evaluates the ten existing networks (five
control, five intervention) and Tikhonov on line phantoms in which structural width, edge-transition
width and curvature are varied separately (src/thin_structures.py), at 14 dB SNR. Nothing is trained.

The design, hypotheses H1 to H4, thresholds, order of evidence, classification rules and stopping rule are in
report/thin_structure_preregistration.md. Every number that defines the experiment is read from
configs/thin_structure_study.json; none is repeated here. The script refuses to run the evaluation
unless those files, this script and the generator are committed and unmodified, and it records the
commit. It checks the checkpoints against report/checkpoint_manifest.json before and after.

    python scripts/thin_structure_study.py

Outputs (written once, never overwritten): the files listed under "outputs" in the configuration.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.evaluate import psnr, ssim  # noqa: E402
from src.image_statistics import image_statistics  # noqa: E402
from src.stats import bootstrap_mean_ci  # noqa: E402
from src.thin_structures import line_parameters, line_phantom, profile_descriptors  # noqa: E402

CONFIG_PATH = "configs/thin_structure_study.json"
TIKHONOV = "tikhonov"
METRICS = (("psnr", psnr), ("ssim", ssim))


class NotLockedError(RuntimeError):
    """The pre-registered files are not committed, or differ from the commit."""


# --------------------------------------------------------------------------- configuration and test set

def load_config(path=os.path.join(ROOT, CONFIG_PATH)):
    with open(path) as f:
        return json.load(f)


def cell_list(config):
    """Every (geometry, core width, edge sigma) of the factorial grid, in a fixed order."""
    p = config["phantom"]
    excluded = {tuple(c) for c in p["excluded_cells"]}
    return [(g, w, s) for g in p["geometries"] for w in p["core_widths_px"] for s in p["edge_sigmas_px"]
            if (w, s) not in excluded]


def profile_key(core_width, edge_sigma):
    return f"w{core_width:g}_s{edge_sigma:g}"


def cell_key(geometry, core_width, edge_sigma):
    return f"{geometry}_{profile_key(core_width, edge_sigma)}"


def image_seeds(config):
    t = config["test_set"]
    return [t["seed_offset"] + i for i in range(t["n_images"])]


def test_phantoms(config, geometry, core_width, edge_sigma):
    """The n_images phantoms of one cell. The seeds, and so the centrelines and amplitudes, are the
    same in every cell; the same array is used for every sensor count."""
    size = config["phantom"]["grid_size"]
    return np.stack([line_phantom(size, seed, core_width, edge_sigma, geometry) for seed in image_seeds(config)])


def long_line_mask(config):
    """Images in which every line has a chord of at least the configured length."""
    size, limit = config["phantom"]["grid_size"], config["test_set"]["long_line_subset_min_chord_px"]
    return np.array([all(line["chord_length"] >= limit for line in line_parameters(size, seed)) for seed in image_seeds(config)])


def add_noise(clean, relative_std, noise_seed, image_index, n_sensors):
    """Gaussian noise of standard deviation relative_std x RMS of the recording (the project's noise
    model). The standard-normal draw depends only on (noise_seed, image_index, n_sensors): it is the
    same for one image in every cell, and independent of the order of evaluation."""
    clean = np.asarray(clean)
    z = np.random.default_rng([noise_seed, image_index, n_sensors]).standard_normal(clean.shape)
    return clean + (relative_std * float(np.sqrt(np.mean(clean ** 2))) * z).astype(clean.dtype)


def network_names(config, role):
    e = config["evaluation"]
    return [f"{e['groups'][role]}_seed{s}" for s in e["network_seeds"]]


# --------------------------------------------------------------------------- evaluation

def evaluate(config, record, reconstruct, progress=None):
    """Per-image PSNR and SSIM of every method in every cell at every sensor count.

    record(phantoms, n_sensors) -> list of clean recordings, one per phantom.
    reconstruct(recordings, n_sensors) -> {method name: (N, size, size) array}; the names must be
    "tikhonov" and the network names of both groups.
    Returns {f"{cell}__{n_sensors}__{method}__{metric}": (n_images,) array}.
    """
    e = config["evaluation"]
    expected = {TIKHONOV, *network_names(config, "control"), *network_names(config, "intervention")}
    per_image = {}
    for geometry, w, s in cell_list(config):
        phantoms = test_phantoms(config, geometry, w, s)
        for k in e["sensor_counts"]:
            clean = record(phantoms, k)
            Y = [add_noise(c, e["relative_noise_std"], e["noise_seed"], i, k) for i, c in enumerate(clean)]
            outputs = reconstruct(Y, k)
            if set(outputs) != expected:
                raise ValueError(f"reconstruct returned {sorted(outputs)}, expected {sorted(expected)}")
            for method, images in outputs.items():
                for metric, f in METRICS:
                    per_image[f"{cell_key(geometry, w, s)}__{k}__{method}__{metric}"] = np.array(
                        [f(images[i], phantoms[i]) for i in range(len(phantoms))])
        if progress:
            progress(geometry, w, s)
    return per_image


def _values(per_image, cell, k, method, metric="psnr"):
    return per_image[f"{cell_key(*cell)}__{k}__{method}__{metric}"]


def group_values(per_image, config, cell, k, role, metric="psnr"):
    """Per-image mean over the five networks of a group."""
    return np.mean([_values(per_image, cell, k, n, metric) for n in network_names(config, role)], axis=0)


def advantage(per_image, config, cell, k, role):
    """Per-image PSNR advantage over Tikhonov of the mean of a group's networks."""
    return group_values(per_image, config, cell, k, role) - _values(per_image, cell, k, TIKHONOV)


def _ci(values, config):
    b = config["bootstrap"]
    return list(bootstrap_mean_ci(values, n_boot=b["n_boot"], level=b["level"], seed=b["seed"]))


def summarise(per_image, config):
    """One row per cell and sensor count, with absolute and relative quality and every factor."""
    size = config["phantom"]["grid_size"]
    rows = []
    for geometry, w, s in cell_list(config):
        cell = (geometry, w, s)
        stats = [image_statistics(p) for p in test_phantoms(config, *cell)]
        for k in config["evaluation"]["sensor_counts"]:
            row = {"geometry": geometry, "core_width_px": w, "edge_sigma_px": s, "sensors": k,
                   "noise_label": config["evaluation"]["noise_label"], "n_images": config["test_set"]["n_images"],
                   "grid_size": size, "profile": profile_descriptors(w, s),
                   "measured": {"thickness_px": float(np.mean([x["thickness"] for x in stats])),
                                "max_gradient": float(np.mean([x["max_gradient"] for x in stats]))},
                   TIKHONOV: {m: _ci(_values(per_image, cell, k, TIKHONOV, m), config) for m, _ in METRICS}}
            for role in ("control", "intervention"):
                names = network_names(config, role)
                tik = _values(per_image, cell, k, TIKHONOV)
                per_network = {m: [float(_values(per_image, cell, k, n, m).mean()) for n in names] for m, _ in METRICS}
                row[role] = {
                    "group": config["evaluation"]["groups"][role],
                    **{m: _ci(group_values(per_image, config, cell, k, role, m), config) for m, _ in METRICS},
                    **{f"{m}_per_network": per_network[m] for m, _ in METRICS},
                    **{f"{m}_network_sd": float(np.std(per_network[m], ddof=1)) for m, _ in METRICS},
                    "psnr_minus_tikhonov": _ci(advantage(per_image, config, cell, k, role), config),
                    "psnr_minus_tikhonov_per_network": [float((_values(per_image, cell, k, n) - tik).mean()) for n in names],
                }
                row[role]["psnr_minus_tikhonov_network_sd"] = float(np.std(row[role]["psnr_minus_tikhonov_per_network"], ddof=1))
            row["intervention_minus_control"] = _ci(group_values(per_image, config, cell, k, "intervention")
                                                    - group_values(per_image, config, cell, k, "control"), config)
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- pre-registered analysis

def contrast(per_image, config, k, role, plus, minus, mask=None, scale=1.0):
    """`scale` times (mean advantage over the cells `plus` minus that over the cells `minus`), paired
    image by image: estimate with a bootstrap interval over images, and the same contrast for each
    network alone. Also split into the change of the networks' PSNR and the change of Tikhonov's."""
    def mean_over(cells, f):
        return scale * np.mean([f(c) for c in cells], axis=0)
    sel = slice(None) if mask is None else mask
    diff = (mean_over(plus, lambda c: advantage(per_image, config, c, k, role))
            - mean_over(minus, lambda c: advantage(per_image, config, c, k, role)))[sel]
    tik = (mean_over(plus, lambda c: _values(per_image, c, k, TIKHONOV)) - mean_over(minus, lambda c: _values(per_image, c, k, TIKHONOV)))[sel]
    per_network = []
    for n in network_names(config, role):
        d = mean_over(plus, lambda c: _values(per_image, c, k, n)) - mean_over(minus, lambda c: _values(per_image, c, k, n))
        per_network.append(float((d[sel] - tik).mean()))
    est, lo, hi = _ci(diff, config)
    return {"estimate": est, "ci": [lo, hi], "per_network": per_network,
            "network_psnr_change": float(diff.mean() + tik.mean()), "tikhonov_psnr_change": float(tik.mean())}


def _three_way(effects, t):
    """supported: every effect at least the material threshold with its interval above zero;
    rejected: every effect below the negligible threshold; otherwise indeterminate."""
    if all(e["estimate"] >= t["material_effect"] and e["ci"][0] > 0 for e in effects):
        return "supported"
    if all(e["estimate"] < t["negligible_effect"] for e in effects):
        return "rejected"
    return "indeterminate"


def evaluate_hypotheses(per_image, config, mask=None):
    """The pre-registered quantities and verdicts, for each sensor count. The order of the entries is
    the order of evidence: precondition, primary contrasts (H1 to H4), interactions, and last the
    summary classification, which is derived from the others and never overrides them."""
    c, t = config["cells"], config["thresholds_db"]
    thin, thick = tuple(c["thin"]), tuple(c["thick_same_edge"])
    wide = c["edge_sweep_core_width"]
    sel = slice(None) if mask is None else mask
    out = {}
    for k in config["evaluation"]["sensor_counts"]:
        def effect(role, plus, minus):
            return contrast(per_image, config, k, role, plus, minus, mask)

        def level(role, cell):
            return _ci(advantage(per_image, config, cell, k, role)[sel], config)

        h = {}
        # P0: the residual failure is present in this test set (curved thin line, intervention networks)
        p0 = level("intervention", ("curved", *thin))
        h["P0_failure_present"] = {"advantage": p0, "holds": bool(p0[2] < 0)}

        # H1: thinness, the effect of core width at a fixed edge width, for each geometry
        h["H1_thinness"] = {}
        for g in config["phantom"]["geometries"]:
            primary = effect("intervention", [(g, *thick)], [(g, *thin)])
            secondary = {}
            for pair in c["secondary_width_contrasts"]:
                e = effect("intervention", [(g, *pair["thick"])], [(g, *pair["thin"])])
                e["status"] = _three_way([e], t)
                secondary[f"{profile_key(*pair['thick'])}_minus_{profile_key(*pair['thin'])}"] = e
            h["H1_thinness"][g] = {"primary_thick_minus_thin": primary, "status": _three_way([primary], t),
                                   "secondary_fixed_edge": secondary}

        # H2: edge-profile sensitivity at a thick core (positive control)
        valley, dip = {}, {}
        for role in ("control", "intervention"):
            means = {s: level(role, ("straight", wide, s))[0] for s in c["valley_edge_sigmas"] + [c["smooth_edge_sigma"]]}
            valley[role] = min(means[s] for s in c["valley_edge_sigmas"])
            dip[role] = means[c["smooth_edge_sigma"]] - valley[role]
        cell = ("straight", wide, c["paired_edge_sigma"])
        gain_at_paired = _ci((group_values(per_image, config, cell, k, "intervention")
                              - group_values(per_image, config, cell, k, "control"))[sel], config)
        h["H2_edge_profile"] = {
            "valley_advantage": valley, "dip_below_smooth_edge": dip, "valley_gain": valley["intervention"] - valley["control"],
            "paired_gain_at_paired_edge_sigma": gain_at_paired,
            "advantage_at_paired_edge_sigma": {role: level(role, cell)[0] for role in ("control", "intervention")},
            "status": "supported" if (dip["control"] >= t["material_effect"]
                                      and valley["intervention"] - valley["control"] >= t["material_effect"]
                                      and gain_at_paired[1] > 0) else "not reproduced"}

        # H3: geometry, G = straight minus curved at matched profiles (positive: curved lines are worse)
        geometry = {profile_key(*cell): effect("intervention", [("straight", *cell)], [("curved", *cell)])
                    for cell in map(tuple, c["thin_soft_set"])}
        g_thin = geometry[profile_key(*thin)]
        tol = t["geometry_tolerance"]
        if -tol < g_thin["ci"][0] and g_thin["ci"][1] < tol and all(abs(e["estimate"]) < tol for e in geometry.values()):
            status = GEOMETRY_NEGLIGIBLE
        elif g_thin["estimate"] >= t["material_effect"] and g_thin["ci"][0] > 0:
            status = CURVED_WORSE
        elif g_thin["estimate"] <= -t["material_effect"] and g_thin["ci"][1] < 0:
            status = CURVED_BETTER
        else:
            status = GEOMETRY_MODEST
        h["H3_geometry"] = {"straight_minus_curved": geometry, "primary_cell": profile_key(*thin), "status": status}

        # H4: sharp thin bar against sharp bars of trained width
        familiar = [("straight", *b) for b in map(tuple, c["sharp_familiar_bars"])]
        bar = [("straight", *c["sharp_thin_bar"])]
        h["H4_sharp_thin_bar"] = {role: effect(role, familiar, bar) for role in ("control", "intervention")}
        h["H4_sharp_thin_bar"]["status"] = _three_way([h["H4_sharp_thin_bar"]["intervention"]], t)

        # interactions (differences of differences; scale 2 because each side averages two cells)
        (w_bar, s_bar), s_soft = c["sharp_thin_bar"], c["paired_edge_sigma"]
        inter = {
            "width_x_edge": contrast(per_image, config, k, "intervention", [("straight", wide, s_soft), ("straight", w_bar, s_bar)],
                                     [("straight", w_bar, s_soft), ("straight", wide, s_bar)], mask, scale=2.0),
            "width_x_geometry": contrast(per_image, config, k, "intervention", [("curved", *thick), ("straight", *thin)],
                                         [("curved", *thin), ("straight", *thick)], mask, scale=2.0),
        }
        for e in inter.values():
            e["flagged"] = bool(abs(e["estimate"]) >= t["interaction_flag"] and (e["ci"][0] > 0 or e["ci"][1] < 0))
        h["interactions"] = inter

        # descriptive: edge sensitivity of the intervention networks at the thick core
        edge_levels = [level("intervention", ("straight", wide, s))[0] for s in c["edge_range_sigmas"]]
        h["edge_range_at_thick_core"] = float(max(edge_levels) - min(edge_levels))
        h["summary_classification"] = classify(h, t)
        out[str(k)] = h
    return out


GEOMETRY_NEGLIGIBLE = "negligible"
CURVED_WORSE = "curved materially worse"
CURVED_BETTER = "curved materially better (unexpected direction)"
GEOMETRY_MODEST = "modest or indeterminate"


def classify(h, t):
    """The summary classification of the pre-registration for one sensor count. It is computed from
    the primary contrasts and the interactions and is the last and weakest item of evidence: a
    "dominated" label is withheld when a flagged interaction complicates it."""
    if not h["P0_failure_present"]["holds"]:
        return "failure not reproduced: no classification"
    thinness = {g: v["status"] for g, v in h["H1_thinness"].items()}
    geometry, sharp_bar = h["H3_geometry"]["status"], h["H4_sharp_thin_bar"]["status"]
    edge_matters = h["edge_range_at_thick_core"] >= t["material_effect"]
    flagged = [name for name, e in h["interactions"].items() if e["flagged"]]
    not_established = "{} supported, dominance not established"
    if all(s == "supported" for s in thinness.values()) and geometry == GEOMETRY_NEGLIGIBLE and sharp_bar == "supported":
        label = "thinness-dominated" if not flagged else not_established.format("thinness")
    elif geometry == CURVED_WORSE and thinness["straight"] != "supported":
        # a width x geometry interaction is what this pattern is; only width x edge complicates it
        label = "geometry-dominated" if "width_x_edge" not in flagged else not_established.format("geometry")
    elif thinness["straight"] != "supported" and edge_matters and geometry != CURVED_WORSE:
        label = "edge-profile-dominated" if "width_x_edge" not in flagged else not_established.format("edge profile")
    else:
        factors = [name for name, present in (("thinness", thinness["straight"] == "supported"),
                                              ("geometry", geometry == CURVED_WORSE),
                                              ("edge profile", edge_matters)) if present]
        if len(factors) >= 2:
            label = "mixed: " + ", ".join(factors)
        elif factors:
            label = not_established.format(factors[0])
        else:
            label = "inconclusive"
    if flagged:
        label += f" (interaction flagged: {', '.join(flagged)})"
    if geometry == CURVED_BETTER:
        label += " (curved lines materially better: not evidence that curvature causes the failure)"
    if h["H2_edge_profile"]["status"] != "supported":
        label += " (edge axis not validated: H2 not reproduced)"
    return label


# --------------------------------------------------------------------------- guards

def require_locked(config, root=ROOT):
    """The commit at which every locked file is tracked and unmodified; raises otherwise."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    for path in config["locked_files"]:
        if git("ls-files", "--error-unmatch", path).returncode != 0:
            raise NotLockedError(f"{path} is not committed: the design must be committed before the evaluation")
        if git("status", "--porcelain", "--", path).stdout.strip():
            raise NotLockedError(f"{path} differs from the commit: the evaluation runs only on the committed design")
    return git("rev-parse", "HEAD").stdout.strip()


def config_sha256(path=os.path.join(ROOT, CONFIG_PATH)):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# --------------------------------------------------------------------------- outputs

def _f(t, d=2):
    return f"{t[0]:+.{d}f} [{t[1]:+.{d}f}, {t[2]:+.{d}f}]"


def _e(e):
    return f"{e['estimate']:+.2f} [{e['ci'][0]:+.2f}, {e['ci'][1]:+.2f}]"


def write_text(results, path):
    """The report, in the pre-registered order of evidence: primary contrasts, interactions, the
    complete response surface, and last the summary classification."""
    L = ["Thin-structure diagnostic, phase 1 (scripts/thin_structure_study.py)",
         f"Design: {results['preregistration']} at commit {results['design_commit']}",
         "Advantage = PSNR of the mean of the five intervention networks minus PSNR of Tikhonov, 14 dB SNR; 95 % interval over images.",
         "", "== 1. Primary pre-registered contrasts"]
    for k, h in results["hypotheses"].items():
        L += [f"  {k} sensors",
              f"    P0 failure present: {h['P0_failure_present']['holds']} (curved thin line, advantage {_f(h['P0_failure_present']['advantage'])})"]
        for g, v in h["H1_thinness"].items():
            a = v["primary_thick_minus_thin"]
            L.append(f"    H1 thinness, {g}: {v['status']} (thick minus thin at fixed edge {_e(a)}; networks {a['network_psnr_change']:+.2f}, "
                     f"Tikhonov {a['tikhonov_psnr_change']:+.2f}; secondary " + ", ".join(
                         f"{n} {_e(e)} {e['status']}" for n, e in v["secondary_fixed_edge"].items()) + ")")
        e = h["H2_edge_profile"]
        L.append(f"    H2 edge profile: {e['status']} (control dip {e['dip_below_smooth_edge']['control']:+.2f}, valley gain {e['valley_gain']:+.2f}, "
                 f"paired gain {_f(e['paired_gain_at_paired_edge_sigma'])})")
        L.append(f"    H3 geometry, straight minus curved: {h['H3_geometry']['status']} (" + ", ".join(
            f"{c} {_e(v)}" for c, v in h["H3_geometry"]["straight_minus_curved"].items()) + ")")
        b = h["H4_sharp_thin_bar"]
        L.append(f"    H4 sharp thin bar: {b['status']} (trained-width bars minus thin bar: intervention {_e(b['intervention'])}, "
                 f"control {_e(b['control'])})")
    L += ["", "== 2. Interactions (differences of differences of the advantage)"]
    for k, h in results["hypotheses"].items():
        L.append(f"  {k} sensors: " + ", ".join(f"{n} {_e(v)}{' FLAGGED' if v['flagged'] else ''}" for n, v in h["interactions"].items())
                 + f"; edge range at the thick core {h['edge_range_at_thick_core']:.2f} dB")
    L += ["", "== 3. Complete response surface: PSNR in dB (network standard deviation in parentheses)"]
    for r in results["cells"]:
        c, i = r["control"], r["intervention"]
        L.append(f"  {r['geometry']:8s} w {r['core_width_px']:g} s {r['edge_sigma_px']:g} {r['sensors']:2d} sensors: Tikhonov {r[TIKHONOV]['psnr'][0]:.2f} | "
                 f"control {c['psnr'][0]:.2f} ({c['psnr_network_sd']:.2f}) | intervention {i['psnr'][0]:.2f} ({i['psnr_network_sd']:.2f}) | "
                 f"control - Tik {_f(c['psnr_minus_tikhonov'])} | intervention - Tik {_f(i['psnr_minus_tikhonov'])} | "
                 f"intervention - control {_f(r['intervention_minus_control'])} | SSIM {r[TIKHONOV]['ssim'][0]:.3f} / {c['ssim'][0]:.3f} / {i['ssim'][0]:.3f}")
    L += ["", "== 4. Summary classification (derived from sections 1 and 2; it does not override them)"]
    for k, h in results["hypotheses"].items():
        L.append(f"  {k} sensors: {h['summary_classification']}")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def plot(results, config, path):
    """Width x edge maps, so that interactions are visible: rows are Tikhonov PSNR, intervention PSNR
    and their difference; columns are geometry and sensor count."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = config["phantom"]
    W, S = p["core_widths_px"], p["edge_sigmas_px"]
    panels = [(g, k) for g in p["geometries"] for k in config["evaluation"]["sensor_counts"]]
    quantities = (("Tikhonov PSNR [dB]", lambda r: r[TIKHONOV]["psnr"][0], "viridis"),
                  ("intervention PSNR [dB]", lambda r: r["intervention"]["psnr"][0], "viridis"),
                  ("intervention minus Tikhonov [dB]", lambda r: r["intervention"]["psnr_minus_tikhonov"][0], "RdBu"))
    fig, axes = plt.subplots(len(quantities), len(panels), figsize=(3.6 * len(panels), 3.2 * len(quantities)), squeeze=False)
    for i, (title, value, cmap) in enumerate(quantities):
        grids = {}
        for g, k in panels:
            grid = np.full((len(W), len(S)), np.nan)
            for r in results["cells"]:
                if r["geometry"] == g and r["sensors"] == k:
                    grid[W.index(r["core_width_px"]), S.index(r["edge_sigma_px"])] = value(r)
            grids[(g, k)] = grid
        lim = np.nanmax(np.abs(list(grids.values()))) if cmap == "RdBu" else None
        lo = -lim if lim is not None else np.nanmin(list(grids.values()))
        hi = lim if lim is not None else np.nanmax(list(grids.values()))
        for j, (g, k) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(grids[(g, k)], origin="lower", cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
            for a in range(len(W)):
                for b in range(len(S)):
                    if np.isfinite(grids[(g, k)][a, b]):
                        ax.text(b, a, f"{grids[(g, k)][a, b]:.1f}", ha="center", va="center", fontsize=7)
            ax.set_xticks(range(len(S)), [f"{s:g}" for s in S])
            ax.set_yticks(range(len(W)), [f"{w:g}" for w in W])
            ax.set_title(f"{title}\n{g}, {k} sensors", fontsize=8)
            if i == len(quantities) - 1:
                ax.set_xlabel("edge sigma [px]")
            if j == 0:
                ax.set_ylabel("core width [px]")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- the evaluation itself

def main():
    os.chdir(ROOT)
    t0 = time.time()
    config = load_config()
    design_commit = require_locked(config)
    for path in config["outputs"]:
        if os.path.exists(path):
            raise SystemExit(f"{path} exists: the pre-registered evaluation is run once and is not overwritten")

    import scripts.mixed_phantom_experiment as M
    import scripts.unrolled_evaluation as U
    from scripts.checkpoint_manifest import verify

    e = config["evaluation"]
    with open(e["checkpoint_manifest"]) as f:
        manifest = json.load(f)
    problems = verify(manifest, e["checkpoint_dir"])
    if problems:
        raise SystemExit("checkpoints do not match the manifest:\n  " + "\n  ".join(problems))
    with open("report/mixed_phantom_results.json") as f:
        selected = json.load(f)["tikhonov_weights_selected_on_mixed160"]
    mu = {int(k): v for k, v in e["tikhonov_mu"].items()}
    assert all(selected[f"{e['noise_label']}|{k}"] == mu[k] for k in e["sensor_counts"]), "Tikhonov weights differ from their source"
    assert dict(U.NOISE_LEVELS)[e["noise_label"]] == e["relative_noise_std"] and config["phantom"]["grid_size"] == U.GRID_SIZE

    setup = M.Setup()
    names = network_names(config, "control") + network_names(config, "intervention")
    models = {n: U.load_model(f"{e['checkpoint_dir']}/{n}.pt", setup.device)[0] for n in names}

    def record(phantoms, k):
        return setup.recordings({"phantom": phantoms, "n_sensors": np.full(len(phantoms), k, np.int32)})

    def reconstruct(Y, k):
        n_s = np.full(len(Y), k, np.int32)
        out = U.reconstruct(models, setup.physics, Y, n_s)
        out[TIKHONOV] = M.tikhonov(setup, Y, n_s, lambda kk: mu[int(kk)])
        return out

    per_image = evaluate(config, record, reconstruct,
                         progress=lambda g, w, s: print(f"[{time.time() - t0:.0f}s] {cell_key(g, w, s)} done", flush=True))
    problems = verify(manifest, e["checkpoint_dir"])
    assert not problems, f"a checkpoint changed during the evaluation: {problems}"

    mask = long_line_mask(config)
    results = {"study": config["study"], "preregistration": config["preregistration"], "design_commit": design_commit,
               "config_sha256": config_sha256(), "config": config,
               "image_seeds": image_seeds(config), "long_line_subset": mask.tolist(),
               "evidence_order": ["primary contrasts (P0, H1 to H4)", "interactions", "complete response surface (cells)",
                                  "summary_classification, which never overrides the items before it"],
               "cells": summarise(per_image, config),
               "hypotheses": evaluate_hypotheses(per_image, config),
               "hypotheses_long_line_subset": evaluate_hypotheses(per_image, config, mask)}
    out = config["outputs"]
    np.savez_compressed(out[2], image_seeds=np.array(image_seeds(config)), **per_image)   # the raw scores first
    with open(out[0], "w") as f:
        json.dump(results, f, indent=2)
    write_text(results, out[1])
    plot(results, config, out[3])
    print(f"[{time.time() - t0:.0f}s] wrote {', '.join(out)}")


if __name__ == "__main__":
    main()
