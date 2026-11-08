#!/usr/bin/env python3
# Supplementary file (Online Resource 1) for:
# "Modelling Trustworthy Prescriptive Learning Analytics Dashboards for Student Decision Support in Resource-Constrained Educational Settings Using Human-Centered AI Approaches",
# Education and Information Technologies
# Author, affiliation, contact details and archive DOI: [withheld for blind review]
# Version 1.0.0, licence CC BY 4.0
"""
hc_tplad_analysis.py  (Online Resource 1)

Reproduces every number reported in the text and Tables 6-10 of:

    "Modelling Trustworthy Prescriptive Learning Analytics Dashboards for Student
     Decision Support in Resource-Constrained Educational Settings Using
     Human-Centered AI Approaches"

The script implements the three rival aggregation hypotheses of Section 2.4:
    H1 compensatory  D   = sum_i w_i x_i                                   Eq. (8)
    H2 gated         D_G = min(1, C/c0) * sum_{i<=6} w_i x_i / (1 - mu)    Eq. (11)
    H3 geometric     D   = prod_i x_i ** w_i                               Eq. (12)
with the construct order  T, E, U, P, SRL, F, C  (C = context sensitivity,
weight mu).  ALL construct values are ASSUMED (no empirical data are used).

Usage
-----
    python hc_tplad_analysis.py                    # writes hc_tplad_results.json
    python hc_tplad_analysis.py --out other.json

Randomness: numpy.random.default_rng(SEED) with SEED = 2026, re-seeded for each
Dirichlet concentration (Section 2.6).  Library versions are written into the
output file.

The final block compares each computed value with the value printed in the
manuscript and stores the result under the key "verification".
"""
import argparse
import json
import platform
import sys
from itertools import combinations

import numpy as np
import scipy
from scipy import integrate

# ----------------------------------------------------------------- publication details
ARTICLE_TITLE = ("Modelling Trustworthy Prescriptive Learning Analytics Dashboards for Student "
                 "Decision Support in Resource-Constrained Educational Settings Using "
                 "Human-Centered AI Approaches")
JOURNAL = "Education and Information Technologies"
AUTHOR = "[withheld for blind review]"
AFFILIATION = "[withheld for blind review]"
CORRESPONDING_EMAIL = "[withheld for blind review]"
ARCHIVE_DOI = "[withheld for blind review]"
ARCHIVE_VERSION = "1.0.0"
LICENCE = "CC BY 4.0"

# ----------------------------------------------------------------- constants
SEED = 2026
N_DRAWS = 100_000
CONSTRUCTS = ["T", "E", "U", "P", "SRL", "F", "C"]
N = len(CONSTRUCTS)
EQUAL_W = np.full(N, 1.0 / N)
CONCENTRATIONS = [0.5, 1, 2, 5, 10]
CSTAR_VALUES = [0.3, 0.5, 0.7]

# Table 6 - assumed scenario values (order T, E, U, P, SRL, F, C)
SCENARIOS = {
    "Worst Case":    [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
    "Low-Resource":  [0.5, 0.5, 0.4, 0.4, 0.4, 0.5, 0.3],
    "Baseline":      [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
    "High Resource": [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
    "Optimised":     [0.8, 0.6, 0.6, 0.6, 0.6, 0.7, 0.8],
}
# Profiles constructed to separate the hypotheses (Section 2.4 and Table 9)
DISCRIMINATING = {
    "Profile A (six design constructs 0.9, C = 0.1)": [0.9] * 6 + [0.1],
    "Profile B (six design constructs 0.5, C = 1.0)": [0.5] * 6 + [1.0],
    "Profile D (six design constructs 1.0, C = 0.0)": [1.0] * 6 + [0.0],
}
# Fixed profile used for Figs. 5 and 6 (Section 3.3)
FIXED_PROFILE = [0.9, 0.1, 0.5, 0.5, 0.5, 0.2, 0.8]


# ----------------------------------------------------------------- the model
def d_h1(x, w=EQUAL_W):
    """Eq. (8). x: (..., 7); w: (7,) or (..., 7)."""
    return np.sum(np.asarray(w) * np.asarray(x), axis=-1)


def d_h2(x, cstar, w=EQUAL_W):
    """Eq. (11): gated form; weights of the six design constructs renormalised by 1 - mu."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    mu = w[..., 6]
    design = np.sum(w[..., :6] * x[..., :6], axis=-1) / (1.0 - mu)
    gate = np.minimum(1.0, x[..., 6] / cstar)
    return gate * design


def d_h3(x, w=EQUAL_W):
    """Eq. (12): weighted geometric mean. 0 ** w = 0 for w > 0."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    return np.prod(np.power(x, w), axis=-1)


# ----------------------------------------------------------------- closed forms (Section 2.5)
def sum_sq_dev(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum((x - x.mean()) ** 2))


def closed_form_sd(x, c, n=N):
    """(ii) SD of D under symmetric Dirichlet(c) weights: sqrt( sum(x-xbar)^2 / [n (n c + 1)] )."""
    return float(np.sqrt(sum_sq_dev(x) / (n * (n * c + 1))))


def closed_form_variance_share(c, n=N):
    """(iv) share of Var(D) due to construct values: (n c + 1) / [n (c + 1)]."""
    return (n * c + 1) / (n * (c + 1))


def exact_reversal_probability():
    """(iii) P(alpha+mu > beta+gamma+delta+theta) with (a, b, lambda) ~ Dirichlet(2, 4, 1).
    Density = Gamma(7)/(Gamma(2)Gamma(4)Gamma(1)) a b^3 = 120 a b^3.
    P = int_0^{1/2} 120 s^3 (1 - 2 s) / 2 ds  (the integral printed in Section 2.5)."""
    val, _ = integrate.quad(lambda s: 120 * s**3 * (1 - 2 * s) / 2, 0, 0.5)
    # independent check by double integration of the Dirichlet density over a > b, a + b <= 1
    val2, _ = integrate.dblquad(lambda a, b: 120 * a * b**3, 0, 0.5, lambda b: b, lambda b: 1 - b)
    return float(val), float(val2)


# ----------------------------------------------------------------- analysis blocks
def phase2_univariate():
    grid = np.round(np.linspace(0, 1, 11), 10)
    out = {}
    for k, name in enumerate(CONSTRUCTS):
        vals = []
        for v in grid:
            x = np.full(N, 0.5)
            x[k] = v
            vals.append(float(d_h1(x)))
        out[name] = vals
    # panel (b): number of constructs raised to 1.0 from a common starting level
    raised = {}
    for start in (0.5, 0.2):
        seq = []
        for m in range(N + 1):
            x = np.full(N, start)
            x[:m] = 1.0
            seq.append(float(d_h1(x)))
        raised[str(start)] = seq
    return {"grid": grid.tolist(), "D_by_construct": out, "raised_to_one": raised,
            "slope_per_unit": (out["T"][-1] - out["T"][0]) / 1.0}


def phase3_scenarios():
    rows = {}
    for name, x in {**SCENARIOS, **DISCRIMINATING}.items():
        x = np.array(x)
        rows[name] = {
            "x": x.tolist(),
            "mean_input": float(x.mean()),
            "H1": float(d_h1(x)),
            "H2": {str(cs): float(d_h2(x, cs)) for cs in CSTAR_VALUES},
            "H3": float(d_h3(x)),
        }
    return rows


def phase4_joint_and_decomposition():
    t = np.linspace(0, 1, 101)
    c = np.linspace(0, 1, 101)
    surf = np.empty((len(c), len(t)))
    for i, cv in enumerate(c):
        for j, tv in enumerate(t):
            x = np.full(N, 0.5)
            x[0] = tv
            x[6] = cv
            surf[i, j] = d_h1(x)
    lines = {}
    for cv in (0, 1 / 3, 2 / 3, 1):
        lines[f"{cv:.4f}"] = [float(d_h1(np.array([tv, .5, .5, .5, .5, .5, cv]))) for tv in t]
    lo, op = np.array(SCENARIOS["Low-Resource"]), np.array(SCENARIOS["Optimised"])
    contrib = ((op - lo) / N)
    total = float(d_h1(op) - d_h1(lo))
    return {
        "TC_surface_min": float(surf.min()), "TC_surface_max": float(surf.max()),
        "TC_line_slope": float((lines["0.0000"][-1] - lines["0.0000"][0]) / 1.0),
        "TC_line_spacing_per_third_of_C": float(lines["0.3333"][0] - lines["0.0000"][0]),
        "decomposition_Low_to_Optimised": {
            "total_difference": total,
            "contribution": dict(zip(CONSTRUCTS, contrib.tolist())),
            "share_of_total": dict(zip(CONSTRUCTS, (contrib / total).tolist())),
        },
    }


def phase5_weights():
    x = np.array(FIXED_PROFILE)
    # two-weight slice: alpha (weight on T) and mu (weight on C); the rest shared equally among E,U,P,SRL,F
    def slice_D(alpha, mu):
        rest = (1.0 - alpha - mu) / 5.0
        w = np.array([alpha, rest, rest, rest, rest, rest, mu])
        return float(d_h1(x, w))
    grid5 = [0.05, 0.15, 0.25, 0.35, 0.45]          # cell grid shown in Fig. 5a
    line_grid = np.linspace(0.05, 0.45, 81)         # lines shown in Fig. 5b
    heat = [[slice_D(a, m) for m in grid5] for a in grid5]
    full_a = np.linspace(0.05, 0.45, 81)
    slice_all = [[slice_D(a, m) for a in full_a] for m in (0.05, 0.25, 0.45)]
    rest_mean = float(np.mean(x[1:6]))
    out = {
        "fixed_profile": x.tolist(),
        "slice_grid": grid5,
        "slice_heat_rows_alpha_cols_mu": heat,
        "slice_min": float(np.min(slice_all)), "slice_max": float(np.max(slice_all)),
        "slope_alpha": float(x[0] - rest_mean), "slope_mu": float(x[6] - rest_mean),
        "mean_of_other_five": rest_mean,
        "range_over_all_admissible_weights": [float(x.min()), float(x.max())],
    }
    # Dirichlet simulations, re-seeded for each concentration
    lo, op, ba = (np.array(SCENARIOS[k]) for k in ("Low-Resource", "Optimised", "Baseline"))
    names = list(SCENARIOS)
    pareto_pairs = []
    for a, b in combinations(names, 2):
        xa, xb = np.array(SCENARIOS[a]), np.array(SCENARIOS[b])
        if np.all(xa >= xb) or np.all(xb >= xa):
            pareto_pairs.append((a, b))
    sims = {}
    for c in CONCENTRATIONS:
        rng = np.random.default_rng(SEED)
        w = rng.dirichlet(np.full(N, c), size=N_DRAWS)
        D_fixed = d_h1(x, w)
        D_low, D_opt, D_base = d_h1(lo, w), d_h1(op, w), d_h1(ba, w)
        diff = D_opt - D_base
        # Monte Carlo check of the variance decomposition (Section 3.4).  Design: construct values are
        # independent and uniform on [0, 1] (N_DRAWS x 7 draws from the same generator, taken after the
        # Dirichlet(c) weights), independent of the weights.  Because E[w_i] = 1/n, E[D | x] = mean(x), so the
        # share of Var(D) due to construct values is Var(mean(x)) / Var(D); weights have no first-order effect.
        xs = rng.random((N_DRAWS, N))
        D_mc = np.sum(w * xs, axis=1)
        share_mc = float(np.var(xs.mean(axis=1)) / np.var(D_mc))
        # pairwise orderings across the five scenarios
        flips = {}
        for a, b in combinations(names, 2):
            da = d_h1(np.array(SCENARIOS[a]), w)
            db = d_h1(np.array(SCENARIOS[b]), w)
            flips[f"{a} > {b}"] = float(np.mean(da > db))
        s = {
            "fixed_profile": {"mean": float(D_fixed.mean()), "sd": float(D_fixed.std()),
                              "p5": float(np.percentile(D_fixed, 5)), "p95": float(np.percentile(D_fixed, 95))},
            "Low-Resource": {"mean": float(D_low.mean()), "sd": float(D_low.std()),
                             "p5": float(np.percentile(D_low, 5)), "p95": float(np.percentile(D_low, 95))},
            "Optimised": {"mean": float(D_opt.mean()), "sd": float(D_opt.std()),
                          "p5": float(np.percentile(D_opt, 5)), "p95": float(np.percentile(D_opt, 95))},
            "Optimised_minus_Baseline": {"mean": float(diff.mean()), "sd": float(diff.std()),
                                         "p5": float(np.percentile(diff, 5)), "p95": float(np.percentile(diff, 95))},
            "P_Optimised_exceeds_Baseline": float(np.mean(diff > 0)),
            "share_draws_with_a_weight_below_0.01": float(np.mean(np.any(w < 0.01, axis=1))),
            "share_draws_with_a_weight_above_0.5": float(np.mean(np.any(w > 0.5, axis=1))),
            "closed_form_sd_fixed_profile": closed_form_sd(x, c),
            "closed_form_sd_Low-Resource": closed_form_sd(lo, c),
            "closed_form_sd_Optimised": closed_form_sd(op, c),
            "closed_form_variance_share_construct_values": closed_form_variance_share(c),
            "mc_variance_share_construct_values": share_mc,
            "pairwise_P_first_exceeds_second": flips,
        }
        sims[str(c)] = s
    out["dirichlet"] = sims
    out["pareto_ordered_pairs"] = [list(p) for p in pareto_pairs]
    out["n_pairs"] = len(list(combinations(names, 2)))
    out["n_pareto_ordered_pairs"] = len(pareto_pairs)
    return out


def phase6_aggregation_curves():
    cgrid = np.linspace(0, 1, 201)
    out = {}
    for label, v in (("others_0.8", 0.8), ("others_0.5", 0.5)):
        X = np.tile(np.full(N, v), (len(cgrid), 1))
        X[:, 6] = cgrid
        out[label] = {
            "C": cgrid.tolist(),
            "H1": d_h1(X).tolist(),
            "H2_cstar_0.5": d_h2(X, 0.5).tolist(),
            "H3": d_h3(X).tolist(),
            "H1_at_C0": float(d_h1(X[0])), "H1_at_C1": float(d_h1(X[-1])),
            "H2_plateau": float(d_h2(X[-1], 0.5)),
            "H3_at_C1": float(d_h3(X[-1])),
        }
    # C = 0 with every other construct at 1 (Section 2.4)
    x = np.array([1.0] * 6 + [0.0])
    out["C0_others1"] = {"H1": float(d_h1(x)), "H2": float(d_h2(x, 0.5)), "H3": float(d_h3(x))}
    # algebra: with the other five constructs at 0.5, D = (T + C + 2.5)/7
    out["max_D_via_T_and_C_only"] = float((1 + 1 + 2.5) / 7)
    return out


# ----------------------------------------------------------------- verification against the manuscript
def verify(res):
    """Each entry: (label, value printed in the manuscript, computed value, tolerance)."""
    sc = res["scenarios"]
    sim1 = res["weights"]["dirichlet"]["1"]
    V = []
    def add(label, reported, computed, tol):
        V.append({"item": label, "reported": reported, "computed": float(computed), "tolerance": tol,
                  "pass": bool(abs(float(computed) - reported) <= tol)})
    # --- Section 3.1 / Table 6
    for name, h1, h3 in (("Worst Case", .2, .2), ("Low-Resource", .4286, .4224), ("Baseline", .7, .7),
                         ("High Resource", .8, .8), ("Optimised", .6714, .6659)):
        add(f"Table 6 linear D, {name}", h1, sc[name]["H1"], 5e-5)
        add(f"Table 6 geometric D, {name}", h3, sc[name]["H3"], 5e-5)
    u = res["univariate"]
    add("Fig. 2a D at construct = 0", .4286, u["D_by_construct"]["T"][0], 5e-5)
    add("Fig. 2a D at construct = 1", .5714, u["D_by_construct"]["T"][-1], 5e-5)
    add("Fig. 2a gain per unit (1/7)", .1429, u["slope_per_unit"], 5e-5)
    add("Fig. 2b start 0.5, step", .0714, u["raised_to_one"]["0.5"][1] - u["raised_to_one"]["0.5"][0], 5e-5)
    add("Fig. 2b start 0.2, step", .1143, u["raised_to_one"]["0.2"][1] - u["raised_to_one"]["0.2"][0], 5e-5)
    # --- Section 3.2
    j = res["joint"]
    add("Fig. 3 minimum D (T = C = 0)", .3571, j["TC_surface_min"], 5e-5)
    add("Fig. 3 maximum D (T = C = 1)", .6429, j["TC_surface_max"], 5e-5)
    add("Fig. 3b line spacing per third of C", .0476, j["TC_line_spacing_per_third_of_C"], 5e-5)
    # --- Section 3.3 / Fig. 4
    d = j["decomposition_Low_to_Optimised"]
    add("Fig. 4 total difference in D", .2429, d["total_difference"], 5e-5)
    for k, v in (("C", .0714), ("T", .0429), ("U", .0286), ("P", .0286), ("SRL", .0286), ("F", .0286), ("E", .0143)):
        add(f"Fig. 4b contribution of {k}", v, d["contribution"][k], 5e-5)
    for k, v in (("C", .29), ("T", .18), ("U", .12), ("E", .06)):
        add(f"Table 10 share of difference, {k}", v, d["share_of_total"][k], 5e-3)
    w = res["weights"]
    add("Fig. 5 slice minimum (alpha = mu = 0.05)", .409, w["slice_min"], 5e-4)
    add("Fig. 5 slice maximum (alpha = mu = 0.45)", .801, w["slice_max"], 5e-4)
    add("Fig. 5 slope with respect to alpha", .54, w["slope_alpha"], 5e-3)
    add("Fig. 5 slope with respect to mu", .44, w["slope_mu"], 5e-3)
    add("Fig. 5 minimum over all admissible weights", .10, w["range_over_all_admissible_weights"][0], 1e-9)
    add("Fig. 5 maximum over all admissible weights", .90, w["range_over_all_admissible_weights"][1], 1e-9)
    # --- Section 2.5 closed forms
    p_exact = res["closed_forms"]["exact_reversal_probability"]
    add("Exact reversal probability 3/16", 3 / 16, p_exact, 1e-12)
    add("Closed-form variance share, c = 1 (8/14)", .571, closed_form_variance_share(1), 5e-4)
    add("Closed-form sum of squares / 56 (variance, fixed profile)", .0089, sum_sq_dev(FIXED_PROFILE) / 56, 5e-5)
    # --- Table 7 (simulation, c = 1, seed 2026)
    for lab, key, e in (("Low-Resource", "Low-Resource", .4286), ("Optimised", "Optimised", .6714)):
        add(f"Table 7 equal weights, {lab}", e, sc[key]["H1"], 5e-5)
    for lab, key, mean, sd, p5, p95 in (
        ("Low-Resource", "Low-Resource", .4285, .0248, .3857, .4676),
        ("Optimised", "Optimised", .6715, .0310, .6248, .7267),
        ("Fixed profile", "fixed_profile", .5003, .0941, .3438, .6564),
        ("Optimised - Baseline", "Optimised_minus_Baseline", -.0285, .0310, -.0752, .0267)):
        s = sim1[key]
        add(f"Table 7 mean, {lab}", mean, s["mean"], 5e-4)
        add(f"Table 7 SD, {lab}", sd, s["sd"], 5e-4)
        add(f"Table 7 5th percentile, {lab}", p5, s["p5"], 5e-4)
        add(f"Table 7 95th percentile, {lab}", p95, s["p95"], 5e-4)
    add("Table 7 closed-form SD, Low-Resource", .0247, sim1["closed_form_sd_Low-Resource"], 5e-5)
    add("Table 7 closed-form SD, Optimised", .0311, sim1["closed_form_sd_Optimised"], 5e-5)
    add("Table 7 closed-form SD, fixed profile", .0945, sim1["closed_form_sd_fixed_profile"], 5e-5)
    add("Section 3.4 simulated P(Optimised > Baseline), c = 1", .186, sim1["P_Optimised_exceeds_Baseline"], 5e-4)
    add("Section 3.4 draws with a weight < 0.01 (c = 1)", .35, sim1["share_draws_with_a_weight_below_0.01"], 5e-3)
    add("Section 3.4 draws with a weight > 0.5 (c = 1)", .11, sim1["share_draws_with_a_weight_above_0.5"], 5e-3)
    # --- Table 8 (all concentrations)
    t8 = {"0.5": (.1260, .246, .787, .347, .429), "1": (.0945, .186, .353, .107, .571),
          "2": (.0690, .114, .050, .012, .714), "5": (.0445, .031, .000, .000, .857),
          "10": (.0317, .004, .000, .000, .922)}
    for c, (sd, pr, lw, hw, vs) in t8.items():
        s = res["weights"]["dirichlet"][c]
        add(f"Table 8 SD of D (c = {c})", sd, s["closed_form_sd_fixed_profile"], 5e-5)
        add(f"Table 8 P(Optimised > Baseline) (c = {c})", pr, s["P_Optimised_exceeds_Baseline"], 5e-4)
        add(f"Table 8 draws with weight < 0.01 (c = {c})", lw, s["share_draws_with_a_weight_below_0.01"], 5e-4)
        add(f"Table 8 draws with weight > 0.5 (c = {c})", hw, s["share_draws_with_a_weight_above_0.5"], 5e-4)
        add(f"Table 8 variance share, closed form (c = {c})", vs, s["closed_form_variance_share_construct_values"], 5e-4)
    add("Section 3.4 Monte Carlo variance share, c = 1 (simulated, 57.6%)", .576, sim1["mc_variance_share_construct_values"], 5e-4)
    add("Section 3.4 Monte Carlo share within 1 point of closed form (57.1%)", .571, sim1["mc_variance_share_construct_values"], 1e-2)
    add("Section 3.4 pairs that are Pareto-ordered (nine of ten)", 9, w["n_pareto_ordered_pairs"], 0)
    # --- Table 9
    t9 = {"Worst Case": (.200, .133, .080, .057, .200), "Low-Resource": (.429, .450, .270, .193, .422),
          "Baseline": (.700, .700, .700, .700, .700), "High Resource": (.800, .800, .800, .800, .800),
          "Optimised": (.671, .650, .650, .650, .666),
          "Profile A (six design constructs 0.9, C = 0.1)": (.786, .300, .180, .129, .658),
          "Profile B (six design constructs 0.5, C = 1.0)": (.571, .500, .500, .500, .552),
          "Profile D (six design constructs 1.0, C = 0.0)": (.857, .000, .000, .000, .000)}
    for name, (h1, a, b, c_, h3) in t9.items():
        r = sc[name]
        add(f"Table 9 H1, {name}", h1, r["H1"], 5e-4)
        add(f"Table 9 H2 c0=0.3, {name}", a, r["H2"]["0.3"], 5e-4)
        add(f"Table 9 H2 c0=0.5, {name}", b, r["H2"]["0.5"], 5e-4)
        add(f"Table 9 H2 c0=0.7, {name}", c_, r["H2"]["0.7"], 5e-4)
        add(f"Table 9 H3, {name}", h3, r["H3"], 5e-4)
    # --- Fig. 7 and Section 2.4
    a8, a5 = res["aggregation"]["others_0.8"], res["aggregation"]["others_0.5"]
    add("Fig. 7a H1 at C = 0", .686, a8["H1_at_C0"], 5e-4)
    add("Fig. 7b H1 at C = 0", .429, a5["H1_at_C0"], 5e-4)
    add("Fig. 7a H2 plateau", .8, a8["H2_plateau"], 5e-4)
    add("Fig. 7b H2 plateau", .5, a5["H2_plateau"], 5e-4)
    add("Fig. 7a H3 at C = 1", .826, a8["H3_at_C1"], 5e-4)
    add("Fig. 7b H3 at C = 1", .552, a5["H3_at_C1"], 5e-4)
    add("Section 2.4 C = 0, other constructs 1: H1 (6/7)", .857, res["aggregation"]["C0_others1"]["H1"], 5e-4)
    add("Section 2.4 C = 0, other constructs 1: H2", 0.0, res["aggregation"]["C0_others1"]["H2"], 1e-12)
    add("Section 2.4 C = 0, other constructs 1: H3", 0.0, res["aggregation"]["C0_others1"]["H3"], 1e-12)
    add("Section 3.5 maximum D from T and C only", .6429, res["aggregation"]["max_D_via_T_and_C_only"], 5e-5)
    # scenario ordering identical under H1, H2 and H3 (Section 3.5)
    order = ["Worst Case", "Low-Resource", "Optimised", "Baseline", "High Resource"]
    ok = True
    for key in (lambda r: r["H1"], lambda r: r["H2"]["0.3"], lambda r: r["H2"]["0.5"],
                lambda r: r["H2"]["0.7"], lambda r: r["H3"]):
        v = [key(sc[n]) for n in order]
        ok &= all(v[i] < v[i + 1] for i in range(len(v) - 1))
    V.append({"item": "Section 3.5 scenarios ordered identically under H1, H2 (all c0) and H3",
              "reported": True, "computed": ok, "tolerance": None, "pass": bool(ok)})
    # Profiles A and B are ordered differently by H1 and H2 (Section 2.4)
    A, B = sc["Profile A (six design constructs 0.9, C = 0.1)"], sc["Profile B (six design constructs 0.5, C = 1.0)"]
    cond = (A["H1"] > B["H1"]) and (A["H3"] > B["H3"]) and (A["H2"]["0.5"] < B["H2"]["0.5"])
    V.append({"item": "Section 2.4 H1 and H3 rank Profile A above B; H2 ranks B above A",
              "reported": True, "computed": bool(cond), "tolerance": None, "pass": bool(cond)})
    return V


# ----------------------------------------------------------------- main
def run():
    res = {
        "meta": {
            "seed": SEED, "n_draws": N_DRAWS, "concentrations": CONCENTRATIONS,
            "rng": "numpy.random.default_rng(seed), re-seeded for each concentration",
            "equal_weights": EQUAL_W.tolist(), "constructs": CONSTRUCTS,
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "note": "All construct values are assumed; no empirical data were used.",
        },
        "supplementary_file": {
            "online_resource": "Online Resource 3 (output of hc_tplad_analysis.py, Online Resource 1)",
            "article_title": ARTICLE_TITLE,
            "journal": JOURNAL,
            "author": AUTHOR,
            "affiliation": AFFILIATION,
            "corresponding_author_email": CORRESPONDING_EMAIL,
            "archive_doi": ARCHIVE_DOI,
            "archive_version": ARCHIVE_VERSION,
            "licence": LICENCE,
        },
        "univariate": phase2_univariate(),
        "scenarios": phase3_scenarios(),
        "joint": phase4_joint_and_decomposition(),
        "weights": phase5_weights(),
        "aggregation": phase6_aggregation_curves(),
    }
    p1, p2 = exact_reversal_probability()
    res["closed_forms"] = {"exact_reversal_probability": p1, "exact_reversal_probability_double_integral": p2}
    res["verification"] = verify(res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="hc_tplad_results.json")
    args = ap.parse_args()
    res = run()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    v = res["verification"]
    bad = [r for r in v if not r["pass"]]
    print(f"python {res['meta']['python']}  numpy {res['meta']['numpy']}  scipy {res['meta']['scipy']}")
    print(f"verification: {len(v) - len(bad)} of {len(v)} checks agree with the manuscript")
    for r in bad:
        print("  DIFFERS:", r["item"], "| manuscript:", r["reported"], "| computed:", round(r["computed"], 6)
              if isinstance(r["computed"], float) else r["computed"])
    print(f"results written to {args.out}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
