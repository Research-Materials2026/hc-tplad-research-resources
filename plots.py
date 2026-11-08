#!/usr/bin/env python3
# Supplementary file (Online Resource 2) for:
# "Modelling Trustworthy Prescriptive Learning Analytics Dashboards for Student Decision Support in Resource-Constrained Educational Settings Using Human-Centered AI Approaches",
# Education and Information Technologies
# Author, affiliation, contact details and archive DOI: [withheld for blind review]
# Version 1.0.0, licence CC BY 4.0
"""
plots.py  (Online Resource 2)

Generates Figs. 1-7 of the manuscript from the model in hc_tplad_analysis.py
(same folder).  No values are typed in here except the figure layout: every
plotted quantity is computed by the functions of the analysis script, with the
same seed (2026) as in the results file.

Usage:   python plots.py            # writes figures/Fig1.png ... Fig7.png (+ .pdf)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

import hc_tplad_analysis as m

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.0, "savefig.dpi": 300,
})
BLUE, RED, ORANGE, BLACK = "#1f4e9c", "#c00000", "#e87c1a", "#111111"


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def panel_label(ax, s):
    ax.text(-0.02, 1.07, s, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom", ha="right")


# ------------------------------------------------------------------ Fig. 1 (conceptual structure)
def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 6.9))
    ax.set_xlim(0, 10); ax.set_ylim(0.8, 11.2); ax.axis("off")
    boxes = [
        (9.1, "Seven construct indices T, E, U, P, SRL, F, C, each in [0, 1]\n"
              "Six are built from three proposed sub-indicators each; C is a single index\n"
              "Measurement protocols (Table 3) are proposed, not validated", "#f2f2f2", BLACK, "-", 0.5, 9.5),
        (6.4, "Three rival aggregation hypotheses (Section 2.4)\n"
              "H1 mean (Eq. 8)  \u00b7  H2 gated (Eq. 11)  \u00b7  H3 geometric (Eq. 12)\n"
              "Weights sum to 1; equal weights are a default, not a neutral choice", "#e6ecf8", BLUE, "-", 0.5, 9.5),
        (3.7, "Decision-support readiness index D\n"
              "analytic properties and numerical checks with assumed inputs\n(this study)", "white", BLACK, "-", 0.5, 9.5),
        (1.0, "Not estimated here: learner decisions, trust in use,\n"
              "self-regulation, performance. Needed for the proposed\n"
              "retrospective test of H1\u2013H3 (Section 2.4)", "white", BLACK, "--", 1.4, 8.6),
    ]
    for y, txt, fc, ec, ls, x0, x1 in boxes:
        ax.add_patch(FancyBboxPatch((x0, y), x1 - x0 if False else (10 - 2 * x0), 1.9, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc=fc, ec=ec, lw=2.4, ls=ls))
        ax.text(5, y + 0.95, txt, ha="center", va="center", fontsize=11.5)
    for y in (9.1, 6.4, 3.7):
        ax.annotate("", xy=(5, y - 0.75), xytext=(5, y - 0.05), arrowprops=dict(arrowstyle="-|>", lw=2.2, color=BLACK))
    save(fig, "Fig1")


# ------------------------------------------------------------------ Fig. 2
def fig2():
    u = m.phase2_univariate()
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.4, 4.0))
    a.plot(u["grid"], u["D_by_construct"]["T"], "-o", color=BLACK, ms=4, lw=1.8,
           label="any one construct (all seven coincide)")
    a.set_ylim(0.40, 0.60); a.set_xlabel("Value of one construct"); a.set_ylabel("Decision-support readiness index (D)")
    a.legend(frameon=False, fontsize=9, loc="upper left"); panel_label(a, "(a)")
    k = np.arange(m.N + 1)
    b.plot(k, u["raised_to_one"]["0.5"], "-s", color=BLUE, ms=5, lw=1.8, label="all constructs start at 0.5")
    b.plot(k, u["raised_to_one"]["0.2"], "--^", color=RED, ms=5, lw=1.8, label="all constructs start at 0.2")
    b.set_xlabel("Number of constructs raised to 1.0"); b.set_ylabel("D")
    b.legend(frameon=False, fontsize=9, loc="lower right"); panel_label(b, "(b)")
    fig.tight_layout(); save(fig, "Fig2")


# ------------------------------------------------------------------ Fig. 3
def fig3():
    t = np.linspace(0, 1, 101); c = np.linspace(0, 1, 101)
    Z = np.array([[(tv + cv + 2.5) / m.N for tv in t] for cv in c])
    # computed with the model itself (not the algebraic shortcut)
    Z = np.array([[float(m.d_h1(np.array([tv, .5, .5, .5, .5, .5, cv]))) for tv in t] for cv in c])
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.4, 4.0))
    im = a.imshow(Z, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="Blues", vmin=0.3, vmax=0.7)
    cs = a.contour(t, c, Z, levels=[0.4, 0.5, 0.6], colors="black", linewidths=1.2, linestyles=["-", "--", ":"])
    a.clabel(cs, fmt="%.1f", fontsize=7)
    cb = fig.colorbar(im, ax=a); cb.set_label("D")
    a.set_xlabel("Trustworthiness (T)"); a.set_ylabel("Context sensitivity (C)"); panel_label(a, "(a)")
    styles = [(0, RED, "--"), (1 / 3, ORANGE, "-."), (2 / 3, BLUE, "-"), (1, BLACK, ":")]
    for cv, col, ls in styles:
        y = [float(m.d_h1(np.array([tv, .5, .5, .5, .5, .5, cv]))) for tv in t]
        b.plot(t, y, color=col, ls=ls, lw=2.0, label=f"C = {cv:.2f}")
    b.set_xlabel("Trustworthiness (T)"); b.set_ylabel("D"); b.legend(frameon=False, fontsize=9, loc="lower right")
    panel_label(b, "(b)")
    fig.tight_layout(); save(fig, "Fig3")


# ------------------------------------------------------------------ Fig. 4
def fig4():
    d = m.phase4_joint_and_decomposition()["decomposition_Low_to_Optimised"]["contribution"]
    lo, op = m.SCENARIOS["Low-Resource"], m.SCENARIOS["Optimised"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.4, 4.0))
    x = np.arange(m.N)
    a.plot(x, lo, "--o", color=RED, ms=7, lw=1.8, label="Low-Resource")
    a.plot(x, op, "-s", color=BLUE, ms=7, lw=1.8, label="Optimised")
    a.set_xticks(x); a.set_xticklabels(m.CONSTRUCTS); a.set_ylim(0.2, 0.9)
    a.set_ylabel("Assumed construct value"); a.legend(frameon=False, loc="lower left"); panel_label(a, "(a)")
    vals = [d[k] for k in m.CONSTRUCTS]
    bars = b.bar(x, vals, color=ORANGE, edgecolor="black", lw=0.8)
    for xi, v in zip(x, vals):
        b.text(xi, v + 0.0015, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    b.set_xticks(x); b.set_xticklabels(m.CONSTRUCTS); b.set_ylim(0, 0.085)
    b.set_ylabel("Contribution to the difference in D"); panel_label(b, "(b)")
    fig.tight_layout(); save(fig, "Fig4")


# ------------------------------------------------------------------ Fig. 5
def fig5():
    x = np.array(m.FIXED_PROFILE)
    def slice_D(alpha, mu):
        rest = (1 - alpha - mu) / 5
        return float(m.d_h1(x, np.array([alpha, rest, rest, rest, rest, rest, mu])))
    g = [0.05, 0.15, 0.25, 0.35, 0.45]
    H = np.array([[slice_D(a_, m_) for m_ in g] for a_ in g])
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.4, 4.2))
    im = a.imshow(H, origin="lower", cmap="Blues", vmin=0.2, vmax=1.0, aspect="auto")
    for i in range(5):
        for j in range(5):
            a.text(j, i, f"{H[i, j]:.3f}", ha="center", va="center", fontsize=9,
                   color="white" if H[i, j] > 0.6 else "black")
    a.set_xticks(range(5)); a.set_xticklabels([f"{v:.2f}" for v in g])
    a.set_yticks(range(5)); a.set_yticklabels([f"{v:.2f}" for v in g])
    a.set_xlabel("Weight on context sensitivity (\u03bc)"); a.set_ylabel("Weight on trustworthiness (\u03b1)")
    for s in a.spines.values(): s.set_visible(True)
    panel_label(a, "(a)")
    al = np.linspace(0.05, 0.45, 81)
    for mu, col, ls in ((0.05, RED, "--"), (0.25, BLUE, "-"), (0.45, BLACK, ":")):
        b.plot(al, [slice_D(v, mu) for v in al], color=col, ls=ls, lw=2.0, label=f"\u03bc = {mu:.2f}")
    b.axhline(x.max(), color="grey", ls="--", lw=1.2); b.axhline(x.min(), color="grey", ls="--", lw=1.2)
    b.text(0.06, x.max() + 0.015, f"maximum over all admissible weights ({x.max():.2f})", color="grey", fontsize=8)
    b.text(0.06, x.min() + 0.015, f"minimum over all admissible weights ({x.min():.2f})", color="grey", fontsize=8)
    b.set_xlabel("Weight on trustworthiness (\u03b1)"); b.set_ylabel("D")
    b.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.0, 0.88)); panel_label(b, "(b)")
    fig.tight_layout(); save(fig, "Fig5")


# ------------------------------------------------------------------ Fig. 6
def fig6():
    rng = np.random.default_rng(m.SEED)
    w = rng.dirichlet(np.full(m.N, 1.0), size=m.N_DRAWS)       # identical draw to the analysis script (c = 1)
    x = np.array(m.FIXED_PROFILE)
    Df = m.d_h1(x, w)
    diff = m.d_h1(np.array(m.SCENARIOS["Optimised"]), w) - m.d_h1(np.array(m.SCENARIOS["Baseline"]), w)
    eq_diff = float(m.d_h1(np.array(m.SCENARIOS["Optimised"])) - m.d_h1(np.array(m.SCENARIOS["Baseline"])))
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.4, 4.0))
    a.hist(Df, bins=60, color="#8aa5d6", edgecolor="white", lw=0.3)
    a.axvline(float(m.d_h1(x)), color=RED, ls="--", lw=2, label=f"equal weights ({float(m.d_h1(x)):.3f})")
    p5, p95 = np.percentile(Df, [5, 95])
    a.axvline(p5, color=BLACK, ls=":", lw=2, label="5th and 95th percentiles"); a.axvline(p95, color=BLACK, ls=":", lw=2)
    a.set_xlabel("D (fixed profile of Fig. 5)"); a.set_ylabel("Number of draws"); a.legend(frameon=False, fontsize=8, loc="upper left")
    panel_label(a, "(a)")
    b.hist(diff, bins=60, color="#8aa5d6", edgecolor="white", lw=0.3)
    b.axvline(0, color=BLACK, lw=2, label="no difference")
    b.axvline(eq_diff, color=RED, ls="--", lw=2, label=f"equal weights (\u2212{abs(eq_diff):.4f})")
    b.set_xlabel("D(Optimised) \u2212 D(Baseline)"); b.set_ylabel("Number of draws"); b.legend(frameon=False, fontsize=8)
    panel_label(b, "(b)")
    fig.tight_layout(); save(fig, "Fig6")


# ------------------------------------------------------------------ Fig. 7
def fig7():
    agg = m.phase6_aggregation_curves()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    for ax, key, title, lab in ((axes[0], "others_0.8", "Other six constructs at 0.8", "(a)"),
                                (axes[1], "others_0.5", "Other six constructs at 0.5", "(b)")):
        r = agg[key]; c = r["C"]
        ax.plot(c, r["H1"], color=BLUE, lw=2.2, label="H1 compensatory (Eq. 8)")
        ax.plot(c, r["H2_cstar_0.5"], color=RED, ls="--", lw=2.2, label="H2 gated, c₀ = 0.5 (Eq. 11)")
        ax.plot(c, r["H3"], color=BLACK, ls=":", lw=2.6, label="H3 geometric (Eq. 12)")
        ax.set_ylim(0, 1.0); ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel("Context sensitivity (C)"); ax.set_ylabel("D"); ax.set_title(title, fontsize=10)
        panel_label(ax, lab)
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout(); save(fig, "Fig7")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        f()
    import json, platform
    with open(os.path.join(OUT, "figure_environment.json"), "w") as fh:
        json.dump({"python": platform.python_version(), "numpy": np.__version__,
                   "matplotlib": matplotlib.__version__, "seed": m.SEED, "n_draws": m.N_DRAWS}, fh, indent=2)
    print("Figures written to", os.path.abspath(OUT))
