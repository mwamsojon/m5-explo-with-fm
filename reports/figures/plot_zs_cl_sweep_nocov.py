"""
plot_zs_cl_sweep_nocov.py
=========================
Regenerate WRMSSE vs CL comparison plots (no covariates vs with covariates).
Same style, colours, and dimensions as plot_zs_cl_sweep_v2.py.

Outputs (300 DPI, 6.5-inch wide — Word-ready):
  zs_cl_sweep_nocov_geo.png        — 4-line geo-mean overview
  zs_cl_sweep_nocov_per_cutoff.png — 3-panel per-cutoff breakdown
  zs_cl_sweep_nocov_delta.png      — covariate-effect delta bars (xcl=False)

Run:
  cd /home/nmwamsojo/tsfm-explo
  .venv/bin/python reports/figures/plot_zs_cl_sweep_nocov.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.dpi":       300,
})

# ── Load data ─────────────────────────────────────────────────────────────────
with open("/mnt/lab/nmwamsojo/zs_cl_sweep_nocov_result.json") as f:
    nc = json.load(f)
with open("/mnt/lab/nmwamsojo/zs_cl_sweep_v2_result.json") as f:
    cv = json.load(f)

nc0 = nc["results_xcl0"]   # no-cov xcl=False
nc1 = nc["results_xcl1"]   # no-cov xcl=True sorted
cr  = cv["results"]         # with-cov (eP_sF)

CLs   = [1, 7, 14, 21, 28, 56, 112, 364]
weeks = ["—", "1w", "2w", "3w", "4w", "8w", "16w", "52w"]
x      = np.arange(len(CLs))
labels = [f"{c}\n({w})" for c, w in zip(CLs, weeks)]
NAIVE  = 1.6310
OUT    = "/home/nmwamsojo/tsfm-explo/reports/figures"

# v2 palette
C_BLUE = "#2166ac"
C_RED  = "#d6604d"
C_GRAY = "#555555"

def _vals(src, key, metric="geo"):
    if isinstance(src, dict) and str(CLs[0]) in src:
        return [src[str(c)][metric] for c in CLs]
    return [src[f"cl{c}_{key}"][metric] for c in CLs]

geo_nc_f = [nc0[str(c)]["geo"] for c in CLs]
geo_nc_t = [nc1[str(c)]["geo"] for c in CLs]
geo_cv_f = [cr[f"cl{c}_xcl0"]["geo"]        for c in CLs]
geo_cv_t = [cr[f"cl{c}_xcl100_sdst"]["geo"] for c in CLs]

# ── Figure 1: Geo-mean overview ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4))

# with-cov as light reference lines (same colour, thinner, no marker fill)
ax.plot(x, geo_cv_f, "o-",  color=C_BLUE, lw=1.2, ms=5, alpha=0.35, zorder=2)
ax.plot(x, geo_cv_t, "s--", color=C_RED,  lw=1.2, ms=5, alpha=0.35, zorder=2)
# no-cov as primary bold lines
ax.plot(x, geo_nc_f, "o-",  color=C_BLUE, lw=2.2, ms=7, zorder=3,
        label="xcl=False · no cov  (sales only)")
ax.plot(x, geo_nc_t, "s--", color=C_RED,  lw=2.2, ms=7, zorder=3,
        label="xcl=True sdst · no cov  (sales only)")
# ghost entries for legend
ax.plot([], [], "o-",  color=C_BLUE, lw=1.2, ms=4, alpha=0.5,
        label="xcl=False · with cov  (eP_sF, ref.)")
ax.plot([], [], "s--", color=C_RED,  lw=1.2, ms=4, alpha=0.5,
        label="xcl=True sdst · with cov  (eP_sF, ref.)")
ax.axhline(NAIVE, color=C_GRAY, ls=":", lw=1.4, zorder=2,
           label=f"Seasonal naive  ({NAIVE})")

best_f = int(np.argmin(geo_nc_f))
best_t = int(np.argmin(geo_nc_t))
ax.annotate(f"{geo_nc_f[best_f]:.4f}",
            (x[best_f], geo_nc_f[best_f]),
            textcoords="offset points", xytext=(4, 8),
            color=C_BLUE, fontsize=9, fontweight="bold")
ax.annotate(f"{geo_nc_t[best_t]:.4f}",
            (x[best_t], geo_nc_t[best_t]),
            textcoords="offset points", xytext=(4, -14),
            color=C_RED, fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel("Context length (observations) / weekly equivalent")
ax.set_ylabel("WRMSSE  (3-cutoff geometric mean)")
ax.set_title("Global Zero-Shot WRMSSE vs Context Length — Sales Only\n"
             "Chronos-2-small · 30,490 M5 series · spring / winter / autumn cutoffs")
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.35, lw=0.7)
ax.set_ylim(0.88, 2.35)
fig.tight_layout()
fig.savefig(f"{OUT}/zs_cl_sweep_nocov_geo.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: zs_cl_sweep_nocov_geo.png")

# ── Figure 2: Per-cutoff breakdown ────────────────────────────────────────────
cutoffs_info = [
    ("Spring 2016-04-24",
     [nc0[str(c)]["spring"] for c in CLs],
     [nc1[str(c)]["spring"] for c in CLs],
     [cr[f"cl{c}_xcl0"]["spring"] for c in CLs],
     [cr[f"cl{c}_xcl100_sdst"]["spring"] for c in CLs],
     1.4639),
    ("Winter 2016-01-03",
     [nc0[str(c)]["winter"] for c in CLs],
     [nc1[str(c)]["winter"] for c in CLs],
     [cr[f"cl{c}_xcl0"]["winter"] for c in CLs],
     [cr[f"cl{c}_xcl100_sdst"]["winter"] for c in CLs],
     1.4216),
    ("Autumn 2015-10-04",
     [nc0[str(c)]["autumn"] for c in CLs],
     [nc1[str(c)]["autumn"] for c in CLs],
     [cr[f"cl{c}_xcl0"]["autumn"] for c in CLs],
     [cr[f"cl{c}_xcl100_sdst"]["autumn"] for c in CLs],
     2.0848),
]

fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.6), sharey=True)
for i, (ax, (title, nc_f, nc_t, cv_f, cv_t, naive_c)) in enumerate(zip(axes, cutoffs_info)):
    ax.plot(x, cv_f, "o-",  color=C_BLUE, lw=1.0, ms=4, alpha=0.35, zorder=2)
    ax.plot(x, cv_t, "s--", color=C_RED,  lw=1.0, ms=4, alpha=0.35, zorder=2)
    ax.plot(x, nc_f, "o-",  color=C_BLUE, lw=1.8, ms=5, zorder=3,
            label="xcl=F no cov")
    ax.plot(x, nc_t, "s--", color=C_RED,  lw=1.8, ms=5, zorder=3,
            label="xcl=T no cov")
    ax.axhline(naive_c, color=C_GRAY, ls=":", lw=1.2, label=f"Naive ({naive_c:.4f})")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in CLs], fontsize=8, rotation=45, ha="right")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("CL", fontsize=9)
    ax.grid(axis="y", alpha=0.35, lw=0.7)
    ax.legend(fontsize=7.5, loc="upper left")

axes[0].set_ylabel("WRMSSE")
fig.suptitle("Per-Cutoff WRMSSE vs Context Length — Sales Only vs With Covariates",
             fontsize=10, y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/zs_cl_sweep_nocov_per_cutoff.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: zs_cl_sweep_nocov_per_cutoff.png")

# ── Figure 3: Covariate delta bars (xcl=False: with-cov minus no-cov) ────────
delta  = [cv - nc for cv, nc in zip(geo_cv_f, geo_nc_f)]
colors = ["#4dac26" if d < 0 else C_RED for d in delta]

fig, ax = plt.subplots(figsize=(6.5, 3.2))
bars = ax.bar(x, delta, color=colors, edgecolor="white", width=0.62, zorder=3)
ax.axhline(0, color="black", lw=0.9, zorder=4)
for bar, d in zip(bars, delta):
    va   = "bottom" if d >= 0 else "top"
    yoff = 0.003 if d >= 0 else -0.003
    ax.text(bar.get_x() + bar.get_width() / 2, d + yoff,
            f"{d:+.3f}", ha="center", va=va, fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel("Context length (observations) / weekly equivalent")
ax.set_ylabel("Δ WRMSSE  (with cov − no cov,  xcl=False)")
ax.set_title("Covariate Effect by Context Length\n"
             "Green = covariates help · Red = covariates hurt")
ax.grid(axis="y", alpha=0.35, lw=0.7, zorder=0)
ax.set_ylim(min(delta) - 0.03, max(delta) + 0.03)
fig.tight_layout()
fig.savefig(f"{OUT}/zs_cl_sweep_nocov_delta.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: zs_cl_sweep_nocov_delta.png")
