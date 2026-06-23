import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
})

with open("/mnt/lab/nmwamsojo/zs_cl_sweep_v2_result.json") as f:
    data = json.load(f)

r = data["results"]
CLs   = [1, 7, 14, 21, 28, 56, 112, 364]
weeks = ["—", "1w", "2w", "3w", "4w", "8w", "16w", "52w"]

geo_f  = [r[f"cl{c}_xcl0"]["geo"]        for c in CLs]
geo_t  = [r[f"cl{c}_xcl100_sdst"]["geo"] for c in CLs]

spring_f = [r[f"cl{c}_xcl0"]["spring"] for c in CLs]
winter_f = [r[f"cl{c}_xcl0"]["winter"] for c in CLs]
autumn_f = [r[f"cl{c}_xcl0"]["autumn"] for c in CLs]
spring_t = [r[f"cl{c}_xcl100_sdst"]["spring"] for c in CLs]
winter_t = [r[f"cl{c}_xcl100_sdst"]["winter"] for c in CLs]
autumn_t = [r[f"cl{c}_xcl100_sdst"]["autumn"] for c in CLs]

NAIVE = 1.6310
OUT   = "/home/nmwamsojo/tsfm-explo/reports/figures"
os.makedirs(OUT, exist_ok=True)

x      = np.arange(len(CLs))
labels = [f"{c}\n({w})" for c, w in zip(CLs, weeks)]

C_BLUE = "#2166ac"
C_RED  = "#d6604d"
C_GRAY = "#555555"

# ── Figure 1: Geo-mean overview (6.5 × 4 in — standard Word text width) ──────
fig, ax = plt.subplots(figsize=(6.5, 4))

ax.plot(x, geo_f, "o-",  color=C_BLUE, lw=2,   ms=6, zorder=3,
        label="xcl=False (geo-mean)")
ax.plot(x, geo_t, "s--", color=C_RED,  lw=2,   ms=6, zorder=3,
        label="xcl=True, store×dept sorted (geo-mean)")
ax.axhline(NAIVE, color=C_GRAY, ls=":", lw=1.4, zorder=2,
           label=f"Seasonal naive  ({NAIVE})")

best_f_i = int(np.argmin(geo_f))
best_t_i = int(np.argmin(geo_t))
ax.annotate(f"{geo_f[best_f_i]:.4f}",
            (x[best_f_i], geo_f[best_f_i]),
            textcoords="offset points", xytext=(4, 8),
            color=C_BLUE, fontsize=9, fontweight="bold")
ax.annotate(f"{geo_t[best_t_i]:.4f}",
            (x[best_t_i], geo_t[best_t_i]),
            textcoords="offset points", xytext=(4, -14),
            color=C_RED, fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel("Context length (observations) / weekly equivalent")
ax.set_ylabel("WRMSSE  (3-cutoff geometric mean)")
ax.set_title("Global Zero-Shot WRMSSE vs Context Length\n"
             "Chronos-2-small · 30,490 M5 series · spring / winter / autumn cutoffs")
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.35, lw=0.7)
ax.set_ylim(0.88, 2.15)
fig.tight_layout()
fig.savefig(f"{OUT}/zs_cl_sweep_v2_geo.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: zs_cl_sweep_v2_geo.png  (300 dpi, 6.5×4 in)")

# ── Figure 2: Per-cutoff breakdown (6.5 × 4 in, 3 panels) ────────────────────
fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.6), sharey=True)
cutoffs_info = [
    ("Spring 2016-04-24", spring_f, spring_t, 1.4639),
    ("Winter 2016-01-03", winter_f, winter_t, 1.4216),
    ("Autumn 2015-10-04", autumn_f, autumn_t, 2.0848),
]
for ax, (title, yf, yt, naive_c) in zip(axes, cutoffs_info):
    ax.plot(x, yf, "o-",  color=C_BLUE, lw=1.8, ms=5, label="xcl=False")
    ax.plot(x, yt, "s--", color=C_RED,  lw=1.8, ms=5, label="xcl=True sdst")
    ax.axhline(naive_c, color=C_GRAY, ls=":", lw=1.2, label=f"Naive ({naive_c:.4f})")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in CLs], fontsize=8, rotation=45, ha="right")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("CL", fontsize=9)
    ax.grid(axis="y", alpha=0.35, lw=0.7)
    ax.legend(fontsize=7.5, loc="upper left")

axes[0].set_ylabel("WRMSSE")
fig.suptitle("Per-Cutoff WRMSSE vs Context Length", fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/zs_cl_sweep_v2_per_cutoff.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: zs_cl_sweep_v2_per_cutoff.png  (300 dpi, 6.5×3.6 in)")

# ── Figure 3: Cross-learning delta bar chart (6.5 × 3.2 in) ──────────────────
delta  = [t - f for t, f in zip(geo_t, geo_f)]
colors = [C_RED if d > 0 else "#4dac26" for d in delta]

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
ax.set_ylabel("Δ WRMSSE  (xcl=True sdst − xcl=False)")
ax.set_title("Cross-Learning Effect by Context Length\n"
             "Green = xcl=True helps · Red = xcl=True hurts")
ax.grid(axis="y", alpha=0.35, lw=0.7, zorder=0)
ax.set_ylim(min(delta) - 0.02, max(delta) + 0.02)
fig.tight_layout()
fig.savefig(f"{OUT}/zs_cl_sweep_v2_delta.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: zs_cl_sweep_v2_delta.png  (300 dpi, 6.5×3.2 in)")
