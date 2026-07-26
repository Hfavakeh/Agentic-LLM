import os, json, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "analysis/motion_plots"
os.makedirs(OUT, exist_ok=True)
NAVY, TEAL, CORAL, AMBER, INK = "#243B53", "#1F7A6F", "#D9655B", "#E0A340", "#1A1A1A"
plt.rcParams.update({"font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})

# ---- radar training-split speed (rows 0:1200, hz = 4) ----
df = pd.read_csv("preprocessed-RadarEXP1(in).csv", header=None)
y = df.iloc[:1200, -2:].values.astype(float)
dy = np.diff(y, axis=0)
speed = np.linalg.norm(dy, axis=1) * 4.0
e1, e2 = float(np.quantile(speed, 1/3)), float(np.quantile(speed, 2/3))
p95 = float(np.quantile(speed, 0.95))
print(f"tercile edges = {e1:.3f}, {e2:.3f}   p95 = {p95:.3f}")

# === A. speed histogram with bin edges ============================================
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.hist(speed, bins=50, color=NAVY, edgecolor="white", alpha=0.85)
xmax = float(speed.max())
ax.axvspan(0,     e1,    alpha=0.10, color=TEAL,  label=f"slow  (≤ {e1:.2f})")
ax.axvspan(e1,    e2,    alpha=0.10, color=AMBER, label=f"medium ({e1:.2f}-{e2:.2f})")
ax.axvspan(e2,    xmax,  alpha=0.10, color=CORAL, label=f"fast  (> {e2:.2f})")
for x in (e1, e2):
    ax.axvline(x, color=INK, lw=1.2, ls="--")
ax.axvline(p95, color=CORAL, lw=1.5, ls=":", label=f"p95 = {p95:.2f} m/s")
ax.set_xlabel("true speed (m/s)"); ax.set_ylabel("count")
ax.set_title("A — Radar training-split speed distribution with bin edges")
ax.set_xlim(0, min(1.5, xmax))
ax.legend(loc="upper right", frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/A_speed_distribution.png", dpi=150); plt.close()

# === B. velocity-plausibility penalty ============================================
fig, ax = plt.subplots(figsize=(7.5, 4.2))
sp = np.linspace(0, 2.0, 500)
colors = [(0.66, NAVY), (1.0, TEAL), (2.0, AMBER)]
for vm, col in colors:
    ax.plot(sp, np.maximum(0, sp - vm) ** 2, color=col, lw=2, label=f"v_max = {vm}")
    ax.axvline(vm, color=col, lw=0.8, ls=":", alpha=0.6)
ax2 = ax.twinx()
ax2.hist(speed, bins=60, color="lightgray", alpha=0.55, edgecolor="white")
ax2.set_ylabel("observed speed (count)", color="gray")
ax.set_xlabel("predicted speed (m/s)")
ax.set_ylabel(r"penalty  $(\max(0,\,\hat{v}-v_{max}))^2$")
ax.set_title(r"B — Velocity-plausibility penalty  $\mathrm{ReLU}(\hat{v}-v_{max})^2$  "
             r"(per-sample, before $\lambda_{vel}$)")
ax.set_xlim(0, 2.0); ax.legend(loc="upper left", frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/B_velocity_penalty.png", dpi=150); plt.close()

# === C. bin-weight staircase =====================================================
fig, ax = plt.subplots(figsize=(7.5, 4.0))
sp = np.linspace(0, max(xmax, 1.5), 500)
def step(sp, w):
    idx = np.digitize(sp, [e1, e2])
    return np.array(w)[idx]
ax.step(sp, step(sp, [1.0, 1.5, 1.0]), where="post", color=NAVY,
        lw=2.5, label="LLM proposal  [1.0, 1.5, 1.0]  (up-weight medium)")
ax.step(sp, step(sp, [1.0, 1.0, 1.0]), where="post", color="gray",
        lw=2, ls="--", label="Plain MSE  [1, 1, 1]")
for x in (e1, e2):
    ax.axvline(x, color=INK, lw=1.0, ls=":", alpha=0.6)
ax.set_xlabel("true sample speed (m/s)"); ax.set_ylabel(r"per-sample weight  $w_i$")
ax.set_title("C — Bin-weight as a step function of true speed (tercile edges)")
ax.set_xlim(0, sp.max()); ax.set_ylim(0, 2.0)
ax.legend(loc="upper right", frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/C_bin_weight_staircase.png", dpi=150); plt.close()

# === D. smoothness on two example trajectories ===================================
T = 40; t = np.arange(T)
smooth_x = 0.25 * t / 4.0
smooth_y = 0.15 * np.sin(2 * np.pi * t / 20.0)
traj_smooth = np.stack([smooth_x, smooth_y], axis=1)
np.random.seed(7)
mask = (t % 4 == 0).astype(float)
jerky_x = smooth_x + mask * np.random.uniform(-0.08, 0.08, T)
jerky_y = smooth_y + mask * np.random.uniform(-0.10, 0.10, T)
traj_jerky = np.stack([jerky_x, jerky_y], axis=1)
def sm_loss(tr):
    a = tr[2:] - 2 * tr[1:-1] + tr[:-2]
    return float(np.mean(np.sum(a ** 2, axis=1)))
Ls = sm_loss(traj_smooth); Lj = sm_loss(traj_jerky)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
for ax, tj, label, L, col in [
    (axes[0], traj_smooth, "smooth trajectory", Ls, TEAL),
    (axes[1], traj_jerky,  "jerky trajectory",  Lj, CORAL),
]:
    ax.plot(tj[:, 0], tj[:, 1], "-o", color=col, ms=4, lw=1.5)
    ax.set_xlabel("x (m)"); ax.set_title(f"D — {label}    mean$\\|$accel$\\|^2$ = {L:.4f}")
axes[0].set_ylabel("y (m)")
fig.suptitle(r"D — Smoothness term  $\mathrm{mean}\|\hat{y}_t - 2 y_{t-1} + y_{t-2}\|^2$  "
             f"(before $\\lambda_{{smooth}}$;  jerky / smooth ratio ≈ {Lj/Ls:.0f}x)",
             y=1.02, fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/D_smoothness_trajectories.png", dpi=150,
                                bbox_inches="tight"); plt.close()

# === E. proposed v_max vs observed p95 (qwen3:8b logs) ===========================
D = "outputs-point3-qwen38bb"
vmaxes = []
for f in sorted(glob.glob(D + "/seed_*/conversation_log_run1.json")):
    for r in json.load(open(f, encoding="utf-8")):
        ch = (r.get("raw_parsed_proposal") or {}).get("proposed_changes", {}) or {}
        if "v_max" in ch:
            vmaxes.append(float(ch["v_max"]))
print(f"qwen3 v_max proposals: n = {len(vmaxes)}  mean = {np.mean(vmaxes):.3f}")
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.hist(vmaxes, bins=np.arange(0.55, 0.78, 0.01), color=NAVY,
        edgecolor="white", alpha=0.9)
ax.axvline(p95, color=CORAL, lw=2, label=f"observed p95 = {p95:.3f}")
ax.axvline(p95 * 1.1, color=AMBER, lw=2, ls="--",
           label=f"recipe v_max = p95 × 1.1 = {p95*1.1:.3f}")
ax.axvline(float(np.mean(vmaxes)), color=TEAL, lw=2, ls=":",
           label=f"LLM mean v_max = {np.mean(vmaxes):.3f}")
ax.set_xlabel("proposed v_max (m/s)"); ax.set_ylabel("count")
ax.set_title("E — qwen3:8b proposed v_max vs observed radar p95  "
             f"(mean v_max / p95 = {np.mean(vmaxes)/p95:.3f}, n = {len(vmaxes)})")
ax.legend(frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT}/E_vmax_vs_p95.png", dpi=150); plt.close()

# === F. total-loss decomposition on a real radar minibatch =======================
np.random.seed(0)
y_true = y[2:]; prev = y[1:-1]; prev2 = y[:-2]
sigma = 0.18
y_pred = y_true + np.random.normal(0, sigma, y_true.shape)
per_sample_mse = ((y_pred - y_true) ** 2).mean(axis=1)
true_speed_b = np.linalg.norm(y_true - prev, axis=1) * 4.0
bin_idx = np.digitize(true_speed_b, [e1, e2]).clip(0, 2)
def base_loss(weights):
    w = np.array(weights)[bin_idx]
    return float((per_sample_mse * w).sum() / w.sum())
def vel_loss(vmax):
    pred_speed = np.linalg.norm(y_pred - prev, axis=1) * 4.0
    return float(np.mean(np.maximum(0, pred_speed - vmax) ** 2))
def sm_loss_real():
    a = y_pred - 2 * prev + prev2
    return float(np.mean(np.sum(a ** 2, axis=1)))
plain_mse = float(per_sample_mse.mean())
b_plain, v_plain, s_plain = plain_mse, 0.0, 0.0
b_llm = base_loss([1.0, 1.5, 1.0])
v_llm = 0.2 * vel_loss(0.66)
s_llm = 0.2 * sm_loss_real()
fig, ax = plt.subplots(figsize=(7.5, 4.5))
labels = ["Plain MSE\n(defaults)",
          "qwen3 proposal\n($\\lambda_{vel}=\\lambda_{sm}=0.2$, $v_{max}=0.66$,\nbin_w=[1.0,1.5,1.0])"]
b = [b_plain, b_llm]; v = [v_plain, v_llm]; s = [s_plain, s_llm]
ax.bar(labels, b, color=NAVY, label=r"$L_{base}$ (position MSE, bin-weighted)")
ax.bar(labels, v, bottom=b, color=AMBER, label=r"$\lambda_{vel}\cdot L_{vel}$")
ax.bar(labels, s, bottom=[bi+vi for bi, vi in zip(b, v)], color=CORAL,
       label=r"$\lambda_{smooth}\cdot L_{smooth}$")
for i, total in enumerate([b[i]+v[i]+s[i] for i in range(2)]):
    ax.text(i, total * 1.02, f"total = {total:.4f}",
            ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("loss value")
ax.set_title("F — Total-loss decomposition on a real radar minibatch  "
             f"(predictions = targets + N(0, {sigma}); n = {len(y_true)} samples)")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/F_loss_decomposition.png", dpi=150); plt.close()

print(f"wrote 6 PNGs to {OUT}")
