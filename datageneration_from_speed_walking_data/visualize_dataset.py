"""
visualize_dataset.py
────────────────────
Lädt dataset.npz und plottet für einige zufällige Simulationen
die Fußpositionen (calcn_r / calcn_l) sowie den COM über die Zeit.

Usage:
    python visualize_dataset.py                    # dataset.npz
    python visualize_dataset.py --data mein.npz
    python visualize_dataset.py --n 5              # 5 zufällige Sims
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse

def main(path, n_sims):
    npz  = np.load(path)
    foot = npz["foot"]   # (S, T, 6)
    com  = npz["com"]    # (S, T, 3)
    S, T, _ = foot.shape
    hz   = 50
    times = np.linspace(0, T / hz, T)

    print(f"Dataset: {S} Simulationen, {T} Frames ({T/hz:.2f}s bei {hz} Hz)")

    # Drift-Korrektur nur für Fuß (für Draufsicht), COM bleibt absolut
    foot_plot = foot - foot[:, :1, :]   # (S, T, 6) – nur für Plot
    com_plot  = com                      # (S, T, 3) – absolut lassen

    rng  = np.random.default_rng(42)
    idx  = rng.choice(S, size=min(n_sims, S), replace=False)
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(idx)))

    coord_labels = ["X (vorwärts)", "Y (vertikal)", "Z (lateral)"]

    # ── Figure 1: Fußpositionen ────────────────────────────────────────────
    fig1, axes1 = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    fig1.suptitle("Fußpositionen über die Zeit  (calcn_r / calcn_l)",
                  fontsize=13, fontweight="bold")

    for ax_i, label in enumerate(coord_labels):
        ax_r = axes1[ax_i, 0]
        ax_l = axes1[ax_i, 1]
        for k, (sim, c) in enumerate(zip(idx, colors)):
            ax_r.plot(times, foot_plot[sim, :, ax_i]     * 100, color=c,
                      lw=1.2, alpha=0.8, label=f"Sim {sim}")
            ax_l.plot(times, foot_plot[sim, :, ax_i + 3] * 100, color=c,
                      lw=1.2, alpha=0.8)
        ax_r.set_ylabel(f"{label}  [cm]", fontsize=9)
        ax_r.grid(alpha=0.3)
        ax_l.grid(alpha=0.3)
        if ax_i == 0:
            ax_r.set_title("Rechter Fuß (calcn_r)", fontsize=10)
            ax_l.set_title("Linker Fuß (calcn_l)",  fontsize=10)
            ax_r.legend(fontsize=7, loc="upper right")
        if ax_i == 2:
            ax_r.set_xlabel("Zeit [s]", fontsize=9)
            ax_l.set_xlabel("Zeit [s]", fontsize=9)

    plt.tight_layout()
    plt.savefig("viz_foot.png", dpi=150, bbox_inches="tight")
    print("Fuß-Plot gespeichert → viz_foot.png")

    # ── Figure 2: COM ──────────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig2.suptitle("Center of Mass (COM) über die Zeit",
                  fontsize=13, fontweight="bold")

    for ax_i, (ax, label) in enumerate(zip(axes2, coord_labels)):
        for k, (sim, c) in enumerate(zip(idx, colors)):
            ax.plot(times, com_plot[sim, :, ax_i] * 100, color=c,
                    lw=1.2, alpha=0.8, label=f"Sim {sim}")
        ax.set_ylabel(f"COM {label}  [cm]", fontsize=9)
        ax.grid(alpha=0.3)
        if ax_i == 0:
            ax.legend(fontsize=7, loc="upper right")
    axes2[-1].set_xlabel("Zeit [s]", fontsize=9)

    plt.tight_layout()
    plt.savefig("viz_com.png", dpi=150, bbox_inches="tight")
    print("COM-Plot gespeichert  → viz_com.png")

    # ── Figure 3: Fußposition XZ (Draufsicht Gangbahn) ────────────────────
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
    fig3.suptitle("Gangbahn Draufsicht  (X–Z Ebene)",
                  fontsize=13, fontweight="bold")
    for k, (sim, c) in enumerate(zip(idx, colors)):
        axes3[0].plot(foot_plot[sim, :, 0] * 100, foot_plot[sim, :, 2] * 100,
                      color=c, lw=1.2, alpha=0.8, label=f"Sim {sim}")
        axes3[1].plot(foot_plot[sim, :, 3] * 100, foot_plot[sim, :, 5] * 100,
                      color=c, lw=1.2, alpha=0.8)
    for ax, title in zip(axes3, ["Rechter Fuß", "Linker Fuß"]):
        ax.set_xlabel("X (vorwärts) [cm]", fontsize=9)
        ax.set_ylabel("Z (lateral) [cm]",  fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes3[0].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig("viz_topdown.png", dpi=150, bbox_inches="tight")
    print("Draufsicht gespeichert → viz_topdown.png")

    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dataset.npz")
    p.add_argument("--n",    type=int, default=6,
                   help="Anzahl zufälliger Simulationen (default: 6)")
    args = p.parse_args()
    main(args.data, args.n)