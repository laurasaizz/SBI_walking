"""
run_pipeline.py

Predicts center of mass (COM) position from foot coordinates using a
sliding window regression network.

Training input:  foot positions over a 0.3s window, expressed relative
                 to the first frame of the window
Training target: COM position at the first frame, relative to the
                 midpoint between both feet

Set the three paths below and run:
    python run_pipeline.py
"""

DATASET_PATH     = "dataset.npz"
REAL_MODEL_PATH  = "gait2392_simbody.osim"
REAL_MOTION_PATH = "normal.mot"

SAMPLE_RATE   = 50
WINDOW_SEC    = 0.3
WINDOW_FRAMES = int(SAMPLE_RATE * WINDOW_SEC)

EPOCHS         = 5
BATCH_SIZE     = 1024
LR             = 1e-3
HIDDEN_DIMS    = [256, 256, 128, 64]
DROPOUT        = 0.2
CHECKPOINT     = "com_model.pt"
PLOT_OUTPUT    = "validation_plot.png"

SIMS_PER_EPOCH = 2000
VAL_SIMS       = 2000

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

try:
    import opensim as osim
    osim.Logger.setLevelString("error")
    OPENSIM_AVAILABLE = True
except ImportError:
    OPENSIM_AVAILABLE = False
    print("OpenSim not found – validation step will be skipped.")


class COMRegressionNet(nn.Module):
    """
    Fully connected regression network.
    Input:  WINDOW_FRAMES * 6 foot coordinates (flattened)
    Output: 3 COM coordinates relative to foot midpoint
    """
    def __init__(self, input_dim, hidden_dims=None, dropout=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 128, 64]
        layers, in_dim = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h),
                       nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_windows(foot, com, W):
    """
    Converts raw foot and COM trajectories into sliding window samples.

    For each window starting at frame t:
      - foot positions are expressed relative to foot[t]
      - COM is expressed relative to the midpoint of both feet at t

    This makes the representation invariant to absolute position in space,
    so the network generalises across different recordings.

    Args:
        foot: (T, 6) array of foot positions in metres
        com:  (T, 3) array of COM positions in metres
        W:    window length in frames

    Returns:
        X: (T-W+1, W*6) input vectors
        y: (T-W+1, 3)   COM targets
    """
    T = foot.shape[0]
    n = T - W + 1
    foot_win = np.lib.stride_tricks.sliding_window_view(foot, (W, 6))[:, 0]

    foot_ref   = foot_win[:, 0, :]
    foot_rel   = foot_win - foot_ref[:, np.newaxis, :]
    foot_mid_t = (foot_ref[:, :3] + foot_ref[:, 3:]) / 2
    com_rel    = com[:n] - foot_mid_t

    X = foot_rel.reshape(n, -1).astype(np.float32)
    y = com_rel.astype(np.float32)
    return X, y


def compute_stats(path):
    """
    Estimates normalisation statistics from the dataset without loading
    everything into memory. Uses memory-mapped file access and samples
    500 simulations to compute mean and std of the windowed data.
    """
    print(f"\n{'─'*60}")
    print(f"  Step 1/3 – computing normalisation stats")
    print(f"{'─'*60}")
    print(f"  Dataset       : {path}")
    print(f"  Window        : {WINDOW_FRAMES} frames = {WINDOW_SEC*1000:.0f} ms")

    npz  = np.load(path, mmap_mode="r")
    foot = npz["foot"]
    com  = npz["com"]
    S, T, _ = foot.shape
    print(f"  Simulations   : {S:,}   frames/sim : {T}")

    rng = np.random.default_rng(0)
    idx = rng.choice(S, size=min(500, S), replace=False)
    W   = WINDOW_FRAMES

    foot_windows, com_windows = [], []
    for si in idx:
        X_i, y_i = make_windows(foot[si].astype(np.float32),
                                 com[si].astype(np.float32), W)
        foot_windows.append(X_i)
        com_windows.append(y_i)

    foot_all = np.concatenate(foot_windows)
    com_all  = np.concatenate(com_windows)

    foot_mean = foot_all.mean(0).astype(np.float32)
    foot_std  = foot_all.std(0).astype(np.float32)  + 1e-8
    com_mean  = com_all.mean(0).astype(np.float32)
    com_std   = com_all.std(0).astype(np.float32)   + 1e-8

    print(f"  COM std       : {(com_std*100).round(1)} cm")

    return dict(foot_mean=foot_mean, foot_std=foot_std,
                com_mean=com_mean,   com_std=com_std,
                S=S, T=T, W=W)


def _sims_to_tensors(npz_foot, npz_com, indices, stats):
    """Loads simulations, builds windows and normalises."""
    W         = stats["W"]
    foot_mean = np.array(stats["foot_mean"], dtype=np.float32)
    foot_std  = np.array(stats["foot_std"],  dtype=np.float32)
    com_mean  = np.array(stats["com_mean"],  dtype=np.float32)
    com_std   = np.array(stats["com_std"],   dtype=np.float32)

    Xs, ys = [], []
    for i in range(len(indices)):
        X_i, y_i = make_windows(npz_foot[indices[i]].astype(np.float32),
                                 npz_com[indices[i]].astype(np.float32), W)
        Xs.append((X_i - foot_mean) / foot_std)
        ys.append((y_i - com_mean)  / com_std)

    return torch.from_numpy(np.concatenate(Xs)), \
           torch.from_numpy(np.concatenate(ys))


def train_model(stats):
    print(f"\n{'─'*60}")
    print(f"  Step 2/3 – training")
    print(f"{'─'*60}")

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S, T, W   = stats["S"], stats["T"], stats["W"]
    input_dim = W * 6

    print(f"  Device        : {device}")
    print(f"  Input dim     : {W} x 6 = {input_dim}")
    print(f"  Epochs        : {EPOCHS}   batch : {BATCH_SIZE}   lr : {LR}")
    print(f"  Sims/epoch    : {SIMS_PER_EPOCH:,} ({SIMS_PER_EPOCH/S*100:.1f}% of dataset)")
    print(f"  Val sims      : {VAL_SIMS:,}")

    rng        = np.random.default_rng(42)
    all_idx    = rng.permutation(S)
    val_idx    = all_idx[:VAL_SIMS]
    train_pool = all_idx[VAL_SIMS:]

    npz      = np.load(DATASET_PATH, mmap_mode="r")
    npz_foot = npz["foot"]
    npz_com  = npz["com"]

    Xv, yv     = _sims_to_tensors(npz_foot, npz_com, val_idx, stats)
    val_loader = DataLoader(TensorDataset(Xv, yv), BATCH_SIZE,
                            shuffle=False, num_workers=0, pin_memory=True)
    n_val = len(Xv)

    model    = COMRegressionNet(input_dim, HIDDEN_DIMS, DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters    : {n_params:,}\n")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=25, T_mult=2, eta_min=LR * 1e-2)

    train_losses, val_losses = [], []
    best_val  = float("inf")
    log_every = max(1, EPOCHS // 20)

    for epoch in range(1, EPOCHS + 1):
        epoch_idx    = rng.choice(train_pool, size=min(SIMS_PER_EPOCH, len(train_pool)),
                                  replace=False)
        Xt, yt       = _sims_to_tensors(npz_foot, npz_com, epoch_idx, stats)
        train_loader = DataLoader(TensorDataset(Xt, yt), BATCH_SIZE,
                                  shuffle=True, num_workers=0, pin_memory=True)

        model.train()
        run = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            run += loss.item() * xb.size(0)
        tl = run / len(Xt)

        model.eval()
        run = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                run += criterion(model(xb), yb).item() * xb.size(0)
        vl = run / n_val
        scheduler.step()

        train_losses.append(tl)
        val_losses.append(vl)

        if vl < best_val:
            best_val   = vl
            safe_stats = {k: (v.tolist() if isinstance(v, np.ndarray) else int(v))
                          for k, v in stats.items()}
            torch.save({"epoch":       epoch,
                        "model_state": model.state_dict(),
                        "stats":       safe_stats,
                        "args":        {"hidden":    HIDDEN_DIMS,
                                        "dropout":   DROPOUT,
                                        "input_dim": input_dim}},
                       CHECKPOINT)

        if epoch % log_every == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{EPOCHS}  "
                  f"train={tl:.5f}  val={vl:.5f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    print(f"\n  Best val MSE : {best_val:.6f}  ->  {CHECKPOINT}")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(train_losses, label="train"); ax.plot(val_losses, label="val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (normalised)")
    ax.set_title("Training loss"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    curve_path = Path(CHECKPOINT).with_suffix(".loss_curve.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"  Loss curve   -> {curve_path}")


def extract_kinematics(model_path, mot_path):
    """
    Runs forward kinematics on a real OpenSim model and motion file.
    Returns raw foot positions and COM in metres (no drift correction –
    make_windows handles the local referencing).
    """
    print(f"\n  Model  : {model_path}")
    print(f"  Motion : {mot_path}")

    model = osim.Model(model_path)
    state = model.initSystem()

    table = osim.TimeSeriesTable(mot_path)
    proc  = osim.TableProcessor(table)
    try:
        if table.getTableMetaDataString("inDegrees").strip().lower() == "yes":
            proc.append(osim.TabOpConvertDegreesToRadians())
    except Exception:
        pass
    table = proc.process(model)

    coordSet = model.getCoordinateSet()
    labels   = table.getColumnLabels()
    coord_indices, coords = [], []
    for j in range(len(labels)):
        if coordSet.contains(labels[j]):
            coord_indices.append(j)
            coords.append(coordSet.get(labels[j]))

    foot_r = model.getBodySet().get("calcn_r")
    foot_l = model.getBodySet().get("calcn_l")

    N         = table.getNumRows()
    times     = list(table.getIndependentColumn())
    foot_data = np.zeros((N, 6), dtype=np.float32)
    com_data  = np.zeros((N, 3), dtype=np.float32)

    print(f"  Frames : {N}  ({times[0]:.3f}s – {times[-1]:.3f}s)")
    print("  Running forward kinematics...", end="", flush=True)

    for i in range(N):
        state.setTime(times[i])
        row = table.getRowAtIndex(i)
        for k, coord in enumerate(coords):
            coord.setValue(state, row[coord_indices[k]], False)
        model.realizePosition(state)

        pr = foot_r.getPositionInGround(state)
        pl = foot_l.getPositionInGround(state)
        foot_data[i] = [pr.get(0), pr.get(1), pr.get(2),
                        pl.get(0), pl.get(1), pl.get(2)]
        c = model.calcMassCenterPosition(state)
        com_data[i]  = [c.get(0), c.get(1), c.get(2)]

    print(" done.")
    return np.array(times, dtype=np.float32), foot_data, com_data


def validate_and_plot(times, foot_data, com_true):
    print(f"\n{'─'*60}")
    print(f"  Step 3/3 – validation")
    print(f"{'─'*60}")

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt      = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    stats     = ckpt["stats"]
    args      = ckpt["args"]
    W         = stats["W"]

    foot_mean = np.array(stats["foot_mean"], dtype=np.float32)
    foot_std  = np.array(stats["foot_std"],  dtype=np.float32)
    com_mean  = np.array(stats["com_mean"],  dtype=np.float32)
    com_std   = np.array(stats["com_std"],   dtype=np.float32)

    X_rel, com_rel_true = make_windows(foot_data, com_true, W)
    X_val       = (X_rel        - foot_mean) / foot_std
    y_true_norm = (com_rel_true - com_mean)  / com_std
    times_win   = times[:len(times) - W + 1]

    print(f"  Window        : {W} frames = {W/SAMPLE_RATE*1000:.0f} ms")
    print(f"  Input range   : [{X_val.min():.2f}, {X_val.max():.2f}]")

    net = COMRegressionNet(input_dim   = args["input_dim"],
                           hidden_dims = args.get("hidden", HIDDEN_DIMS),
                           dropout     = 0.0).to(device)
    net.load_state_dict(ckpt["model_state"])
    net.eval()

    with torch.no_grad():
        y_pred_norm = net(torch.from_numpy(X_val).to(device)).cpu().numpy()

    com_pred = y_pred_norm  * com_std + com_mean
    com_gt   = com_rel_true

    axis_labels = ["X – forward (AP)", "Y – vertical", "Z – lateral (ML)"]
    axis_short  = ["X", "Y", "Z"]
    c_gt        = ["#2166ac", "#1a9850", "#d6604d"]
    c_pred      = ["#74add1", "#a6d96a", "#f4a582"]

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        f"COM relative to foot midpoint  –  validation on real motion  "
        f"(window: {W} frames = {W/SAMPLE_RATE*1000:.0f} ms)",
        fontsize=12, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, width_ratios=[3, 1], hspace=0.50, wspace=0.35)

    rmse_list, mae_list, maxe_list = [], [], []

    for i in range(3):
        gt   = com_gt[:,   i]
        pred = com_pred[:, i]
        err  = pred - gt
        rmse = np.sqrt(np.mean(err**2)) * 100
        mae  = np.mean(np.abs(err))     * 100
        maxe = np.max(np.abs(err))      * 100
        rmse_list.append(rmse); mae_list.append(mae); maxe_list.append(maxe)

        ax_ts = fig.add_subplot(gs[i, 0])
        ax_ts.plot(times_win, gt   * 100, color=c_gt[i],   lw=1.8, label="ground truth")
        ax_ts.plot(times_win, pred * 100, color=c_pred[i], lw=1.8, ls="--", label="predicted")
        ax_ts.set_xlabel("time [s]", fontsize=9)
        ax_ts.set_ylabel(f"COM {axis_short[i]} [cm]", fontsize=9)
        ax_ts.set_title(f"{axis_labels[i]}   RMSE = {rmse:.2f} cm   MAE = {mae:.2f} cm",
                        fontsize=9)
        ax_ts.legend(fontsize=8, loc="upper right")
        ax_ts.grid(alpha=0.3)

        ax_er = fig.add_subplot(gs[i, 1])
        ax_er.plot(times_win, err * 100, color="#555555", lw=1.0)
        ax_er.axhline(0, color="black", lw=0.8, ls="--")
        ax_er.fill_between(times_win, err * 100, 0, alpha=0.25, color="#d62728")
        ax_er.set_xlabel("time [s]", fontsize=9)
        ax_er.set_ylabel("error [cm]", fontsize=9)
        ax_er.set_title(f"error (max = {maxe:.2f} cm)", fontsize=9)
        ax_er.grid(alpha=0.3)

    plt.savefig(PLOT_OUTPUT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved -> {PLOT_OUTPUT}")

    print(f"\n  {'axis':<6}  {'RMSE [cm]':>10}  {'MAE [cm]':>9}  {'max [cm]':>9}")
    print(f"  {'─'*42}")
    for i, a in enumerate(axis_short):
        print(f"  {a:<6}  {rmse_list[i]:>10.3f}  {mae_list[i]:>9.3f}  {maxe_list[i]:>9.3f}")
    overall = np.sqrt(np.mean((com_pred - com_gt)**2)) * 100
    print(f"  {'─'*42}")
    print(f"  {'total':<6}  {overall:>10.3f}")


if __name__ == "__main__":
    print("=" * 60)
    print("  COM prediction from foot coordinates")
    print("=" * 60)

    stats = compute_stats(DATASET_PATH)
    train_model(stats)

    if not OPENSIM_AVAILABLE:
        print("\nValidation skipped (OpenSim not installed).")
    else:
        print(f"\n{'─'*60}")
        print(f"  Step 3/3 – loading real motion data")
        print(f"{'─'*60}")
        times, foot_data, com_true = extract_kinematics(REAL_MODEL_PATH, REAL_MOTION_PATH)
        validate_and_plot(times, foot_data, com_true)

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)

    curve_path = Path(CHECKPOINT).with_suffix(".loss_curve.png")
    for p in [curve_path, PLOT_OUTPUT]:
        if Path(p).exists():
            img = plt.imread(str(p))
            fig, ax = plt.subplots(figsize=(12, 4) if "loss" in str(p) else (15, 10))
            ax.imshow(img); ax.axis("off"); ax.set_title(str(p))
            plt.tight_layout()
    plt.show()