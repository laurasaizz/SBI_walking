import torch
import matplotlib.pyplot as plt



#-----------------------SLIDING WINDOW-----------------------
#sampling from posterior for each window, plotting against true com
def plot_posterior_samples(model, test_dataset, idx, n_samples = 1000,device="cpu"):
    window, com = test_dataset[idx]
    window = window.unsqueeze(0).to(device)        # shape [1, window_dim], before [window_dim]
    com = com.unsqueeze(0).to(device) 
    com = com.numpy().flatten()
    with torch.no_grad():
        samples = model.sample(window, n_samples) 

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    labels = ["CoM forward direction", "CoM vertical direction", "CoM lateral direction"]

    for d in range(3):
        axes[d].hist(samples[:, d].numpy(), bins=50, density=True)
        axes[d].axvline(com[d].item(), linestyle="--")
        axes[d].set_title(labels[d])

    plt.tight_layout()
    plt.show()

#sampling posterior for each window in a trajectory, plotting mean and credible intervals against true com trajectory
def get_posterior_samples_over_time(model, windows, n_samples=200, device="cpu"):
    """
    windows: [N_windows, window_dim]
    returns:
        all_samples: [N_windows, n_samples, 3]
    """
    model.eval()
    all_samples = []

    with torch.no_grad():
        for t in range(len(windows)):
            X = windows[t].unsqueeze(0).to(device)   # [1, window_dim]
            samples = model.sample(X, n_samples)  

            # possible cleanup depending on shape
            if samples.dim() == 3:
                samples = samples.squeeze(1)

            all_samples.append(samples.cpu())  # [n_samples, 3]

    return torch.stack(all_samples) 

def summarize_posterior_trajectory(all_samples):
    """
    all_samples: [N_windows, n_samples, 3]
    returns:
        mean:   [N_windows, 3]
        median: [N_windows, 3]
        q05:    [N_windows, 3]
        q95:    [N_windows, 3]
    """
    mean = all_samples.mean(dim=1)
    median = all_samples.median(dim=1).values
    q05 = torch.quantile(all_samples, 0.05, dim=1)
    q95 = torch.quantile(all_samples, 0.95, dim=1)

    return mean, median, q05, q95

def plot_posterior_trajectory(true_com, mean, q05, q95):
    """
    true_com: [N_windows, 3]
    mean:     [N_windows, 3]
    q05:      [N_windows, 3]
    q95:      [N_windows, 3]
    """
    t = range(len(mean))
    labels = ["x", "y", "z"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for d in range(3):
        axes[d].plot(t, true_com[:, d].numpy(), label="true")
        axes[d].plot(t, mean[:, d].numpy(), label="posterior mean")
        axes[d].fill_between(
            t,
            q05[:, d].numpy(),
            q95[:, d].numpy(),
            alpha=0.3,
            label="90% credible band"
        )
        axes[d].set_ylabel(labels[d])
        axes[d].legend()

    axes[-1].set_xlabel("time index")
    plt.tight_layout()
    plt.show()

#sample whole com trajectories from the model
def sample_trajectory_paths(all_samples, n_traj=20):
    """
    all_samples: [N_windows, n_samples, 3]
    returns:
        trajs: [n_traj, N_windows, 3]
    """
    N_windows, n_samples, dim = all_samples.shape
    trajs = []

    for _ in range(n_traj):
        idx = torch.randint(low=0, high=n_samples, size=(N_windows,))
        traj = all_samples[torch.arange(N_windows), idx]   # [N_windows, 3]
        trajs.append(traj)

    return torch.stack(trajs)  # [n_traj, N_windows, 3]

def plot_sampled_trajectories(true_com, mean, trajs, max_trajs=20):
    """
    true_com: [N_windows, 3]
    mean:     [N_windows, 3]
    trajs:    [n_traj, N_windows, 3]
    """
    t = range(len(mean))
    labels = ["x", "y", "z"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    n_plot = min(max_trajs, trajs.shape[0])

    for d in range(3):
        for k in range(n_plot):
            axes[d].plot(t, trajs[k, :, d].numpy(), alpha=0.2)

        axes[d].plot(t, true_com[:, d].numpy(), linewidth=2, label="true")
        axes[d].plot(t, mean[:, d].numpy(), linewidth=2, label="posterior mean")
        axes[d].set_ylabel(labels[d])
        axes[d].legend()

    axes[-1].set_xlabel("time index")
    plt.tight_layout()
    plt.show()

#trajectory density per dimension
def plot_trajectory_density(true_com, all_samples):
    """
    all_samples: [T, S, 3]
    """
    T = all_samples.shape[0]
    labels = ["x", "y", "z"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for d in range(3):
        data = all_samples[:, :, d].numpy()  # [T, S]

        axes[d].imshow(
            data.T,
            aspect='auto',
            origin='lower',
            extent=[0, T, data.min(), data.max()],
        )

        axes[d].plot(true_com[:, d].numpy(), color="red", label="true")
        axes[d].set_ylabel(labels[d])
        axes[d].legend()

    axes[-1].set_xlabel("time")
    plt.tight_layout()
    plt.show()

#2D trajectory distribution
def plot_spatial_trajectory_density(true_com, trajs):
    """
    trajs: [n_traj, T, 3]
    """
    x = trajs[:, :, 0].reshape(-1)
    z = trajs[:, :, 2].reshape(-1)

    plt.figure(figsize=(6, 6))
    plt.hexbin(x, z, gridsize=50, cmap="Blues")

    plt.plot(true_com[:, 0], true_com[:, 2], color="red", linewidth=2, label="true")

    plt.xlabel("x")
    plt.ylabel("z")
    plt.legend()
    plt.title("Trajectory distribution (spatial)")
    plt.show()



#-------------------------PCA-------------------------








#-------------PROPER SCORING RULES----------------
#score for windows, how well does the posterior match the true com? (lower is better)
def energy_score(samples, y):
    """
    samples: [S, 3]
    y: [3]
    """
    S = samples.shape[0]

    term1 = torch.norm(samples - y, dim=1).mean()

    diff = samples.unsqueeze(0) - samples.unsqueeze(1)
    term2 = torch.norm(diff, dim=2).mean()

    return term1 - 0.5 * term2

def evaluate_energy(model, test_dataset, n_samples=200):
    scores = []

    for window, y in test_dataset:
        samples = model.sample(window.unsqueeze(0), n_samples).squeeze(0)

        score = energy_score(samples, y)
        scores.append(score.item())

    return sum(scores) / len(scores)

#mean error
import torch
import numpy as np
import matplotlib.pyplot as plt

def evaluate_mean_error(model, dataset, n_samples=200, device="cpu"):
    """
    Returns:
        mean_error: scalar (average over dataset)
        all_errors: list of per-window errors
    """
    model.eval()
    errors = []

    with torch.no_grad():
        for window, y in dataset:
            window = window.unsqueeze(0).to(device) 
            y = y.to(device)                         

            samples = model.sample(window, n_samples) 

            if samples.dim() == 3:
                samples = samples.squeeze(0)

            mean_pred = samples.mean(dim=0) 

            error = torch.norm(mean_pred - y) 
            errors.append(error.item())

    mean_error = np.mean(errors)

    print(f"Mean l2 error: {mean_error:.4f}")

    #histogram of errors
    plt.hist(errors, bins=50)
    plt.title("Mean error distribution")
    plt.xlabel("l2 error")
    plt.ylabel("count")
    plt.show()

    return mean_error, errors

def trajectory_energy_score(trajs, true_com):
    """
    trajs: [K, T, 3]
    true_com: [T, 3]
    """
    K = trajs.shape[0]

    trajs_flat = trajs.reshape(K, -1)        # [K, 3T]
    true_flat = true_com.reshape(-1)         # [3T]

    term1 = torch.norm(trajs_flat - true_flat, dim=1).mean()

    diff = trajs_flat.unsqueeze(0) - trajs_flat.unsqueeze(1)
    term2 = torch.norm(diff, dim=2).mean()

    return term1 - 0.5 * term2