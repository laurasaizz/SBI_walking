from typing import Callable
import numpy as np


def build_warmup_epochs(warmup_epochs: int = 10, epochs: int = 100) -> Callable:
    """
    builds a lr-factor lambda  function for lr schedule
    """
    decay_epochs = epochs - warmup_epochs

    def lr_lambda(epoch: int) -> float:
        if epoch < 10:
            return (epoch + 1) / warmup_epochs
            # goes from 0.1 to 1 on warmup_epochs=10
        else:

            return (
                0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs - 1) / decay_epochs))
            ).item()  # cos from 0 to pi

    return lr_lambda


if __name__ == "__main__":
    warmup_epochs = 10
    num_epochs = 100
    x = np.arange(num_epochs)
    lam_func = build_warmup_epochs(warmup_epochs=warmup_epochs, epochs=num_epochs)
    y = [lam_func(epoch) for epoch in x]
    import matplotlib.pyplot as plt

    plt.plot(x, y, label="lr factor over epoch")
    plt.legend()
    plt.grid()
    plt.show()
