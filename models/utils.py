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

def make_gif(frame_list:list,filename,title=""):
    num_frames = len(frame_list)


    frames = []

    lambda_list = np.linspace(0,1.2,num_frames)


    # 2. Loop to generate each Matplotlib figure (frame)
    for i in enumerate(frame_list):
        # Create a figure and axis for the plot
        fig, ax = plt.subplots()
        
        # --- Capture the plot as a numpy array (image data) ---
        # The figure is rendered, and the data is captured directly into a numpy array.
        # This avoids saving a file for every frame.
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())

        # Close the figure to free up memory
        plt.close(fig)

        # Add the captured image to the frames list
        frames.append(image)
    # 3. Save the list of frames as a GIF
    # 'duration' is in milliseconds for imageio.v3
    DURATION_MS = 100

    import imageio.v2 as iio2
    iio2.mimsave(
        filename,          # Output file name
        frames,            # List of frames (NumPy arrays)
        duration=DURATION_MS,
        loop=0             # 0 means loop forever
    )
