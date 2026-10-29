import math

import matplotlib.pyplot as plt


def display(img, cmap=None):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap)
    plt.show()


def display_grid(images: list, cmap=None, max_cols=2, titles: None | list[str] = None):
    """
    Display images in a dynamicc grid.
    - images: list of numpy arrays
    - cmap: optional colormap
    - max_cols: maximum number of columns before wrapping
    """

    n = len(images)
    cols = min(n, max_cols)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows))

    # normalize axes to a flat list for easy iteration
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.ravel()

    # hide unused axes
    i = 0
    for ax, img in zip(axes, images):
        ax.imshow(img, cmap=cmap)
        ax.axis("off")
        if titles:
            ax.set_title(titles[i])
            i += 1

    plt.tight_layout()
    plt.show()
