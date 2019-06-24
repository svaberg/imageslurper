import time

import IPython.display
import numpy as np
from matplotlib import pyplot as plt


def plot_updater(ax=None, cmap=('viridis', 'magma'), update_freq=1):
    img = yield

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))

    img_img = ax.imshow(img, cmap=cmap[0])
    ax.set_xlabel('Pixels')
    ax.set_ylabel('Pixels')
    ax.set_title('Finding nearest colors')
    img_cb = plt.colorbar(img_img, ax=ax)
    img_cb.set_clim(0, np.max(img))
    img_cb.set_label('Index of nearest pixel in colormap')

    # err_img = axs[1].imshow(err, cmap=cmap[1])
    # err_cb = plt.colorbar(err_img, ax=axs[1])
    # err_cb.set_clim(0, np.max(err))

    last_update = time.time()

    while True:
        img = yield

        if img[-1, -1] < 0 and last_update + update_freq > time.time():
            continue

        if np.isclose(img_img.get_clim()[1], np.max(img)):
            img_img.set_data(img)
        else:
            img_img = ax.imshow(img, cmap=cmap[0])
            img_cb.set_clim(0, np.max(img))
            img_cb.update_normal(img_img)

        # if np.isclose(err_cb.get_clim()[1], np.max(err)):
        #     err_img.set_data(err)
        # else:
        #     err_img = axs[1].imshow(err, cmap=cmap[1])
        #     err_cb.set_clim(0, np.max(err))
        #     err_cb.update_normal(err_img)

        IPython.display.clear_output(wait=True)
        if img[-1, -1] <= 0:
            IPython.display.display(plt.gcf())

        last_update = time.time()
