import datetime
import pickle
import time
from os.path import basename

import IPython.display
import PIL.Image
import PIL.ImageChops
import PIL.ImageDraw
import PIL.ImageOps
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

__version__ = "0.1.0"


def auto_crop(image, threshold=200, show_steps=False):
    """
    Automatically crop image if background is black or white.
    :param image: Image
    :param threshold: Adjustable threshold value
    :param show_steps: Show image processing steps
    :return:
    """
    difference = PIL.ImageChops.difference(image,
                                           PIL.ImageOps.colorize(PIL.ImageOps.grayscale(image),
                                                                 black=(0, 0, 0),
                                                                 white=(255, 255, 255)))

    if show_steps:
        IPython.display.display('Difference', difference)

    difference = PIL.ImageChops.add(difference, difference, 2.0, -threshold)

    if show_steps:
        IPython.display.display('Difference', difference)

    bbox = difference.getbbox()
    return image.crop(bbox)


def auto_rotate(image, angle=-90):
    """
    Rotate image if it is taller than it is wide.
    :param image: Image
    :param angle: Rotation angle (-90 for clockwise rotation)
    :return: rotated image
    """
    width, height = image.size
    if height > width:
        return image.rotate(angle, expand=True)
    else:
        return image


def auto_resize(image, max_pixels=1e99, resample=PIL.Image.NEAREST):
    """
    Proportionally rescale image if it is larger than max_pixels
    :param image: image
    :param max_pixels: max number of pixels
    :param resample: resampling method
    :return: scaled image
    """
    image_size = np.asarray(image.size)
    image_pixels = np.prod(image_size)
    scaling_factor = np.sqrt(image_pixels / max_pixels)
    if scaling_factor < 1:
        return image
    else:
        new_size = np.floor(image_size / scaling_factor).astype(int)
        return image.resize(new_size, resample)


def auto_scale(nearest_indices, residual_norm, colorbar_data, xlim, ylim, clim):
    """
    Scale the image given by unmap_nearest to the original image's colorbar units and axis units.
    :param nearest_indices: image produced by unmap_image
    :param residual_norm: residuals produced by unmap_image
    :param colorbar_data: pixel image of the colorbar
    :param xlim: original image x-axis limits
    :param ylim: original image y-axis limits
    :param clim: original image colorbar limits.
    :return: vector of x positions, vector of y positions, scaled image and error estimate.
    """
    colorbar_length_pixels = colorbar_data.shape[0] - 1

    vmin, vmax = clim[0], clim[1]
    scaled_image = nearest_indices / colorbar_length_pixels * (vmax - vmin) + vmin
    scaled_error = residual_norm / colorbar_length_pixels * (vmax - vmin)
    assert not np.any(np.isnan(scaled_image)), "No NaN values expected in reconstructed image."
    assert not np.any(np.isnan(scaled_error)), "No NaN values expected in reconstructed image error."

    x = np.linspace(xlim[0], xlim[1], scaled_image.shape[1])
    y = np.linspace(ylim[0], ylim[1], scaled_image.shape[0])

    return x, y, scaled_image, scaled_error


def auto_hole_fill(data, error, threshold, radius=5):
    """
    A basic hole filling algorithm. Pixels whose error is larger than the error threshold
    are replaced by the median of their neighbours.
    :param data: image data
    :param error: error data
    :param threshold: error threshold
    :param radius: 
    :return: 
    """"))

    data_padded = np.empty([i + 2 * radius for i in data.shape])
    data_padded.fill(np.nan)

    data_padded[radius:-radius, radius:-radius] = data
    data_padded[radius:-radius, radius:-radius][error > threshold] = np.nan

    result = data
    ids = np.where(error > threshold)
    ids = np.array(ids)
    for x_id, y_id in ids.T:
        x_id += radius
        y_id += radius

        result[x_id - radius, y_id - radius] = \
            np.nanmedian(data_padded[x_id - radius:x_id + radius, y_id - radius:y_id + radius])

    return result


def make_header(file, size):
    """
    Make a header for csv output.
    :param file: Original input file name.
    :param size: Image dimensions.
    :return: header as string.
    """"

    now = datetime.datetime.now()
    string = ""
    string += "Created on %s from file \"%s\" \n" % (now, file)
    string += "using https://github.com/svaberg/imageslurper version %s\n" % __version__
    string += "Image dimensions " + str(size)
    return string


# See https://stackoverflow.com/questions/43843381/digitize-a-colormap
def unmap_nearest(image, colorbar, norm_order):
    """
    For each pixel in the image, find the index of the pixel in the colormap whose color value is closest
    in the provided norm.

    The image should by a numpy array of shape (..., c) and the colorbar is a numpy array of shape (k, c).
    The last dimension is assumed to be color (i.e. RGB) dimension
    The algorithm uses brute force to find the closest color of each input pixel.

    :param image: plot area image of shape (..., c)
    :param colorbar: colorbar color values of shape (k, c)
    :param norm_order: order of vector norm used to calculate color distance.
    :return: array of colorbar pixel indices of shape (...) with values in [0, k-1]
    """

    # Fully expanded distance array of shape (k, ..., 3).
    rgb_distances = np.linalg.norm(np.abs(image[np.newaxis, ...] - colorbar[:, np.newaxis, :]),
                                   ord=norm_order,
                                   axis=-1)

    # Minimum taken over k (axis 0).
    min_index = np.argmin(rgb_distances, axis=0)

    return min_index


def buffered_unmap(image, colorbar, chunk_size=1000, updater=None, norm_order=1):
    """
    Buffering wrapper for unmap_nearest.

    For larger images unmap_nearest may run out of memory. The buffered wrapper flattens the input
    array over all axes but the color axis and processes the flattened image in chunks.

    :param image: plot area image of shape (..., c)
    :param colorbar: colorbar color values of shape (k, c)
    :param chunk_size: buffer size in pixels
    :param updater: coroutine that can be used for progress monitoring
    :param norm_order: order of vector norm used to calculate color distance.
    :return:
    """
    if updater is not None:
        next(updater)

    small_negative = -1

    unmapped_image = np.ones(image.shape[:2], dtype=int) * small_negative

    rec_flat = unmapped_image.ravel()
    img_flat = image.reshape(-1, 3)

    pixel_id = 0
    while pixel_id <= unmapped_image.size:
        _slice = np.s_[pixel_id:pixel_id + chunk_size]

        rec_flat[_slice] = unmap_nearest(img_flat[_slice], colorbar, norm_order)

        if updater:
            updater.send(unmapped_image)

        pixel_id += chunk_size

    if updater:
        updater.send(unmapped_image)
        updater.close()

    return unmapped_image


def save_pickle(file, unmapped_filled_image, colorbar_data):
    """
    Helper function to save the unmapped image as a Python picke file.
    Saves the image data and the colorbar data.
    :param file: input file name
    :param unmapped_filled_image: image data
    :param colorbar_data: colorbar data
    :return: name of pickle file
    """
    save_file = basename(file) + "-slurped.p"

    with open(save_file, 'wb') as save_handle:
        pickle.dump([unmapped_filled_image, colorbar_data],
                    save_handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    # This reads the objects back:
    with open(save_file, 'rb') as load_handle:
        unmapped_filled_image, colorbar_data = pickle.load(load_handle)

    return save_file


def text_updater(file, update_freq=1):
    """
    Coroutine that prints the completion percentage of buffered_unmap
    :param file: input file name
    :param update_freq: update frequency in seconds.
    :return:
    """
    img = yield

    last_update = time.time()

    while True:
        img = yield

        if img[-1, -1] < 0 and last_update + update_freq > time.time():
            continue

        print("File " + file + " %4.2f%% complete." % (100 * np.count_nonzero(np.where(img >= 0)[0]) / img.size))
        last_update = time.time()


def plot_input(image_data, colorbar_image_data, axs):
    """
    Create a plot of the input image and the colorbar image data.
    :param image_data: image data
    :param colorbar_image_data: colorbar image data
    :param axs: plot axes
    :return:
    """
    ax = axs[0]
    ax.imshow(image_data)
    ax.set_title("Image array shape " + str(image_data.shape))
    ax.set_xlabel('Pixels')
    ax.set_ylabel('Pixels')

    ax = axs[1]
    ax.imshow(colorbar_image_data, interpolation='nearest', aspect='auto')

    ax2 = ax.twinx()
    for _id, color in enumerate('rgb'):
        x = np.arange(0, len(colorbar_image_data[0, :, 0]))
        ax2.fill_between(x, y1=np.min(colorbar_image_data[..., _id], axis=0),
                         y2=np.max(colorbar_image_data[..., _id], axis=0), color=color, alpha=.25)
        ax2.plot(x, np.median(colorbar_image_data[..., _id], axis=0), color=color)
    ax.invert_yaxis()
    ax.set_title("Colorbar image shape " + str(colorbar_image_data.shape))
    ax2.set_ylim(0, 256)
    ax2.set_yticks(np.linspace(0, 256, 9))
    ax.set_xlim(0, len(colorbar_image_data[0, :, 0]))
    ax2.set_ylabel('RGB intensity range and median')
    ax.set_xlabel('Pixels')
    ax.set_ylabel('Pixels')


def autoslurp(file,
              map_corners,
              colorbar_corners,
              error_threshold=None,
              xlim=(0, 1),
              ylim=(0, 1),
              clim=(0, 1),
              norm_order=2,
              updater=None,
              ):
    """
    Convenience function that goes through the full process.
    :param file: image input file
    :param map_corners: pixel corners of the plot area
    :param colorbar_corners: pixel corners of the colorbar area
    :param error_threshold: error threshold for hole filling
    :param xlim: plot x axis limits
    :param ylim: plot y axis limits
    :param clim: plot colorbar limits
    :param norm_order: order of vector norm used to calculate color distance.
    :param updater: Coroutine that prints the completion percentage of buffered_unmap
    :return:
    """
    if updater is None:
        updater = text_updater(file)

    full_image = PIL.Image.open(file)

    map_image = full_image.crop(map_corners)
    colorbar_image = full_image.crop(colorbar_corners)

    map_image = auto_crop(map_image, threshold=100, show_steps=False)
    colorbar_image = auto_crop(colorbar_image, threshold=100, show_steps=False)
    colorbar_image = auto_rotate(colorbar_image)

    image_data = np.asarray(map_image)

    assert not np.any(np.isnan(image_data)), "No NaN values expected in plot area rgb image."

    colorbar_image_data = np.asarray(colorbar_image)
    assert not np.any(np.isnan(colorbar_image_data)), "No NaN values expected in colorbar rgb image"

    colorbar_data = np.median(colorbar_image_data, axis=0)  # One pixel wide

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    plot_input(image_data, colorbar_image_data, axs)
    plt.show()

    nearest_indices = buffered_unmap(image_data, colorbar_data, updater=updater, norm_order=1)
    assert (nearest_indices.shape[:2] == image_data.shape[:2])

    mapped_colors = colorbar_data[nearest_indices]

    residual_rgb = image_data - mapped_colors
    residual_norm = np.linalg.norm(residual_rgb, ord=norm_order, axis=-1)

    x, y, scaled_image, scaled_residual = auto_scale(nearest_indices, residual_norm,
                                                     colorbar_data,
                                                     xlim=xlim,
                                                     ylim=ylim,
                                                     clim=clim)

    fig, ax = plt.subplots(figsize=(14, 6))
    img = ax.pcolormesh(x, y, scaled_image, cmap='viridis')
    ax.set_title('Reconstructed dataset')
    plt.colorbar(img, ax=ax)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title('Reconstruction residual')
    img = ax.pcolormesh(x, y, scaled_residual, cmap='magma')
    plt.colorbar(img, ax=ax)

    #
    # Hole filling
    #
    if error_threshold is not None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        ax.set_title('Bad pixels')
        ax.imshow(1.0 * (scaled_residual > error_threshold), cmap='magma')

        unmapped_filled_image = auto_hole_fill(scaled_image, scaled_residual, error_threshold)

        if np.any(np.isnan(unmapped_filled_image)):
            print("Some NaN values could not be filled.")

    else:
        unmapped_filled_image = scaled_image

    #
    # Error analysis
    #
    fig, ax = plt.subplots()
    plot_residual_histogram(ax, norm_order, residual_norm, residual_rgb)
    plt.show()

    #
    # Comparison with original
    #
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(full_image)
    ax.set_title("Original image area")

    fig, ax = plt.subplots(figsize=(14, 6))
    original_colormap = ListedColormap(colorbar_data / 255)

    img = ax.pcolormesh(x, y, unmapped_filled_image, cmap=original_colormap)
    fig.colorbar(img)
    plt.title('Reconstructed dataset using original colorbar')

    # Save pickle and csv file.
    save_pickle(file, unmapped_filled_image, colorbar_data)

    np.savetxt(basename(file) + "-slurped.csv",
               unmapped_filled_image,
               delimiter=", ",
               fmt="%0.6e",
               header=make_header(file, unmapped_filled_image.shape))

    return locals()


def plot_residual_histogram(ax, norm_order, residual_norm, residual_rgb):
    """
    Plot a histogram of the unmapping residuals.
    :param ax: plot axis
    :param norm_order: order of vector norm used to calculate color distance.
    :param residual_norm: residual norm
    :param residual_rgb: residual rgb values
    :return:
    """
    hatch = ['|||', '///', '---']
    np.min(residual_norm)
    bins = np.logspace(np.log10(np.min(residual_norm[np.where(residual_norm > 0)])),
                       np.log10(np.max(residual_norm)), 50)
    ax.hist(residual_norm.ravel(),
            bins=bins,
            histtype='stepfilled',
            color='gray',
            label=str(norm_order) + " norm")
    for _id, color in enumerate(('red', 'green', 'blue')):
        ax.hist(residual_rgb[..., _id].ravel(),
                bins=bins,
                histtype='step',
                color=color,
                label=color,
                hatch=hatch[_id],
                edgecolor=color,
                alpha=0.3)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('Pixels')
    ax.set_xlabel('RGB residual')
    ax.set_title('Residuals histogram')
    plt.legend()
