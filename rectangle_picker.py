import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector

import PIL.Image

_boxes = [Rectangle((1, 1), 0, 0, facecolor='green', edgecolor='green', alpha=0.4),
          Rectangle((1, 1), 0, 0, facecolor='blue', edgecolor='blue', alpha=0.4)]

_user_box_count = 0

_rectangle_selector = None


def line_select_callback(eclick, erelease):
    global _user_box_count
    'eclick and erelease are the press and release events'
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata

    bid = _user_box_count % len(_boxes)
    _boxes[bid].set_x(x1)
    _boxes[bid].set_y(y1)
    _boxes[bid].set_width(x2 - x1)
    _boxes[bid].set_height(y2 - y1)

    _user_box_count += 1

    print(_user_box_count)


def select_rectangles(file="img/world-temp.jpg"):
    global _rectangle_selector
    full_image = PIL.Image.open(file)

    fig, current_ax = plt.subplots(figsize=(10, 4))
    current_ax.imshow(full_image)
    current_ax.axis('off')

    current_ax.add_patch(_boxes[0])
    current_ax.add_patch(_boxes[1])

    _rectangle_selector = RectangleSelector(current_ax, line_select_callback,
                            drawtype='box', useblit=True,
                            button=[1],  # don't use middle button
                            minspanx=5, minspany=5,
                            spancoords='pixels',
                            interactive=True,
                            rectprops=dict(facecolor='none', edgecolor='black',
                                           alpha=1, fill=True))


def print_picker_result():
    print("User picked map corners " + str(map_corners()) +
          " and colorbar corners " + str(bar_corners()) +
          " in %d actions." % _user_box_count)

def map_corners():
    return _boxes[0].get_bbox().extents


def bar_corners():
    return _boxes[1].get_bbox().extents
