import pytest
import PIL
import numpy as np
import matplotlib.pyplot as plt
import imageslurper

@pytest.mark.parametrize("error_threshold", (None, 20))
def test_autoslurp(error_threshold):
    """
    Test of the whole process.
    """
    plt.ion()

    imageslurper.autoslurp(file="img/world-temp.jpg",
                           map_corners=np.array([  50.,   50., 1000.,  550.]),
                           colorbar_corners=np.array([150., 560., 900., 576.]),
                           error_threshold=error_threshold,  # In clim units
                           xlim = (-180, 180),
                           ylim = (90, -90),
                           clim = (180, 280),
                           norm_order=2,
                           updater=None,
                           )


@pytest.mark.parametrize("max_pixels", (100, 1e99))
def test_auto_resize(max_pixels):
    """
    Test of auto resize as it is not part of the usual slurp.
    """
    imageslurper.auto_resize(
        PIL.Image.open("img/world-temp.jpg"), 
        max_pixels=max_pixels, 
        resample=PIL.Image.NEAREST)
