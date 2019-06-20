# The image slurper
Are you **tired** of asking people for the data behind their published false color plots? Use the image slurper to reconstruct the underlying matrix of values from a false-color (heat map) image and its colorbar! 

The image slurper can be used on many types of false color plots, and works best on relatively smooth data.

The image slurper is a [Jupyter worksheet](imageslurper.ipynb) that takes an image like this one

![Source image](img/world-temp.jpg)

and reconstructs the original pixel values by comparing each pixel to the colors of the colorbar, giving a `numpy` array
```python
array([[232.39477504, 234.42670537, 234.86211901, ..., 234.86211901,
        234.42670537, 233.84615385],
       ...,
       [206.85050798, 207.28592163, 207.57619739, ..., 208.30188679,
        207.28592163, 209.17271408]])
```
which can be plotted with `matplotlib` using the same colorbar as the image, giving a result like below.

![Result image](world-temp.jpg.reconstructed.jpg)

For a well executed slurp the input image plot areas should be visually indistinguishable from the result plot area.

The brute force algorithm used is based on this Stack Overflow answer: ["Digitize a colormap"](https://stackoverflow.com/a/43844204/3198895).