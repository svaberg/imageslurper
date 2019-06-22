import pickle
from os.path import basename

import IPython.display

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import HTML
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from matplotlib.collections import PatchCollection

import PIL.Image
import PIL.ImageDraw
import PIL.ImageOps
import PIL.ImageChops

import datetime

__version__ = "0.1.0"

def crop(image, thresh=200):

    difference = PIL.ImageChops.difference(image, 
                                           PIL.ImageOps.colorize(PIL.ImageOps.grayscale(image), 
                                                                 black=(0,0,0), 
                                                                 white=(255, 255, 255)))

    difference = PIL.ImageChops.add(difference, difference, 2.0, -thresh)

    bbox = difference.getbbox()
    return image.crop(bbox)


def make_header(file, size):
    now = datetime.datetime.now()
    string = ""
    string += "Created on %s from file \"%s\" \n" % (now, file)
    string += "using https://github.com/svaberg/imageslurper version %s\n" % __version__
    string += "Image dimensions " + str(size)
    return string
