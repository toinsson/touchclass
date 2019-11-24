import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import h5py

from IPython.display import display
from ipywidgets import interactive
import ipywidgets as widgets

from matplotlib.patches import Rectangle

from matplotlib import cm
import scipy.interpolate as si
def int16_to_rgb(frame):
    # calculate depth histogram
    hist, edges = np.histogram(frame, bins=100)

    # calculate cumulative depth histogram
    hist = np.cumsum(hist)
    hist -= hist[0]

    rgb_frame = np.zeros(frame.shape[:2] + (3,), dtype=np.uint8)
    zeros = frame==0
    non_zeros = frame!=0
    f_hist = si.interp1d(edges[1:], hist/hist.max())
    f = f_hist(frame[non_zeros])

    rgb_frame[non_zeros] = cm.viridis(f, alpha=None, bytes=True)[:, :3]
    rgb_frame[zeros, 0] = 0
    rgb_frame[zeros, 1] = 5
    rgb_frame[zeros, 2] = 20

    return rgb_frame


def get_data_from_dataset(filename, frame_number):
    with h5py.File(filename, 'r') as hf:
        frame = hf['data/depth'][frame_number]
        frame[0,0] = frame[0,1] = 0
        mat = np.array(hf['matrix'], dtype=np.float32)
        org = np.array(hf['origin'], dtype=np.float32)
        ext = np.array(hf['extrema'], dtype=np.float32)

    return frame, mat, org, ext
