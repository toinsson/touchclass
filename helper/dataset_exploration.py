import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import h5py

def begin_table():
    return """<table>
        <tr>
        <th>name</th>
        <th>shape</th>
        <th>dtype</th>
        </tr>
    """

def create_row(row):
    return """<tr>
        <td>{}</td>
        <td>{}</td>
        <td>{}</td>
    </tr>
    """.format(*row)

def end_table():
    return """</table>"""

def hdf5_to_md(hf):

    d = []
    def save_structure(name, obj):
        if obj.__class__ is h5py.Dataset:
            d.append([name, hf[name].shape, hf[name].dtype])
    hf.visititems(save_structure)

    return begin_table()+"".join([create_row(row) for row in d])+end_table()

from matplotlib import cm
import scipy.interpolate as si

def int16_to_rgb(frame):
    # calculate depth histogram
    hist, edges = np.histogram(frame, bins=100)
    # center = (edges[:-1] + edges[1:]) / 2

#     plt.figure(figsize=(16, 4))
#     plt.subplot(1, 2, 1)
#     plt.scatter(edges[:-1], hist, s=1)
#     plt.title('Depth histogram')

    # calculate cumulative depth histogram
    hist = np.cumsum(hist)
    hist -= hist[0]
#     plt.subplot(1, 2, 2)
#     plt.scatter(edges[:-1], hist, s=1)
#     plt.title('Cumulative depth histogram')
#     plt.tight_layout()

    rgb_frame = np.zeros(frame.shape[:2] + (3,), dtype=np.uint8)

    zeros = frame==0
    non_zeros = frame!=0

    f_hist = si.interp1d(edges[1:], hist/hist.max())
    f = f_hist(frame[non_zeros])

    # f = hist[frame[non_zeros]] * 255 / hist[0xFFFF]
    # rgb_frame[non_zeros, 0] = f*255
    # rgb_frame[non_zeros, 1] = 200
    # rgb_frame[non_zeros, 2] = 255 - f

    rgb_frame[non_zeros] = cm.viridis(f, alpha=None, bytes=True)[:, :3]

    rgb_frame[zeros, 0] = 0
    rgb_frame[zeros, 1] = 5
    rgb_frame[zeros, 2] = 20

    return rgb_frame