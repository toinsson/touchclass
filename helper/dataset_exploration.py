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

def display_dataset_md(hf):
    from IPython.display import display, Markdown
    return display(Markdown(hdf5_to_md(hf)))
