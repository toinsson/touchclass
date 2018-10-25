import numpy as np
import vtk
import pointobject
from matplotlib.pyplot import cm


def plot_objects(*args, **kwargs):
    ren = vtk.vtkRenderer()

    colors = cm.rainbow(np.linspace(0,1,len(args)))
    objs = []
    for i, arg in enumerate(args):

        # make sure input is ok
        if len(arg.shape) > 2:
            arg = arg.reshape(-1,3)
        if not arg.flags.contiguous:
            arg = np.ascontiguousarray(arg)

        obj = pointobject.VTKObject()
        obj.CreateFromArray(arg)

        color_cur = (colors[i][:3]*255).astype(int)
        obj.AddColors(np.ones((arg.shape[0],1), dtype=int) * color_cur)

        ren.AddActor(obj.GetActor())

        objs.append(obj)

    if ('axis' in kwargs) and (kwargs['axis'] is not False):
        axes = pointobject.VTKObject()
        axes.CreateAxes(1)
        ren.AddActor(axes.GetActor())

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)
    iren.Initialize()
    iren.Start()
