import numpy as np
import pyrealsense2 as rs


import yaml
with open('sr300_610205001689.param', 'r') as fh:
    d = yaml.load(fh)
intr_ = d['610205001689']

intr = rs.intrinsics()
intr.coeffs = intr_['coeffs']
intr.width  = intr_['width']
intr.height = intr_['height']
intr.ppx    = intr_['ppx']
intr.ppy    = intr_['ppy']
intr.fx     = intr_['fx']
intr.fy     = intr_['fy']
# intr.model  = intr_['model']
intr.coeffs = intr_['coeffs']
ds = intr_['depth_scale']


# def compute(short[:,:]depth_image):

#     # intrinsics and depth_scale are global
#     cdef double depth_scale = ds

#     cdef float[:,:,:] pointcloud = np.zeros((intr.height, intr.width, 3), dtype=np.float32)

#     cdef int height = intr.height
#     cdef int width  = intr.width
#     cdef int dx, dy
#     cdef float depth_value

#     for dy in range(height):
#         for dx in range(width):
#             depth_value = depth_image[dy, dx] * depth_scale
#             if depth_value == 0: continue
#             res = rs.rs2_deproject_pixel_to_point(intr, [dx, dy], depth_value)
#             for i in range(3):
#                 pointcloud[dy, dx][i] = res[i]
#     return pointcloud

cimport numpy as np

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

def compute(short[:,:]depth_image):

    # intrinsics and depth_scale are global
    cdef double depth_scale = ds


    cdef int height = intr.height
    cdef int width  = intr.width
    cdef int dx, dy
    cdef float depth_value

    cdef np.ndarray[DTYPE_t, ndim=3] pointcloud = np.zeros((height, width, 3), dtype=np.float32)

    cdef float x,y, r2, f, ux, uy
    cdef float ppx, ppy, fx, fy, c0, c1, c2, c3, c4

    ppx = intr.ppx
    ppy = intr.ppy
    fx  = intr.fx
    fy  = intr.fy
    c0  = intr.coeffs[0]
    c1  = intr.coeffs[1]
    c2  = intr.coeffs[2]
    c3  = intr.coeffs[3]
    c4  = intr.coeffs[4]

    for dy in range(height):
        for dx in range(width):
            depth_value = depth_image[dy, dx] * depth_scale
            if depth_value == 0: continue

            x = (dx - ppx) / fx;
            y = (dy - ppy) / fy;
            r2  = x*x + y*y;
            f = 1 + c0*r2 + c1*r2*r2 + c4*r2*r2*r2;
            ux = x*f + 2*c2*x*y + c3*(r2 + 2*x*x);
            uy = y*f + 2*c3*x*y + c2*(r2 + 2*y*y);

            x = ux;
            y = uy;

            pointcloud[dy, dx, 0] = depth_value * x;
            pointcloud[dy, dx, 1] = depth_value * y;
            pointcloud[dy, dx, 2] = depth_value;

    return pointcloud
