#cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np
cimport numpy as np

import cython

import yaml
with open('./helper/deproject/sr300_610205001689.param', 'r') as fh:
    d = yaml.load(fh, yaml.Loader)
intr_ = d['610205001689']

# intr = rs.intrinsics()
_coeffs = intr_['coeffs']
cdef float c0, c1, c2, c3, c4
c0  = _coeffs[0]
c1  = _coeffs[1]
c2  = _coeffs[2]
c3  = _coeffs[3]
c4  = _coeffs[4]

cdef int width  = intr_['width']
cdef int height = intr_['height']
cdef float ppx    = intr_['ppx']
cdef float ppy    = intr_['ppy']
cdef float fx     = intr_['fx']
cdef float fy     = intr_['fy']

cdef double depth_scale = intr_['depth_scale']

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

@cython.cdivision(True)
def compute(short[:,:]depth_image):

    cdef np.ndarray[np.float32_t, ndim=3] pointcloud = np.zeros((height, width, 3), dtype=np.float32)
    cdef float x,y, r2, f, ux, uy
    cdef int dx, dy
    cdef float depth_value

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
