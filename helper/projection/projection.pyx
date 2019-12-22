#cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np
cimport numpy as np

import cython

import yaml
with open('./helper/projection/sr300_610205001689.param', 'r') as fh:
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
@cython.wraparound(False)
@cython.boundscheck(False)
def pixel_to_point(short[:,:]depth_image):

    cdef:
        np.ndarray[np.float32_t, ndim=3] pointcloud = np.zeros((height, width, 3), dtype=np.float32)
        float x,y, r2, f, ux, uy
        int dx, dy
        float depth_value

    for dy in range(height):
        for dx in range(width):
            depth_value = depth_image[dy, dx] * depth_scale
            if depth_value == 0: continue

            x = (dx - ppx) / fx
            y = (dy - ppy) / fy
            r2  = x*x + y*y
            f = 1 + c0*r2 + c1*r2*r2 + c4*r2*r2*r2
            ux = x*f + 2*c2*x*y + c3*(r2 + 2*x*x)
            uy = y*f + 2*c3*x*y + c2*(r2 + 2*y*y)

            x = ux
            y = uy

            pointcloud[dy, dx, 0] = depth_value * x
            pointcloud[dy, dx, 1] = depth_value * y
            pointcloud[dy, dx, 2] = depth_value

    return pointcloud
    # https://github.com/IntelRealSense/librealsense/blob/7332ecadc057552c178addd577d24a2756f8789a/include/librealsense/rsutil.h#L11
    # rs_deproject_pixel_to_point
    # float x = (pixel[0] - intrin->ppx) / intrin->fx;
    # float y = (pixel[1] - intrin->ppy) / intrin->fy;
    # if(intrin->model == RS_DISTORTION_INVERSE_BROWN_CONRADY)
    # {
    #     float r2  = x*x + y*y;
    #     float f = 1 + intrin->coeffs[0]*r2 + intrin->coeffs[1]*r2*r2 + intrin->coeffs[4]*r2*r2*r2;
    #     float ux = x*f + 2*intrin->coeffs[2]*x*y + intrin->coeffs[3]*(r2 + 2*x*x);
    #     float uy = y*f + 2*intrin->coeffs[3]*x*y + intrin->coeffs[2]*(r2 + 2*y*y);
    #     x = ux;
    #     y = uy;
    # }
    # point[0] = depth * x;
    # point[1] = depth * y;
    # point[2] = depth;


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def point_to_pixel(float[:,:] pc):

    cdef:
        np.ndarray[np.float32_t, ndim=2] frame = np.zeros((height, width), dtype=np.float32)
        float x,y, r2, f, dx, dy
        int px, py
        int k
        float[:] point
        float p0, p1, p2

    for k in range(pc.shape[0]):
        point = pc[k, :3]

        x = point[0]/point[2]
        y = point[1]/point[2]

        r2  = x*x + y*y
        f = 1 + c0*r2 + c1*r2*r2 + c4*r2*r2*r2;
        x *= f
        y *= f
        dx = x + 2*c2*x*y + c3*(r2 + 2*x*x)
        dy = y + 2*c3*x*y + c2*(r2 + 2*y*y)

        px = int(x * fx + ppx)
        py = int(y * fy + ppy)

        if (0 < py < 480) and (0 < px < 640):
            if (frame[py, px] < point[2]):
                frame[py, px] = point[2]

    return frame
    # https://github.com/IntelRealSense/librealsense/blob/7332ecadc057552c178addd577d24a2756f8789a/include/librealsense/rsutil.h#L33
    # rs_deproject_pixel_to_point
    # float x = point[0] / point[2], y = point[1] / point[2];
    # if(intrin->model == RS_DISTORTION_MODIFIED_BROWN_CONRADY)
    # {
    #     float r2  = x*x + y*y;
    #     float f = 1 + intrin->coeffs[0]*r2 + intrin->coeffs[1]*r2*r2 + intrin->coeffs[4]*r2*r2*r2;
    #     x *= f;
    #     y *= f;
    #     float dx = x + 2*intrin->coeffs[2]*x*y + intrin->coeffs[3]*(r2 + 2*x*x);
    #     float dy = y + 2*intrin->coeffs[3]*x*y + intrin->coeffs[2]*(r2 + 2*y*y);
    #     x = dx;
    #     y = dy;
    # }
    # pixel[0] = x * intrin->fx + intrin->ppx;
    # pixel[1] = y * intrin->fy + intrin->ppy;
