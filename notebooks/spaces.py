import logging
logger = logging.getLogger(__name__)

import numpy as np
import cv2

import warnings

pcl_available = False
try:
    import pcl
    pcl_available = True
except ImportError:
    # warnings.warn("PCL not available.")
    pcl_available = False


# from fistwriter.utils import ProcessingError
class ProcessingError(Exception):
    """Error thrown during the processing in case the processing chain needs to be exited."""
    def __init__(self, msg, value):
        self.msg = msg
        self.value = value
    def __str__(self):
        return repr(self.msg)



H0 = 16.93
W0 = 25.4
surfacePro3 = { 
                'S1':{'width':H0*4, 'height':W0*2},
                'S2':{'width':W0*2, 'height':H0*2},
                'S3':{'width':H0*2, 'height':W0  },
                'S4':{'width':W0  , 'height':H0  },
                'S5':{'width':H0  , 'height':W0/2},
                'S6':{'width':W0/2, 'height':H0/2},
                }


def cm_to_pixel(cm, dpi=72):
    CENTIMETERS_IN_INCH = 2.54
    """Convert centimeters into pixel at 72dpi."""
    return int(np.round(cm * dpi/CENTIMETERS_IN_INCH))


class Pattern(object):
    def __init__(self, filepathOrArray, image, shape='S4'):

        if type(filepathOrArray) == str:
            self.filePath = filepathOrArray
            self.pattern = cv2.imread(self.filePath,0)
        elif type(filepathOrArray) == np.ndarray:
            self.pattern = filepathOrArray
        else:
            raise TypeError("Wrong pattern input, must be file descriptor or ndarray.")

        self.shape = shape
        if shape in ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']:
            self.heightCm = surfacePro3[shape]['height']
            self.widthCm = surfacePro3[shape]['width']
            self.heightPxl = cm_to_pixel(self.heightCm)
            self.widthPxl = cm_to_pixel(self.widthCm)
        else:
            raise ValueError('wrong shape name %s'%shape)

        self.image = None
        self.kp1 = None
        self.kp2 = None
        self.good = None
        self._homography = None
        self.mask = None

        self.find_homography(image)

    @property
    def homography(self):
        return self._homography
    @homography.setter
    def homography(self, value):
        self._homography = value


    def find_homography(self, image):
        """Find and save the homography between pattern and image. If not found mask is None.
        """
        FLANN_INDEX_KDTREE = 0
        MIN_MATCH_COUNT = 5

        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(self.pattern,None)
        kp2, des2 = sift.detectAndCompute(image,None)

        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        bfMatcher = cv2.BFMatcher()

        ## use KNN matcher and apply distance check between 2 neighbour
        matches = bfMatcher.knnMatch(des1, des2, k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            self.homography, self.mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        else:
            logger.warning('not enough good match for the pattern homography')
            raise ProcessingError('Homography not found', len(good))

        ## save for plotting
        self.image = image
        self.kp1, self.kp2 = kp1, kp2
        self.good = good


    # absCoords
    @property
    def pattern_corners(self):
        """Corners expressed in pixels in the pattern coordinate system.
        """
        x = self.heightPxl
        y = self.widthPxl
        return np.float32([ [0,0],[0,y],[x,y],[x,0] ]).reshape(-1,1,2)


    @property
    def image_corners(self):
        """Corners expressed as position in the image space, using the homography computed from
        the attached pattern.
        """
        C, H = self.pattern_corners[:,:,::-1], self.homography
        corners = cv2.perspectiveTransform(C, H).astype(dtype=int)
        return corners[:,:,::-1]  # put corner back to np convention


    def draw_matches(self):
        """Draw the matches between pattern and image used to find the homography.
        """
        return cv2.drawMatches(self.pattern,self.kp1,
                               self.image,self.kp2,
                               self.good,None)


    def draw_contours(self, image):
        """Draw the recorded pattern contour in image.
        """
        dst = self.image_corners[:,:,::-1]
        return cv2.polylines(image,[np.int32(dst)],True,255,3, cv2.LINE_AA)



class System(object):
    def __init__(self, pattern, pcl):
        super(System, self).__init__()
        self.pattern = pattern
        self.find_model(pcl)


    def find_model(self, pcl):
        self.pcl = pcl.copy()

        # corners_mean = np.mean(self.pattern.pattern_corners, axis=0).reshape(1,1,2)[...,::-1]
        # self.depth_center = cv2.perspectiveTransform(corners_mean, self.pattern.homography)
        # self.depth_center = self.depth_center.astype(int).reshape(-1)[...,::-1]
        # self.pcl_center = self.pcl[tuple(self.depth_center)]

        # self.depth_corners = self.pattern.image_corners
        # self.pcl_corners = np.array([self.pcl[tuple(c[0])] for c in self.depth_corners])
        self.depth_corners = self.pattern.image_corners
        self.depth_center = cv2.perspectiveTransform(np.mean(self.pattern.pattern_corners, axis=0).reshape(1,1,2)[...,::-1], self.pattern.homography).astype(int).reshape(-1)[...,::-1]
        self.pcl_center = self.pcl[tuple(self.depth_center)]

        self.pcl_corners = np.array([self.pcl[tuple(c[0])] for c in self.depth_corners])

        if (self.pcl_corners == [0,0,0]).all(axis = 1).any():
            raise ProcessingError("System corner points to undefined", "")

        self._set_plane_model()
        self._set_trans_matrix()


    @property
    def plane_model(self):
        """Fit a plane with RANSAC, get model ax+by+cz+d = 0 on a cloud around surface center.
        """
        return self._planeModel
    def _set_plane_model(self):
        ## algo parameter
        SURFACE_RADIUS = np.sqrt(np.power(self.pattern.heightCm/100/2,2) + 
                                 np.power(self.pattern.widthCm/100/2,2))

        norm = np.linalg.norm(self.pcl - self.pcl_center, axis=2)
        smallcloud = self.pcl[np.where(norm < (SURFACE_RADIUS))]


        if pcl_available:

            planePcl = pcl.PointCloud(smallcloud)
            seg = planePcl.make_segmenter()
            seg.set_optimize_coefficients(True)
            seg.set_model_type(pcl.SACMODEL_PLANE)
            # seg.set_normal_distance_weight(0.01)
            seg.set_method_type(pcl.SAC_RANSAC)
            # seg.set_max_iterations(1000)
            seg.set_distance_threshold(0.01)
            indices, model = seg.segment()
            self._planeModel = model

        else:
            import sklearn
            from sklearn import linear_model

            X, y = np.c_[np.ones(smallcloud.shape[0]), smallcloud[:,:2]], smallcloud[:,2:]

            linear_regression = linear_model.LinearRegression(fit_intercept=False)
            model_ransac = linear_model.RANSACRegressor(linear_regression, residual_threshold=0.1)
            model_ransac.fit(X, y)
            inlier_mask = model_ransac.inlier_mask_
            outlier_mask = np.logical_not(inlier_mask)

            model = model_ransac.estimator_.coef_[0]

            self._planeModel = np.r_[model[1], model[2], -1, model[0]]


    @property
    def coord_system(self):
        """Return the 3d coordinates system of the surface as 3d dimentional
        unit vectors x,y,z. Make sure the system is direct.
        """
        x,o,y,z = self.pcl_corners
        normal = np.array(self.plane_model[:3])

        # compute and normalize x,y
        xv = (x-o) / np.linalg.norm(x-o)
        yv = (y-o) / np.linalg.norm(y-o)

        # Check normal sign to make it direct with x,y
        dot = np.sum(np.cross(xv, yv) * normal)
        if dot < 0:
            normal = -normal

        # normalise normal
        normal /= np.linalg.norm(normal)

        return xv, yv, normal


    @property
    def trans_matrix(self):
        """Return the 3d transformation matrix between the camera and the 
        surface 3d coordinate system.
        """
        return self._trans_matrix
    def _set_trans_matrix(self):
        x,o,y,z = self.pcl_corners
        xv, yv, normal = self.coord_system

        ## scaling matrix - do not scale against the pattern
        S = np.eye(3) # * [1/np.linalg.norm(x-o), 1/np.linalg.norm(y-o), 1]

        ## rotation matrix
        RM = np.vstack((xv, yv, normal))

        ## final matrix
        SRM = np.dot(S, RM)

        self._trans_matrix = SRM.astype(np.float32)


    def normalise(self, cloud):
        # get coordinates and transformation matrix
        org = self.pcl_center
        trans_matrix = self.trans_matrix

        # translate and transform
        return np.dot(trans_matrix, (cloud - org).reshape((-1,3)).T).T


    def denormalise(self, cloud):
        # get coordinates and transformation matrix
        org = self.pcl_center
        trans_matrix_inv = np.linalg.inv(self.trans_matrix)

        # translate and transform
        return np.dot(trans_matrix_inv, cloud.reshape((-1,3)).T).T + org


class Scene(object):
    def __init__(self, camera, pattern, shape = 'S4'):

        self.camera = camera

        ## find the pattern
        self.camera.wait_for_frames()
        self.cad = self.camera.cad.copy()
        self.pcl = self.camera.pointcloud.copy()

        self.pattern = Pattern(pattern, self.cad)

        ## get the x,y,z system
        self.system = System(self.pattern, self.pcl)

        self.find_shape(shape)


    def find_shape(self, shape):
        self.shape = shape

        ## using SX convention
        if shape in ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']:

            ## scene size in relation to the system size
            self.heightCm = surfacePro3[shape]['height']
            self.widthCm = surfacePro3[shape]['width']

        ## using "WxH" width and height convention in cm
        elif 'x' in shape:
            self.widthCm, self.heightCm = map(float, shape.split('x'))

        else:
            raise ValueError("""wrong shape, must be S{1,2,3,4,5,6} or {'h':Y, 'w':X}.""")


        # import ipdb; ipdb.set_trace()

        self.heightPxl = cm_to_pixel(self.heightCm)
        self.widthPxl = cm_to_pixel(self.widthCm)

        self.xmin = - self.widthCm / 100 / 2
        self.xmax = self.widthCm / 100 / 2
        self.ymin = - self.heightCm / 100 / 2
        self.ymax = self.heightCm / 100 / 2

        ## add 5cm margin
        margin = 0.03
        self.xmin_ = self.xmin - margin
        self.xmax_ = self.xmax + margin
        self.ymin_ = self.ymin - margin
        self.ymax_ = self.ymax + 0.02


    @property
    def pcl_corners(self):
        return np.array([
            [self.xmax, self.ymin, 0],
            [self.xmin, self.ymin, 0],
            [self.xmin, self.ymax, 0],
            [self.xmax, self.ymax, 0],
        ])


    @property
    def image_corners(self):
        ## denormalise scene corners
        self.pcl_corners_ = self.system.denormalise(self.pcl_corners)

        ## deproject points to pixel
        image_corners = np.zeros((4,2))
        for i, point in enumerate(self.pcl_corners_):
            image_corners[i] = self.camera.project_point_to_pixel(point)

        ## might need to store in NOT opencv convention
        return image_corners[:,None,:].astype(int)


    ## specifically to draw on input stream image (from homography)
    def draw_surface(self, image):
        return cv2.polylines(image, [self.image_corners], True, (255,0,255), 3, cv2.LINE_AA)

    @property
    def pcl_corners_margins(self):
        return np.array([
            [self.xmax_, self.ymin_, 0],
            [self.xmin_, self.ymin_, 0],
            [self.xmin_, self.ymax_, 0],
            [self.xmax_, self.ymax_, 0],
        ])
    @property
    def image_corners_margins(self):
        ## denormalise scene corners
        self.pcl_corners_ = self.system.denormalise(self.pcl_corners_margins)

        ## deproject points to pixel
        image_corners = np.zeros((4,2))
        for i, point in enumerate(self.pcl_corners_):
            image_corners[i] = self.camera.project_point_to_pixel(point.astype(np.float32))

        ## might need to store in NOT opencv convention
        return image_corners[:,None,:].astype(int)
    ## specifically to draw on input stream image (from homography)
    def draw_surface_margins(self, image):
        return cv2.polylines(image, [self.image_corners_margins], True, (0,255,0), 3, cv2.LINE_AA)


    @property
    def volume(self):
        """Latest pointcloud from the scene, filtered."""
        self.camera.wait_for_frames()
        norm = self.system.normalise(self.camera.pointcloud)
        filt = self._filter(norm)
        # scal = self._scale(filt)
        return filt


    def get_volume(self, pc):
        """Latest pointcloud from the scene, filtered."""
        norm = self.system.normalise(pc)
        filt = self._filter(norm)
        # scal = self._scale(filt)
        return filt


    def _filter(self, obj):
        xm = (obj[:,0] >= self.xmin_) & (obj[:,0] <= self.xmax_)
        ym = (obj[:,1] >= self.ymin_) & (obj[:,1] <= self.ymax_)
        return obj[xm & ym]


    # def _scale(self, obj):
    #     obj = np.add(obj, (self.xmin, self.ymin, 0), dtype=np.float32)
    #     obj = np.multiply(obj, (1/self.scale_x, 1/self.scale_y, 1), dtype=np.float32)
    #     return obj


    def norm_coord(self, x, y):
        x = (x - self.widthCm/100/2)/(self.widthCm/100) + 1
        y = (y - self.heightCm/100/2)/(self.heightCm/100) + 1

        return x,y


    ## not good
    # def draw_point(self, image, x,y):
    #     """Draw contours on the RGB stream.
    #     """
    #     # transform viewer to opencv convention
    #     x1 = self.widthPxl * x
    #     y1 = self.heightPxl * (1-y)
    #     touchpoint = np.float32([[x1, y1]]).reshape(-1,1,2)

    #     return cv2.circle(image, tuple(touchpoint.reshape(2)), 2, (0,0,0), 30)

