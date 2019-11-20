import numpy as np
from sklearn import pipeline, base
import open3d as o3d

from helper.deproject import deproject

# change name
class Deproject(base.BaseEstimator, base.TransformerMixin):
    def fit(self, x, y=None): return self

    def _normalise(self, X, mat, org):
        return np.dot(mat, (X - org).reshape((-1,3)).T).T

    # def _filter(self, pc, ext):
    #     xm = (pc[:,0] >= ext[0]) & (pc[:,0] <= ext[1])
    #     ym = (pc[:,1] >= ext[2]) & (pc[:,1] <= ext[3])
    #     return pc[xm & ym]

    def transform(self, dataset):
        # for g in groups:
            for X in dataset.X:
                pc = deproject.compute(X)
                X_ = self._normalise(pc, dataset.mat, dataset.org)
                # X_ = self._filter(X_, dataset.ext)
                yield X_.astype(np.float64)



class VoxelGridFilter(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, leaf_size = 0.01, *args):
        super(VoxelGridFilter, self).__init__(*args)
        self.leaf_size = leaf_size

    def fit(self, x, y=None): return self

    def _transform(self, volume):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(volume)
        pcd = pcd.voxel_down_sample(voxel_size=self.leaf_size)
        return np.asarray(pcd.points)

        # vpcl = pcl.PointCloud(volume.astype(np.float32))
        # vgf = vpcl.make_voxel_grid_filter()
        # vgf.set_leaf_size(self.leaf_size, self.leaf_size, self.leaf_size)
        # return vgf.filter().to_array()

    def transform(self, volumes):
        return (self._transform(volume) for volume in volumes)



class RoiExtractor(base.BaseEstimator, base.TransformerMixin):
    def fit(self, x, y=None): return self

    def _transform(self, volume):

        ## ALGO PARAMS
        CLOUD_UPPER_MIN = 0.02  # 1cm, which exclude the plane
        CLOUD_UPPER_MAX = 0.10  # 10cm

        ## split in 2 clouds Low and High
        zm_upper_min = volume[:,2] > CLOUD_UPPER_MIN
        zp_upper_max = volume[:,2] < CLOUD_UPPER_MAX
        cloud_upper = volume[zm_upper_min & zp_upper_max]

        if cloud_upper.shape[0] > 0:

            ## find point with max y - make sure it is not isolated
            # track_point = cloud_upper[np.argmax(cloud_upper[:,1])]
            track_point = np.zeros((1,3))

            for i in np.argsort(cloud_upper[:,1])[::-1][:50]:
                n = np.count_nonzero(np.linalg.norm(cloud_upper[i] - cloud_upper, axis=1) < 0.01)
                if (n > 50):
                    track_point = cloud_upper[i]
                    break

            if (track_point == np.zeros(3)).all():
                track_point = cloud_upper[i]  #np.zeros((1,3))

            ## get points in full volume
            track_points = volume[np.argwhere(np.linalg.norm(volume[:, :2] - track_point[:2], axis = 1) < 0.05)].reshape(-1,3)
        else:
            track_points = np.zeros((1,3), dtype=np.float32)

        return track_points


    def transform(self, volumes):
        return (self._transform(volume) for volume in volumes)





# if pcl_available:

#     planePcl = pcl.PointCloud(smallcloud)
#     seg = planePcl.make_segmenter()
#     seg.set_optimize_coefficients(True)
#     seg.set_model_type(pcl.SACMODEL_PLANE)
#     # seg.set_normal_distance_weight(0.01)
#     seg.set_method_type(pcl.SAC_RANSAC)
#     # seg.set_max_iterations(1000)
#     seg.set_distance_threshold(0.01)
#     indices, model = seg.segment()
#     self._planeModel = model

# else:
#     import sklearn
#     from sklearn import linear_model

#     X, y = np.c_[np.ones(smallcloud.shape[0]), smallcloud[:,:2]], smallcloud[:,2:]

#     linear_regression = linear_model.LinearRegression(fit_intercept=False)
#     model_ransac = linear_model.RANSACRegressor(linear_regression, residual_threshold=0.1)
#     model_ransac.fit(X, y)
#     inlier_mask = model_ransac.inlier_mask_
#     outlier_mask = np.logical_not(inlier_mask)

#     model = model_ransac.estimator_.coef_[0]

#     self._planeModel = np.r_[model[1], model[2], -1, model[0]]


import sklearn
from sklearn import linear_model


class FingerPlaneExtractor(base.BaseEstimator, base.TransformerMixin):
    """Extract the plane and the finger from the input ROI."""
    def fit(self, x, y=None):
        return self

    def _transform(self, volume):

        if volume.shape[0] < 2: return (np.zeros((1,3)), np.zeros((1,3)))

        linear_regression = linear_model.LinearRegression(fit_intercept=False)
        model_ransac = linear_model.RANSACRegressor(linear_regression, residual_threshold=0.002)
        X, y = np.c_[np.ones(volume.shape[0]), volume[:,:2]], volume[:,2:]
        model_ransac.fit(X, y)
        inlier_mask = model_ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        finger = volume[outlier_mask]
        plane = volume[inlier_mask]

        # planePcl = pcl.PointCloud(volume.astype(np.float32))
        # seg = planePcl.make_segmenter()
        # seg.set_optimize_coefficients(True)
        # seg.set_model_type(pcl.SACMODEL_PLANE)
        # seg.set_method_type(pcl.SAC_RANSAC)
        # seg.set_distance_threshold(0.002)
        # indices, model = seg.segment()
        # finger = planePcl.extract(indices, negative=True).to_array()
        # plane  = planePcl.extract(indices, negative=False).to_array()

        ## run statistical outlier
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(finger)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.01)

        finger_clean = np.asarray(cl.points)

        # fpcl = pcl.PointCloud(finger)
        # sof = fpcl.make_statistical_outlier_filter()
        # sof.set_mean_k(50)
        # sof.set_std_dev_mul_thresh(0.01)
        # sof.set_negative(False)
        # finger_clean = sof.filter()
        # finger_clean = finger_clean.to_array()

        return (finger_clean, plane)

    def transform(self, volumes):
        return (self._transform(volume) for volume in volumes)


from scipy.spatial import ckdtree


class FingerTipExtractor(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, ball_size = 0.02, *args):
        super(FingerTipExtractor, self).__init__(*args)
        self.ball_size = ball_size

    def fit(self, x, y=None):
        return self

    def _transform(self, volume):

        finger = volume

        if finger.shape[0] < 2: return np.zeros((1,3))

        ## find the max y with positive z
        y_sorted = finger[finger[:,1].argsort()]
        candidate = y_sorted[y_sorted[:, 2] > 0]

        if candidate.shape[0] < 1:
            # logger.warning('no candidates')
            return np.zeros(3)

        vol_max_y = candidate[-1]

        vol_kdt = ckdtree.cKDTree(finger)
        tip_idx = vol_kdt.query_ball_point(vol_max_y, r=self.ball_size)
        return finger[tip_idx]

    def transform(self, volumes):
        return np.array([self._transform(volume) for volume in volumes])




class RegFeatureExtractor(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, leaf_size = 0.01, *args):
        super(RegFeatureExtractor, self).__init__(*args)
        self.fingertipextractor = FingerTipExtractor()

    def fit(self, x, y=None):
        return self

    def _transform(self, volume):
        fingers, plane = volume

        if fingers.shape[0] < 2: return np.zeros(3)

        ## extract fingertip
        finger = self.fingertipextractor._transform(fingers)

        return finger.mean(axis=0)

    def transform(self, volumes):
        return np.array([self._transform(volume) for volume in volumes])


class ClassFeatureExtractor(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, bins_size = 20, *args):
        super(ClassFeatureExtractor, self).__init__(*args)
        self.fingertipextractor = FingerTipExtractor()
        self.BINS_SIZE = bins_size

        self.normalisation = bins_size / 0.04

    def fit(self, x, y=None):
        return self

    def _transform(self, volume):

        (fingers, plane) = volume

        # print(fingers.shape, plane.shape)

        if fingers.shape[0] < 2: return np.zeros(self.BINS_SIZE)

        ## extract fingertip
        finger = self.fingertipextractor._transform(fingers)
        if finger.shape[0] < 2: return np.zeros(self.BINS_SIZE)

        # try:
        #     # plane_mean = np.mean(plane[:,2])
        finger_z = finger[:,2]

        hist, bins = np.histogram(finger_z,
                                  bins=self.BINS_SIZE,
                                  range=(-0.01, 0.03))
                                  # density=1)

        if hist.sum() == 0:
            return np.zeros(self.BINS_SIZE)

        else:
            db = np.array(np.diff(bins), float)
            hist = hist/db/hist.sum()

        # except ValueError:
        #     return np.zeros(self.BINS_SIZE)
        # except IndexError:
        #     return np.zeros(self.BINS_SIZE)
        # if np.isnan(hist).all():
        #     hist = np.zeros(hist.shape)

        features = hist / self.normalisation

        return features

    def transform(self, volumes):
        return np.array([self._transform(volume) for volume in volumes])


class FeatureExtractor(base.BaseEstimator, base.TransformerMixin):
    def __init__(self):
        self.cfe = ClassFeatureExtractor()
        self.rfe = RegFeatureExtractor()

    def fit(self, x, y=None): return self

    def _transform(self, volume):
        f = self.cfe._transform(volume)
        r = self.rfe._transform(volume)
        return np.r_[f,r]

    def transform(self, volumes):
        return np.array([self._transform(volume) for volume in volumes])

