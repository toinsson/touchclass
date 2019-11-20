import numpy as np
from sklearn import pipeline, base
import pcl
from scipy.spatial import ckdtree


from helper.deproject import deproject

# change name
class Deproject(base.BaseEstimator, base.TransformerMixin):
    def fit(self, x, y=None): return self

    def _normalise(self, X, mat, org):
        return np.dot(mat, (X - org).reshape((-1,3)).T).T

    def _filter(self, pc, ext):
        xm = (pc[:,0] >= ext[0]) & (pc[:,0] <= ext[1])
        ym = (pc[:,1] >= ext[2]) & (pc[:,1] <= ext[3])
        return pc[xm & ym]

    def transform(self, groups):
        for g in groups:
            for X in g.X:
                pc = deproject.compute(X)
                X_ = self._normalise(pc, g.mat, g.org)
                X_ = self._filter(X_, g.ext)
                yield X_


class VoxelGridFilter(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, leaf_size = 0.01):
        super(VoxelGridFilter, self).__init__()
        self.leaf_size = leaf_size

    def fit(self, x, y=None): return self

    def _transform(self, volume):
        vpcl = pcl.PointCloud(volume.astype(np.float32))
        vgf = vpcl.make_voxel_grid_filter()
        vgf.set_leaf_size(self.leaf_size, self.leaf_size, self.leaf_size)
        return vgf.filter().to_array()

    def transform(self, volumes):
        return (self._transform(volume) for volume in volumes)


class RoiExtractor(base.BaseEstimator, base.TransformerMixin):
    def fit(self, x, y=None): return self

    def _transform(self, volume):

        CLOUD_UPPER_MIN = 0.02
        CLOUD_UPPER_MAX = 0.10

        ## split in 2 clouds Low and High
        zm_upper_min = volume[:,2] > CLOUD_UPPER_MIN
        zp_upper_max = volume[:,2] < CLOUD_UPPER_MAX
        cloud_upper = volume[zm_upper_min & zp_upper_max]

        if cloud_upper.shape[0] > 0:

            track_point = np.zeros((1,3))

            for i in np.argsort(cloud_upper[:,1])[::-1][:50]:
                n = np.count_nonzero(np.linalg.norm(cloud_upper[i] - cloud_upper, axis=1) < 0.01)
                if (n > 50):
                    track_point = cloud_upper[i]
                    break

            if (track_point == np.zeros(3)).all():
                track_point = cloud_upper[i]

            track_points = volume[np.argwhere(np.linalg.norm(volume[:, :2] - track_point[:2], axis = 1) < 0.05)].reshape(-1,3)
        else:
            track_points = np.zeros((1,3), dtype=np.float32)

        return track_points


    def transform(self, volumes):
        return (self._transform(volume) for volume in volumes)


class FingerPlaneExtractor(base.BaseEstimator, base.TransformerMixin):
    """Extract the plane and the finger from the input ROI."""
    def fit(self, x, y=None):
        return self

    def _transform(self, volume):

        if volume.shape[0] < 2: return (np.zeros((1,3)), np.zeros((1,3)))

        planePcl = pcl.PointCloud(volume.astype(np.float32))
        seg = planePcl.make_segmenter()
        seg.set_optimize_coefficients(True)
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_distance_threshold(0.002)
        indices, model = seg.segment()

        finger = planePcl.extract(indices, negative=True).to_array()
        plane  = planePcl.extract(indices, negative=False).to_array()

        ## run statistical outlier
        fpcl = pcl.PointCloud(finger)
        sof = fpcl.make_statistical_outlier_filter()
        sof.set_mean_k(50)
        sof.set_std_dev_mul_thresh(0.01)
        sof.set_negative(False)
        finger_clean = sof.filter()
        finger_clean = finger_clean.to_array()

        return (finger_clean, plane)

    def transform(self, volumes):
        return (self._transform(volume) for volume in volumes)


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

        if fingers.shape[0] < 2: return np.zeros(self.BINS_SIZE)

        ## extract fingertip
        finger = self.fingertipextractor._transform(fingers)
        if finger.shape[0] < 2: return np.zeros(self.BINS_SIZE)

        try:
            plane_mean = np.mean(plane[:,2])
            finger_z = finger[:,2]

            hist, bins = np.histogram(finger_z,
                                      bins=self.BINS_SIZE,
                                      range=(-0.01, 0.03))

            if hist.sum() == 0:
                return np.zeros(self.BINS_SIZE)
            else:
                db = np.array(np.diff(bins), float)
                hist = hist/db/hist.sum()


        except ValueError:
            return np.zeros(self.BINS_SIZE)
        except IndexError:
            return np.zeros(self.BINS_SIZE)

        if np.isnan(hist).all():
            hist = np.zeros(hist.shape)

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


class TouchClassifier(base.BaseEstimator):
    def __init__(self, modelpath):
        from keras.models import load_model
        self.classfeatureextractor = ClassFeatureExtractor()
        self.clf = load_model(modelpath)

    def fit(self, x, y=None): return self

    def predict(self, X):
        classfeature = self.classfeatureextractor.transform(X)
        return self.clf.predict(classfeature)


class TouchRegressor(base.BaseEstimator):
    def __init__(self):
        self.regfeatureextractor = RegFeatureExtractor()

    def fit(self, x, y=None): return self

    def predict(self, X):
        regfeature = self.regfeatureextractor.transform(X)
        return regfeature[:,:2]


pip = pipeline.Pipeline([
    ('deproject', Deproject()),
    ('voxelgridfilter', VoxelGridFilter(leaf_size=0.002)),
    ('roiextractor', RoiExtractor()),
    ('fingerPlaneExtractor', FingerPlaneExtractor()),
    ('featureextractor', FeatureExtractor()),
], verbose=True)

class Dataset(object):
    def __init__(self, X, org, mat, ext):
        self.X = X
        self.org = org
        self.mat = mat
        self.ext = ext

import datetime
import h5py

def extract_feature(args):
    i, row = args

    ## X
    filename = row['filename']
    hdf5file = h5py.File('./dataset/'+filename, 'r')

    x = hdf5file['data/depth']
    org = np.array(hdf5file['origin'], dtype=np.float32)
    mat = np.array(hdf5file['matrix'], dtype=np.float32)
    ext = np.array(hdf5file['extrema'], dtype=np.float32)

    t1 = datetime.datetime.now()
    data = pip.transform([Dataset(x, org, mat, ext)])

    print("classification {} for {} frames".format(datetime.datetime.now() - t1, x.shape[0]))
    return data[:,:-3]
