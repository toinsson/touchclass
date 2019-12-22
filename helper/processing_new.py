import numpy as np

from sklearn import pipeline, base
from helper.projection import projection
import pcl


from sklearn.pipeline import FeatureUnion

import datetime
import h5py


def process_dataset(dataset, pipeline):
    i, row = args

    ## X
    filename = row['filename']

    hf = h5py.File('./dataset/'+filename, 'r')
    x = hf['data/depth']
    org = np.array(hf['origin'], dtype=np.float32)
    mat = np.array(hf['matrix'], dtype=np.float32)
    ext = np.array(hf['extrema'], dtype=np.float32)


    t1 = datetime.datetime.now()

    # pl = pipeline.Pipeline([
    #         ('track_and_identity', FeatureUnion([('user_pointer', UserPointer()), ('identity', Identity())])),
    #         ('clip_frame', ClipFrame()),
    #         ('preprocess', PreProcess()),
    #     ], verbose=True)

    td = TraverseDataset(x, mat, org, ext, pipeline)
    res = td.compute()


    print("classification {} for {} frames".format(datetime.datetime.now() - t1, x.shape[0]))

    return res


def extract_feature(args):
    i, row = args

    ## X
    filename = row['filename']

    hf = h5py.File('./dataset/'+filename, 'r')
    x = hf['data/depth']
    org = np.array(hf['origin'], dtype=np.float32)
    mat = np.array(hf['matrix'], dtype=np.float32)
    ext = np.array(hf['extrema'], dtype=np.float32)


    t1 = datetime.datetime.now()

    pl = pipeline.Pipeline([
            ('track_and_identity', FeatureUnion([('user_pointer', UserPointer()), ('identity', Identity())])),
            ('clip_frame', ClipFrame()),
            ('preprocess', PreProcess()),
        ], verbose=True)

    td = TraverseDataset(x, mat, org, ext, pl)
    res = td.compute()


    print("classification {} for {} frames".format(datetime.datetime.now() - t1, x.shape[0]))

    return res


class Dataset(object):
    def __init__(self, X, org, mat, ext):
        self.X = X
        self.org = org
        self.mat = mat
        self.ext = ext


class TraverseDataset(object):
    def __init__(self, X, mat, org, ext, pl):
        self.X = X
        self.mat = mat
        self.org = org
        self.ext = ext
        self.pl = pl

    def compute(self):
        return np.array([self.pl.transform((x, self.mat, self.org, self.ext)) for x in self.X])


class Identity(base.BaseEstimator, base.TransformerMixin):
    def fit(self, x, y=None): return self
    def transform(self, X): return X


class Deproject(base.BaseEstimator, base.TransformerMixin):
    def fit(self, x, y=None): return self

    def _normalise(self, X, mat, org):
        return np.dot(mat, (X - org).reshape((-1,3)).T).T

    def _filter(self, pc, ext):
        xm = (pc[:,0] >= ext[0]) & (pc[:,0] <= ext[1])
        ym = (pc[:,1] >= ext[2]) & (pc[:,1] <= ext[3])
        return pc[xm & ym]

    def transform(self, X):
        depth = X['frame']
        mat, org, ext = X['mat'], X['org'], X['ext']

        pc = projection.pixel_to_point(X)
        X_ = self._normalise(pc, g.mat, g.org)
        X_ = self._filter(X_, g.ext)
        return X_



class UserPointer(base.BaseEstimator, base.TransformerMixin):
    def fit(self, x, y=None): return self

    def track(self, X):
        x, mat, org, ext = X

        pc = projection.pixel_to_point(x)
        pc = np.dot(mat, (pc - org).reshape((-1,3)).T).T
        pc_org = pc.copy()

        CLOUD_UPPER_MIN = 0.01  # 1cm, which exclude the plane
        CLOUD_UPPER_MAX = 0.10
        xm = (pc[:,0] >= ext[0]) & (pc[:,0] <= ext[1])
        ym = (pc[:,1] >= ext[2]) & (pc[:,1] <= ext[3])
        zm = (pc[:,2] > CLOUD_UPPER_MIN) & (pc[:,2] < CLOUD_UPPER_MAX)
        pc = pc[xm & ym & zm]

        if pc.shape[0] == 0:
            return np.zeros(2)

        LEAF_SIZE = 0.05

        vpcl = pcl.PointCloud(pc.astype(np.float32))
        vgf = vpcl.make_voxel_grid_filter()
        vgf.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
        voxel = vgf.filter().to_array()

        closest_y = voxel[np.argsort(voxel[:,1])[-1]]
        track_point_idx_ = np.argmin(np.linalg.norm(pc_org - closest_y, axis=1))
        track_point_idx = np.unravel_index(track_point_idx_, x.shape)

        return track_point_idx

    def transform(self, X):
        return self.track(X)

class ClipFrame(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, abcd=(-25, 75, -50, 50), size=100):
        super(ClipFrame, self).__init__()
        self.abcd = abcd
        self.size = size
    #     add init with output size parameter
    def fit(self, x, y=None): return self

    def clip_frame(self, frame, point):
        a, b, c, d = self.abcd
        s = self.size
        if (np.array(point) == np.zeros(2)).all():
            res = np.zeros((s,s))
        else:
            res = frame[point[0]+a:point[0]+b, point[1]+c:point[1]+d]
        return res

    def transform(self, X):
        track_point_idx = X[:2]
        frame = X[2]
        return self.clip_frame(frame, track_point_idx)

from skimage.transform import downscale_local_mean
from sklearn.preprocessing import StandardScaler

class PreProcess(base.BaseEstimator, base.TransformerMixin):
    def fit(self, x, y=None): return self

    def transform(self, X):
        res = downscale_local_mean(X, factors=(10,10))
        res = StandardScaler().fit_transform(res)
        return res
