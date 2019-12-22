
import numpy as np

from sklearn import pipeline, base
from helper.projection import projection

# import open3d as o3d

import pcl

class Dataset(object):
    def __init__(self, X, org, mat, ext):
        self.X = X
        self.org = org
        self.mat = mat
        self.ext = ext

from sklearn import pipeline, base


class TraverseDataset(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, X, mat, org, ext):
        self.X = X
        self.mat = mat
        self.org = org
        self.ext = ext

    def transform(self, X):
        return ((x, self.mat, self.org, self.ext) for x in self.X)


class Identity(base.BaseEstimator, base.TransformerMixin):
    def fit(self, x, y=None): return self
    def transform(self, X): return X


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
        track_point_idx = np.unravel_index(track_point_idx_, X.shape)

        return track_point_idx

    def transform(self, X):
        track_point_idx = self.identify_tracker(X)
        return clipped_frame

class Clean(base.BaseEstimator, base.TransformerMixin):
    def fit(self, x, y=None): return self


    def clip_frame(self, frame, point):
        if (np.array(point) == np.zeros(2)).all():
            res = np.zeros((50,50))
        else:
            res = frame[point[0]:point[0]+50, point[1]-25:point[1]+25]
            # print(res.shape)
        return res

    def transform(self, X):

        clipped_frame = self.clip_frame(X, track_point_idx)
        return clipped_frame



class ExtractFeature(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, dataset, leaf_size=0.02):
        self.dataset = dataset
        self.leaf_size = 0.02

    def fit(self, x, y=None): return self

    def identify_tracker(self, X):

        mat = self.dataset.mat
        org = self.dataset.org
        ext = self.dataset.ext

        pc = projection.pixel_to_point(X)
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

        leaf_size = self.leaf_size #0.02

        vpcl = pcl.PointCloud(pc.astype(np.float32))
        vgf = vpcl.make_voxel_grid_filter()
        vgf.set_leaf_size(leaf_size, leaf_size, leaf_size)
        voxel = vgf.filter().to_array()

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pc)
        # # pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        # pcd = pcd.voxel_down_sample(voxel_size=0.05)
        # voxel = np.asarray(pcd.points)

        closest_y = voxel[np.argsort(voxel[:,1])[-1]]
        track_point_idx_ = np.argmin(np.linalg.norm(pc_org - closest_y, axis=1))
        track_point_idx = np.unravel_index(track_point_idx_, X.shape)

        return track_point_idx

    def clip_frame(self, frame, point):
        if (np.array(point) == np.zeros(2)).all():
            res = np.zeros((50,50))
        else:
            res = frame[point[0]:point[0]+50, point[1]-25:point[1]+25]
            # print(res.shape)
        return res

    def transform(self, X):
        track_point_idx = self.identify_tracker(X)
        clipped_frame = self.clip_frame(X, track_point_idx)
        return clipped_frame

    def transform_all(self):
        return np.array([self.transform(x) for x in self.dataset.X])

    def transform_all_gen(self):
        return (self.transform(x) for x in self.dataset.X)


import datetime
import h5py

# this does not work with interact used before ...
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
    ef = ExtractFeature(Dataset(x, org, mat, ext))
    res = [x for x in ef.transform_all_gen()]

    print("classification {} for {} frames".format(datetime.datetime.now() - t1, x.shape[0]))

    return res

