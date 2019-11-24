
import numpy as np

from sklearn import pipeline, base
from helper.deproject import deproject

# import open3d as o3d

import pcl

class Dataset(object):
    def __init__(self, X, org, mat, ext):
        self.X = X
        self.org = org
        self.mat = mat
        self.ext = ext

from sklearn import pipeline, base


class ExtractFeature(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, dataset):
        self.dataset = dataset

    def fit(self, x, y=None): return self

    def identify_tracker(self, X):

        mat = self.dataset.mat
        org = self.dataset.org
        ext = self.dataset.ext

        pc = deproject.compute(X)
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

        leaf_size = 0.05

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
        print(self.dataset.X)
        return (self.transform(x) for x in self.dataset.X)


import datetime
import h5py

# this does not work with interact used before ...
def extract_feature(args):

    i, row = args
    print(i)
    ## X
    filename = row['filename']
    hdf5file = h5py.File('./dataset/'+filename, 'r')

    print(hdf5file)

    x = hdf5file['data/depth']
    org = np.array(hdf5file['origin'], dtype=np.float32)
    mat = np.array(hdf5file['matrix'], dtype=np.float32)
    ext = np.array(hdf5file['extrema'], dtype=np.float32)

    t1 = datetime.datetime.now()
    ef = ExtractFeature(Dataset(x, org, mat, ext))

    res = ef.transform_all_gen()
    res = [x for x in res]
    print("classification {} for {} frames".format(datetime.datetime.now() - t1, x.shape[0]))

    return res

