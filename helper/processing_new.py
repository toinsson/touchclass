
from sklearn import pipeline, base
from helper.deproject import deproject

import open3d as o3d

class Dataset(object):
    def __init__(self, X, org, mat, ext):
        self.X = X
        self.org = org
        self.mat = mat
        self.ext = ext

class Deproject(base.BaseEstimator, base.TransformerMixin):
    def fit(self, x, y=None): return self

    def transform(self, dataset):
        for X in dataset.X:
            pc = deproject.compute(X)
            pc = np.dot(dataset.mat, (pc - dataset.org).reshape((-1,3)).T).T
            yield pc.astype(np.float64)


class Track(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, ext, *args):
        super(Track, self).__init__(*args)
        self.ext = ext
        self.CLOUD_UPPER_MIN = 0.02  # 1cm, which exclude the plane
        self.CLOUD_UPPER_MAX = 0.10  # 10cm

    def fit(self, x, y=None): return self

    def _transform(self, volume):
        CLOUD_UPPER_MIN = 0.02  # 1cm, which exclude the plane
        CLOUD_UPPER_MAX = 0.10

        org = volume.copy()

        xm = (volume[:,0] >= self.ext[0]) & (volume[:,0] <= self.ext[1])
        ym = (volume[:,1] >= self.ext[2]) & (volume[:,1] <= self.ext[3])
        zm = (volume[:,2] > CLOUD_UPPER_MIN) & (volume[:,2] < CLOUD_UPPER_MAX)
        volume = volume[xm & ym & zm]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(volume)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd = pcd.voxel_down_sample(voxel_size=0.05)

        voxel = np.asarray(pcd.points)
        closest_y = voxel[np.argsort(voxel[:,1])[-1]]

        track_point_idx_ = np.argmin(np.linalg.norm(org - closest_y, axis=1))
        track_point_idx = np.unravel_index(track_point_idx_, (480, 640))

        return track_point_idx

    def transform(self, volumes):
        return (self._transform(volume) for volume in volumes)
