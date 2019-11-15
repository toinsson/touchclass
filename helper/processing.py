import numpy as np
from sklearn import pipeline, base
from helper.deproject import deproject


class Step1(base.BaseEstimator, base.TransformerMixin):
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
