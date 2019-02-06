import numpy as np
from omg import *

def feature_mat(vertices, linedefs, vec_backs=False, normalize=True):
    ''' Instantiates feature matrix of coordinates of linedefs.

    @param: vertices: list of vertex objects for a map from omg lib
    @param: linedefs: list of linedef objects for a map from omg lib
    @param: vec_backs: create an additional, reversed feature vector for
    linedefs that have two sides
    @param: normalize: center the data or not
    @return: feature matrix for map of linedefs based on coordinates
    '''
    feature_vecs = []
    for line in linedefs:
        src = edit.vertexes[line.vx_a]
        snk = edit.vertexes[line.vx_b]

        feature_vecs.append(np.array([src.x, src.y, snk.x, snk.y]))
        if line.back > 0 and vec_backs:
            feature_vecs.append(np.array([snk.x, snk.y, src.x, src.y]))

    mat = np.vstack(feature_vecs)
    if normalize:
        from sklearn.preprocessing import normalize
        return normalize(mat, axis=1)
    return mat
