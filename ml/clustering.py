""" Functions for performing clustering
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import (
    utils,
    rendering,
)
from ml.feature_extraction import feature_mat
import sklearn.cluster as skcluster
from sklearn import preprocessing as skpreprocessing


def cluster_and_plot(verts, lines, sides, secs, N, clustering_=skcluster.KMeans, normalize=True, feature_indices=None, X_=None, figsize=None):
    ''' Cluster linedefs using the features returned by ml.feature_mat, after which the clustered linedefs are drawn
    @param verts (utils.wadreader.Vertex list)
    @param lines (utils.linedef.Linedef list): A list of one sided line defs.
    @param sides (utils.sidedef.Sidedef list): List of sidedefs which can be
        indexed into by corresponding linedef.
    @param secs (utils.sector.Sector list): List of sectors
    @param N (int): Number of clusters to create.
    @param clustering_ (class): sklearn-like clustering estimator class
    @param normalize (bool, default=True): normalize feature matrix
    @param feature_indices (list): indices of feature_mat (along dim=1)
        to use as features
    @param X_ (ndarray) custom feature matrix to use for clustering.
        normalize and feature_indices are still used.
    @param figsize (int tuple or None): If provided, a matplotlib figure
        of provided size is plotted.

    @return X (ndarray) feature matrix passed to estimator.
    @return estimator (instance of clustering_): Instance of clustering_
    @return labels (array-like): Cluster labels for each line
    '''
    if X_ is None:
        X = feature_mat(verts, lines, sides, secs)
    else:
        X = X_

    if feature_indices is not None:
        X = X[:, feature_indices]

    if normalize:
        # X = skpreprocessing.StandardScaler().fit_transform(X)
        X_part_0 = skpreprocessing.StandardScaler().fit_transform(X[:, :4])
        X_part_1 = X[:, 4:]
        X = np.concatenate([X_part_0, X_part_1], axis=1)

    if (len(lines) != X.shape[0]):
        raise RuntimeError("linedefs must be one sided. \
                           Please use utils.utils.duplicate_linedefs \
                           function first.")

    estimator = clustering_(N).fit(X)
    labels = estimator.predict(X)

    if figsize is not None:
        colors = utils.get_color_palette(N)
        plt.figure(figsize=figsize)
        full_image = None
        for c in range(N):
            color = tuple([int(colors[c][i] * 255) for i in range(3)])
            cluster_linedefs = [lines[i] for i in range(len(lines)) if labels[i] == c]
            cluster_image = rendering.draw_linedefs(verts, cluster_linedefs, color=color)
            cluster_image = cv2.resize(cluster_image, (int(cluster_image.shape[0] / 8), int(cluster_image.shape[1] / 8)))
            if full_image is None:
                full_image = cluster_image
            else:
                full_image += cluster_image
        plt.imshow(full_image / 255.0)
    return X, estimator, labels
