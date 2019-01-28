""" Functions for performing clustering
"""
import matplotlib.pyplot as plt
import cv2
from utils import (
    utils,
    rendering,
)
from ml.feature_extraction import feature_mat
import sklearn.cluster as skcluster

def cluster_and_plot(verts, lines, sides, secs, N, clustering_=skcluster.KMeans, feature_indices=None, plot=False):
    ''' Cluster linedefs using the features returned by ml.feature_mat, after which the clustered linedefs are drawn
    @param verts (utils.wadreader.Vertex list)
    @param lines (utils.linedef.Linedef list): A list of one sided line defs.
    @param sides (utils.sidedef.Sidedef list): List of sidedefs which can be indexed into by corresponding linedef.
    @param secs (utils.sector.Sector list): List of sectors
    @param N (int): Number of clusters to create.
    @param clustering_ (class): sklearn-like clustering estimator class
    @param feature_indices (list): indices of feature_mat (along dim=1) to use as features
    @param plot (bool): If True, then clusters are plotted.

    @return estimator (instance of clustering_): Instance of clustering_
    @return labels (array-like): Cluster labels for each line
    '''
    X = feature_mat(verts, lines, sides, secs)

    if feature_indices is not None:
        X = X[:, feature_indices]

    if (len(lines) != X.shape[0]):
        raise RuntimeError("linedefs must be one sided. \
                           Please use utils.utils.duplicate_linedefs \
                           function first.")

    estimator = clustering_(n_clusters=N).fit(X)
    labels = estimator.labels_

    if plot:
        colors = utils.get_color_palette(N)
        plt.figure(figsize=(10, 10))
        full_image = None
        for c in range(N):
            color = tuple([int(colors[c][i] * 255) for i in range(3)])
            cluster_linedefs = [lines[i] for i in range(len(lines)) if labels[i] == c]
            cluster_image = rendering.draw_linedefs(verts, cluster_linedefs, color=color)
            cluster_image = cv2.resize(cluster_image, (int(cluster_image.shape[0]/8),int(cluster_image.shape[1]/8)))
            if full_image is None:
                full_image = cluster_image
            else:
                full_image += cluster_image
        plt.imshow(full_image / 255.0)
    return estimator, labels
