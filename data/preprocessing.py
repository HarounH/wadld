import os
import copy
import numpy as np
import pandas as pd
import cv2
import torch
from torch import nn
import omg
from omg import mapedit
from ml import feature_extraction
import sklearn
import sklearn.cluster


class MapEditorPreprocessor(mapedit.MapEditor):
    ''' Wrapper around map editor for preprocessing
    > All the members from mapedit.MapEditor
    with the following changes:

    data members:
        parent_wad: Original WAD. Necessary to allow easy export.
        linedefs: list of one sided linedefs
        original_linedefs: list of original linedefs
        recognized_clustering_algorithms: algorithms recognized by
            MapEditorPreprocessor.split
        enclosed: bool, denoting whether MapEditor
            has been enclosed by 4 linedefs.
        used_vxi: list of vertices that are use by linedefs.
    '''
    recognized_clustering_algorithms = {
        'kmeans': sklearn.cluster.KMeans
    }

    def __init__(self, wad, *args, **kwargs):
        ''' All arguments are passed onto mapedit.MapEditor
        '''
        super().__init__(*args, **kwargs)
        self.construct_one_sided_linedefs()
        self.enclosed = False
        self.wad = wad

    def construct_one_sided_linedefs(self):
        self.is_original = []
        duplicated_linedefs = []
        for linedef in self.linedefs:
            if (linedef.front != -1) and (linedef.back != -1):
                back_linedef = copy.deepcopy(linedef)
                back_linedef.vx_a, back_linedef.vx_b = linedef.vx_b, linedef.vx_a
                back_linedef.front = linedef.back
                back_linedef.back = -1
                linedef.back = -1
                duplicated_linedefs.append(linedef)
                duplicated_linedefs.append(back_linedef)
                self.is_original.extend([False, False])
            elif (linedef.front != -1):
                duplicated_linedefs.append(linedef)
                self.is_original.extend([True])
            elif linedef.back != -1:
                linedef.vx_a, linedef.vx_b = linedef.vx_b, linedef.vx_a
                linedef.front = linedef.back
                linedef.back = -1
                duplicated_linedefs.append(linedef)
                self.is_original.extend([True])
            else:  # Neither front norback exists.
                duplicated_linedefs.append(linedef)
                self.is_original.extend([True])
        self.original_linedefs = self.linedefs
        self.linedefs = duplicated_linedefs
        self.used_vxi = self.construct_vxi()

    def construct_vxi(self, linedefs=None):
        ''' Function to construct list of vertex indices that are used
        '''
        if linedefs is None:
            linedefs = self.linedefs
        return list(set(line.vx_a for line in linedefs).union(
                set(line.vx_b for line in linedefs)))

    def get_range(self, used_vxi=None):
        if used_vxi is None:
            used_vxi = range(len(self.vertexes))
        elif used_vxi:
            used_vxi = self.used_vxi
        min_x, min_y, max_x, max_y = None, None, None, None
        for vxi in used_vxi:
            x = self.vertexes[vxi].x
            y = self.vertexes[vxi].y
            if (min_x is None) or (x < min_x):
                min_x = x
            if (max_x is None) or (x > max_x):
                max_x = x
            if (min_y is None) or (y < min_y):
                min_y = y
            if (max_y is None) or (y > max_y):
                max_y = y
        return min_x, min_y, max_x, max_y

    def enclose(self, in_place=False, force=False):
        ''' Function to surround the wad file using 4 extra lines.
        Also sets a flag denoting that this MapEditor has been enclosed.
        Unless the force flag is set to True, it will not be enclosed a second time.
        '''
        if in_place:
            target = self
        else:
            target = copy.deepcopy(self)

        if self.enclosed and not(force):
            return target

        # Enclosing time!

        # First, construct necessary vertices.
        min_x, min_y, max_x, max_y = self.get_range()
        boundary_width = 100
        bl_vertex = mapedit.Vertex(x=(min_x - boundary_width), y=(min_y - boundary_width))
        tl_vertex = mapedit.Vertex(x=(min_x - boundary_width), y=(max_y + boundary_width))
        br_vertex = mapedit.Vertex(x=(max_x + boundary_width), y=(min_y - boundary_width))
        tr_vertex = mapedit.Vertex(x=(max_x + boundary_width), y=(max_y + boundary_width))

        # TODO: @chase: Need to create a LineDef for each of these points
        raise NotImplementedError()
        return target

    def split(self, n_clusters, standardize=True, feature_indices=None, algorithm='kmeans'):
        ''' Function that creates `n_clusters` number of SplittableMapEditors
        @param: n_clusters (int): number of clusters to split map into
        @param: standardize (bool): standardize feature matrix before using clustering algorithm
        @param: feature_indices (None or int list): which indices of feature_extraction.feature_mat
            to use for clustering. If None, all indices are used.
        @param: algorithm (str): Name of algorithm used for splitting. Must be registered in
            MapEditorPreprocessor.recognized_clustering_algorithms

        @return: splits (MapEditorPreprocessor list)
        '''
        X = feature_extraction.feature_mat(
            self.vertexes,
            self.linedefs,
            vec_backs=False,
            standardize=standardize
        )  # One sided linedefs.
        if feature_indices is not None:
            X = X[:, feature_indices]

        clustering_ = self.recognized_clustering_algorithms[algorithm]
        estimator = clustering_(n_clusters).fit(X)
        labels = estimator.predict(X)
        splits = []
        for label in range(n_clusters):
            new_map_editor = copy.copy(self)  # Not a deep copy.
            new_map_editor.linedefs = [line for i, line in enumerate(self.linedefs) if labels[i] == label]
            new_map_editor.used_vxi = new_map_editor.construct_vxi()
            splits.append(new_map_editor)
        return splits

    def draw_linedefs(self, color=(255, 255, 255), thickness=8, boundary=500):
        ''' Function that creates an image array of
        shape (W, H, C) with linedefs drawn.
        @param: color (float triplet): triplet representing color to use to draw.
        @param: thickness (int): thickness of lines.

        @return: arr (np.ndarray): a 3d array of shape (W, H, C) representing the image.
        '''
        # color = np.array(color)
        min_x, min_y, max_x, max_y = self.get_range()

        # Vertex translation (makes everything positive basically)
        def wad2cv(x, y):
            return (boundary) + (x - min_x), (boundary) + (y - min_y)

        arr = np.zeros((
            (max_x - min_x) + (2 * boundary),
            (max_y - min_y) + (2 * boundary),
            3
        ))

        for line in self.linedefs:
            vx_a = self.vertexes[line.vx_a]
            vx_a = wad2cv(vx_a.x, vx_a.y)
            vx_b = self.vertexes[line.vx_b]
            vx_b = wad2cv(vx_b.x, vx_b.y)
            # cv2 is a little weird in terms of what boundaries it uses.
            cv2.line(arr, vx_a, vx_b, color, thickness=thickness)
        return arr

    def append_to_wad(self, map_name):
        self.wad.maps[map_name] = self.to_lumps()


def load_map_editors(wad_or_wad_file):
    ''' Function to load map editors from wad (or filename)
    '''
    if isinstance(wad_or_wad_file, str):
        wad = omg.WAD(wad_or_wad_file)
    elif isinstance(wad_or_wad_file, omg.WAD):
        wad = wad_or_wad_file
    else:
        raise TypeError("argument {} is of type {} but expected str or omg.WAD".format('wad_or_wad_file', type(wad_or_wad_file)))
    return {k: MapEditorPreprocessor(wad, v) for k, v in wad.maps.items()}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("wad_path", type=str, help="path to wad file to load")
    args = parser.parse_args()
    mapeditors = load_map_editors(args.wad_path)
    print("Loaded {} maps".format(len(mapeditors)))
    # TODO: Write code to test split, draw_linedefs functions
    base_map = mapeditors[0]
    clusters = base_map.split(4)
    import pdb; pdb.set_trace()
