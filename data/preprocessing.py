import os
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from omg import mapedit
from ml import feature_extraction


class MapEditorPreprocessor(mapedit.MapEditor):
    ''' Wrapper around map editor for preprocessing
    > All the members from mapedit.MapEditor
    with the following changes:
        - linedefs: list of one sided linedefs
        - original_linedefs: list of original linedefs
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.construct_one_sided_linedefs()

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

    def enclose(self):
        ''' Function to surround the wad file using 4 extra lines.
        '''
        raise NotImplementedError()

    def split(self, n_clusters, algorithm='kmeans'):
        ''' Function that creates `n_clusters` number of SplittableMapEditors
        '''
        X = feature_extraction.feature_mat(self.linedefs)  # One sided linedefs.
        raise NotImplementedError("Need to cluster and create children MapEditorPreprocessor objects")

    def draw_linedefs(self):
        ''' Function that creates an image array of shape (W, H, C) with linedefs drawn.
        '''
        raise NotImplementedError()


def load_map_editors(wad_or_wad_file):
    ''' Function to
    '''
    if isinstance(wad_or_wad_file, str):
        wad = omg.WAD()
        wad.from_file(wad_or_wad_file)
    elif isinstance(wad_or_wad_file, omg.WAD):
        wad = wad_or_wad_file
    else:
        raise TypeError("argument {} is of type {} but expected str or omg.WAD".format('wad_or_wad_file', type(wad_or_wad_file)))
    return {k: MapEditorPreprocessor(v) for k, v in wad.maps.items()}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("wad_path", type=str, help="path to wad file to load")
    args = parser.parse_args()
    mapeditors = load_map_editors(args.wad_path)
    print("Loaded {} maps".format(len(mapeditors)))
