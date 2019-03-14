import os
import copy
import numpy as np
import pandas as pd
import pickle
import json
import preprocessor
import joblib
import multiprocessing
import argparse


def dataframe_filter(df, average_rating_threshold=4.0, votes_threshold=5):
    # Returns a view into original df.
    new_df = df.loc[df.average_rating >= average_rating_threshold]
    new_df = new_df.loc[df.votes >= votes_threshold]
    return new_df


def binarize(conf_obj, df):
    assert conf_obj['preproc_type'] in ['binarize'], 'Wrong preprocessing function called - type is {} but function is binarize'.format(conf_obj['preproc_type'])
    # Filter df into acceptable df
    df = dataframe_filter(
        df,
        average_rating_threshold=conf_obj['min_average_rating'],
        votes_threshold=conf_obj['min_votes'])

    n_vertices = conf_obj['nodes_per_graph']

    def process_wad(wad_paths):
        for wad_path in wad_paths:
            mapeditors = preprocessor.load_map_editors(wad_path)
            Vs = []
            Es = []
            for mapeditor in mapeditors:
                n_clusters = (len(mapeditor.vertexes) // n_vertices)
                clusters = mapeditor.split(n_clusters)
                for cluster in clusters:
                    V, E = cluster.binarize()
                    Vs.append(V)
                    Es.append(E)
            return Vs, Es

    njobs = multiprocessing.cpu_count() // 4
    samples = joblib.Parallel(njobs)(
        joblib.delayed(_process_wad)(
            df.iloc[i].wad_file,
        ) for i in range(njobs)
    )
    Vs, Es = list(zip(*samples))
    return {'V': Vs, 'E': Es}


if __name__ == '__main__':
    preproc_functions = {
        'binarize': binarize,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="path to CSV to preprocess or wad file to load for debugging")
    parser.add_argument('-o', '--output_path', help='Path to dump the pickle')
    parser.add_argument('-conf', '--config_file', help='JSON File containing configuration to use for preprocessing')
    args = parser.parse_args()

    # Load CSV as dataframe
    df = pd.read_csv(args.path)

    with import(args.config_file, 'r') as f:
        conf_obj = json.load(f)
    preproc_type = conf_obj['preproc_type']
    data = preproc_functions[preproc_type](conf_obj, df)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'wb') as f:
        pickle.dump(data)
