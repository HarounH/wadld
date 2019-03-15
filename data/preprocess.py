import os
import copy
import numpy as np
import pandas as pd
import pickle
import json
import data.preprocessor as preprocessor
import joblib
import multiprocessing
import argparse
import time


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

    def _process_wad(wad_paths):
        Vs = []
        Es = []
        for wad_path in wad_paths:
            try:
                mapeditors = preprocessor.load_map_editors(os.path.join('data', wad_path))
            except:  # noqa
                print('Error while loading {}'.format(wad_path))
                continue
            for map_name, mapeditor in mapeditors.items():
                n_clusters = (len(mapeditor.vertexes) // n_vertices)
                try:
                    clusters = mapeditor.split(n_clusters)
                    for cluster in clusters:
                        V, E = cluster.binarize()
                        Vs.append(V)
                        Es.append(E)
                except:  # noqa
                    print('Error while loading map {} of {}'.format(map_name, wad_path))
                    continue

        print('Loaded {} clusters from {}'.format(len(Vs), wad_paths))
        return Vs, Es

    njobs = multiprocessing.cpu_count() // 4
    samples = joblib.Parallel(njobs)(
        joblib.delayed(_process_wad)(
            df.iloc[i].wad_file,
        ) for i in range(df.shape[0])
    )
    Vs, Es = list(zip(*samples))
    print('Binarized {} clusters'.format(len(Vs)))
    return {'V': sum(Vs, []), 'E': sum(Es, [])}


if __name__ == '__main__':
    preproc_functions = {
        'binarize': binarize,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="path to CSV to preprocess or wad file to load for debugging")
    parser.add_argument('-o', '--output_path', default='preprocessed_data/binarized.pkl', help='Path to dump the pickle')
    parser.add_argument('-conf', '--config_file', default='default_preprocess_conf.json', help='JSON File containing configuration to use for preprocessing')
    args = parser.parse_args()

    # Load CSV as dataframe
    print('Loading dataframe from {}'.format(args.path))
    df = pd.read_pickle(args.path)

    print('Loading configuration from {}'.format(args.config_file))
    with open(args.config_file, 'r') as f:
        conf_obj = json.load(f)
    preproc_type = conf_obj['preproc_type']
    tic = time.time()
    data = preproc_functions[preproc_type](conf_obj, df)
    print('Preprocessing took {}s'.format(time.time() - tic))
    print('Writing preprocessed data to {}'.format(args.output_path))
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'wb') as f:
        pickle.dump(data, f)
