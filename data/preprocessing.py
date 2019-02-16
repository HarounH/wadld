import os
import copy
import numpy as np
import pandas as pd
import cv2
import torch
from torch import nn
import omg
import sklearn
import sklearn.cluster
import pickle


def dataframe_filter(df, average_rating_threshold=4.0, votes_threshold=5):
    raise NotImplementedError()


def binarize(args, df):
    raise NotImplementedError()


if __name__ == '__main__':
    preproc_functions = {
        'binarize': binarize,
    }
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="path to CSV to preprocess or wad file to load for debugging")
    parser.add_argument("-pt", "--preproc_type", choices=list(preproc_functions.keys()), default='binarize', help="Preprocessing mode")
    args = parser.parse_args()

    # Load CSV as dataframe
    df = pd.read_csv(args.path)
    # Filter df into acceptable df
    df = dataframe_filter(
        df,
        average_rating_threshold=args.average_rating_threshold,
        votes_threshold=args.votes_threshold)

    dataset = preproc_functions[args.preproc_type](args, df)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'wb') as f:
        pickle.dump(dataset)
