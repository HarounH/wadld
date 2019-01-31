import os
import struct
from collections import defaultdict
import numpy as np


def encode_wad(vertices, linedefs, sidedefs, sectors, *args, **kwargs):
    ''' Inverse of decode_wad
    '''
