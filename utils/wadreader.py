import os
import struct
from collections import defaultdict
import numpy as np

from utils.data_utils import *
from utils.linedef import LineDef
from utils.sidedef import Sidedef


# offsets for reading WAD
WAD_TYPE_END = 4
NUM_LUMPS_END = 8
DIR_OFF_END = 12
LUMP_ENT_SIZE = 16
LUMP_NAME_IDX = -8

# struct.unpack returns a tuple even if the data is only
# one element. This constant indexes the data in that tuple.
DATA_IDX = 0

# WAD spec constants
LINEDEF_SIZE = 14
VERTEX_SIZE = 4
SIDEDEF_SIZE = 30

SIDEDEF_TEXTURE_START = 20
SIDEDEF_TEXTURE_END = 28

class Vertex():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vec = np.array([x,y])


def get_wad_data(wad):
    ''' Reads wad file into byte string.

    @param: wad: the wad file
    @return: wad_data: byte string of wad data
    '''
    wad_data = None
    with open(wad, "rb") as f:
        wad_data = f.read()
    return wad_data


def get_wad_index(wad_data):
    ''' Indexes lumps in wad data. Returns a map which maps each lump to its
    corresponding byte string of data.

    @param: wad_data: byte string of wad data
    @return: lump_index: index of lumps to data
    '''
    num_lumps = struct.unpack("<L",
                              wad_data[WAD_TYPE_END:NUM_LUMPS_END])[DATA_IDX]
    dir_offset = struct.unpack("<L",
                               wad_data[NUM_LUMPS_END:DIR_OFF_END])[DATA_IDX]
    lump_index = {}

    lump_start = dir_offset
    lump_end = lump_start + LUMP_ENT_SIZE
    for _ in range(int(num_lumps)):
        lump = wad_data[lump_start:lump_end]
        lump_offset = struct.unpack("<l", lump[:4])[DATA_IDX]
        lump_size = struct.unpack("<l", lump[4:LUMP_NAME_IDX])[DATA_IDX]
        # The name of the lump is zero-padded from the right
        lump_name = lump[LUMP_NAME_IDX:].decode('UTF8').rstrip("\x00")
        lump_index[lump_name] = (lump_offset, lump_size)
        lump_start = lump_end
        lump_end += LUMP_ENT_SIZE

    return lump_index


def parse_linedefs(linedefs_data):
    ''' Generates a list of Linedefs from a byte string of linedef data.

    @param: linedefs_data: byte array of linedefs data
    @return: linedefs: a list of Linedef objects
    '''
    num_sides = int(len(linedefs_data)/LINEDEF_SIZE)
    print("num_sides={}".format(num_sides))
    linedefs = []

    for i in range(num_sides):
        linedef_start = LINEDEF_SIZE*i
        linedefs.append(LineDef(
            linedefs_data[linedef_start:linedef_start+LINEDEF_SIZE]))

    return linedefs

def parse_sidedefs(sidedefs_string):
    ''' Parses sidedefs data.

    @param: sidedefs_string: (str) byte string of sidedefs data
    @return: (list) of Sidedefs
    '''
    num_sidedefs = int(len(sidedefs_string) / SIDEDEF_SIZE)
    sidedefs = []
    for i in range(num_sidedefs):
        sidedef_start = i*SIDEDEF_SIZE
        texture = \
            sidedefs_string[
                sidedef_start+SIDEDEF_TEXTURE_START:sidedef_start+SIDEDEF_TEXTURE_END].decode('UTF8').rstrip("\x00")
        sector = struct.unpack("<h",
            sidedefs_string[
                sidedef_start+SIDEDEF_TEXTURE_END:sidedef_start+SIDEDEF_SIZE])[DATA_IDX]
        sidedefs.append(Sidedef(texture, sector))
    return sidedefs

def parse_sectors(sectors_string):
    ''' Parses sectors data.

    @param: sectors_string: (str) byte string of sectors data
    @return: (list) of Sectors
    '''
    pass


def parse_vertexes(vertexes_string):
    ''' Parses vertexes byte string.

    @param: vertexes_string: (str) vertexes byte string
    @return: vertexes: list of tupes of vertices
    @return: x_coord: abs max x coordinate
    @return: y_coord: abs max y coordinate
    '''
    num_vertexes = int(len(vertexes_string) / VERTEX_SIZE)
    vertexes = []
    for i in range(num_vertexes):
        # Each sequence of 4 bytes is a vertex tuple
        vertex_start = 4*i
        vertex_string = vertexes_string[vertex_start:vertex_start+VERTEX_SIZE]
        x_coord = struct.unpack("<h",
                                vertex_string[0:2])[DATA_IDX]
        y_coord = struct.unpack("<h",
                                vertex_string[2:4])[DATA_IDX]

        vertexes.append(Vertex(x_coord, y_coord))
    return vertexes


def chunk_data(offsets, wad_data):
    ''' Extracts the data for a particular lump from full block of wad data.

    @param: offsets: tuple of start and end offsets of the lump
    @param: wad_data: byte string of wad data
    @return: byte string of lump data
    '''
    data_start = offsets[0]
    data_end = data_start+offsets[1]

    return wad_data[data_start:data_end]


def decode_wad(wad):
    ''' Extracts data from wad.

    @param: wad: the wad data as a byte-array
    '''
    wad_data = get_wad_data(wad)
    wad_index = get_wad_index(wad_data)
    vertices = parse_vertexes(chunk_data(wad_index['VERTEXES'], wad_data))
    linedefs = parse_linedefs(chunk_data(wad_index['LINEDEFS'], wad_data))
    sidedefs = parse_sidedefs(chunk_data(wad_index['SIDEDEFS'], wad_data))

    return vertices, linedefs, sidedefs

