import os, struct

from data_transformation import *

from collections import defaultdict

import numpy as np

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

class Vertex():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vec = np.array([x,y])

    #def __lt__(self, other):


class LineDef():
    def __init__(self, start_vertex_idx, end_vertex_idx, flags, spec_type, sec_tag,
                  front_sidedef, back_sidedef):
        self.start_idx = start_vertex_idx
        self.end_idx = end_vertex_idx
        self.flags = flags
        self.spec_type = spec_type
        self.sec_tag = sec_tag
        self.front_sidedef = front_sidedef
        self.back_sidedef = back_sidedef


def get_wad_paths(wad_dir):
    ''' Return a list of relative paths to all .wad or .WAD files
    found in the subdirectories of WAD_DIR
    '''
    wad_paths = []
    for d in os.listdir(wad_dir):
        if os.path.isdir(wad_dir+"/"+d):
            for f in os.listdir(wad_dir+"/"+d):
                if "wad" in f or "WAD" in f:
                    wad_paths.append(wad_dir + "/" + d + "/" + f)
    return wad_paths


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
    wad_type = wad_data[:WAD_TYPE_END].decode("ASCII")
    num_lumps = struct.unpack("<L",
                              wad_data[WAD_TYPE_END:NUM_LUMPS_END])[DATA_IDX]
    dir_offset = struct.unpack("<L",
                               wad_data[NUM_LUMPS_END:DIR_OFF_END])[DATA_IDX]
    lump_index = {}

    lump_start = dir_offset
    lump_end = lump_start + LUMP_ENT_SIZE
    for i in range(int(num_lumps)):
        lump = wad_data[lump_start:lump_end]
        lump_offset = struct.unpack("<l", lump[:4])[DATA_IDX]
        lump_size = struct.unpack("<l", lump[4:LUMP_NAME_IDX])[DATA_IDX]
        # The name of the lump is zero-padded from the right
        lump_name = lump[LUMP_NAME_IDX:].decode('UTF8').rstrip("\x00")
        lump_index[lump_name] = (lump_offset, lump_size)
        lump_start = lump_end
        lump_end+=LUMP_ENT_SIZE

    return lump_index

def adjacency_matrix(linedefs):
    ''' Instantiates an adjacency matrix over the nodes in the level. This is
    represented as a dictionary of lists. Each key is a vertex which indexes
    a list of adjacent nodes.

    @param: linedefs: the linedefs for the map
    @return: adj_mat: the adjacency matrix for the map.
    '''
    adj_mat = defaultdict(list)
    for linedef in linedefs:
        adj_mat[linedef.start_idx].append(linedef.end_idx)
        adj_mat[linedef.end_idx].append(linedef.start_idx)
    return adj_mat


def get_linedefs(linedefs_data):
    ''' Generates a list of Linedefs from a byte string of linedef data.

    @param: linedefs_data: byte array of linedefs data
    @return: linedefs: a list of Linedef objects
    '''
    num_sides = int(len(linedefs_data)/LINEDEF_SIZE)
    linedefs = []

    for i in range(num_sides):
        linedef_start = LINEDEF_SIZE*i
        vertex_start_idx = struct.unpack("<h",
                        linedefs_data[linedef_start:linedef_start+2])[DATA_IDX]
        vertex_end_idx = struct.unpack("<h",
                        linedefs_data[linedef_start+2:linedef_start+4])[DATA_IDX]
        flags = struct.unpack("<h",
                        linedefs_data[linedef_start+4:linedef_start+6])[DATA_IDX]
        special_type = struct.unpack("<h",
                        linedefs_data[linedef_start+6:linedef_start+8])[DATA_IDX]
        sector_tag = struct.unpack("<h",
                        linedefs_data[linedef_start+8:linedef_start+10])[DATA_IDX]
        front_sidedef = struct.unpack("<h",
                        linedefs_data[linedef_start+10:linedef_start+12])[DATA_IDX]
        back_sidedef = struct.unpack("<h",
                        linedefs_data[linedef_start+12:linedef_start+14])[DATA_IDX]
        linedefs.append(LineDef(vertex_start_idx, vertex_end_idx, flags,
                                special_type, sector_tag, front_sidedef,
                                back_sidedef))

    return linedefs


def get_vertex(vertex_string):
    ''' Extracts vertex from byte string.
    @param: vertex_string: byte string of data
    @return tuple of x,y coordinate of vertex
    '''
    x_coord = struct.unpack("<h",
                            vertex_string[0:2])[DATA_IDX]
    y_coord = struct.unpack("<h",
                            vertex_string[2:4])[DATA_IDX]
    return x_coord, y_coord

def find_max_coords(vertices):
    ''' Finds maximum magnitudes along each axis.

    @param: vertices: vertices for the map
    @return: max
    '''
    max_x = 0
    max_y = 0
    for vertex in vertices:
        if abs(vertex.x) > max_x:
            max_x = abs(vertex.x)
        if abs(vertex.y) > max_y:
            max_y = abs(vertex.y)
    return max_x, max_y

def get_vertexes(vertexes_string):
    ''' Parses vertexes byte string.

    @param: vertexes_string: vertexes byte string
    @return: vertexes: list of tupes of vertices
    @return: x_coord: abs max x coordinate
    @return: y_coord: abs max y coordinate
    '''
    num_vertexes = int(len(vertexes_string) / VERTEX_SIZE)
    vertexes = []
    for i in range(num_vertexes):
        # Each sequence of 4 bytes is a vertex tuple
        vertex_start = 4*i
        x_coord, y_coord = \
            get_vertex(vertexes_string[vertex_start:vertex_start+VERTEX_SIZE])
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


def decode_wad_test():
    vertexes = [(200,200), (200,300), (300,300), (300,200),(200,500),(300,500),
               (0,600),(100,600),(0,900),(100,900)]
    max_coord = 1000
    linedefs =\
    [(0,1,None,None),(1,2,None,None),(2,3,None,None),(3,0,None,None),
    (1,4,None,None),(2,5,None,None),(4,6,None,None),(5,7,None,None),(6,7,None,None),
    (8,9,None,None),(7,9,None,None),(8,9,None,None),(6,8,None,None)]
    adj_list = {0:[1,3],1:[0,2,4],2:[1,3,5],3:[0,2],4:[1,6],5:[2,7],
                6:[4,7,8],7:[5,6,9],8:[6,9],9:[7,8]}
    ordered_incident_vertices = init_ordered_incident_vertices(vertexes, linedefs,
                                                         adj_list)
    linedefs_by_vertices = [((vertexes[line[0]], vertexes[line[1]]), line[2]) \
                            for line in linedefs]
    return vertexes, max_coord, linedefs_by_vertices, ordered_incident_vertices


def decode_wad(wad):
    ''' Extracts data from wad.

    @param: wad: the wad data as a byte-array
    '''
    wad_data = get_wad_data(wad)
    wad_index = get_wad_index(wad_data)

    vertexes = get_vertexes(chunk_data(wad_index['VERTEXES'], wad_data))
    max_coord_x, max_coord_y = find_max_coords(vertexes)
    linedefs = get_linedefs(chunk_data(wad_index['LINEDEFS'], wad_data))
    adj_mat = adjacency_matrix(linedefs)
    ordered_incident_vertices = init_ordered_incident_vertices(
        vertexes, linedefs, adj_mat)
    linedefs_by_vertices = [((vertexes[line.start_idx],
                              vertexes[line.end_idx]),\
                             line) for line in linedefs]
    return vertexes, max_coord_x, max_coord_y, linedefs_by_vertices, ordered_incident_vertices

