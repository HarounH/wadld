from collections import defaultdict
import numpy as np
import copy


def duplicate_linedefs(linedefs):
    new_linedefs = []
    for linedef in linedefs:
        new_linedefs.append(copy.deepcopy(linedef))
        new_linedefs[-1].back_sidedef = -1
        if linedef.back_sidedef != -1:
            nld = (copy.deepcopy(linedef))
            nld.vertex_start_idx, nld.vertex_end_idx = nld.vertex_end_idx, nld.vertex_start_idx
            nld.front_sidedef = nld.back_sidedef
            nld.back_sidedef = -1
            new_linedefs.append(nld)
    return new_linedefs


def translate_vertex(vertex, offset_x, offset_y):
    ''' Transforms coordinates from wad coordinate space
    to PIL coordinate space.

    @param: vertex (int tuple): the vertex to translate
    @param: offset_x: translation along x axis
    @param: offset_y: translation along y axis
    '''
    return (offset_x + vertex[0], offset_y - vertex[1])


def translate_edge_list(edge_list, offset_x, offset_y):
    ''' Transforms coordinates from wad coordinate space
    to PIL coordinate space.

    @param: edge_list: the edges to translate
    @param: offset_x: translation along x axis
    @param: offset_y: translation along y axis
    '''
    translated_edges = [(offset_x+x.vec[0],offset_y-x.vec[1])\
                            for x in edge_list]
    return translated_edges


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

def init_ordered_incident_vertices(vertexes, linedefs, adj_list):
    ''' Creates a cyclical list of incident vertices in clockwise order.
    '''
    ordered_incident_edges = {}
    for i, vertex in enumerate(vertexes):
        incident_vertices = [vertexes[j] for j in adj_list[i]]

        if not incident_vertices:
            continue

        cyc_vertex_list = []
        for i in range(0, len(incident_vertices)):
            next_vec = vertex.vec - incident_vertices[i].vec
            angle = np.arctan2(next_vec[1], next_vec[0])
            cyc_vertex_list.append((angle, incident_vertices[i]))

        cyc_vertex_list.sort()
        ordered_incident_edges[vertex] = [x[1] for x in cyc_vertex_list]

    return ordered_incident_edges

def adjacency_lists(linedefs):
    ''' Instantiates an dict of adjacency lists over the nodes in the level.
    Each key is a vertex which indexes a list of adjacent nodes.

    @param: linedefs: the linedefs for the map
    @return: adj_mat: the adjacency matrix for the map.
    '''
    adj_mat = defaultdict(list)
    for linedef in linedefs:
        start = linedef.vertex_start_idx
        end = linedef.vertex_end_idx
        if end not in adj_mat[start]:
            adj_mat[start].append(end)
        if start not in adj_mat[end]:
            adj_mat[end].append(start)
    return adj_mat
