import cv2
import numpy as np

from utils.data_utils import *

BLACK = (0, 0, 0)
WHITE = "#ffffff"
RED = "#ff0000"

def get_next_edge(cur_edge, ordered_incident_vertices):
    tail_vertex = cur_edge[0]
    head_vertex = cur_edge[1]
    head_incident_vertices = ordered_incident_vertices[head_vertex]
    next_hop = get_next_hop(tail_vertex, head_incident_vertices)
    return (head_vertex, next_hop)


def get_next_hop(tail_vertex, head_incident_edges):
    tail_vertex_idx = get_index(tail_vertex, head_incident_edges)
    next_hop_idx = (tail_vertex_idx + 1) % len(head_incident_edges)
    return head_incident_edges[next_hop_idx]


def get_index(vertex, vertexes):
    for i in range(len(vertexes)):
        if np.array_equal(vertexes[i], vertex):
            return i
    return None


def find_covering_face(line, ordered_incident_vertices):
    cur_edge = line
    face = [cur_edge]
    found_face = False
    while not found_face:
        next_edge = get_next_edge(cur_edge, ordered_incident_vertices)
        if next_edge == line:
            found_face = True
        else:
            face.append(next_edge)
            cur_edge = next_edge
    return face


def draw_linedefs(vertices, linedefs, color=(255, 255, 255), thickness=32, **kwargs):
    ''' Draws the linedefs provided in specified color onto a numpy array
    @param vertices (utils.wadreader.Vertex list): list of vertices to use.
    @param linedefs (utils.linedef.LineDef list): list of linedefs to draw
    '''
    max_coord_x, max_coord_y = find_max_coords(vertices)
    image = np.zeros((2*max_coord_x+100, 2*max_coord_y+100, 3))
    for linedef in linedefs:
        start = translate_vertex(vertices[linedef.vertex_start_idx].vec, max_coord_x, max_coord_y)
        end = translate_vertex(vertices[linedef.vertex_end_idx].vec, max_coord_x, max_coord_y)
        cv2.line(image, start, end, color, thickness=thickness, **kwargs)
    return image


def draw_traversable_space(vertices, linedefs, debug=False):
    ''' Draws the traversable space of the level as a black and white image
    where the areas which the player can walk in are white.
    @param linedefs: the linedefs from the DOOM was
    @param ordered_incident_vertices: incident vertices for each
    '''
    max_coord_x, max_coord_y = find_max_coords(vertices)
    adj_lists = adjacency_lists(linedefs)
    if debug:
        print("Initialized adj_lists variable")
    ordered_incident_vertices = init_ordered_incident_vertices(vertices,
                                                               linedefs,
                                                               adj_lists)
    if debug:
        print("Initialized ordered_incident_vertices variable")
    image = np.zeros((2*max_coord_x+100, 2*max_coord_y+100))
    if debug:
        print("Created image of size {}".format(image.shape))
    faces = []
    line_is_wall = {}
    linedefs_by_vertices = [((vertices[line.vertex_start_idx],
                              vertices[line.vertex_end_idx]),\
                             line) for line in linedefs]
    if debug:
        print("Initialized linedefs_by_vertices variable")
    for linedef in linedefs_by_vertices:
        line = linedef[0]
        linedef_flags = linedef[1].flags
        if (1 & linedef_flags):# | (1 & linedef_flags >> 7):
            line_is_wall[line] = True
            line_is_wall[(line[1],line[0])] =True
            faces.append(find_covering_face(line, ordered_incident_vertices))
        else:
            line_is_wall[line] = False
            line_is_wall[(line[1],line[0])] = False
            faces.append(find_covering_face((line[1],line[0]),
                                            ordered_incident_vertices))
            faces.append(find_covering_face(line, ordered_incident_vertices))
    if debug:
        print("Initialized faces variable")
    for face in faces:
        edge_list = []
        for edge in face:
            edge_list.append(edge[0])
        translated_edge_list = translate_edge_list(edge_list, max_coord_x,
                                                   max_coord_y)
        cv2.fillConvexPoly(image, np.array(translated_edge_list,
                                           dtype=np.int32),255)
    return image
