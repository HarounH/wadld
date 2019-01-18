from PIL import Image, ImageDraw

#from wadreader import *
import numpy as np

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
    print(face)
    return face


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


def draw_traversable_space(linedefs, ordered_incident_vertices, max_coord_x,
                           max_coord_y):
    ''' Draws the traversable space of the level as a black and white image
    where the areas which the player can walk in are white.
    @param linedefs: the linedefs from the DOOM was
    @param ordered_incident_vertices: incident vertices for each 
    '''

    image = Image.new('RGB', (2*max_coord_x+100, 2*max_coord_y+100), BLACK)
    draw = ImageDraw.Draw(image)
    faces = []
    line_is_wall = {}
    for linedef in linedefs:
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

    already_drawn = {}
    for face in faces:
        edge_list = []
        #hash_string = str(sorted(face))
        #if hash_string in already_drawn:
        #    continue
        #already_drawn[hash_string] = True

        #white_space = False
        for edge in face:
            edge_list.append(edge[0])
            #if not line_is_wall[edge]:
            #    white_space = True
        translated_edge_list = translate_edge_list(edge_list, max_coord_x,
                                                   max_coord_y)
        white_space = True
        if white_space:
            draw.polygon(translated_edge_list, fill=WHITE, outline=RED)
    return image


