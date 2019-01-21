import cv2
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

    image = np.zeros((2*max_coord_x+100, 2*max_coord_y+100))
    #img = cv2.fromarray(image)
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
        for edge in face:
            #edge_list.append(edge[0].vec)
            #edge_list.append(edge[1].vec)
            edge_list.append(edge[0])
        translated_edge_list = translate_edge_list(edge_list, max_coord_x,
                                                   max_coord_y)
        cv2.fillConvexPoly(image, np.array(translated_edge_list,
                                     dtype=np.int32),255)
    return image


def wad_space2cv_space(point, offset_x, offset_y):
    """ Transforms coordinates from wad coordinate space
    to PIL coordinate space. This transformation corresponds to a translation

    @param: point: the point to translate
        can either be a tuple ((x, y)), list ([x, y]), or any object with x and y attributes
    @param: offset_x: translation along x axis
    @param: offset_y: translation along y axis

    return
    ----
        a tuple representing the x and y coordinates
    """
    if isinstance(point, tuple) or isinstance(point, list):
        x, y = point
    else:
        x = point.x
        y = point.y
    return (x + offset_x, offset_y - y)


def draw_linedefs(linedefs, max_coord_x, max_coord_y, criterion=lambda linedef: True, color=(255, 255, 255), **kwargs):
    """ Draws linedefs that satisfy a criterion and returns corresponding image array in (M, N, 3) format
    Takes in optional kwargs that are passed to cv2.line call
    Args
        linedefs: iterable of ((Vertex, Vertex), LineDef) type. (See wadreader.py)
        max_coord_x (int)
        max_coord_y (int)
        criterion: fn: type(element of linedefs) -> bool
        color (int tuple of length 3 - BGR channels)
        kwargs: optional arguments passed to cv2.line

    Returns
        image: numpy array that captures linedefs. values are in the range: [0, 225]
    """
    image = np.zeros((2*max_coord_x+100, 2*max_coord_y+100, 3)).astype(np.float32)
    if 'thickness' not in kwargs:
        kwargs['thickness'] = 32  # Need a big thickness for large images.
    count = 0
    for linedef in linedefs:
        if not(criterion(linedef)):
            continue
        count += 1
        points = [wad_space2cv_space(point, max_coord_x, max_coord_y) for point in linedef[0]]
        cv2.line(image, (points[0]), (points[1]), color, **kwargs)
    print("Drew {} lines".format(count))
    return image
