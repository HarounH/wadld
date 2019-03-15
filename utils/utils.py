import copy
import numpy as np
import seaborn
seaborn.reset_orig()


def get_color_palette(n, name='husl'):
    ''' Returns a palette of `n` colors
    @params n (int): number of colors
    @params name (str, optional): name of color scheme to use - husl by default
    @returns colors: [n] indexable set of float triplets in the range [0, 1].
    '''
    return seaborn.color_palette(name, n_colors=n)


def draw_binary_lines(V, E, boundary=100, color=(255, 255, 255), thickness=8):
    '''
        Given `n` 2-d points, and an adjancency matrix between them,
        draw corresponding planar graphs

        @param V (np.ndarray of shape (n, 2)): position of points
        @param E (np.ndarray of shape (n, n)): adjacency graph
        @param boundary (int): boundary around image
        @param boundary (int): color of lines
        @param thickness (int): thickness of lines

        @return arr: 3 dimensional array representing image,
            with dimensions in (X,Y,Color) order.
            image is cropped to content, leaving a boundary behind
    '''
    min_x, min_y = V.min(0)
    max_x, max_y = V.max(0)
    arr = np.zeros((
        int(max_x - min_x) + (2 * boundary),
        int(max_y - min_y) + (2 * boundary),
        3
    ))
    min_point = np.array([min_x, min_y])
    V = (V - min_point).astype(np.int)
    Eai, Ebi = np.where(E)
    for ei in range(len(Eai)):
        vx_a = tuple(V[Eai[ei]].tolist())
        vx_b = tuple(V[Ebi[ei]].tolist())
        cv2.line(arr, vx_a, vx_b, color, thickness=thickness)
    return arr
class Metadata:
    """
    Simple wrapper around a dict.
    Usage:
        >> metadata = Metadata(ratings=4.2, url="lolol.com")
    """
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
