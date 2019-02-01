import copy
import seaborn
seaborn.reset_orig()


def get_color_palette(n, name='husl'):
    ''' Returns a palette of `n` colors
    @params n (int): number of colors
    @params name (str, optional): name of color scheme to use - husl by default
    @returns colors: [n] indexable set of float triplets in the range [0, 1].
    '''
    return seaborn.color_palette(name, n_colors=n)


class Metadata:
    """
    Simple wrapper around a dict.
    Usage:
        >> metadata = Metadata(ratings=4.2, url="lolol.com")
    """
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
