class Metadata:
    """
    Simple wrapper around a dict.
    Usage:
        >> metadata = Metadata(ratings=4.2, url="lolol.com")
    """
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
