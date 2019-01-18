from data_transformation import *
from wadreader import *

def image_tensor(wad_path):
    vertexes, max_coord_x, max_coord_y, linedefs, ordered_incident_vertices = \
            decode_wad(wad_path)
    print(linedefs)
    return draw_traversable_space(linedefs,
                                  ordered_incident_vertices,
                                  max_coord_x,
                                  max_coord_y)

def main(wad_path):
    image = image_tensor(wad_path)
    image.show()

if __name__=="__main__":
    import sys
    wad_dir = sys.argv[1]
    main(get_wad_paths(wad_dir)[3])
