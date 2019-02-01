import cv2

from utils.data_transformation import *
from utils.wadreader import *
from matplotlib import pyplot as plt


def image_tensor(wad_path):
    vertexes, max_coord_x, max_coord_y, linedefs, ordered_incident_vertices = \
            decode_wad(wad_path)
    return draw_traversable_space(linedefs,
                                  ordered_incident_vertices,
                                  max_coord_x,
                                  max_coord_y)

def main(wad_path):
    image = image_tensor(wad_path)
    image = cv2.resize(image, (int(image.shape[0]/8),int(image.shape[1]/8)))
    cv2.imshow('image',image)
    cv2.waitKey(0)
    #plt.imshow(image)
    #plt.show()

if __name__=="__main__":
    import sys
    wad_dir = sys.argv[1]
    main(get_wad_paths(wad_dir)[3])
