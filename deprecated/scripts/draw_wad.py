import cv2

from utils.rendering import draw_traversable_space
from utils.wadreader import decode_wad
from utils.io_utils import get_wad_paths
from ml.feature_extraction import feature_mat

def image_tensor(wad_path):
    vertices, linedefs, sidedefs, sectors = decode_wad(wad_path)
    print(feature_mat(vertices, linedefs, sidedefs, sectors))
    return draw_traversable_space(vertices, linedefs)

def main(wad_path):
    image = image_tensor(wad_path)
    image = cv2.resize(image, (int(image.shape[0]/8),int(image.shape[1]/8)))
    cv2.imshow('image',image)
    cv2.waitKey(0)

if __name__=="__main__":
    import sys
    wad_dir = sys.argv[1]
    main(get_wad_paths(wad_dir)[3])
