import os

def get_wad_paths(wad_dir):
    ''' Return a list of relative paths to all .wad or .WAD files
    found in the subdirectories of WAD_DIR
    '''
    wad_paths = []
    for d in os.listdir(wad_dir):
        if os.path.isdir(wad_dir+"/"+d):
            for f in os.listdir(wad_dir+"/"+d):
                if "wad" in f or "WAD" in f:
                    wad_paths.append(wad_dir + "/" + d + "/" + f)
    return wad_paths

