import numpy as np

NUM_COORDS = 4
NUM_SECTOR_FEATURES = 3

def init_texture_lex(sidedefs, sectors):
    textures = [side.texture for side in sidedefs]
    for sector in sectors:
        textures.append(sector.floor_texture)
        textures.append(sector.ceiling_texture)
    textures_set = set(textures)
    texture_lex = {}
    for i, texture in enumerate(textures_set):
        texture_lex[texture] = i
    return texture_lex

def init_feature_vec(vertex_start, vertex_end, textures, sector, num_textures):
    feature_vec = np.zeros(NUM_COORDS + 3*num_textures + NUM_SECTOR_FEATURES)

    # set coordinate values
    feature_vec[0:2] = vertex_start.vec
    feature_vec[2:4] = vertex_end.vec

    # set texture features
    for i, texture in enumerate(textures):
        offset = 4+i*num_textures+texture
        feature_vec[offset]=1

    #set floor_height, ceiling_height, and brightness
    feature_vec[-3]=sector.floor_height
    feature_vec[-2]=sector.ceiling_height
    feature_vec[-1]=sector.brightness

    return feature_vec

def extract_features(line, vertices, sidedefs, sectors, texture_lex):
    vertex_start = vertices[line.vertex_start_idx]
    vertex_end = vertices[line.vertex_end_idx]

    sidedef = sidedefs[line.front_sidedef]
    sector = sectors[sidedef.sector]

    # set texture features
    textures = [texture_lex[sidedef.texture],
                texture_lex[sector.floor_texture],
                texture_lex[sector.ceiling_texture]]
    front_feature_vec = init_feature_vec(vertex_start, vertex_end, textures,
                                         sector, len(texture_lex))
    if line.back_sidedef == -1:
        return front_feature_vec, None, False

    sidedef = sidedefs[line.back_sidedef]
    sector = sectors[sidedef.sector]

    # set texture features
    textures = [texture_lex[sidedef.texture],
                texture_lex[sector.floor_texture],
                texture_lex[sector.ceiling_texture]]

    back_feature_vec = init_feature_vec(vertex_end, vertex_start, textures,
                                        sector, len(texture_lex))

    return front_feature_vec, back_feature_vec, True


def feature_mat(vertices, linedefs, sidedefs, sectors):
    texture_lex = init_texture_lex(sidedefs, sectors)
    feature_vecs = []
    for line in linedefs:
        front_feature_vec, back_feature_vec, has_back = extract_features(line, vertices,
                                                               sidedefs, sectors,
                                                               texture_lex)
        feature_vecs.append(front_feature_vec)
        if has_back:
            feature_vecs.append(back_feature_vec)
    return np.vstack(feature_vecs)
