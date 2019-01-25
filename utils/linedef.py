import struct

'''
Defines linedef class which manages data pertaining to linedefs in a wad file.
'''
IMPASSABLE = int('0b000000001', 2)
MONSTER_BLOCKER = int('0b000000010', 2)

# struct.unpack returns a tuple even if the data is only
# one element. This constant indexes the data in that tuple.
DATA_IDX = 0

class LineDef():
    ''' Stores and retrieves data for linedefs.'''
    def __init__(self, linedef_bytestring):
        '''
        '''
        #TODO: All these magic numbers
        self.vertex_start_idx = struct.unpack("<h", linedef_bytestring[0:2])[DATA_IDX]
        self.vertex_end_idx = struct.unpack("<h", linedef_bytestring[2:4])[DATA_IDX]
        self.flags = struct.unpack("<h", linedef_bytestring[4:6])[DATA_IDX]
        self.special_type = struct.unpack("<h", linedef_bytestring[6:8])[DATA_IDX]
        self.sector_tag = struct.unpack("<h", linedef_bytestring[8:10])[DATA_IDX]
        self.front_sidedef = struct.unpack("<h", linedef_bytestring[10:12])[DATA_IDX]
        self.back_sidedef = struct.unpack("<h", linedef_bytestring[12:14])[DATA_IDX]

    def impassable(self):
        ''' Is the linedef impassable?

        @return: True if the linedef is impassable, False if not.
        '''
        return self.flags & IMPASSABLE

    def monster_blocker(self):
        ''' Does the linedef block monsters?

        @return: True if the linedef blocks monsters, False if not.
        '''
        return self.flags & MONSTER_BLOCKER
