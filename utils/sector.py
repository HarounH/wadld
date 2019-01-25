from dataclasses import dataclass

@dataclass
class Sector:
    floor_height: int
    ceiling_height: int
    floor_texture: str
    ceiling_texture: str
    brightness: int
