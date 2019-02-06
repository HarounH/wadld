""" File to handle Dataset for maps.
"""

import os
import numpy as np
import pandas as pd
from data import (
    preprocessing,
)
import torch
from torch import nn
from omg import mapedit
