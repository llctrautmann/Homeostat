import numpy as np
import math
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms

from typing import Dict, Any, Union, List, Callable, Tuple

pygame = None
try:
    import os
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
    import pygame
except:
    print("WARNING: pygame module not found, visualisations will not be shown. " +
          "You should be able to install pygame with:\n" +
          "     pip install pygame")

####################################################################################
#                           utility functions begin
####################################################################################

# for any two angles, return difference in the interval of [-pi, pi]
def angle_difference(angle1: float, angle2: float) -> float:
    diff = (angle1 - angle2) % (2*math.pi)
    if diff > math.pi:
        diff -= (2*math.pi)
    return diff

# generate random number from uniform interval
# - numpy already has a function for this, but I wrote this and used it in many places before thinking to check that
def random_in_interval(minimum: float=0, maximum: float=1) -> float:
    width = maximum - minimum
    return (width * np.random.random()) + minimum

####################################################################################
#                           utility functions end
####################################################################################
