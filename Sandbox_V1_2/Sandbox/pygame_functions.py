from .base import *
from .Arena import *

# set up the pygame window, if we are animating the simulation
def setup_pygame_window(screen_width: int) -> None:
    """


        :param screen_width:
        :type screen_width: int
    """
    # initialise pygame and set parameters
    pygame.init()
    screen = pygame.display.set_mode([screen_width, screen_width])

    return screen

def pygame_drawline(screen, shiftx: float, shifty: float, scale: float, x1: float, y1: float, x2: float, y2: float, colour: str, width: float) -> None:
    """


        :param screen:
        :type screen: PyGame screen

        :param shiftx:
        :type shiftx: float

        :param shifty:
        :type shifty: float

        :param scale:
        :type scale: float

        :param x1:
        :type x1: float

        :param y1:
        :type y2: float

        :param x2:
        :type x2: float

        :param y2:
        :type y2: float

        :param width:
        :type width: float

        :param colour:
        :type colour: str
    """
    pygame.draw.line(surface=screen,
                     color=colour,
                     start_pos=(scale * x1 + shiftx, scale * y1 + shifty),
                     end_pos=(scale * x2 + shiftx, scale * y2 + shifty),
                     width=width)

def pygame_drawcircle(screen, shiftx: float, shifty: float, scale: float, centre_x: float, centre_y: float, radius: float, colour: str) -> None:
    """


        :param screen:
        :type screen: PyGame screen

        :param shiftx:
        :type shiftx: float

        :param shifty:
        :type shifty: float

        :param scale:
        :type scale: float

        :param centre_x:
        :type centre_x: float

        :param centre_y:
        :type centre_y: float

        :param radius:
        :type radius: float

        :param colour:
        :type colour: str
    """
    pygame.draw.circle(surface=screen, center=(scale*centre_x+shiftx, scale*centre_y+shifty), color=colour, radius=scale*radius)
