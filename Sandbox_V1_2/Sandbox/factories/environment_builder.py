from .environment_features import *
from ..base import *
from ..Arena import *

def circle_with_arena(radius, num, x_left, x_right, y_bottom, y_top, arena_agents=[], brightness=1):

    arena = Arena(agents=arena_agents, x_left=x_left, x_right=x_right, y_top=y_top, y_bottom=y_bottom)

    light_sources = sources_circle(n=num, r=radius, brightness=brightness)

    return arena, light_sources


def ellipse_with_arena(a, b, num, x_left, x_right, y_bottom, y_top, arena_agents=[], brightness=1):

    arena = Arena(agents=arena_agents, x_left=x_left, x_right=x_right, y_top=y_top, y_bottom=y_bottom)

    light_sources = sources_ellipse(n=num, a=a, b=b, brightness=brightness)

    return arena, light_sources

def rect_with_arena(x_left, x_right, y_bottom, y_top, x_n, y_n, width, height, arena_agents=[], brightness=1):

    arena = Arena(agents=arena_agents, x_left=x_left, x_right=x_right, y_top=y_top, y_bottom=y_bottom)

    light_sources = sources_rectangle(x_n=x_n, y_n=x_n, x=0, y=0, width=width, height=height, brightness=brightness)

    return arena, light_sources

def an_EBA_arena(x_left, x_right, y_bottom, y_top, n1, n2, arena_agents=[], brightness=1):

    arena = Arena(agents=arena_agents, x_left=x_left, x_right=x_right, y_top=y_top, y_bottom=y_bottom)

    light_sources = random_sources(x_min=x_left, x_max=x_right, y_min=y_bottom, y_max=y_top, brightness=1, n=n1, colour='yellow', label='yellow')

    light_sources += random_sources(x_min=x_left, x_max=x_right, y_min=y_bottom, y_max=y_top, brightness=1, n=n2, colour='red', label='red')

    return arena, light_sources


def double_circle(x1, y1, rad1, num1, x2, y2, rad2, num2):

    circle1 = sources_circle(n=num1, r=rad1, x=x1, y=y1, brightness=1, label='yellow', colour='yellow')

    circle2 = sources_circle(n=num2, r=rad2, x=x2, y=y2, brightness=1, label='red', colour='red')

    return circle1, circle2

def perturbable_double_circle(x1, y1, rad1, num1, x2, y2, rad2, num2):

    circle1 = perturbable_sources_circle(n=num1, r=rad1, x=x1, y=y1, brightness=1, label='yellow', colour='yellow')

    circle2 = perturbable_sources_circle(n=num2, r=rad2, x=x2, y=y2, brightness=1, label='red', colour='red')

    return circle1, circle2
