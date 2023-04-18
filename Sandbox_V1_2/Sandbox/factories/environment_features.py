from ..base import *
from ..stimuli import *

# generate a circular arrangement of light sources
def sources_circle(n=20, r=9, x=0, y=0, brightness=1, label: str=None, colour='yellow'):
    sources = []
    for i in range(n):
        a = i * 2*np.pi / n
        sources.append(LightSource(x + r * np.cos(a), y + r * np.sin(a), label=label, colour=colour))
    return sources

# generate a circular arrangement of light sources
def sources_ellipse(n=20, a=9, b=12, x=0, y=0, brightness=1, label: str=None, colour='yellow'):
    sources = []
    for i in range(n):
        theta = i * 2*np.pi / n
        sources.append(LightSource(x + a * np.cos(theta), y + b * np.sin(theta), label=label, colour=colour))
    return sources

def sources_rectangle(x_n=10, y_n=10, x=0, y=0, width=20, height=20, brightness=1, label: str=None, colour='yellow'):
    sources = []

    x_left = x - (width/2)
    x_right = x_left + width
    x_step = width / (x_n-1)

    y_bottom = y - (height/2)
    y_top = y_bottom + height
    y_step = height / (y_n-1)

    y_co = y_bottom
    for _ in range(y_n-2):
        y_co += y_step
        sources.append(LightSource(x_left, y_co, label=label, colour=colour))
        sources.append(LightSource(x_right, y_co, label=label, colour=colour))

    x_co = x_left
    for _ in range(x_n):
        sources.append(LightSource(x_co, y_bottom, label=label, colour=colour))
        sources.append(LightSource(x_co, y_top, label=label, colour=colour))
        x_co += x_step

    return sources

def sources_hor_line(x, y_min, y_max, n):
    pass

def sources_ver_line(y, x_min, x_max, n):
    pass

def sources_line(x1, x2, y1, y2, n):
    pass

def random_sources(x_min=-10, x_max=10, y_min=-10, y_max=10, brightness=1, n=20, colour='yellow', label='yellow'):
    sources = []
    for _ in range(n):
        x = random_in_interval(minimum=x_min, maximum=x_max)
        y = random_in_interval(minimum=x_min, maximum=x_max)
        sources.append(LightSource(x, y, colour=colour, label=label))

    return sources

# generate a circular arrangement of light sources
def perturbable_sources_circle(n=20, r=9, x=0, y=0, brightness=1, label: str=None, colour='yellow'):
    sources = []
    for i in range(n):
        a = i * 2*np.pi / n
        sources.append(PerturbableLightSource(x + r * np.cos(a), y + r * np.sin(a), label=label, colour=colour))
    return sources
