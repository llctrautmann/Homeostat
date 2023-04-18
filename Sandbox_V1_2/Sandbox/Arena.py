from .System import *
from .Agent import *

# a class to represent a rectangular arena, which will constrain the
# position of agents inside of it
# - this class relies on agents having a radius attribute

 # Should this inherit from System, and possible some intermediary Env class?

class Arena(System):
    """
        .. figure:: images/arena.png
          :width: 600
          :align: center
          :alt: Robot in arena

          A robot inside an Arena, which will not allow it to escape.

        A class to represent a rectangular arena, which will confine any agents
        which are inside of its walls and in its list.
    """
    # construct Arena with a list of agents to constrain and coordinates of walls
    def __init__(self, agents: List[Agent], x_left: float, x_right: float, y_top: float, y_bottom: float):
        """
            __init__(agents: List[Agent], x_left: float, x_right: float, y_top: float, y_bottom: float)

            Note: in the current implementation, the code does not check that x_right > x_left and y_top > y_bottom - you have to make sure you get this right yourself.

            :param agents: List of agents which are confined by the Arena's walls. Agents must have a radius attribute for Arena to work.
            :type agents: List of :class:`Agent`

            :param x_left: The x-coordinate of the arena's left wall.
            :type x_left: float

            :param x_right: The x-coordinate of the arena's right wall.
            :type x_right: float

            :param y_top: The y-coordinate of the arena's top wall.
            :type y_top: float

            :param y_bottom: The y-coordinate of the arena's bottom wall.
            :type y_bottom: float
        """

        # call System constructor
        super().__init__(x=0, y=0, theta=None)
        # set attributes
        self.agents = agents
        self.x_left = x_left
        self.x_right = x_right
        self.y_top = y_top
        self.y_bottom = y_bottom

    # step arena
    def step(self, dt: float, x_move: float=None, y_move: float=None) -> None:
        """
            Step :class:`Arena` forwards in time. If any :class:`Agent` s in its ``agents`` list have crossed its walls, they will be pushed back inside.

            :param dt: The interval of time to integrate the :class:`Arena` over. Currently unused here.
            :type dt: float

            :param x_move: The distance to move the :class:`Arena` by in the x-axis.
            :type x_move: float

            :param y_move: The distance to move the :class:`Arena` by in the y-axis.
            :type y_move: float
        """
        # if move parameters are passed to step, then shift the arena
        if x_move and y_move:
            self.move(x_move, y_move)
        # call step of System, so that new xy-coordinates are stored
        super().step(dt)

        # for all agents, constrain them to remain inside the box
        for agent in self.agents:
            # constrain in y
            if (agent.y + agent.radius) > self.y_top:
                # agent.y = self.y_top - agent.radius
                agent.push(y=self.y_top - agent.radius)
            elif (agent.y - agent.radius) < self.y_bottom:
                # agent.y = self.y_bottom + agent.radius
                agent.push(y=self.y_bottom + agent.radius)
            # constrain in x
            if (agent.x + agent.radius) > self.x_right:
                # agent.x = self.x_right - agent.radius
                agent.push(x=self.x_right - agent.radius)
            elif (agent.x - agent.radius) < self.x_left:
                # agent.x = self.x_left + agent.radius
                agent.push(x=self.x_left + agent.radius)
            # agent.update_children_positions()

    # move (translate) arena by specifed increments
    def move(self, x_move: float, y_move: float) -> None:
        """
            A method which can be used to move an :class:`Arena` by the distance specified in the x and y dimensions. This method would normally be called from an :class:`Arena`'s ``step`` method.

            :param x_move: The distance to move the :class:`Arena` by in the x-axis.
            :type x_move: float

            :param y_move: The distance to move the :class:`Arena` by in the y-axis.
            :type y_move: float
        """
        self.x += x_move
        self.y += y_move
        self.x_left += x_move
        self.x_right += x_move
        self.y_top += y_move
        self.y_bottom += y_move

    # draw arena in the specified matplotlib axes
    def draw(self, ax) -> None:
        """
            A method to draw an :class:`Arena` on Matplotlib axes.

            :param ax: The Matplotlib axes to draw the Arena on.
            :type ax: Matplotlib axes
        """
        ax.plot([self.x_left, self.x_left,
                 self.x_right, self.x_right,
                 self.x_left],
                [self.y_bottom, self.y_top,
                 self.y_top, self.y_bottom,
                 self.y_bottom], 'r', linewidth=4)

    # draw arena in whichever matplotlib plot was last used, or
    # a new window if there aren't any open
    def draw2(self) -> None:
        """
            A method to draw an :class:`Arena` on a Matplotlib figure. If there is no figure already open, a new one will be opened. If there is already one or more figure open, then the arena will be drawn on the last one used.
        """
        plt.plot([self.x_left, self.x_left,
                  self.x_right, self.x_right,
                  self.x_left],
                 [self.y_bottom, self.y_top,
                  self.y_top, self.y_bottom,
                  self.y_bottom], 'r')

    # draw arena in a pygame display
    def pygame_draw(self, screen, scale: float, shiftx: float, shifty: float):
        """
            A method for drawing an :class:`Arena` on a PyGame display.

            :param screen: The PyGame display to draw on.
            :type screen: PyGame display

            :param scale: The scale to draw at.
            :type scale: float

            :param shiftx: The offset from centre in the x-axis for drawing.
            :type shiftx: float

            :param shifty: The offset from centre in the y-axis for drawing.
            :type shifty: float
        """
        self.line_drawer(self.x_left, self.x_left, self.y_bottom, self.y_top,
                         screen, scale, shiftx, shifty)
        self.line_drawer(self.x_left, self.x_right, self.y_top, self.y_top,
                         screen, scale, shiftx, shifty)

        self.line_drawer(self.x_right, self.x_right, self.y_top, self.y_bottom,
                      screen, scale, shiftx, shifty)
        self.line_drawer(self.x_right, self.x_left, self.y_bottom, self.y_bottom,
                         screen, scale, shiftx, shifty)

    # THIS METHOD DUPLICATES CODE FROM pygame_functions?

    # a function for drawing a line in the pygame window
    def line_drawer(self, x1: float, x2: float, y1: float, y2: float, screen, scale: float, shiftx: float, shifty:float) -> None:
        """
            A method for drawing a straight line between two points on a PyGame display.

            :param x1: The x-coordinate of the first point.
            :type x1: float

            :param x2: The x-coordinate of the second point.
            :type x2: float

            :param y1: The y-coordinate of the first point.
            :type y1: float

            :param y2: The y-coordinate of the second point.
            :type y2: float

            :param screen: The PyGame display to draw on.
            :type screen: PyGame display

            :param scale: The scale to draw at.
            :type scale: float

            :param shiftx: The offset from centre in the x-axis for drawing.
            :type shiftx: float

            :param shifty: The offset from centre in the y-axis for drawing.
            :type shifty: float
        """
        pygame.draw.line(screen, color='green',
                         start_pos=(scale * x1 + shiftx, scale * y1 + shifty),
                         end_pos=(scale * x2 + shiftx, scale * y2 + shifty),
                         width=2)

    def get_data(self):
        """
            A placeholder - not yet implemented.
        """
        return {"dummy": [0]}
