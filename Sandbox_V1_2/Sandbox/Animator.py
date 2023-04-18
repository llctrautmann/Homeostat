from .base import *
from .pygame_functions import *

class Animator: # Should this inherit from System?
    """
        An class for animating a simulation using PyGame. To simplify scaling, the display is square, i.e. has equal width and height. Whenever a system moves beyond any edge of the currently displayed area, the view will zoom out accordingly, byt the same amount in both axes. The view will not zoom in again if the system moves back towards the centre.
    """
    def __init__(self, systems: List[System]=None, screen_width: int=700, paused: bool=False, delay: int=0):
        """
            __init__(systems: List[System]=None, screen_width: float=700, paused: bool=False, delay: float=0)

                :param systems: A list of systems to draw on the PyGame display.
                :type systems: List[System]

                :param screen_width: The PyGame display height and width, in pixels.
                :type screen_width: int

                :param paused: A flag, set by keyboard inputs in the animation, to specify whether or not the animation, and underlying simulation, should be paused. It is the job of the simulation to actually implement that pause, e.g. see the ``run_once`` method of :class:`SimulationRunner`
                :type paused: bool

                :param delay: The number of 10s of milliseconds to delay the animation by inbetween each pair of successive simulation steps. Can be used to slow an animation/simulation down in order to see what is happening more clearly.
                :type delay: int

        """
        self.screen_width = screen_width
        self.screen = setup_pygame_window(screen_width)
        self.paused = paused
        self.delay = delay
        self.systems = systems
        self.animate_current = True
        self.animate_any = True

    def draw_frame(self, t: float):
        """
            A method for drawing a frame of of the animation.

            :param t: The simulation time variable, which will be displayed in the title bar of the PyGame display window.
            :type t: float
        """
        self.write_title(t)

        quit = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                quit = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_UP:
                    self.delay -= 1
                elif event.key == pygame.K_DOWN:
                    self.delay += 1
                elif event.key == pygame.K_EQUALS:
                    self.delay = 0
                elif event.key == pygame.K_0:
                    self.animate_any = False
                elif event.key == pygame.K_1:
                    self.animate_current = False

        # pause inbetween frames
        delay = max([self.delay, 0])
        time.sleep(delay/100)

        # fill display background
        self.screen.fill('black')

        # initial scale factor and offsets for converting simulation coordinates
        # to pygame animation display coordinates
        pygame_x_offset = self.screen_width/2
        pygame_y_offset = self.screen_width/2

        # find extremes of system trajectories for resizing animation window
        max_xs = []
        max_ys = []
        for system in self.systems:
            if system.has_position:
                max_xs.append(max(np.abs(system.xs)))
                max_ys.append(max(np.abs(system.ys)))
            if isinstance(system, Arena):
                max_xs.append(max(abs(system.x_left), abs(system.x_right)))
                max_ys.append(max(abs(system.y_bottom), abs(system.y_top)))

        # reset scale according to where systems are and have been
        pygame_scale = self.screen_width / (2 * max(max(max_xs), max(max_ys)) + 1)

        # draw all systems
        for system in self.systems:
            system.pygame_draw(self.screen, scale=pygame_scale, shiftx=pygame_x_offset, shifty=pygame_y_offset)

        # flip the pygame display
        self.screen.blit(pygame.transform.flip(self.screen, False, True), (0, 0))
        # update the pygame display
        pygame.display.update()

        return quit

    def stop(self):
        """
            Stop the animation.
        """
        pygame.display.quit()

    def shutdown(self):
        """
            Stop the animation and close the PyGame window.
        """
        self.stop()
        pygame.quit()

    def write_title(self, t: float):
        """
            A method for writing simuation info to the title bar of the PyGame display window.

            :param t: The time to be displayed.
            :type t: float
        """

        # build title bar string
        title_str = 'Sandbox: '
        if self.paused:
            title_str += 'Paused'
        else:
            title_str += 'Running'
        t_str = f'{t:.2f}'
        title_str += ': t = ' + t_str
        title_str += ': frame delay: ' + str(self.delay/100) + 's'
        # write string
        pygame.display.set_caption(title_str)
