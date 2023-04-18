General information
*******************

Dependencies
------------

Sandbox has been written with the following dependencies:

* **matplotlib** (for plotting, https://matplotlib.org)
* **numpy** (for linear algebra, pseudo-random number generation, and lots more, https://numpy.org)
* **PyGame** (for animation, https://www.pygame.org/news)
* **copy** (for deepcopy, which is useful when copying objects, https://docs.python.org/3.8/library/copy.html#module-copy)

I have found that the easiest way to install all of these dependencies is to use an Anaconda Python 3.x environment. PyGame, in particular, can be difficult to install otherwise. When using Anaconda, PyGame cannot be installed directly from the Anaconda navigator, or by using the ``conda`` command, but it can be installed using ``pip install pygame`` from a terminal or command line with the correct Anaconda environment active.

Type hinting
------------

I have recently started using type hinting (https://docs.python.org/3.8/library/typing.html) in Sandbox. This helps me debug my code using *mypy* (https://mypy.readthedocs.io/en/stable/), but I also hope it will help anyone using the code to see the expected types of parameters to and returns from functions.

Abstraction
-----------

Some object-oriented programming languages have *abstract* classes. It is possible to have similar classes in Python using ``abc`` (https://docs.python.org/3/library/abc.html). I have chosen not to use ``abc`` in *Sandbox* for now, but you will notice that I make various references to abstract classes below.

In this documentation, if I refer to one of the classes as abstract, what I mean is that the class is not designed to be used directly, e.g. in the cases of :class:`System` or :class:`Agent`. For example, there is nothing in the code to stop you, but it would make no sense to construct a :class:`System` directly - we would normally only construct subclasses of :class:`System` which represent specific types of system.

Stepping
--------

Most classes in *Sandbox* will have a ``step`` method, which will move a system forwards in time by a single simulation step. Many systems make use of a ``dt`` parameter. Strictly speaking, I should have called this variable ``delta_t``, as it represents the discrete interval of time that a system is stepped forwards by. Some systems don't use ``dt`` at all, but I leave it in the method parameters for consistency (I have a poor memory, so the more consistent I am with how I code my classes, the easier it is for me to remember how to use them). The step methods in some classes will take other parameters as well as ``dt``, e.g. any inputs to the system during the current time step.

Integrating
-----------

In some classes, such as those which represent agents, what happens as a result of a call to ``step`` is split up into a number of other methods. This is to make it easier to make custom subclasses. The ``integrate`` method is one of those methods, and is used solely to model a system's "physical" dynamics (as opposed to aspects such as sensing and control, which will have their dynamics, where they have any, modelled in other classes).

Resetting objects for reuse
---------------------------

When running a batch of simulation runs, it is normally going to be more convenient to reset all of the systems in the simulation to put them back in their initial states, rather than constructing them again. For this reason, almost all *Sandbox* classes implement a ``reset`` method. If you are adding a class to *Sandbox*, you should probably be implementing a reset method.

Getting data
------------

When a system is used in multiple simulation runs, we need to keep the data from those different runs separate. We do this by getting the systems' data from them, in the form of a dict, before resetting them and using them again. For this reason, many *Sandbox* classes implement a ``get_data`` method. If you are adding a class to *Sandbox*, you should consider implementing a reset method. You can also call the ``get_data_and_reset`` method on any  :class:`System` which implements both the ``get_data`` and the ``reset`` methods, to combine the get data and reset operations.
