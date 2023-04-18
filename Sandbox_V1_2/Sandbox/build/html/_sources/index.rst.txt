.. Sandbox documentation master file, created by
   sphinx-quickstart on Thu Jun  9 18:44:42 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the Sandbox documentation!
======================================

.. figure:: images/double_rings.png
  :width: 600
  :align: center
  :alt: Robot and lights in PyGame animation.

  A robot and lights in a *Sandbox* PyGame animation.

Introduction to Sandbox
-----------------------

*Sandbox* is the latest version of a simulation framework for agent-based modelling which I created for teaching the *Adaptive Systems* module at the University of Sussex, in 2020. In 2020, it was loosely known as *the simulation with no name*. In 2021 it became *SituSim* (An abbreviation for *Situated and Embodied Agent Simulation*), and was used on the *Intelligence in Animals and Machines (IAM)* module for the first time. In 2022, I have renamed it again, to *Sandbox*. The reason for this new name is that I want to make it clear that we use this simulation framework as a sandbox for experimenting with adaptive systems, and agent-based models of animal behaviour.

In labs involving *Sandbox*, we will work with simulations of mobile agents, which will often be modelled on robots, but that does not mean that our objective is to become good robot programmers. Our true objective is to use the various agents and systems in Sandbox as vehicles for exploring and learning about processes of adaptation, and agent-based biological modelling.

As *Sandbox* is developed for the purposes of experimentation, it is programmed in an object-oriented and highly customisable style. It is relatively easy to set up simulations of agents and environments for our experiments, and it is also relatively easy to extend the existing code to add new types of agents and environmental features.

I have recently reprogrammed large parts of *Sandbox*. I have tested and debugged a lot of the old and new code, but it is possible that there are still some small bugs. If you find anything in the code which you think might be a bug, please let me know.

I have also been working on this completely new documentation for *Sandbox*. This has proven to be a much bigger task than I anticipated, and so I have not yet been able to spend the amount of time I would have liked on proofreading and rewriting. Please let me know (Chris, c.a.johnson@sussex.ac.uk) if any parts of the documentation are unclear or seem to be incorrect.

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Contents:

   intro
   systems
   stimuli_sensors
   agents
   actuators
   controllers
   environment
   noise_sources
   disturbance_sources
   simulation

Indices and tables
------------------

* :ref:`genindex`
.. * :ref:`modindex`
* :ref:`search`
