# cpp-AIF
``cpp-AIF`` is a C++ header-only library for simulating active inference agents in
discrete space and time, using partially-observed Markov Decision Processes
(POMDPs) as a generative model class.

The library implements a multicore parallelization of the most demanding computational kernels. Note that most of the computational complexity of active inference depends on the multidimensional inner products involved both in expected free energy computation and state estimation.

One of the key benefits of using ``cpp-AIF`` is that it allows you to abstract away the underlying mathematical details of [active inference](doc/active_inference.md). This can be particularly useful if you are not familiar with the underlying theory, or if you simply want to focus on building your system rather than worrying about the math.

Overall, ``cpp-AIF`` provides a powerful and flexible tool for implementing active inference in your projects. By allowing you to focus on the high-level aspects of your system and abstracting away the mathematical nuances, it can help you save time and improve the quality of your work.

<!--
realizzare un tutorial così? https://pymdp-rtd.readthedocs.io/en/latest/notebooks/active_inference_from_scratch.html

poi c'è quest'altro tutorial https://pymdp-rtd.readthedocs.io/en/latest/notebooks/using_the_agent_class.html

aggiungere una cosa così in mdp class? https://pymdp-rtd.readthedocs.io/en/latest/notebooks/pymdp_fundamentals.html#constructing-factorized-distributions-with-object-arrays
-->

## Quick-start: Usage
``cpp-AIF`` is a header-only C++ library. No installation is required; in order to use ``cpp-AIF`` to build and develop active inference agents, just include the header:

```c++
#include "mdp.hpp"
```

Tested C++ compiler: GNU C++ compiler.

The recommended command-line compiler options for `g++` are:

`-std=c++11 -O3`

To enable OpenMP-based parallelization add:

`-fopenmp`

Preprocessor directive macros:
- `-D DEBUG` print out debugging information
- `-D FULL` temporal horizon and time length policy coincide
- `-D BEST_AS_MAX` the selected action is chosen as the maximum of the posterior over actions
- `-D WITH_GP` if the generative model is not a veridical representation of the generative process
- `-D LEARNING` to use the functions for updating parameters of posteriors in POMDP generative models

For example to compile the [T-Maze](doc/tmaze_doc/tmaze.md) example you can type:

`g++ -O3 -D FULL -o Tmaze main_Tmaze.cpp`

## The main API

The [`MDP`](doc/mdp_class.md) class is the main API offered by ``cpp-AIF`` for running active inference processes.

It provides a high-level abstraction that allows you to specify the generative model for your system, and then perform inference and generate actions.

Once you have created your generative model, you can use it to create an `MDP` object. The `MDP` Class provides necessary methods for performing active inference, including methods for computing the free energy, updating beliefs about the state of the system, and generating actions based on those beliefs.

The generative model is a key component of active inference, as it describes how the internal states of the system are generated and how those states give rise to observations. By providing specifically designed classes for building the generative model, we help users focus on the high-level aspects of their system and reduce the amount of low-level programming needed to define the model. Users can more easily create and configure their generative models, as well as reuse and modify them as needed.

We have implemented specific [classes](doc/generative_model_classes.md) for building the generative model and [here](doc/data_structure.md) we focus on how we build it (e.g. the components of the generative model).

## Examples

We provide two hands-on examples of active inference agents in action. By providing them for users to work through, we help them build their understanding of the underlying concepts and how to apply them in practice.

The [T-Maze](doc/tmaze_doc/tmaze.md) and [Epistemic Chaining](doc/cue_chaining_doc/cue_chaining.md) demo are both excellent examples of how active inference can be applied to different types of tasks. The T-Maze is a classic task used in neuroscience to study decision making, while the Epistemic Chaining demo provides a more complex example that showcases the power of active inference in solving sequential decision-making problems.

Examples can help users understand how to set up and configure active inference agents for different types of tasks and therefore how to implement and explore active inference in practice. By leveraging these examples, users are helped in building their understanding of active inference and how it can be applied to solve real-world problems.

## Table of Contents
-  [Active Inference](doc/active_inference.md)
-  [Data Structure and Factorized Distributions](doc/data_structure.md)
  -   [Generative Model Classes](doc/generative_model_classes.md)
-  [`MDP` class](doc/mdp_class.md)
-  Examples
  -  [T-Maze](doc/tmaze_doc/tmaze.md)
  -  [Epistemic Chaining](doc/cue_chaining_doc/cue_chaining.md)



