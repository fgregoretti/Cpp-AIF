# cpp-AcI
``cpp-AcI`` is a C++ header library for simulating active inference agents in
discrete space and time, using partially-observed Markov Decision Processes
(POMDPs) as a generative model class.

The library implements a multicore parallelisation of the most demanding computational kernels. Note that most of the computational complexity of active inference depends on the multidimensional inner products involved both in expected free energy computation and state estimation.

realizzare un tutorial così? https://pymdp-rtd.readthedocs.io/en/latest/notebooks/active_inference_from_scratch.html

poi c'è quest'altro tutorial https://pymdp-rtd.readthedocs.io/en/latest/notebooks/using_the_agent_class.html

aggiungere una cosa così in mdp class? https://pymdp-rtd.readthedocs.io/en/latest/notebooks/pymdp_fundamentals.html#constructing-factorized-distributions-with-object-arrays

## Quick-start: Usage
``cpp-AcI`` is a header-only C++ library. No installation is required; in order to use ``cpp-AcI`` to build and develop active inference agents, just include the headers:

```c++
#include "common.h"
#include "mdp.hpp"
```

Tested C++ compiler: GNU C++ compiler.

The recommended command-line compiler options for `g++` are:

`-std=c++11 -O3`

To enable OpenMP-based parallelization add:

`-fopenmp`

## The main API

The [`MDP` Class](doc/mdp_class.md) is the main API offered by ``cpp-AcI`` for running active inference processes.

It provides a high-level abstraction that allows you to specify the generative model for your system, and then perform inference and generate actions.

Once you have created your generative model, you can use it to create an `MDP` object. The `MDP` Class provides necessary methods for performing active inference, including methods for computing the free energy, updating beliefs about the state of the system, and generating actions based on those beliefs.

One of the key benefits of using ``cpp-AcI`` is that it allows you to abstract away the underlying mathematical details of active inference. This can be particularly useful if you are not familiar with the underlying theory, or if you simply want to focus on building your system rather than worrying about the math.

Overall, ``cpp-AcI`` provides a powerful and flexible tool for implementing active inference in your projects. By allowing you to focus on the high-level aspects of your system and abstracting away the mathematical nuances, it can help you save time and improve the quality of your work.

[custom array Classes](doc/custom_array_classes.md)

[Demo: T-Maze](doc/tmaze_doc/tmaze.md)

[Demo: Epistemic Chaining](doc/cue_chaining_doc/cue_chaining.md)
