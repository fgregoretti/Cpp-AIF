# cpp-AcI
``cpp-AcI`` is a C++ header library for simulating active inference agents in
discrete space and time, using partially-observed Markov Decision Processes
(POMDPs) as a generative model class.

The library implements a multicore parallelisation of the most demanding computational kernels. Note that most of the computational complexity of active inference depends on the multidimensional inner products involved both in expected free energy computation and state estimation.

realizzare un tutorial così? https://pymdp-rtd.readthedocs.io/en/latest/notebooks/active_inference_from_scratch.html

poi c'è quest'altro tutorial https://pymdp-rtd.readthedocs.io/en/latest/notebooks/using_the_agent_class.html

aggiungere una cosa così in mdp class? https://pymdp-rtd.readthedocs.io/en/latest/notebooks/pymdp_fundamentals.html#constructing-factorized-distributions-with-object-arrays

## Quick-start: Usage
In order to use ``cpp-AcI`` to build and develop active inference agents, you have to write your own main program file that have these includes:

```c++
#include "common.h"
#include "mdp.hpp"
```

Tested C++ compiler: GNU C++ compiler.

The recommended command-line compiler options for `g++` are:

`-std=c++11 -O3`

To enable OpenMP-based parallelization add:

`-fopenmp`

[MDP Class](doc/mdp_class.md)

[custom array Classes](doc/custom_array_classes.md)

[Demo: T-Maze](doc/tmaze_doc/tmaze.md)

[Demo: Epistemic Chaining](doc/cue_chaining_doc/cue_chaining.md)
