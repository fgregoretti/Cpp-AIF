# MDP class
```c++
template <typename Ty, std::size_t M> class MDP
```

`MDP` class allows you to run active inference processes. Firstly, you need to is build a generative model in terms of **$\bf{A}$**, **$\bf{B}$**, **$\bf{C}$**, and **$\bf{D}$** arrays, true inital state **$S^0$** arrays and optionally a vector of policies. if the latter is empty, the constructor will generate the policies. Then you have to pass them as parameters to the `MDP` constructor, and then start running active inference processes using the desired functions of the `MDP` class, like `infer_states`, `infer_policies`, `sample_action`, `sample_state`, and `sample_observation`.
 
***Constructor:***
```c++
MDP(std::vector<Beliefs<Ty>*>& __D,
      std::vector<States*>& __S,
      std::vector<std::vector<Transitions<Ty>*>>& __B,
      std::vector<std::vector<likelihood<Ty,M>*>>& __A,
      std::vector<Priors<Ty>*>& __C,
      std::vector<std::vector<int>>& __V,
      unsigned int T_ = 10, Ty alpha_ = 8, Ty beta_ = 4,
      Ty lambda_ = 0, Ty gamma_ = 1, unsigned int N_ = 4,
#ifndef FULL
      unsigned int policy_len_ = 1,
#endif
      unsigned int seed_ = 0);
```

**Parameters**
- `__D` agent's prior over initial states (initial beliefs)
- `__S` agent's true inital state
- `__B` transition model
- `__A` observation model
- `__C` preferred outcomes
- `__V` policies
- `T_` temporal horizon
- `alpha_` gamma hyperparameter
- `beta_` gamma hyperparameter
- `lambda` precision update rate
- `gamma_`
- `N_` number of variational iterations
- `policy_len_` when not compiled with macro FULL is the time length policy, otherwise temporal horizon and time length policy coincide 
- `seed_` number to initialize a pseudorandom number generator

In `Cpp-AcI` generative model distributions as well as expectations of hidden states, states and observations are represented as vector of vector (**$\bf{A}$** and **$\bf{B}$**) or vector (all the others) of "custom objects". These are instances of [classes] specifically designed to handle active inference data, with an array as member. 

Understanding the representation of factorized probability distributions as vector (of vector) of class instances is critical for understanding and constructing generative models in `Cpp-AcI`. In particular, we use vector of vector of specific class instances to encode the observation and transition models of the agent’s generative model. We represent them as a vector of vector due to the convention of factorizing the observation space into multiple observation factors and the hidden states into multiple hidden state factors and to express dependency from control states **$u$** executable by the agent.

This factorization of observations across modalities and hidden states across hidden state factors, carries forward into the specification of the A and B arrays, the representation of the conditional distributions $P(\mathbf{o}_t|\mathbf{s}t)$ and $P(\mathbf{s}t|\mathbf{s}{t-1}, \mathbf{u}{t-1})$ in pymdp. These two arrays of conditional distributions can also be factorized by modality and factor, respectively.

The A array, for instance, contains the agent’s observation model, that relates hidden states $\mathbf{s}_t$ to observations $\mathbf{o}_t$:

$$ \mathbf{A} = {A^1, A^2, …, A^M }, \hspace{5mm} A^m = P(o^m_t | s^1_t, s^2_t, …, s^F_t) $$

Therefore, we represent the A array as an object array whose constituent arrays are multidimensional numpy.ndarrays that encode conjunctive relationships between combinations of hidden states and observations.

This is best explained using the example in the original example code at the top of this tutorial. It is custom to build lists of the dimensionalities of the modalities (resp. hidden state factors) of your model, typically using lists named num_obs (for the dimensionalities of the observation modalities) and num_states (dimensionalities of the hidden state factors). These lists can then be used to automatically construct the A array with the correct shape.

For example, A_array[0] stores the conditional relationships between the hidden states $\mathbf{s}$ and observations within the first modality $o^1_t$, which has dimensionality 3. This explains why the shape of A_array[0] is (3, 4, 2, 3) – it stores the conditional relationships between each setting of the hidden state factors (which have dimensionalities [4, 2, 3]) and the observations within the first modality, which has dimensionality 3. Crucially, each sub-array A[m] stores the conditional dependencies between all the hidden state factor combinations (configurations of $s^1, s^2, …, s^F$) and the observations along modality m.

In this case, we used the pymdp.utils function random_A_matrix() to generate a random A array, but in most cases users will want to design their own bespoke observation models (or at least initialize them to some reasonable starting values). In such a scenario, the usual route is to initialize the A array to a series of identically-valued multidimensional arrays (e.g. arrays filled with 0’s or uniform values), and then fill out the conditional probability entries “by hand”, according to the task the user is interested in modelling.

For this purpose, utility functions like obj_array_zeros and obj_array_uniform come in handy. These functions takes as inputs list of shapes, where each shape contains the dimensionality (e.g. [2, 3, 4]) of one of the multi-dimensional arrays that will populate the final object array. For example, creating this shape list for the A array, given num_obs and num_states is quite straightforward:
