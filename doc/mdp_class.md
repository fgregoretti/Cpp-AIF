# MDP class
```c++
template <typename Ty, std::size_t M> class MDP
```

`MDP` class allows you to run active inference processes. Firstly, you need to is build a generative model in terms of **$\bf{A}$**, **$\bf{B}$**, **$\bf{C}$**, and **$\bf{D}$** arrays, true inital state **$\bf{S}$** arrays and optionally a vector of policies. if the latter is empty, the constructor will generate the policies. Then you have to pass them as parameters to the `MDP` constructor, and then start running active inference processes using the desired functions of the `MDP` class, like `infer_states`, `infer_policies`, `sample_action`, `sample_state`, and `sample_observation`.
 
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

