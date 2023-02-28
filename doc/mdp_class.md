# MDP class
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
- `__D` agent's initial beliefs
- `__S` agent's true inital state
- `__B` transition model
