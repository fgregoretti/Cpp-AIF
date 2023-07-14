# MDP class
```c++
template <typename Ty, std::size_t M> class MDP
```

template `MDP` class: Ty is the template argument which is a placeholder for the data type used while M is the number of the observation  multidimensional array dimensions. `MDP` class allows you to run active inference processes. Firstly, you need to build a generative model in terms of **$\bf{A}$**, **$\bf{B}$**, **$\bf{C}$**, and **$\bf{D}$** arrays, initialize state **$\bf{S}$** arrays by setting up initial states **$\bf{s}_ 0$** and optionally build a vector of policies. if the latter is empty, the constructor will generate the policies. Then you have to pass them as parameters to the `MDP` constructor, and then start running active inference processes using the desired functions of the `MDP` class, like `infer_states`, `infer_policies`, `sample_action`, `sample_state`, and `sample_observation`.
 
***Constructor:***
```c++
  MDP(std::vector<Beliefs<Ty>*>& __D, /* initial state probabilities */
      std::vector<States*>& __S, /* true initial state */
      std::vector<std::vector<Transitions<Ty>*>>& __B, /* transition probabilities */
      std::vector<std::vector<likelihood<Ty,M>*>>& __A, /* observation model */
#ifdef WITH_GP
      std::vector<std::vector<likelihood<Ty,M>*>>& __AA, /* observation process */
#endif
      std::vector<Priors<Ty>*>& __C, /* terminal cost probabilities */
      std::vector<std::vector<int>>& __V, /* policies */
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
- `__AA` when compiled with macro WITH_GP, observation likelihood of the generative process 
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

The basic usage is as follows:

```c++
MDP<double,N> *mdp = new MDP<double,N>(__D,__S,__B,__A,__C,__V,T_,<more_params>);
unsigned int tt = 0;                                                                                       
while (tt < T)
{                                                                                             
  infer_states(tt);
  std::vector<Ty> G = infer_policies(tt);
  int a = sample_action(tt);                                                                               
  if (tt < T-1)                                                                                            
  {                                                                                                        
    sample_state(tt+1, a);                                                                                 
    sample_observation(tt+1, a);                                                                           
  }                                                                                                        
  tt += 1;                                                                                                 
}
```
**Public members:**
- `unsigned int Nf` number of hidden-states factors
- `unsigned int Ng` number of outcome factors
- `unsigned int Np` number of allowable policies
- `std::vector<std::vector<int>> _st` $\\_st[i][j]$ sampled state for factor $j$ at time step $i$
- `std::vector<std::vector<int>> _ot` $\\_ot[i][j]$ observed state for factor $j$ at time step $i$

***Public methods:***
```c++
void infer_states(unsigned int t)
```
Compute expectations of allowable policies and current state, assigning results to class members `_ut` and `_X`, respectively. `_X[f]->getArray(t)` refers to beliefs about factor `f` expected at timepoint `t`.

**Parameters**
- `t` time step

```c++
std::vector<Ty> infer_policies(unsigned int t)
```
Return negative expected free energy $\bf{G}$ of each policy `std::vector<Ty> G`. Update class members posterior precision `_W` and posterior beliefs about control `_P`.
 
**Parameters**
- `t` time step

```c++
int sample_action(unsigned int t)
```
[Sample](utils.md#Sampling) or select (when not compiled with macro BEST_AS_MAX) next action (the action that minimizes expected free energy) from the posterior over control states. This function both assigns the action to the class member `U` and returns it.

**Parameters**
- `t` time step

```c++
void sample_state(unsigned int t, int action)
```
Next [sampled](utils.md#Sampling) state under the action `action`. When not compiled with macro SAMPLE_AS_MAX select a random action from the posterior over control states using a cumulative distribution function (CDF), otherwise the selected action is chosen as the maximum of the posterior over actions. If there are multiple maxima, then randomly select one of them.

**Parameters**
- `t` time step
- `action` sampled at previous time step

```c++
void sample_observation(unsigned int t, int action)
```
Next observed state under the action `action`.

**Parameters**
- `t` time step
- `action` sampled at previous time step

```c++
void actiive inference()
```
basic active inference procedure 

## Learning
The following public methods update parameters of posteriors in POMDP generative models.

```c++
std::vector<std::vector<likelihood<Ty,M>*>>& MDP<Ty,M>::update_A(                                       
                std::vector<std::vector<likelihood<Ty,M>*>>& _a,                                             
                Ty eta, unsigned int tt
```
Update parameters of the observation likelihood distribution.

**Parameters**
- `_a` observation likelihood to be updated
- `eta` learning rate
- `tt` time step

```c++
std::vector<std::vector<Transitions<Ty>*>>& MDP<Ty,M>::update_B(                                             
                std::vector<std::vector<Transitions<Ty>*>>& _b,                                              
                Ty eta, unsigned int tt)
```
Update parameters of the transition distribution. 

**Parameters**
- `_b` transition distribution to be updated
- `eta` learning rate
- `tt` time step


```c++
std::vector<Priors<Ty>*>& MDP<Ty,M>::update_C(                                                               
                std::vector<Priors<Ty>*>& _c,                                                                
                Ty eta, unsigned int tt)
```
Update prior preferences.

**Parameters**
- `_c` prior preferences to be updated
- `eta` learning rate
- `tt` time step

```c++
std::vector<Beliefs<Ty>*>& MDP<Ty,M>::update_D(                                                              
                std::vector<Beliefs<Ty>*>& _d,                                                               
                Ty eta, unsigned int tt)
```
Update parameters of the initial hidden state distribution (prior beliefs about hidden states).

**Parameters**
- `_d` initial beliefs to be updated
- `eta` learning rate
- `tt` time step
