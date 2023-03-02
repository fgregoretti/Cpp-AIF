# MDP class
```c++
template <typename Ty, std::size_t M> class MDP
```

`MDP` class allows you to run active inference processes. Firstly, you need to is build a generative model in terms of **$\bf{A}$**, **$\bf{B}$**, **$\bf{C}$**, and **$\bf{D}$** arrays, initialize state **$\bf{S}$** arrays by setting up initial states **$\bf{s}_ 0$** and optionally build a vector of policies. if the latter is empty, the constructor will generate the policies. Then you have to pass them as parameters to the `MDP` constructor, and then start running active inference processes using the desired functions of the `MDP` class, like `infer_states`, `infer_policies`, `sample_action`, `sample_state`, and `sample_observation`.
 
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

In `Cpp-AcI` generative model distributions as well as expectations of hidden states, states and observations are represented as vector of vector (**$\bf{A}$** and **$\bf{B}$**) or vector (all the others) of "custom objects". These are instances of [classes](custom_array_classes.md) specifically designed to handle active inference data, with an array as member. 

Understanding the representation of factorized probability distributions as vector (of vector) of class instances is critical for understanding and constructing generative models in `Cpp-AcI`. In particular, we use vector of vector of specific class instances to encode the observation and transition models of the agent’s generative model. This representation is chosen because the observation space is typically factorized into multiple observation factors, and the hidden states are similarly factorized into multiple hidden state factors. Additionally, this allows for the expression of dependencies on control states **$u$** that the agent can execute.

**$\bf{A}$** and **$\bf{B}$** represent the conditional distributions **$P(\mathbf{o}_ t|\mathbf{s}_ t, u_ t)$** and **$P(\mathbf{s}_ t|\mathbf{s}_ {t-1}, u_ {t-1})$**, being $\mathbf{s}_ t$ the hidden states and  $\mathbf{o}_ t$ the observations. These arrays of conditional distributions can also be factorized by observation and hidden state factor, respectively.

**$\bf{A}$**, for instance, contains the agent’s observation model, that relates hidden states $\mathbf{s}_t$ to observations $\mathbf{o}_t$:

$$ \mathbf{A} = {A^0_u, A^1_u, …, A^{N_g}_ u }, \hspace{5mm} A^m_u = P(o^m_t | s^0_t, s^1_t, …, s^{N_f}_ t,u_t) $$

where **$N_g$** is the number of observation factors and **$N_f$** is the number of hidden state factors.

Therefore, we represent it as a vector of size **$N_g$** whose each element will contain a vector of **$N_u$** (or **$1$** if the factor is uncontrollable) of `likelihood` class instances. This class handles a multidimensional array that encodes conjunctive relationships between combinations of hidden states and observations.

For example, if **$N_f=3$**, **$N_g=2$**, the number of hidden states for each factor is **$\bf{N_s}=[4,2,3]$** and the number of observations for each factor is **$\bf{N_o}=[3,5]$**, **$A^0_u$** stores the conditional relationships between the hidden states $\mathbf{s}$ and observations within the first factor $o^1_t$, which has dimensionality 3. Therfore **$A^0_u$** is a 4D array of dimensions **$(3, 4, 2, 3)$** – it stores the conditional relationships between each setting of the hidden state factors (which have dimensionalities **$[4, 2, 3]$**) and the observations within the first factor, which has dimensionality **$3$**. Then, each array **$A^m_u$** stores the conditional dependencies between all the hidden state factor combinations (configurations of $s^0, s^1, …, s^{N_f}$) and the observations along factor **$m$**.

We can use the instruction
```c++
likelihood<double,4> *A0 = new likelihood<double,4>(3,4,2,3);
```
to create a 4D array of double with dimensions **$(3, 4, 2, 3)$**

Users may desire to create their own customized observation models or at least set them up with appropriate initial values. Typically, in such situations, users begin by initializing the **$A$** arrays with identical multidimensional arrays filled with zeros, using `Zeros` class method. They would then fill out the conditional probability entries “by hand”, using `()` operator or `setValue` class method, according to the task the user is interested in modelling.
