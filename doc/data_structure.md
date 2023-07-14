# Data structure and factorized distributions 
In `cpp-AIF` generative model distributions as well as expectations of hidden states, states and observations are represented as vector of vectors (**$\bf{A}$** and **$\bf{B}$**) or vector (all the others) of "specific objects". These are instances of [classes](generative_model_classes.md) specifically designed to handle active inference data, with an array as member. 

Understanding the representation of factorized probability distributions as vector (of vectors) of class instances is critical for understanding and constructing generative models in `cpp-AIF`. In particular, we use vector of vectors of specific class instances to encode the observation and transition models of the agent’s generative model. This representation is chosen because the observation space is typically factorized into multiple observation factors, and the hidden states are similarly factorized into multiple hidden state factors. Additionally, this allows for the expression of dependencies on control states **$u$** that the agent can execute.

**$\bf{A}$** and **$\bf{B}$** represent the conditional distributions **$P(\mathbf{o}_ t|\mathbf{s}_ t, u_ t)$** and **$P(\mathbf{s}_ t|\mathbf{s}_ {t-1}, u_ {t-1})$**, being $\mathbf{s}_ t$ the hidden states and  $\mathbf{o}_ t$ the observations. These arrays of conditional distributions can also be factorized by observation and hidden state factor, respectively.

**$\bf{A}$**, for instance, contains the agent’s observation model, that relates hidden states $\mathbf{s}_ t$ to observations $\mathbf{o}_ t$:

$$ \mathbf{A} = {A^0_u, A^1_u, …, A^{N_g}_ u }, \hspace{5mm} A^m_u = P(o^m_t | s^0_t, s^1_t, …, s^{N_f}_ t,u_t) $$

where **$N_g$** is the number of observation factors and **$N_f$** is the number of hidden state factors.

Therefore, we represent it as a vector of size **$N_g$** whose each element will contain a vector of **$N_u$** (or **$1$** if the factor is uncontrollable) of [`likelihood`](generative_model_classes.md#template-typename-t-typename-s-class-likelihood)  class instances. **$N_u$** is the number of control states. The `likelihood` class handles a multidimensional array that encodes conjunctive relationships between combinations of hidden states and observations, eventually further conditioned on actions or control states along observation factor.

For example, if **$N_f=3$**, **$N_g=2$**, the number of hidden states for each factor is **$\bf{N_s}=[4,2,3]$** and the number of observations for each factor is **$\bf{N_o}=[3,5]$**, **$A^0_u$** stores the conditional relationships between the hidden states $\mathbf{s}$ and observations within the first factor $o^1_t$, which has dimensionality 3. Therefore **$A^0_u$** is a four-dimensional array with dimensions **$(3, 4, 2, 3)$** – it stores the conditional relationships between each setting of the hidden state factors (with dimensionalities of **$[4, 2, 3]$**) and the observations within the first factor, which has dimensionality **$3$**. Then, each array **$A^m_u$** stores the conditional dependencies between all the hidden state factor combinations (configurations of $s^0, s^1, …, s^{N_f}$) and the observations along factor **$m$**.

We can use the instruction
```c++
likelihood<double,4> *A0 = new likelihood<double,4>(3,4,2,3);
```
to create a four-dimensional array of double with dimensions **$(3, 4, 2, 3)$**.

Users may desire to create their own customized observation models or at least set them up with appropriate initial values. Typically, in such situations, users begin by initializing the **$A$** arrays with identical multidimensional arrays filled with zeros, using `Zeros` class method. They would then fill out the conditional probability entries “by hand”, using `()` operator or `setValue` class method, according to the task the user is interested in modelling.

After having filled out `A0` we define a vector of vectors of pointers to `likelihood` objects 

`std::vector<std::vector<likelihood<FLOAT_TYPE,4>*>> A;`

to be passed to `MDP` constructor. We also need to define a vector of pointers to `likelihood` objects

`std::vector<likelihood<FLOAT_TYPE,4>*> a0;`

Assuming **$A^0_u$** is uncontrollable, namely `A0` cannot be changed by **$u$** and is therefore the same for each **$u$**, we just push back  `A0` into `a0` and then `a0` into `A`

```c++
a0.push_back(A0);
A.push_back(a0);
```

Analogous instructions are needed to add **$A^1_u$**.

Similarly, we can create **$\bf{B}$** that encodes the temporal dependencies between the hidden state factors over time, further conditioned on control states. Mathematically, **$\bf{B}$** can be formulated as:

$$\mathbf{B} = {B^0_u, B^1_U, …, B^{N_f}_ u}, \hspace{5mm} B^f_u = P(s^f_t | s^f_{t-1}, u^f_{t-1})$$

where $u^f_{t-1}$ denotes the control state for control factor $f$, taken at time $t-1$.

Therefore, we represent it as a vector of size **$N_f$** whose each element will contain a vector of **$N_u$** (or **$1$** if the factor is uncontrollable) of [`Transitions`](generative_model_classes.md#template-typename-t-class-transitions) class instances. This class handles a matrix of size $Ns[f] \times Ns[f]$ encoding the conditional dependencies between states for a given factor $f$ at subsequent timepoints, further conditioned on actions or control states along that factor. This method of construction implies that the hidden state factors are statistically uncorrelated with each other. In other words, the hidden state factor $f$ is influenced solely by its own state at the previous time step, as well as the state of the $f-th$ control factor.

If **$N_f=3$** and the number of hidden states for each factor is **$\bf{N_s}=[4,2,3]$**, **$B^0_u$** is a transition matrix with dimensions $(4, 4)$. 

We can use the instruction
```c++
Transitions<double> *B0 = new Transitions<double>(4,4);
```
to create a transition matrix of double with dimensions **$(4, 4)$** and $4$ non-zero values (first parameter is size of the matrix, while second parameter is number of non-zero values). `Transitions` matrices are stored in CSR format, therefore the easyest way to design a customized transition model is to build a 2D matrix using vector of vectors and then pass it to the constructor with vector of vectors as parameter. Alternatively it would be possible to write a custom function that uses `SetCol`, `SetRowPtr` and `SetData` class methods to fill out the arrays (**$col$**, **$row\\_ ptr$**, **$data$**) used to represent transition matrix in CSR format.

To create **$\bf{B}$** we define a vector of vectors of pointers to `Transitions` objects 
```c++
std::vector<std::vector<Transitions<double>*>> B;
```
to be passed to `MDP` constructor. We also need to define a vector of pointers to `Transitions` objects

`std::vector<Transitions<FLOAT_TYPE,4>*> b0;`

if **$N_u=4$** and assuming that, regarding first hidden state factor, control states determine transitions from one state to another we need to create a transition matrix for each control state:

```c++
for (unsigned int a = 0; a < Nu; a++) {                                                
  Transitions<doubel> *B0 = new Transitions<double>(4, 4);
  /* function that fill out B0 depending on control state */
  ...
  b0.push_back(B0);
}
```
Then we just push back `b0` into `B`

```c++
B.push_back(b0);
```

Analogous instructions are needed to add **$B^1_u$** and **$B^2_u$**.
