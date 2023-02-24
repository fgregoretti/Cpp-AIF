# Epistemic chaining

## 1. The problem

In this problem a rat is tasked to solve a spatial puzzle: must visit two cues located at different positions within a 2D grid world of size **$dim_x \times dim_y$**, in a specific order, to uncover the locations of two reward outcomes - one will give the rat a positive reward (the "Cheese"), and the other will give a negative reward (the "Shock"). 

Cue 1 is located in one location of the grid world, while there are four additional locations that could potentially contain Cue 2. However, only one of these four locations actually has Cue 2, while the other three are empty. Once the agent reaches Cue 1, it will receive one of four unambiguous signals (L1, L2, L3, L4), which indicate the exact location of Cue 2. After discovering the location of Cue 2, the agent can visit that location to receive one of two possible signals, which reveal the location of the reward or punishment. The reward and punishment are located in two distinct positions: "First" reward position, or "Second" reward position.

The optimal strategy to maximize reward while minimizing risk in this task involves the following approach: first, the agent needs to visit Cue 1 to obtain the signal that reveals the location of Cue 2. Once the location of Cue 2 is determined, the agent can then visit that location to receive the signal that indicates the location of the reward or punishment.

## 2. The generative model

The hidden states are factorized into three factors **$S^0, S^1$**, and **$S^2$**

1. Agent location: **$S^0$** encodes the agent's location in the grid world Cue with as many elements as there are the grid locations. Therefore it has cardinality **$dim_x \times dim_y$** and the tuples of **$(x, y)$** coordinate locations are mapped to linear indices by using **$y \times dim_x+x$**. It follows an example for a grid world of size **$7 \times 5$**
<img src=s0.png width=300>

2. Cue2 location: **$S^1$** has cardinality **$4$**, encoding in which of the four possible location Cue 2 is actually located (**$[L1, L2, L3, L4]$**).

3. Reward location: **$S^2$** has cardinality **$2$**, encoding which of the two reward positions ("First" or "Second") the "Cheese" has to be found in (**$[First, Second]$**).

The vector **$\bf{N_s}$** listing the dimensionality of the hidden states is **$[dim_x \times dim_y, 4, 2]$**.

Observations **$\bf{O}$** are organized in four factors **$O^0, O^1, O^2$**, and **$O^3$**

1. Location observation, **$O^0$**, representing the agent’s observation of its location in the grid world, with as many elements as there are the grid locations.

2. Cue1 observation, **$O^1$**, only obtained at the Cue 1 location, that signals in which of the 4 possible locations Cue 2 is located. When not at the Cue 1 location, the agent sees **Null** observation. Therefore it has cardinality **$5$** (**$[Null, L1, L2, L3, L4]$**).

3. Cue2 observation, **$O^2$**, only obtained at the Cue 2 location, that signals in which of the two reward locations (“First” or “Second”) the “Cheese” is located. When not at the Cue 2 location, the agent sees **Null**  observation. Therefore it has cardinality **$3$** (**$[Null, First, Second]$**).

4. Reward observation, **$O^3$**, only received when occupying one of the two reward locations (“Cheese” or “Shock”), and Null otherwise. Therefore it has cardinality **$3$** (**$[Null, Cheese, Shock]$**).

The vector **$\bf{N_o}$** listing the number of outcomes for each factor is **$[dim_x \times dim_y, 5, 3, 3]$**.

The control states **$U$** encode the actions of the agent. In this 2D grid world the agent have the ability to make movements in the **$4$** cardinal directions (NORTH, EAST, SOUTH, WEST)

### The transition model
To create the transition model we have to define the arrays **$B^0, B^1$**, and **$B^2$**, one for each state factor. The control states **$U$** determine the transitions from one state to another for the first hidden state factor only. Therefore only **$B^0$** will be dependent from action **$u$**, namely **$B^0_u$**

We create the transition model defining a vector of vector of objects `Transitions`. Specifically a vector with size **$N_f$**, and each element **$i$** will contain a vector of **$num\\_controls[i]$** of objects `Transitions` **$B$** with size **$N_s[f] \times N_s[f]$**

Being only the first hidden state factor controllable by the agent, **$num\\_controls[0]=4$**, while the other uncontrollable hidden state factors can be encoded as control factors of dimension **$1$**. **$num\\_controls=[4,1,1]$**

We need to write a derived Transitions class that adds a specialized method to fill out **$B^0_u$** according to the expected outcomes of the **$4$** actions. Note that the rows correspond to the ending state and the columns correspond to the starting state of a transition. Therefore the easyeast way to fill out the transition matrix **$B^0_u$** is to build it as a CSC sparse matrix and then converting it to CSR format by using the `void csc_tocsr(unsigned int col_ptr[], unsigned int row[])` method of the `Transitions` class.

In [`epistemic_chaining.hpp`](../../epistemic_chaining.hpp) we write:

```c++
class _Transitions : public Transitions<Ty>
{
public:
  using Transitions<Ty>::Transitions;

  void epistemic_chaining_init(int action, Grid<int> grid_)
  {
    for(unsigned int i = 0; i < this->Ns; i++)
    {
      this->col[i] = 0;
      this->row_ptr[i] = 0;
      this->data[i] = 0.0;
    }
    this->row_ptr[this->Ns] = 0;

    unsigned int row[this->Ns];
    memset(row, 0, this->Ns*sizeof(unsigned int));
    unsigned int col_ptr[this->Ns+1];
    memset(col_ptr, 0, (this->Ns+1)*sizeof(unsigned int));

    for (unsigned int s = 0; s < this->Ns; s++)
    {
      col_ptr[s] = s;
      row[s] = NextState(s, action, grid_);
      this->data[s] = 1;
    }
    col_ptr[this->Ns] = this->Ns;

    this->csc_tocsr(col_ptr, row);
  }
};
```

Where the function `NextState` is

```c++
/* return the state obtained performing action 'action'
from state 'state' */
int NextState(unsigned int state, unsigned int action,
              Grid<int> grid_)
{
  Coord current_pos = grid_.IndexToCoord(state);
  current_pos += MoveTo[action];

  if (grid_.Inside(current_pos)) {
    int next_pos = grid_.CoordToIndex(current_pos);
    return next_pos;
  }
  else
  {
    return state;
  }
}
```

In [`main_epistemic_chaining.cpp`](../../examples/main_epistemic_chaining.cpp) we write:

```c++
  std::vector<std::vector<Transitions<FLOAT_TYPE>*>> __B;

  std::vector<Transitions<FLOAT_TYPE>*> _b1;
  for (unsigned int a = 0; a < Nu; a++) {
    _Transitions<FLOAT_TYPE> *__b1 = new _Transitions<FLOAT_TYPE>(Ns[0], Ns[0]);
    __b1->epistemic_chaining_init(a, grid_);
    _b1.push_back((Transitions<FLOAT_TYPE> *) __b1);
  }
  __B.push_back(_b1);                                                                             
```

Fill out **$B^1$**, and **$B^2$** as identity matrices, encoding the fact that those hidden states are uncontrollable
```c++
  std::vector<Transitions<FLOAT_TYPE>*> _b2;
  Transitions<FLOAT_TYPE> *__b2 = new Transitions<FLOAT_TYPE>(Ns[1], Ns[1]);
  __b2->Eye();
  _b2.push_back(__b2);                                                            
  __B.push_back(_b2);
  
  std::vector<Transitions<FLOAT_TYPE>*> _b3;
  Transitions<FLOAT_TYPE> *__b3 = new Transitions<FLOAT_TYPE>(Ns[2], Ns[2]);
  __b3->Eye();
  _b3.push_back(__b3);
  __B.push_back(_b3);
```
