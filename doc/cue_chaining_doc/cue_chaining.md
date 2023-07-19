# Epistemic chaining

## 1. The problem

In this problem a rat is tasked to solve a spatial puzzle: must visit two cues located at different positions within a 2D grid world of size **$dim_x \times dim_y$**, in a specific order, to uncover the locations of two reward outcomes - one will give the rat a positive reward (the "Cheese"), and the other will give a negative reward (the "Shock"). 

Cue 1 is located in one location of the grid world, while there are four additional locations that could potentially contain Cue 2. However, only one of these four locations actually has Cue 2, while the other three are empty. Once the agent reaches Cue 1, it will receive one of four unambiguous signals (L1, L2, L3, L4), which indicate the exact location of Cue 2. After discovering the location of Cue 2, the agent can visit that location to receive one of two possible signals, which reveal the location of the reward or punishment. The reward and punishment are located in two distinct positions: "First" reward position, or "Second" reward position.

The optimal strategy to maximize reward while minimizing risk in this task involves the following approach: first, the agent needs to visit Cue 1 to obtain the signal that reveals the location of Cue 2. Once the location of Cue 2 is determined, the agent can then visit that location to receive the signal that indicates the location of the reward or punishment.

For the implementation of this problem using `cpp-AIF` refer to the files [`main_epistemic_chaining.cpp`](../../examples/main_epistemic_chaining.cpp) and [`epistemic_chaining.hpp`](../../examples/epistemic_chaining.hpp). Here, we have built 
`Beliefs`, `Transitions` and `likelihood` derived classes in order to add specialized methods to fill the arrays but one can also write these methods elsewhere.

## 2. The 2D grid world
To create the physical environment inhabited by the agent we defined a 2D grid world within a specific class `Grid` (header file [`grid.hpp`](../../examples/grid.hpp)). Locations on the grid are identified using **$(x, y)$** tuples, which correspond to a specific row and column, respectively, on the grid.

Let's create a grid world with dimensions **$7 \times 5$**. To this purpose, we can write a function whose parameters are `Grid<int> grid_(size_x, size_y)`, `Coord cue1_pos_`, `std::vector<Coord> cue2_pos_`, `Coord start_position_`, `std::vector<Coord> reward_pos_`, `unsigned int reward`, respectively the `Grid` object, the Cue 1 location, the vector of four Cue2 locations, the agent start location, the vector of the reward locations, and the variable indicating where the positive reward is located, as follows:

```c++
void Init_7_5(Grid<int>& grid_, Coord& cue1_pos_,
              std::vector<Coord>& cue2_pos_, Coord& start_pos_,
              std::vector<Coord>& reward_pos_, unsigned int reward)
{
  std::cout << "epistemic_chaining(7, 5)" << std::endl;
  cue1_pos_ = Coord(0, 2);
  cue2_pos_ = { Coord(2, 4), Coord(3, 3), Coord(3, 1), Coord(2, 0) };
  start_pos_ = Coord(0, 4);
  reward_pos_ = { Coord(5, 3), Coord(5, 1) };
  grid_.SetAllValues(-1);
  grid_(cue1_pos_) = 1;
  for (unsigned int i = 0; i < cue2_pos_.size(); ++i) {
    grid_(cue2_pos_[i]) = 2+i;
  }
  grid_(reward_pos_[reward]) = 100;
  grid_(reward_pos_[1-reward]) = -100;
}
```
<img src=grid_world_7x5.png width=300>

## 3. The generative model
The hidden states are factorized into three factors **$S^0, S^1$**, and **$S^2$**. **$N_f=3$**.

1. Agent location: **$S^0$** encodes the agent's location in the grid world with as many elements as there are the grid locations. Therefore it has cardinality **$dim_x \times dim_y$** and the tuples of **$(x, y)$** coordinate locations are mapped to linear indices by using the function **$y \times dim_x+x$** (`CoordToIndex` method of the class `Grid`). It follows an example for the grid world of size **$7 \times 5$**
<img src=s0.png width=300>

2. Cue2 location: **$S^1$** has cardinality **$4$**, encoding in which of the four possible locations Cue 2 is actually located (**$[L1, L2, L3, L4]$**).

3. Reward location: **$S^2$** has cardinality **$2$**, encoding which of the two reward positions ("First" or "Second") the "Cheese" has to be found in (**$[First, Second]$**).

The vector **$\bf{N_s}$** listing the dimensionality of the hidden states is **$[dim_x \times dim_y, 4, 2]$**.

Observations **$\bf{O}$** are organized in four factors **$O^0, O^1, O^2$**, and **$O^3$**. **$N_g=4$**.

1. Location observation, **$O^0$**, representing the agent’s observation of its location in the grid world, with as many elements as there are the grid locations.

2. Cue1 observation, **$O^1$**, only obtained at the Cue 1 location, that signals in which of the 4 possible locations Cue 2 is located. When not at the Cue 1 location, the agent sees **Null** observation. Therefore it has cardinality **$5$** (**$[Null, L1, L2, L3, L4]$**).

3. Cue2 observation, **$O^2$**, only obtained at the Cue 2 location, that signals in which of the two reward locations (“First” or “Second”) the “Cheese” is located. When not at the Cue 2 location, the agent sees **Null**  observation. Therefore it has cardinality **$3$** (**$[Null, First, Second]$**).

4. Reward observation, **$O^3$**, only received when occupying one of the two reward locations (“Cheese” or “Shock”), and Null otherwise. Therefore it has cardinality **$3$** (**$[Null, Cheese, Shock]$**).

The vector **$\bf{N_o}$** listing the number of outcomes for each factor is **$[dim_x \times dim_y, 5, 3, 3]$**.

The control states **$U$** encode the actions of the agent. In this 2D grid world, the agent has the ability to make movements in the **$4$** cardinal directions (NORTH, EAST, SOUTH, WEST)

### The initial beliefs: **$\bf{D}$**
The agent's initial belief is defined over the multi-factor hidden states, therefore we have to define three arrays **$D^0, D^1$**, and **$D^2$**, each corresponding to a specific hidden state factor.

**$D^0$** encodes the prior beliefs over the initial location of the agent in the grid world, while **$D^1$** and **$D^2$** are arrays of ones implying that all the states are equally probable.
 
We create the initial beliefs defining a vector of objects `Beliefs`. Specifically, a vector with size **$N_f$**, and each element will contain an object `Beliefs` with size **$N_s[f]$**.

We can build a derived `Beliefs` class that adds a specialized method to fill out **$D^0$**, which is a method that assigns **$1$** to the state corresponding to the initial location of the agent and **$0$** elsewhere.

In [`epistemic_chaining.hpp`](../../examples/epistemic_chaining.hpp) we wrote:

```c++
template <typename T>
class _Beliefs : public Beliefs<T>
{
public:
  using Beliefs<T>::Beliefs;

  void epistemic_chaining_init(Coord start_pos_,
                               Grid<int> grid_)
  {
    this->Zeros();

    int pos = grid_.CoordToIndex(start_pos_);

    this->setValue(1.0,pos)
  }
};
```
In [`main_epistemic_chaining.cpp`](../../examples/main_epistemic_chaining.cpp) we wrote:

```c++
  std::vector<Beliefs<FLOAT_TYPE>*> __D;

  _Beliefs<FLOAT_TYPE> *_d1 = new _Beliefs<FLOAT_TYPE>(Ns[0]);
  _d1->epistemic_chaining_init(start_position_, grid_);
  __D.push_back((Beliefs<FLOAT_TYPE> *) _d1);

  _Beliefs<FLOAT_TYPE> *_d2 = new _Beliefs<FLOAT_TYPE>(Ns[1]);
  _d2->Ones();
  __D.push_back((Beliefs<FLOAT_TYPE> *) _d2);

  _Beliefs<FLOAT_TYPE> *_d3 = new _Beliefs<FLOAT_TYPE>(Ns[2]);
  _d3->Ones();
  __D.push_back((Beliefs<FLOAT_TYPE> *) _d3);
```

### The transition model: **$\bf{B}$**
To create the transition model we have to define the arrays **$B^0, B^1$**, and **$B^2$**, one for each state factor. The control states **$U$** determine the transitions from one state to another for the first hidden state factor only. Therefore only **$B^0$** will be dependent from action **$u$**, namely **$B^0_u$**

We create the transition model defining a vector of vectors of objects `Transitions`. Specifically a vector with size **$N_f$**, and each element **$i$** will contain a vector of **$num\\_controls[i]$** of objects `Transitions` **$B$** with size **$N_s[f] \times N_s[f]$**

Being only the first hidden state factor controllable by the agent, **$num\\_controls[0]=4$**, while the other uncontrollable hidden state factors can be encoded as control factors of dimension **$1$**. **$num\\_controls=[4,1,1]$**

We can build a derived `Transitions` class that adds a specialized method to fill out **$B^0_u$** according to the expected outcomes of the **$4$** actions. Note that the rows correspond to the ending state and the columns correspond to the starting state of a transition. Therefore the easiest way to fill out the transition matrix **$B^0_u$** is to build it as a CSC (in which values are read first by column) sparse matrix and then convert it to CSR format by using the `void csc_tocsr(unsigned int col_ptr[], unsigned int row[])` method of the `Transitions` class.

In [`epistemic_chaining.hpp`](../../examples/epistemic_chaining.hpp) we wrote:

```c++
template <typename Ty>                                                                                       
class _Transitions : public Transitions<Ty>                                                                  
{                                                                                                            
public:                                                                                                      
  using Transitions<Ty>::Transitions;                                                                        
                                                                                                             
  void epistemic_chaining_init(int action, Grid<int> grid_)                                                    {                                                                                                          
    for(unsigned int i = 0; i < this->Ns; i++)                                                               
    {                                                                                                              this->SetCol(0,i);                                                                                     
      this->SetRowPtr(0,i);                                                                                  
      this->SetData(0.0,i);                                                                                      }                                                                                                        
    this->SetRowPtr(0,this->Ns);                                                                             
                                                                                                             
    unsigned int row[this->Ns];                                                                              
    memset(row, 0, this->Ns*sizeof(unsigned int));                                                           
    unsigned int col_ptr[this->Ns+1];                                                                        
    memset(col_ptr, 0, (this->Ns+1)*sizeof(unsigned int));                                                   
                                                                                                             
    for (unsigned int s = 0; s < this->Ns; s++)                                                              
    {                                                                                                        
      col_ptr[s] = s;                                                                                        
      row[s] = NextState(s, action, grid_);                                                                  
      this->SetData(1,s);                                                                                    
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

In [`main_epistemic_chaining.cpp`](../../examples/main_epistemic_chaining.cpp) we wrote:

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

Fill out **$B^1$** and **$B^2$** as identity matrices, encoding the fact that those hidden states are uncontrollable
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

### The observation model: **$\bf{A}$**
**$\bf{A}$** has four components, encoding the agent's beliefs about how hidden states probabilistically cause observations within each factor of the observations: the multidimensional arrays **$A^0, A^1, A^2$**, and **$A^3$**. Their dimensions are **$N_o[0] \times N_s[0] \times N_s[1] \times N_s[2]$**, 
**$N_o[1] \times N_s[0] \times N_s[1] \times N_s[2]$**, **$N_o[2] \times N_s[0] \times N_s[1] \times N_s[2]$**, and **$N_o[3] \times N_s[0] \times N_s[1] \times N_s[2]$** respectively. All the components are therefore 4-dimensional arrays and are the same for each action **$u$**.

We create the observation model defining a vector of vectors of objects `likelihood`. Specifically a vector with size **$N_g$**, and each element **$i$** will contain a vector of one object `likelihood` **$A$** with size **$N_o[g] \times N_s[0] \times N_s[1] \times N_s[2]$**.

We can build a derived `likelihood` class that adds specialized methods to fill out **$A^0, A^1, A^2$**, and **$A^3$**.

In [`epistemic_chaining.hpp`](../../examples/epistemic_chaining.hpp) we wrote:

```c++
template <typename T, std::size_t N>
class _likelihood : public detail::likelihood<T, typename gen_seq<N>::type>
{
public:
  using likelihood<T,N>::likelihood;

  /* location observation */
  void Observe(std::vector<int> num_states);
  /* cue1 observation */
  void Observe(std::vector<int> num_states,
               Grid<int> grid_, Coord cue1_location,
               std::vector<Coord> cue2_location);
  /* cue2 observation */
  void Observe(std::vector<int> num_states, Grid<int> grid_,
               std::vector<Coord> cue2_location);
  /* reward observation  */
  void Observe(std::vector<int> num_states, Grid<int> grid_,
               std::vector<Coord> reward_location, T a);
};
```

Fill out **$A^0$** making the location observation only depending on the location state:

```c++
template <typename T, std::size_t N>
void _likelihood<T,N>::Observe(std::vector<int> num_states)
{
  this->Zeros();

  /* make the location observation only depend on the location
     state  */
  for (int s = 0; s < num_states[0]; s++)
    for (int j = 0; j < num_states[1]; ++j)
      for (int k = 0; k < num_states[2]; ++k)
        this->setValue(1,s,s,j,k);
}
```

Fill out **$A^1$** making the cue1 observation depending on both the agent being at cue1 location and the location of cue2:

```c++
/* cue1 observation */
template <typename T, std::size_t N>
void _likelihood<T,N>::Observe(std::vector<int> num_states,
               Grid<int> grid_, Coord cue1_location,
               std::vector<Coord> cue2_location)
{
  this->Zeros();

  /* make Null the most likely observation everywhere */
  for (int i = 0; i < num_states[0]; ++i)
    for (int j = 0; j < num_states[1]; ++j)
      for (int k = 0; k < num_states[2]; ++k)
        this->setValue(1,0,i,j,k);

  /* make the cue1 signal to be contingent upon both the agent's
     presence at the cue 1 location and the location of cue2 */
  for (unsigned int i = 0; i < cue2_location.size(); ++i)
    for (int k = 0; k < num_states[2]; ++k)
    {
      this->setValue(0,0,grid_.CoordToIndex(cue1_location),i,k);
      this->setValue(1,i+1,grid_.CoordToIndex(cue1_location),i,k);
    }
}
```

Fill out **$A^2$** making cue2 observation depending on both the agent's presence at the correct cue2 location and the reward location:

```c++
/* cue2 observation */
template <typename T, std::size_t N>
void _likelihood<T,N>::Observe(std::vector<int> num_states, Grid<int> grid_,
               std::vector<Coord> cue2_location)
{
  this->Zeros();

  /* make Null the most likely observation everywhere */
  for (int i = 0; i < num_states[0]; ++i)
    for (int j = 0; j < num_states[1]; ++j)
      for (int k = 0; k < num_states[2]; ++k)
        this->setValue(1,0,i,j,k);

  /* if the agent is located at the cue2 location, provide 
     a signal indicating the location of the reward */
  for (unsigned int i = 0; i < cue2_location.size(); ++i)
  {
    int loc_index = grid_.CoordToIndex(cue2_location[i]);

    for (int k = 0; k < num_states[2]; ++k)
      this->setValue(0,0,loc_index,i,k);
    this->setValue(1,1,loc_index,i,0);
    this->setValue(1,2,loc_index,i,1);
  }
}
```

Fill out **$A^3$** making the reward observation to be contingent upon both the agent's presence at the reward location and the reward location:

```c++
/* reward observation  */
template <typename T, std::size_t N>
void _likelihood<T,N>::Observe(std::vector<int> num_states, Grid<int> grid_,
               std::vector<Coord> reward_location, T a)
{
  this->Zeros();                                                                                             
                                                                                                             
  /* make Null the most likely observation everywhere */                                                     
  for (int i = 0; i < num_states[0]; ++i)                                                                    
    for (int j = 0; j < num_states[1]; ++j)                                                                  
      for (int k = 0; k < num_states[2]; ++k)                                                                
        this->setValue(1,0,i,j,k);                                                                           
                                                                                                             
  int reward_first_index = grid_.CoordToIndex(reward_location[0]);                                           
  int reward_second_index = grid_.CoordToIndex(reward_location[1]);                                          
                                                                                                             
  /* fill out the contingences arising when the agent is located                                             
     in the reward location identified as 'first' */                                                         
  for (int j = 0; j < num_states[1]; ++j)                                                                    
  {                                                                                                          
    this->setValue(a,1,reward_first_index,j,0);                                                              
    this->setValue((1-a)/2,1,reward_first_index,j,1);                                                        
    this->setValue(a,2,reward_first_index,j,1);                                                              
    this->setValue((1-a)/2,2,reward_first_index,j,0);                                                        
    for (int k = 0; k < num_states[2]; ++k)                                                                  
      this->setValue((1-a)/2,0,reward_first_index,j,k);                                                      
  }                                                                                                          
                                                                                                             
  /* fill out the contingences arising when the agent is located                                             
     in the reward location identified as 'second' */                                                        
  for (int j = 0; j < num_states[1]; ++j)                                                                    
  {                                                                                                          
    this->setValue(a,1,reward_second_index,j,1);                                                             
    this->setValue((1-a)/2,1,reward_second_index,j,0);                                                       
    this->setValue(a,2,reward_second_index,j,0);                                                             
    this->setValue((1-a)/2,2,reward_second_index,j,1);                                                       
    for (int k = 0; k < num_states[2]; ++k)                                                                  
      this->setValue((1-a)/2,0,reward_second_index,j,k);                                                     
  }
}
```

In [`main_epistemic_chaining.cpp`](../../examples/main_epistemic_chaining.cpp) we wrote:

```c++
  std::vector<std::vector<likelihood<FLOAT_TYPE,4>*>> __A;

  std::vector<likelihood<FLOAT_TYPE,4>*> _a1;
  _likelihood<FLOAT_TYPE,4> *__a1 = new _likelihood<FLOAT_TYPE,4>(Ns[0],Ns[0],Ns[1],Ns[2]);
  __a1->Observe(Ns);
  _a1.push_back((likelihood<FLOAT_TYPE,4> *) __a1);
  __A.push_back(_a1);

  std::vector<likelihood<FLOAT_TYPE,4>*> _a2;
  _likelihood<FLOAT_TYPE,4> *__a2 = new _likelihood<FLOAT_TYPE,4>(5,Ns[0],Ns[1],Ns[2]);
  __a2->Observe(Ns, grid_, cue1_pos_, cue2_pos_);
  _a2.push_back((likelihood<FLOAT_TYPE,4> *) __a2);
  __A.push_back(_a2);

  std::vector<likelihood<FLOAT_TYPE,4>*> _a3;
  _likelihood<FLOAT_TYPE,4> *__a3 = new _likelihood<FLOAT_TYPE,4>(3,Ns[0],Ns[1],Ns[2]);
  __a3->Observe(Ns, grid_, cue2_pos_);
  _a3.push_back((likelihood<FLOAT_TYPE,4> *) __a3);
  __A.push_back(_a3);
  
    std::vector<likelihood<FLOAT_TYPE,4>*> _a4;
  _likelihood<FLOAT_TYPE,4> *__a4 = new _likelihood<FLOAT_TYPE,4>(3,Ns[0],Ns[1],Ns[2]);
  __a4->Observe(Ns, grid_, reward_pos_, 0.9);
  _a4.push_back((likelihood<FLOAT_TYPE,4> *) __a4);
  __A.push_back(_a4);
```

### The prior over (preferred) observations: **$\bf{C}$**
Being defined as priors over observations, C will consist of four arrays corresponding to the priors over the different observation factors: **$C^0, C^1, C^2$**, and **$C^3$** with size **$N_s[0]$**, **$5$**, **$3$**, and **$3$** respectively. **$C^3$** encodes the prior preferences for different levels of the Reward observation outcome, while the others are zero arrays.

We create the prior over observations defining a vector of objects `Priors`. Specifically, a vector with size **$N_g$**, and each element will contain an object `Priors` with size **$N_o[g]$**.

In [`main_epistemic_chaining.cpp`](../../examples/main_epistemic_chaining.cpp) we wrote:
```c++
  std::vector<Priors<FLOAT_TYPE>*> __C;

  std::vector<FLOAT_TYPE> C1(Ns[0]);
  std::fill(C1.begin(), C1.end(), 0);
  softmax<FLOAT_TYPE>(C1);
  Priors<FLOAT_TYPE>* _C1 = new Priors<FLOAT_TYPE>(C1);
  __C.push_back(_C1);

  std::vector<FLOAT_TYPE> C2(5);
  std::fill(C2.begin(), C2.end(), 0);
  softmax<FLOAT_TYPE>(C2);
  Priors<FLOAT_TYPE>* _C2 = new Priors<FLOAT_TYPE>(C2);
  __C.push_back(_C2);

  std::vector<FLOAT_TYPE> C3(3);
  std::fill(C3.begin(), C3.end(), 0);
  softmax<FLOAT_TYPE>(C3);
  Priors<FLOAT_TYPE>* _C3 = new Priors<FLOAT_TYPE>(C3);
  __C.push_back(_C3);
  
  std::vector<FLOAT_TYPE> C4(3);
  std::fill(C4.begin(), C4.end(), 0);
  C4[1] = 2.0; /* make the agent want to encounter the "Cheese" observation level */
  C4[2] = -4.0; /* make the agent not want to encounter the "Shock" observation level */
  softmax<FLOAT_TYPE>(C4);
  Priors<FLOAT_TYPE>* _C4 = new Priors<FLOAT_TYPE>(C4);
  __C.push_back(_C4);
```

### True Initial State
If we set up the true Cue 2 location and the location of the positive reward:
```c++
unsigned int cue2 = 0;
unsigned int reward = 0;
```
we can set up the true initial State accordingly. We define a vector of three objects `States`, we initialize them to all 0s and then we use the `Set` method to assign the corresponding true initial state at time step $0$.

```c++
  std::vector<States*> __S;

  States *_s1 = new States(T);
  _s1->Zeros();
  int start_state = grid_.CoordToIndex(start_position_);
  _s1->Set(start_state);
  __S.push_back(_s1);

  States *_s2 = new States(T);
  _s2->Zeros();
  _s2->Set(cue2);
  __S.push_back(_s2);

  States *_s3 = new States(T);
  _s3->Zeros();
  _s3->Set(reward);
  __S.push_back(_s3);
```

## 4. Active Inference

We wrote a specific active inference procedure to provide for an exit state.

  ```c++
  void active_inference() override                                                                           
  {                                                                                                          
    unsigned int tt = 0;                                                                                     
                                                                                                             
    while (tt < this->T)                                                                                     
    {                                                                                                        
#ifdef PRINT                                                                                                 
      std::cout << "active_inference: tt=" << tt << std::endl;                                               
#endif                                                                                                       
                                                                                                             
      this->infer_states(tt);                                                                                
                                                                                                             
      /* value of policies (G) */                                                                            
      std::vector<Ty> G = this->infer_policies(tt);                                                          
                                                                                                             
      /* next action (the action that minimises expected free energy) */                                     
      int a = this->sample_action(tt);                                                                       
#ifdef PRINT                                                                                                 
      PrintAction(a);                                                                                        
#endif                                                                                                       
                                                                                                             
      /* sampling of next state (outcome) */                                                                 
      if (tt < this->T-1)                                                                                    
      {                                                                                                      
        /* next sampled state */                                                                             
        this->sample_state(tt+1, a);                                                                         
                                                                                                             
        /* next observed state */                                                                            
        this->sample_observation(tt+1, a);                                                                   
                                                                                                             
#ifdef PRINT                                                                                                 
        PrintState(this->_S[0]->Get(tt+1), grid);                                                            
        std::cout << "Reward: " << RewardString[this->_O[3]->Get(tt+1)] << std::endl;                        
#endif                                                                                                       
        if (grid(grid.GetCoord(this->_S[0]->Get(tt+1))) == 100 ||                                            
            grid(grid.GetCoord(this->_S[0]->Get(tt+1))) == -100)                                             
          break;                                                                                             
      }
      tt += 1;                                                                                               
    }                                                                                                        
  }    
  ```
to compile the [`main`](../../examples/main_epistemic_chaining.cpp) you can type:

`g++  -std=c++11 -Wall -O3 -D PRINT -D BEST_AS_MAX -o  epistemic_chaining  main_epistemic_chaining.cpp`

Executing the program we obtain the following output:
````
size_x=7
size_y=5
seed=0
T=11
cue2=3
reward=1
epistemic_chaining(7, 5)
Ns=[ 35 4 2 ]
Time taken by MDP constructor is : 0.000130363 sec
State:
# # # # # # # # #
# M . L1. . . . #
# . . . L2. S!. #
# C1. . . . . . #
# . . . L3. C!. #
# . . L4. . . . #
# # # # # # # # #
active_inference: tt=0

Action: South
State:
# # # # # # # # #
# . . L1. . . . #
# M . . L2. S!. #
# C1. . . . . . #
# . . . L3. C!. #
# . . L4. . . . #
# # # # # # # # #
Reward: Null
active_inference: tt=1

Action: South
State:
# # # # # # # # #
# . . L1. . . . #
# . . . L2. S!. #
# M . . . . . . #
# . . . L3. C!. #
# . . L4. . . . #
# # # # # # # # #
Reward: Null
active_inference: tt=2

Action: South
State:
# # # # # # # # #
# . . L1. . . . #
# . . . L2. S!. #
# C1. . . . . . #
# M . . L3. C!. #
# . . L4. . . . #
# # # # # # # # #
Reward: Null
active_inference: tt=3

Action: East
State:
# # # # # # # # #
# . . L1. . . . #
# . . . L2. S!. #
# C1. . . . . . #
# . M . L3. C!. #
# . . L4. . . . #
# # # # # # # # #
Reward: Null
active_inference: tt=4

Action: South
State:
# # # # # # # # #
# . . L1. . . . #
# . . . L2. S!. #
# C1. . . . . . #
# . . . L3. C!. #
# . M L4. . . . #
# # # # # # # # #
Reward: Null
active_inference: tt=5

Action: East
State:
# # # # # # # # #
# . . L1. . . . #
# . . . L2. S!. #
# C1. . . . . . #
# . . . L3. C!. #
# . . M . . . . #
# # # # # # # # #
Reward: Null
active_inference: tt=6

Action: East
State:
# # # # # # # # #
# . . L1. . . . #
# . . . L2. S!. #
# C1. . . . . . #
# . . . L3. C!. #
# . . L4M . . . #
# # # # # # # # #
Reward: Null
active_inference: tt=7

Action: East
State:
# # # # # # # # #
# . . L1. . . . #
# . . . L2. S!. #
# C1. . . . . . #
# . . . L3. C!. #
# . . L4. M . . #
# # # # # # # # #
Reward: Null
active_inference: tt=8

Action: East
State:
# # # # # # # # #
# . . L1. . . . #
# . . . L2. S!. #
# C1. . . . . . #
# . . . L3. C!. #
# . . L4. . M . #
# # # # # # # # #
Reward: Null
active_inference: tt=9

Action: North
State:
# # # # # # # # #
# . . L1. . . . #
# . . . L2. S!. #
# C1. . . . . . #
# . . . L3. M . #
# . . L4. . . . #
# # # # # # # # #
Reward: Cheese
Time taken by active inference is : 0.198082 sec
Total Time is : 0.19829 sec
=========
````
