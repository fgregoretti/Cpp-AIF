# T-Maze

## 1. The problem
In this problem, we examine an agent traversing a T-maze with three arms,
with the agent's initial position being the central location.
One of the bottom arms of the maze includes a cue that provides information about the probable location
of a reward in either of the two top arms, referred to as 'Left' or 'Right'.

## 2. The environment
The environment is described by the joint occurence two distinct 'types' of states at each time step,
which are referred to as hidden state factors.

The first hidden state factor (location) is a discrete variable with values in $(0,1,2,3)$ that encodes the agent's present location.
The four vlaues correspond to the { "Center", "Left", "Right", "Bottom" } location.
For example, if the agent is in the CUE LOCATION, the current state of this factor would be $S^0 = \left\lbrack \matrix{ 0 & 0 & 0 & 1 } \right\rbrack$.

The second hidden state factor (reward condition) is a discrete variable with values in $(0,1)$ that econdes the reward condition of the trial: { "Reward on Left", "Reward on RIght" }. For example, a trial with the Reward on Left condition would be represented as the state $S^1 = \left\lbrack \matrix{ 1 & 0} \right\rbrack$.

These two hidden state factors are independent of one another. As an example, consider a Reward on Right trial and assume the agent begins in the center location. We can encode the state of the environment at the first time step using the following pair of hidden state vectors $S^0 = \left\lbrack \matrix{ 1 & 0 & 0 & 0\} \right\rbrack$ and $S^1 = \left\lbrack \matrix{ 1 & 0} \right\rbrack$. If the agent subsequently moves to the right arm, the hidden state vectors would change to $S^0 = \left\lbrack \matrix{ 0 & 1 & 0 & 0\} \right\rbrack$ and $S^1 = \left\lbrack \matrix{ 1 & 0} \right\rbrack$. This illustrates that the two hidden state factors are independent, meaning that the agent's location ($S^0$) can change without affecting the reward condition ($S^1$).

## 3. The outcome mapping

By examining the probability distributions that map from hidden states to observations, we can gain insight into the rules encoded by the environment, also known as the generative process. We refer to this collection of probabilistic relationships as the **$\bf{A}$** array.

In this T-maze demonstration, the agent's observations consist of two sensory channels or observation factors: Cue (exteroceptive) and Reward (interoceptive).

The exteroceptive outcomes provide cues about location and context (the reward condition of the trial) $\left\lbrack \matrix{ Cue Center & Cue Left & Cue Right & Cue Bottom\} \right\rbrack$.

The interoceptive outcomes denotes different levels of reward $\left\lbrack \matrix{ Cue Left & Cue Right & Reward & No Reward} \right\rbrack$.
When the agent occupies the "Bottom" location, this observation unambiguously signals the reward condition of the trial, and therefore in which arm the Reward observation is more probable.
When the agent occupies the "Center", the Cue observation will be Cue Right or Cue Left with equal probability.
The Reward (index 2) and No Reward (index 3) observations are observed in the right and left arms of the T-maze, with associated probabilities $a$ and $b$. The variables $a$ and $b$ represent the probabilities of obtaining a reward or a loss when choosing the "correct" arm, and the probabilities of obtaining a loss or a reward when choosing the "incorrect" arm. The definition of which arm is considered "correct" or "incorrect" depends on the reward condition, which is determined by the state of the second hidden state factor.

In `Cpp-AcI`, we use a set of `likelihood` class instances to store the set of probability distributions that encode the conditional probabilities of observations under different configurations of hidden states. Each factor-specific **$\bf{A}$** array is stored as a multidimensional array with $N_o[m]$ rows and as many lagging dimensions as there are hidden state factors. $N_o[m]$ refers to the number of observation values for observation factor $m$, i.e. **$\bf{N_o} = [4, 4]$**.

The multidimensional arrays **$A^0$** and **$A^1$** are both 3-dimensional arrays with dimensions **$4 \times 4 \times 2$** and are the same for each action **$u$**.

We create the observation model defining a vector of vector of objects `likelihood`. Specifically a vector with size **$2$**, and each element will contain a vector of one object `likelihood` with size **$4 \times 4 \times 2$**.

```c++
  std::vector<std::vector<likelihood<FLOAT_TYPE,3>*>> __A;

  std::vector<likelihood<FLOAT_TYPE,3>*> _a0;
  likelihood<FLOAT_TYPE,3> __a0(4,4,2);

  std::vector<likelihood<FLOAT_TYPE,3>*> _a1;
  likelihood<FLOAT_TYPE,3> __a1(4,4,2);
```

We then fill out **$A^0$** and **$A^1$** accordingly.

```c++
  __a0.Zeros();                                                                                              
  /* cue start cue left cue right cue down */                                                                
  __a0(0,0,0)=1; __a0(1,1,0)=1; __a0(2,2,0)=1; __a0(3,3,0)=1;                                                
  /* cue start cue left cue right cue down */                                                                
  __a0(0,0,1)=1; __a0(1,1,1)=1; __a0(2,2,1)=1; __a0(3,3,1)=1;                                                
  _a0.push_back(&__a0);                                                                                      
  __A.push_back(_a0);                                                                                        
                                                                                                             
  const FLOAT_TYPE a = .9;                                                                                   
  const FLOAT_TYPE b = 1.-a;                                                                                 
                                                                                                             
  const FLOAT_TYPE d = 1.;                                                                                   
  const FLOAT_TYPE e = 1.-d;                                                                                 
                                                                                                             
  __a1.Zeros();                                                                                              
  /* CS left CS right reward positive reward negative */                                                     
  __a1(0,0,0)=0.5; __a1(0,3,0)=d; __a1(1,0,0)=0.5; __a1(1,3,0)=e;                                            
  __a1(2,1,0)=a;   __a1(2,2,0)=b; __a1(3,1,0)=b;   __a1(3,2,0)=a;                                            
  /* CS left CS right reward positive reward negative */                                                     
  __a2(0,0,1)=0.5; __a2(0,3,1)=e; __a2(1,0,1)=0.5; __a2(1,3,1)=d;                                            
  __a2(2,1,1)=b;   __a2(2,2,1)=a; __a2(3,1,1)=a;   __a2(3,2,1)=b;                                            
  _a2.push_back(&__a2);                                                                                      
  __A.push_back(_a2); 
```

## 4. Transition distribution

We represent the dynamics of the environment (e.g. changes in the location of the agent and changes to the reward condition) as conditional probability distributions. These distributions encode the likelihood of transitions between the states of a given hidden state factor being grouped together into the **$\bf{B}$** array, which is also known as transition distribution. Each matrix **$B^f$** represents the transition probabilities between state-values of a specific hidden state factor with index $f$. These matrices reflect Markovian transition probabilities that encode dynamics, such that the entry $i,j$ in a particular matrix indicates the probability of transitioning to state $i$ at time $t+1$, given that the system was in state $j$ at time $t$.

It is crucial to note that certain hidden state factors can be controlled by the agent. This means that the probability of being in state i at time t+1 is not solely determined by the state at time t, but also by the actions taken (or control states) from the agent's perspective. Consequently, each transition likelihood now incorporates conditional probability distributions over states at time $t+1$, where the conditioning variables comprise both the states at time $t-1$ and the actions at time $t-1$.

For instance, in our scenario, the first hidden state factor (Location) is within the agent's control. Therefore, the corresponding transition distribution **$B^0_u$** can be accessed using both the previous state and action indices.

Being **$U = [ Move to Center, Move to Left, Move to Right, Move to Bottom ]$** (i.e. there are four actions taking the agent directly to each of the four locations), we can create the four transition distribution matrix as follows:

```c++
  std::vector<std::vector<FLOAT_TYPE>> B0_0 {
              { 1, 0, 0, 1 },
              { 0, 1, 0, 0 },
              { 0, 0, 1, 0 },
              { 0, 0, 0, 0 }
          }

  std::vector<std::vector<FLOAT_TYPE>> B0_1 {
              { 0, 0, 0, 0 },
              { 1, 1, 0, 1 },
              { 0, 0, 1, 0 },
              { 0, 0, 0, 0 },
          };

  std::vector<std::vector<FLOAT_TYPE>> B0_2 {
              { 0, 0, 0, 0 },
              { 0, 1, 0, 0 },
              { 1, 0, 1, 1 },
              { 0, 0, 0, 0 },
          };
          
  std::vector<std::vector<FLOAT_TYPE>> B0_3 {
              { 0, 0, 0, 0 },
              { 0, 1, 0, 0 },
              { 0, 0, 1, 0 },
              { 1, 0, 0, 1 },
          };
```

The transition array for the reward condition factor is a "trivial" identity matrix. This implies that the reward condition remains unchanged over time, as it is mapped from its current value to the same value in the next time step.

To account for the conditioning on factors and the conditioning on actions, we represent **$\bf{B}$** as a vector of size **$N_f$** whose each element will contain a vector of **$N_u$** (or **$1$** if the factor is uncontrollable) of `Transitions` class instances. **$N_f=2** is the number of hidden state factors, while **$N_u=4$** is the number of control states. This `Transitions` class handles a transition distribution matrix and in order to build its own instance we can use the costructor with vector of vectors as parameter and pass it the 2D matrices previously created.

```c++
  std::vector<std::vector<Transitions<FLOAT_TYPE>*>> __B;

  std::vector<Transitions<FLOAT_TYPE>*> _b0;
  
  Transitions<FLOAT_TYPE> *__b0 = new Transitions<FLOAT_TYPE>(B0_0);                                         
  _b0.push_back(__b0)
  Transitions<FLOAT_TYPE> *__b1 = new Transitions<FLOAT_TYPE>(B0_1);                                         
  _b0.push_back(__b1);
  Transitions<FLOAT_TYPE> *__b2 = new Transitions<FLOAT_TYPE>(B0_2);                                         
  _b0.push_back(__b2);
  Transitions<FLOAT_TYPE> *__b3 = new Transitions<FLOAT_TYPE>(B0_3);                                         
  _b0.push_back(__b3)
  
  std::vector<Transitions<FLOAT_TYPE>*> _b1;                                                                 
                                                                                                             
  std::vector<std::vector<FLOAT_TYPE>> eye {                                                                 
              { 1., 0. },                                                                                    
              { 0., 1. }                                                                                     
          };
  Transitions<FLOAT_TYPE> *__b = new Transitions<FLOAT_TYPE>(eye);                                         
  _b1.push_back(__b);
  
  __B.push_back(_b0);                                                                                        
  __B.push_back(_b1);
```
