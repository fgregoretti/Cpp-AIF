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
The four vlaues correspond to the $( "Center", "Left", "Right", "Bottom" )$ location.
For example, if the agent is in the CUE LOCATION, the current state of this factor would be $S^0 = \left\lbrack \matrix{ 0 & 0 & 0 & 1 } \right\rbrack$.

The second hidden state factor (reward condition) is a discrete variable with values in $(0,1)$ that econdes the reward condition of the trial: $( "Reward on Left", "Reward on RIght" )$. For example, a trial with the Reward on Left condition would be represented as the state $S^1 = \left\lbrack \matrix{ 1 & 0} \right\rbrack$.

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

## 5. The generative model

Let's move forward with setting up the generative model of the agent, which involves the agent's beliefs or assumptions regarding how hidden states generate observations and transition among themselves.

In most Markov Decision Processes (MDPs), the essential components of this generative model are the agent's representation of the observation likelihood and its representation of the transition distribution.

Assuming that the agent has an accurate representation of the rules governing the T-maze, including how hidden states lead to observations, and its ability to control its movements with predictable consequences (i.e. 'noiseless' transitions), the agent will possess a true representation of the environment's "rules," encoded in the arrays **$\bf{A}$** and **$\bf{B}$** of the generative process.

Let’s encode encode the agent's initial beliefs regarding its starting location and reward condition in the prior over hidden states, which is referred to as the **$\bf{D}$** array.

We have to define two arrays $D^0$ and $D^1$, each corresponding to a specific hidden state factor. We will ensure that the agent begins with precise and accurate prior beliefs about its starting location.

```c++
std::vector<FLOAT_TYPE> D0 = {1., 0., 0., 0.};
std::vector<FLOAT_TYPE> D1 = {1./2, 1./2};
```
We create the initial beliefs defining a vector of objects `Beliefs`. Specifically a vector with size $N_f=2$, and each element will contain an object `Beliefs` with size $4$ and $2$ respectively.

```c++
  std::vector<Beliefs<FLOAT_TYPE>*> __D;                                                                     
  Beliefs<FLOAT_TYPE> *d0 = new Beliefs<FLOAT_TYPE>(D0);                                                     
  __D.push_back(d0);                                                                                         
  Beliefs<FLOAT_TYPE> *d1 = new Beliefs<FLOAT_TYPE>(D1);                                                     
  __D.push_back(d1);
```

To ensure that the agent is motivated to choose the arm that maximizes the probability of receiving a reward, we need to give the agent a sense of reward and loss. We can achieve this by setting up the **$\bf{C}$** array, which represents the agent's prior preferences for each observation facor. We initialize the $C^0$ array to all 1s, indicating that the agent has no preference for any particular outcomes. Instead, since the second factor is the Reward modality, with the Reward outcome having an index of 2 and the No Reward outcome having an index of 3, we can assign values to the corresponding entries that reflect the relative preference for one outcome over the other. Specifically, we use relative log-probabilities to encode these preferences.

```c++
  std::vector<FLOAT_TYPE> C0 = {1., 1., 1., 1.};                                                             
  softmax<FLOAT_TYPE>(C0)
  const FLOAT_TYPE c = 2.;
  std::vector<FLOAT_TYPE> C1 = {0., 0., c, -c};                                                              
  softmax<FLOAT_TYPE>(C1);
```

The ability to modify the agent's prior beliefs and bias it towards observing the Reward outcome more often than the No Reward outcome is what gives the Reward modality its intrinsic value. Without this bias, the Reward modality would be no different from any other arbitrary observation.

## 6. Policies

We can either create an empty vector of policies and in this case the constructor will generate the policies, or we can build our own vector of policies, for example:

```c++
  std::vector<std::vector<int>> V {                                                                        
    { 0,  0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3 },                                     
    { 0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3 },                                     
    { 0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3 }                                      
  }
```

V is a vector of vectors of  `int` with size **$(num\\_timesteps, num\\_policies)$** where **$num\\_timesteps$** is the temporal depth of the policy and **$num\\_policies$** is the number of policies.

## 7. Introducing the `MDP` class

Within `Cpp-AcI`, we have abstracted many of the computations necessary for active inference into the `MDP` class. This flexible object can be used to store essential elements of the generative model, the agent's current observations and actions, and execute action/perception through functions like `infer_states` and `infer_policies`.

To create an instance of the `MDP`, simply call the `MDP` constructor with a list of arguments.

```c++
int seed = 0;
unsigned int T = 3;
MDP<FLOAT_TYPE,3> *mdp = new MDP<FLOAT_TYPE,3>(__D,__S,__B,__A,__C,V,T,64,4,1./4,1,4,1,seed);
```

## 8. Active Inference

We can use the basic active inference  procedure implemented as `MDP` class method

```c++
template <typename Ty, std::size_t M>
void MDP<Ty,M>::active_inference()
{
  unsigned int tt = 0;
  while (tt < T)
  {
    infer_states(tt);

    /* value of policies (G) */
    std::vector<Ty> G = infer_policies(tt);

    /* next action (the action that minimises expected free energy) */
    int a = sample_action(tt);

    /* sampling of next state (outcome) */
    if (tt < T-1)
    {
      /* next sampled state */
      sample_state(tt+1, a);

      /* next observed state */
      sample_observation(tt+1, a);
    }

    tt += 1;
  }
}
```

calling it in the main like this:

```c++
mdp->active_inference();
```

Executing the program we obtain the following output:

T=1 Location: [Center] Observation: [Cue Left] Action: [Move to Bottom]

T=2 Location: [Bottom] Observation: [Cue Left] Action: [Move to Left]

T=3 Location: [Left] Observation: [Reward!] Action: [Move to Bottom]
