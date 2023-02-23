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

The control states **$\bf{U}$** encode the actions of the agent. In this 2D grid world the agent have the ability to make movements in the **$4$** cardinal directions (NORTH, EAST, SOUTH, WEST)

### The transition model: a derived Transition class
The control states **$\bf{U}$** determine the transitions from one state to another for the first hidden state factor. We need to write a derived Transition class that add a specialized method to fill out **$B[0]$** according to the expected outcomes of the **$4$** actions. 
