# Epistemic chaining

## 1. The problem

In this problem a rat is tasked to solve a spatial puzzle: must visit two cues located at different positions within a 2D grid world, in a specific order, to uncover the locations of two reward outcomes - one will give the rat a positive reward (the "Cheese"), and the other will give a negative reward (the "Shock"). 

Cue 1 is located in one location of the grid world, while there are four additional locations that could potentially contain Cue 2. However, only one of these four locations actually has Cue 2, while the other three are empty. Once the agent reaches Cue 1, it will receive one of four unambiguous signals (L1, L2, L3, L4), which indicate the exact location of Cue 2. After discovering the location of Cue 2, the agent can visit that location to receive one of two possible signals, which reveal the location of the reward or punishment. The reward and punishment are located in two distinct positions: "First" reward position, or "Second" reward position.

The optimal strategy to maximize reward while minimizing risk in this task involves the following approach: first, the agent needs to visit Cue 1 to obtain the signal that reveals the location of Cue 2. Once the location of Cue 2 is determined, the agent can then visit that location to receive the signal that indicates the location of the reward or punishment.

## 2. The generative model

The hidden states are factorized into three factors **$S^0, S^1$**, and **$S^2$**

1. Agent location: **$S^0$** encodes the agent's location in the grid world with as many elements as there are the grid locations. Therefore it has cardinality **$35$** and the tuples of **$(x, y)$** coordinate locations are mapped to linear indices by using **$y*dim_x+x$**.

2. Cue2 location: it has cardinality **$4$** encoding in which of the four possible location Cue 2 is actually located.

3. Reward location: it has cardinality **$2$** encoding which of the two reward positions ("First" or "Second") the "Cheese" has to be found in.
