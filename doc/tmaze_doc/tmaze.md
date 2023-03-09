# T-Maze

## 1. The problem
In this problem, we examine an agent traversing a T-maze with three arms,
with the agent's initial position being the central location.
One of the bottom arms of the maze includes a cue that provides information about the probable location
of a reward in either of the two top arms, referred to as 'Left' or 'Right'.

The environment is characterized by two distinct 'types' of states at each time step,
which are referred to as hidden state factors.

The first hidden state factor (location) is a discrete variable with values in $(0,1,2,3)$ that encodes the agent's present location.
The four vlaues correspond to the { "Center", "Left", "Right", "Bottom" } location.
For example, if the agent is in the CUE LOCATION, the current state of this factor would be $S^0 = \left\lbrack \matrix{ 0 & 0 & 0 & 1 } \right\rbrack$.

The second hidden state factor (reward condition) is a discrete variable with values in $(0,1)$ that econdes the reward condition of the trial: { "Reward on Left", "Reward on RIght" }. For example, a trial with the Reward on Left condition would be represented as the state $S^1 = \left\lbrack \matrix{ 1 & 0} \right\rbrack$.

These two hidden state factors are independent of one another. As an example, consider a Reward on Right trial and assume the agent begins in the center location. We can encode the state od the environment at the first time step using the following pair of hidden state vectors $S^0 = \left\lbrack \matrix{ 1 & 0 & 0 & 0\} \right\rbrack$ and $S^1 = \left\lbrack \matrix{ 1 & 0} \right\rbrack$. If the agent subsequently moves to the right arm, the hidden state vectors would change to $S^0 = \left\lbrack \matrix{ 0 & 1 & 0 & 0\} \right\rbrack$ and $S^1 = \left\lbrack \matrix{ 1 & 0} \right\rbrack$. This illustrates that the two hidden state factors are independent, meaning that the agent's location ($S^0$) can change without affecting the reward condition ($S^1$).
