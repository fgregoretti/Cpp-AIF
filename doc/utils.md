# Utility functions

## `softmax`

```c++
template <typename T> void softmax(std::vector<T>& expectation)
```

**Parameters**
- `expectation` numeric array

```c++
template <typename T> void softmax(T *expectation, std::size_t size)  
```
**Parameters**
- `expectation` numeric array
- `size` array size

template softmax (e.g., neural transfer) function: T is the template argument which is a placeholder for the data type. The `softmax` function, also known as softargmax or normalized exponential function, extends the logistic function to multiple dimensions. It is a widely used tool in multinomial logistic regression and is the final activation function in neural networks to normalize the network's output and predict probability distributions over output vectors. It can be applied after activation function outputs to normalize the resulting vector or array.

This normalization ensures that the `softmax` output values, between 0 and 1, represent essential values from the given output vector or array. By applying the standard exponential function to each activation output element and dividing by the sum of all exponentials, `softmax` guarantees that the components of the output vector add up to 1.

For a $x$ vector (or array) which has $n$ members a `softmax` for each member can be written as follows

$$ \text{softmax}(x_i) = \frac{e^{x_i}}{ \sum_{j=1}^n e^{x_j}} $$

To prevent the possibility of the function overflowing due to infinite results, we can modulate $x$ values by subtracting the maximum value, denoted as $m$.

$$ \text{softmax}(x_i) = \frac{e^{x_i-m}}{ \sum_{j=1}^n e^{x_j-m}} $$

## Sampling
We may need to draw samples from categorical probability distributions. For instance, this could be necessary when sampling observations from the generative process or sampling actions from the agent's posterior probability distribution over policies.

We have one function capable of handling the sampling from a single probability distribution.

```c++
template <typename T> int CDFs(std::vector<T> &p, const T rand1)
```
template sampling using cumulative distribution: T is the template argument which is a placeholder for the data type.

**Parameters**
- `p` probability distribution
- `rand1` random number in the interval $[0, 1)$.
