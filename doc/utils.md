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

template softmax activation function: T is the template argument which is a placeholder for the data type. The softmax Function, also known as softargmax or normalized exponential function, extends the logistic function to multiple dimensions. It is a widely used tool in multinomial logistic regression and as the final activation function in neural networks to normalize the network's output and predict probability distributions over output vectors. It can be applied after activation function outputs to normalize the resulting vector or array.

This normalization ensures that the softmax output values, between 0 and 1, represent essential values from the given output vector or array. By applying the standard exponential function to each activation output element and dividing by the sum of all exponentials, softmax guarantees that the components of the output vector add up to 1.

For a x vector (or array) which has $n$ members a Softmax for each member can be written as follows

$$ \text{softmax}(x_i) = \frac{e^{x_i}}{ \sum_{j=1}^n e^{x_j}} $$

To prevent the possibility of the function overflowing due to infinite results, we can modulate $x$ values by subtracting the maximum value, denoted as $m$.

$$ \text{softmax}(x_i) = \frac{e^{x_i-m}}{ \sum_{j=1}^n e^{x_j-m}} $$
