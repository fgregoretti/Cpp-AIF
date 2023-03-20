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

template softmax activation function: T is the template argument which is a placeholder for the data type.

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{k} e^{x_j}}
$$

To avoid overflow due to infinitive results we can modulate x values by subtracting maximum value m.

$$
\text{softmax}(x_i) = \frac{e^{x_i-m}}{\sum_{j=1}^{k} e^{x_j-m}}
$$
