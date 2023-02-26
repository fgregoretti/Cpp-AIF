# Generative model Classes

The generative model used by the Active Inference agent is defined by setting the hidden states **$\bf{S}$**,
the observations **$\bf{O}$**, the control states **$U$**, and the parameters **$\bf{\Theta} = \\{\bf{𝐀}, \bf{𝐁}, \bf{𝐂}, \bf{𝐃}\\}$**.

Hidden States and Observations are represented as instances of the [States](#-class-states) class.
A arrays are represented as istances of the [likelihood](#-class-likelihood) class.

## `class States`
```c++
protected:
  unsigned int T;
private:
  unsigned int *id;
```
integer array (of size T) class 

***Constructor:***
```c++
States(unsigned int T_)
```
**Parameters**
- `T_` size of the integer array

***Public class methods:***
```c++
void Zeros()
```
Set all array elements to **$0$**.

```c++
Set(unsigned int val, unsigned int t)
```
Assign a value to a specific element of the array.

**Parameters**
- `val` value to assign
- `t` element of the array where to assign the value

```c++
Set(unsigned int val)
```
Assign a value to first element of the array.

**Parameters**
- `val` value to assign

```c++
Get(unsigned int t)
```
Retrieve value of a specific element of the array.

**Parameters**
- `t` element of the array whose value we are looking for

```c++
Get()
```
Retrive first element array value.


## `template <typename T, typename S> class likelihood`
```c++
private:
  T *t;
  const std::array<std::size_t, sizeof...(Iseq)> s;
```
template likelihood multidimensional array class: `T` is the template argument which is a placeholder for the data type used while `S` is the number of dimensions.
`s` is the index array so that **$|s|$** is the number of indices (rank) and **$s[i], i=0,...,N$** is the size of each dimension; `t` is the vector containing the **$s[0] \times ... \times s[N]$** components of the multidimensional array. Specifically, **$s[0]$** is the number of observations; **$s[1],...,s[N]$** the number of state factors.
