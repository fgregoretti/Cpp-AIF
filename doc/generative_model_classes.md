# Generative model Classes

The generative model used by the Active Inference agent is defined by setting the hidden states **$\bf{S}$**,
the observations **$\bf{O}$**, the control states **$U$**, and the parameters **$\bf{\Theta} = \\{\bf{𝐀}, \bf{𝐁}, \bf{𝐂}, \bf{𝐃}\\}$**.

Hidden States and Observations are represented as instances of the [States](#-class-states) class.

**$\bf{𝐀}$** arrays are represented as istances of the [likelihood](#template-typename-t-typename-s-class-likelihood) class.

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

***Public methods:***
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

***Constructor:***
```c++
likelihood(decltype(Iseq)... size)
```
**Parameters**
- `decltype(Iseq)... size` integer tuple 

***Public methods:***
```c++
void setValue(T value, decltype(Iseq)... i)
```
Assign a value to a specific element (identified by an integer tuple) of the array.

**Parameters**
- `value` value to assign 
- `i` integer tuple 

```c++
void Zeros()
```
Set all array elements to **$0$**.

```c++
void Eye()
```
Create a two dimentional identity array.

```c++
void Norm()
```
Normalisation.

void sum(const likelihood<T,seq<Iseq...>>& b)
Sum of this object and likelihood object given as a parameter.
b likelihood instance to sum

const std::size_t get_order()
return order (rank)

const std::size_t get_firstdimension()
return size of first dimension

std::size_t *get_dimensions()
return array with sizes for each dimension

const std::size_t get_tnc()
return total number of components

int MaxIndex(const std::vector<std::size_t>& a) const
index of first dimension with maximum value in **$t(:,a[0],...,a[Nf-1])$**.
a

likelihood(std::vector<std::vector<T>> const &matrix)
constructor by passing a vector of vector (matrix).
matrix
  
likelihood(const std::array<std::size_t, sizeof...(Iseq)>& ia)
copy constructor by passing the index array
ia
  
likelihood AlogA()
return a new object obtained by multiplying each element of the array by the logarithm of itself
  
T **Dot(std::vector<int> sq, std::size_t f)
extract the array elements corresponding to the index tuple sq along dimension f
sq index tuples
dimension along which extract elements
