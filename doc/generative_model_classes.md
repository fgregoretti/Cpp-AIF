# Generative model Classes

The generative model used by the Active Inference agent is defined by setting the hidden states **$\bf{S}$**,
the observations **$\bf{O}$**, the control states **$U$**, and the parameters **$\bf{\Theta} = \\{\bf{𝐀}, \bf{𝐁}, \bf{𝐂}, \bf{𝐃}\\}$**.

Hidden States and Observations are represented as instances of the [States](#class-states) class.

**$\bf{𝐀}$** arrays are represented as istances of the [likelihood](#template-typename-t-typename-s-class-likelihood) class.

**$\bf{B}$** arrays are represented as istances of the [Transitions](#class-transitions) class.

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
void Set(unsigned int val, unsigned int t)
```
Assign a value to a specific element of the array.

**Parameters**
- `val` value to assign
- `t` element of the array where to assign the value

```c++
void Set(unsigned int val)
```
Assign a value to first element of the array.

**Parameters**
- `val` value to assign

```c++
unsigned int Get(unsigned int t)
```
Retrieve value of a specific element of the array.

**Parameters**
- `t` element of the array whose value we are looking for

```c++
unsigned int Get()
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
- `size` integer sequence of the size in ecah dimension 

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

```c++
void sum(const likelihood<T,seq<Iseq...>>& b)
```
Sum of this object and likelihood object given as a parameter.

**Parameters**
- `b` likelihood instance to sum

```c++
const std::size_t get_order()
```
Return order (rank).

```c++
const std::size_t get_firstdimension()
```
Return size of first dimension.

```c++
std::size_t *get_dimensions()
```
Return array with sizes for each dimension.

```c++
const std::size_t get_tnc()
```
Return total number of elements.

```c++
int MaxIndex(const std::vector<std::size_t>& a) const
```
Return the index of first dimension with maximum value in **$t(:,a[0],...,a[N_f-1])$**.

**Parameters**
- `a` index tuple 

```c++
likelihood(std::vector<std::vector<T>> const &matrix)
```
Constructor by passing a vector of vector (matrix).

**Parameters**
- `matrix` matrix used to construct a new `likelihood` instance.
  
```c++
likelihood(const std::array<std::size_t, sizeof...(Iseq)>& ia)
```
Copy constructor by passing the index array.

**Parameters**
- `ia` index array
  
```c++
likelihood AlogA()
```
Return a new object obtained by multiplying each element of the array by the logarithm of itself.

```c++
T **Dot(std::vector<int> sq, std::size_t f)
```
Extract the array elements corresponding to the index tuple sq along dimension f and store them in a 2D array.

**Parameters**
- `sq` index tuple
- `f` dimension along which extract elements

```c++
T *HDot(T **xt, likelihood& l, T *H)
```
Multidimensional dot (inner) producT: compute the inner product obtained by summing the products of the likelihood and the vectors **$xt[i], i=0,...,N_f-1$**, along leading dimension of the likelihood and the epistemic value. Return an array.

**Parameters**
- `xt` array of vectors
- `l` likelihood with the products of the likelihood elements by the logarithm of themselves
- `H` epistemic value

```c++
T *HDot(T **xt, T *H)
```
Multidimensional dot (inner) producT: compute the inner product obtained by summing the products of the likelihood and the vectors **$xt[i], i=0,...,N_f-1$**, along leading dimension of the likelihood and the epistemic value. Return an array.

**Parameters**
- `xt` array of vectors
- `H` epistemic value

```c++
void find(std::vector<int> sq, std::vector<T> &p)
```
Find the likelihood elements **$t(:,sq[0],...,sq[N_f-1])$** and store them in the vector **$p$**.

**Parameters**
- `sq` index tuple
- `p` output vector

