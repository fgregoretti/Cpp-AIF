# custom array Classes

The generative model used by the Active Inference agent is defined by setting the hidden states **$\bf{S}$**,
the observations **$\bf{O}$**, the control states **$U$**, and the parameters **$\bf{\Theta} = \\{\bf{𝐀}, \bf{𝐁}, \bf{𝐂}, \bf{𝐃}\\}$**.

Hidden States and Observations are represented as instances of the [States](#class-states) class.

**$\bf{𝐀}$** arrays are represented as istances of the [likelihood](#template-typename-t-typename-s-class-likelihood) class.

**$\bf{B}$** arrays are represented as istances of the [Transitions](#template-typename-t-class-transitions) class.

**$\bf{C}$** arrays are represented as istances of the [Priors](#template-typename-t-class-priors) class.

**$\bf{D}$** arrays are represented as istances of the [Beliefs](#template-typename-ty-class-beliefs) class.

## `class States`
```c++
protected:
  unsigned int T;
private:
  unsigned int *id;
```
Integer array (of size T) class. 

***Constructor:***
```c++
States(unsigned int T_)
```
**Parameters**
- `T_` temporal horizon size of the integer array

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
Extract the array elements corresponding to the index tuple $sq$ along dimension $f$ and store them in a 2D array.

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

## `template <typename T> class Transitions`
```c++
protected:
  unsigned int Ns;
  unsigned int Nnz;
private:
  unsigned int *col;                                                              
  unsigned int *row_ptr;
  T *data;
```
Template transition probabilities matrix class: `T` is the template argument which is a placeholder for the data type used. Transition matrix has size of **$Ns$** by **$Ns$** with **$Nnz$** non-zero values and is stored in CSR format using three (one-dimensional) arrays (**$col$**, **$row\\_ptr$**, **$data$**). 
The arrays **$data$** and **$col$** are of length **$Nnz$**, and contain the non-zero values and the column indices of those values respectively. The array **$row\\_ptr$** is of length **$Ns+1$** and encodes the index in **$data$** and **$col$** where the given row starts.

***Constructor:***
```c++
Transitions(unsigned int Ns_, unsigned int Nnz_)
```

**Parameters**
- `Ns_` size of the matrix
- `Nnz_` number of non-zero values

***Public methods:***
```c++
void SetCol(unsigned int i, unsigned int j)
```
Assign a column index to a specific element of the array **$col$**.

**Parameters**
- `i` column index
- `j` **$col$** array index

```c++
void SetRowPtr(unsigned int p, unsigned int i)
```
Assign an index to a specific element of the array **$row\\_ptr$**.
**Parameters**
- `p` index to be assigned
- `i` **$row\\_ptr$** array index

```c++
void SetData(T value, unsigned int i)
```
Assign a non-zero value to a specific element of the array **$data$**.

**Parameters**
- `value` value to assign
- `i` **$data$** array index

```c++
void Eye()
```
Create a two dimentional identity array.

```c++
void Norm()
```
Normalisation.

```c++
unsigned int get_size()
```
Get matrix size.

```c++
unsigned int get_nnz()
```
Get numbers of non-zero values.

```c++
T *Txv(T *x)
```
Sparse matrix-vector multiplication. Return an array containing the product vector.

**Parameters**
- `x` array containing the vector to be multiplied by the `Transition` matrix

```c++
void Txv(T *x, T *y)
```
Sparse matrix-vector multiplication.

**Parameters**
- `x` array containing the vector to be multiplied by the `Transitions` matrix
- `y` array containing the product vector

```c++
void logTxv(T *x, std::vector<T> &y)
```
**$log$**(matrix)-vector multiplication

**Parameters**
- `x` array containing the vector to be multiplied by the **$log$**(`Transitions` matrix)
- `y` vector containing the product vector

```c++
void extract_column(unsigned int f, std::vector<T> &s)
```
Store the **$f-th$** column in vector **$s$**.

**Parameters**
- `f` column index of the column to store in **$s$**
- `s` stores the extracted column

```c++
int MaxIndex(unsigned int f)
```
Return the index corresponding to the maximum value in the **$f-th$** column

**Parameters**
- `f` column index where the maximum value has to be found

```c++
Transitions(std::vector<std::vector<T>> const &matrix)
```
Constructor by passing a vector of vector (matrix).

**Parameters**
- `matrix` matrix used to construct a new `Transitions` instance.

```c++
Transitions(const Transitions<T> &t)
```
Copy constructor by passing an instance of `Transitions`.

**Parameters**
- `t` instance of `Transitions`

```c++
void csc_tocsr(unsigned int col_ptr[], unsigned int row[])
```
Convert matrix stored in CSC format to CSR.

**Parameters**
- `col_ptr` array storing column pointers of CSC format
- `row` array storing row indices of CSC format

## `template <typename T> class Priors`
```c++
protected:
  unsigned int Ns;
private:
  T *value;
```
Template priors array class: `T` is the template argument which is a placeholder for the data type used. **$Ns$** is the size of the array stored in `value`.

***Constructor:***
```c++
Priors(unsigned int Ns_)
```

**Parameters**
- `Ns_` size of the array

***Public methods:***
```c++
void setValue(T val, unsigned int i)
```
Assign a value to a specific element of the array.

**Parameters**
- `val` value to assign
- `i` element of the array where to assign the value

```c++
T getValue(unsigned int i)
```
Return the **$i-th$** array element.

**Parameters**
- `i` array element to be retrieved

```c++
void Zeros()
```
Set all array elements to **$0$**.

```c++
void NormLog()
```
Logarithmic transformation (after normalisation).

```c++
unsigned int get_size()
```
Return size of the array.

```c++
Priors(std::vector<T> const &v)
```
Constructor by passing a vector.

**Parameters**
- `v` vector used to construct a new `Priors` instance.

```c++
Priors(const Priors<T> &p)
```
Copy constructor by passing an instance of `Priors`.

**Parameters**
- `p` instance of `Priors`

## `template <typename Ty> class Beliefs`
```c++
protected:
  unsigned int Ns;
  unsigned int T;
private:
  Ty *value;
```
Beliefs array (Ns by T) class: `Ty` is the template argument which is a placeholder for the data type used. **$Ns \times T$** is the size of the array stored in `value`.

***Constructor:***
```c++
Beliefs(unsigned int Ns_, unsigned int T_ = 1)
```

**Parameters**
- `Ns_` size of the array for each time step
- `T_` temporal horizon 

***Public methods:***
```c++
void setValue(Ty val, unsigned int i)
```
Assign a value to a specific element of the array at time step **$0$**.

**Parameters**
- `val` value to assign
- `i` element of the array where to assign the value

```c++
void setValue(Ty val, unsigned int i, unsigned int t_)
```
Assign a value to a specific element of the array at time step **$t$**.

**Parameters**
- `val` value to assign
- `i` element of the array where to assign the value
- `t_` time step **$t$**

```c++
Ty getValue(unsigned int i)
```
Return the **$i-th$** array element at time step **$0$**.

**Parameters**
- `i` array element to be retrieved

```c++
Ty getValue(unsigned int i, unsigned int t_)
```
Return the **$i-th$** array element at time step **$t$**.

**Parameters**
- `i` array element to be retrieved
- `t_` time step **$t$**

```c++
Ty *getArray(unsigned int t_)
```
Return first pointer to array at time step **$t$**.

**Parameters**
- `t_` time step **$t$**

```c++
void Zeros()
```
Set all array elements to **$0$**.

```c++
void Ones()
```
Set all array elements to **$1$**.

```c++
void Norm()
```
Normalisation.

```c++
void Log()
```
Logarithmic transformation.

```c++
void NormLog()
```
Logarithmic transformation (after normalisation).

```c++
unsigned int get_size()
```
Return size of the array at each time step.

```c++
Beliefs(std::vector<Ty> D)
```
Constructor by passing a vector.

**Parameters**
- `D` vector used to construct a new `Beliefs` instance.

```c++
Beliefs(const Beliefs<Ty> &b)
```
Copy constructor by passing an instance of `Beliefs`.

**Parameters**
- `b` instance of `Beliefs`
