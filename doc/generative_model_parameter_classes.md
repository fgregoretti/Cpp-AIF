The generative model used by the Active Inference agent is defined by setting the hidden states **$\bf{S}$**,
the observations **$\bf{O}$**, the control states **$U$**, and the parameters **$\bf{\Theta} = \\{\bf{𝐀}, \bf{𝐁}, \bf{𝐂}, \bf{𝐃}\\}$**.

- [States](#-class-states)


## class States

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
