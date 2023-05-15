#ifndef UTIL_HPP
#define UTIL_HPP
#include <iostream>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <cmath>
#include <stdlib.h>
#include <algorithm>
#include "constants.h"
#define ABS(x) ( (x) > 0.0 ? x : -(x) )

/* softmax activation function */
template <typename T>
void softmax(std::vector<T>& expectation)
{
  T m = -INFINITY;
  for (std::size_t i = 0; i < expectation.size(); i++) {
    if (expectation[i] > m) {
      m = expectation[i];
    }
  }

  T sum = 0.0;
  for (std::size_t i = 0; i < expectation.size(); i++) {
    sum += exp(expectation[i] - m);
  }

  const T scale = m + log(sum);
  for (std::size_t i = 0;  i < expectation.size();  i++) {
    expectation[i] = exp(expectation[i] - scale);
  }
}

template <typename T>
void softmax(T *expectation, std::size_t size)
{
  T m = -INFINITY;
  for (std::size_t i = 0; i < size; i++) {
    if (expectation[i] > m) {
      m = expectation[i];
    }
  }

  T sum = 0.0;
  for (std::size_t i = 0; i < size; i++) {
    sum += exp(expectation[i] - m);
  }

  const T scale = m + log(sum);
  for (std::size_t i = 0;  i < size;  i++) {
    expectation[i] = exp(expectation[i] - scale);
  }
}

/* sample using cumulative distribution */
template <typename T>
int CDFs(std::vector<T> &p, const T rand1)
{
  int i = -1;

  std::partial_sum(p.begin(), p.end(), p.begin());

  for (std::size_t j = 0; j < p.size(); ++j)
    if (rand1 < p[j])
    {
      i = j;
      break;
    }

  if (i == -1)
  {
    std::cerr << "CDFs: no sample found:" << i << " rand=" << rand1 << " p = ";
    for (T val: p)
      std::cerr << val << " ";
    std::cerr << std::endl;

    if ( (ABS(1-p.back()) < maxError) && (rand1 <= 1) )
      return p.size()-1; 
    else 
      exit(-2);
  }

  return i;
}

/* sample using cumulative distribution */
template <typename T>
int CDFs(std::vector<T> &p)
{
  int i = -1;

  std::partial_sum(p.begin(), p.end(), p.begin());

  T rand1 = (T)rand() / (T)(RAND_MAX);
  for (std::size_t j = 0; j < p.size(); ++j)
    if (rand1 < p[j])
    {
      i = j;
      break;
    }

  if (i == -1)
  {
    std::cerr << "CDFs: no sample found:" << i << " rand=" << rand1 << " p = ";
    for (T val: p)
      std::cerr << val << " ";
    std::cerr << std::endl;

    if ( (ABS(1-p.back()) < maxError) && (rand1 <= 1) )
      return p.size()-1; 
    else 
      exit(-2);
  }

  return i;
}

/* return all the entries that match the maximum value */
template <typename T>
std::vector<int> findMaxima(std::vector<T> &p)
{
  std::vector<int> maxima;
  T maxValue = *std::max_element(p.begin(), p.end());

  for (std::size_t i = 0; i < p.size(); i++)
    if (p[i] == maxValue)
      maxima.push_back(i);

  return maxima;
}

template <typename T>
T opt_dot(unsigned int N, T *X, T *x)
{
  T dot = 0.0;
  T temp = 0.0;

  if (N == 0)
    return dot;

  unsigned int m = N % 5;

  if (m != 0)
  {
    for (unsigned int i = 0; i < m; i++)
      temp += X[i] * x[i];

    if (N < 5)
    {
      dot = temp;
      return dot;
    }
  }

  unsigned int ns = m;

  for (unsigned int i = ns; i < N; i+=5)
    temp += X[i] * x[i] + X[i+1] * x[i+1] + X[i+2] * x[i+2] +
            X[i+3] * x[i+3] + X[i+4] * x[i+4];

  dot = temp;

  return dot;
}

template <typename T>
T _log(T a)
{
  if (a > 0)
    return log(a);
  else
    return log0;
}

template <typename T>
T opt_hdot(unsigned int N, T *X, T *x)
{
  T h = 0.0;
  T _temp = 0.0;

  if (N == 0)
    return h;

  unsigned int m = N % 5;

  if (m != 0)
  {
    for (unsigned int i = 0; i < m; i++)
      _temp += X[i] * _log(X[i]) * x[i];

    if (N < 5)
    {
      h = _temp;
      return h;
    }
  }

  unsigned int ns = m;

  for (unsigned int i = ns; i < N; i+=5)
  {
    _temp += X[i] * _log(X[i]) * x[i] + X[i+1] * _log(X[i+1]) * x[i+1] +
            X[i+2] * _log(X[i+2]) * x[i+2] + X[i+3] * _log(X[i+3]) * x[i+3] +
            X[i+4] * _log(X[i+4]) * x[i+4];
  }

  h = _temp;

  return h;
}

template <typename T>
T opt_dot(unsigned int N, T *X, T *x, T *h)
{
  T dot = 0.0;
  T temp = 0.0;
  T _temp = 0.0;

  if (N == 0)
    return dot;

  unsigned int m = N % 5;

  if (m != 0)
  {
    for (unsigned int i = 0; i < m; i++)
    {
      temp += X[i] * x[i];
      _temp += X[i] * _log(X[i]) * x[i];
    }

    if (N < 5)
    {
      dot = temp;
      *h = _temp;
      return dot;
    }
  }

  unsigned int ns = m;

  for (unsigned int i = ns; i < N; i+=5)
  {
    temp += X[i] * x[i] + X[i+1] * x[i+1] + X[i+2] * x[i+2] +
            X[i+3] * x[i+3] + X[i+4] * x[i+4];
    _temp += X[i] * _log(X[i]) * x[i] + X[i+1] * _log(X[i+1]) * x[i+1] +
            X[i+2] * _log(X[i+2]) * x[i+2] + X[i+3] * _log(X[i+3]) * x[i+3] +
            X[i+4] * _log(X[i+4]) * x[i+4];
  }

  dot = temp;
  *h = _temp;

  return dot;
}

template <typename T>
T opt_dot(unsigned int N, T *X, T *_X, T *x, T *h)
{
  T dot = 0.0;
  T temp = 0.0;
  T _temp = 0.0;

  if (N == 0)
    return dot;

  unsigned int m = N % 5;

  if (m != 0)
  {
    for (unsigned int i = 0; i < m; i++)
    {
      temp += X[i] * x[i];
      _temp += _X[i] * x[i];
    }

    if (N < 5)
    {
      dot = temp;
      *h = _temp;
      return dot;
    }
  }

  unsigned int ns = m;

  for (unsigned int i = ns; i < N; i+=5)
  {
    temp += X[i] * x[i] + X[i+1] * x[i+1] + X[i+2] * x[i+2] +
            X[i+3] * x[i+3] + X[i+4] * x[i+4];
    _temp += _X[i] * x[i] + _X[i+1] * x[i+1] + _X[i+2] * x[i+2] +
             _X[i+3] * x[i+3] + _X[i+4] * x[i+4];
  }

  dot = temp;
  *h = _temp;

  return dot;
}
#endif
