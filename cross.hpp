#ifndef CROSS_HPP
#define CROSS_HPP
#include <iostream>
#include <cstdlib>
#include <vector>
#include <stdlib.h>
#include <numeric>
#include <array>
#include <cstring>

template <typename T, std::size_t N>
T *cross(T **arr, std::array<std::size_t, N> s)
{
  /* number of arrays */
  std::size_t n = s.size()-1;

  std::size_t m = s[1];
  for (std::size_t i = 2; i <= n; i++)
    m *= s[i];
  T *y = new T[m];
  memset(y, 0, m*sizeof(T));

  if (n == 1)
  {
    for (std::size_t i = 0; i < m; i++)
      y[i] = arr[0][i]; 
    return y;
  }

  std::size_t m1 = s[1];
 
#ifdef _OPENMP                                                                                                    
  #pragma omp parallel for                                                                                    
#endif
  for (std::size_t j = 0; j < m1; j++)
  {
    std::size_t m2 = m/m1;

    /* to keep track of next element in each of
       the n-1 arrays */
    std::size_t *indices = new std::size_t[n];
    indices[0] = j;
    for (std::size_t i = 1; i < n; i++)
      indices[i] = 0;

    for (std::size_t count = 0; count < m2; count++) {
      /* compute current product */
      T product = arr[0][indices[0]];

      for (std::size_t i = 1; i < n; i++)
        product *= arr[i][indices[i]];

      y[j*m2+count] = product;

      /* find the rightmost array that has more
         elements left after the current element
         in that array */
      int next = n - 1;
      while (next >= 1 &&
            (indices[next] + 1 >= s[next+1]))
        next--;
 
      if (next >=1) {
        /* if found move to next element in that
           array */
        indices[next]++;
 
        /* for all arrays to the right of this
           array current index again points to
           first element */
        for (std::size_t i = next + 1; i < n; i++)
          indices[i] = 0;
      }
    }

    delete [] indices;
  }

  return y;
}
#endif
