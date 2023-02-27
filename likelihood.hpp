#ifndef LIKELIHOOD_HPP
#define LIKELIHOOD_HPP
#include <iostream>
#include <vector>
#include <utility>
#include <array>
#include <cstddef>
#include <functional>
#include <numeric>
#include <assert.h>
#include <algorithm>
#include <cmath>
#include <ctime>
#include "common.h"
#include "cross.hpp"
#include "util.hpp"
#include "constants.h"
#ifdef _OPENMP
#include "omp.h"
#endif

template <std::size_t...> struct seq {};
template <std::size_t N, std::size_t... Iseq> struct gen_seq : gen_seq<N-1, N-1, Iseq...> {};
template <std::size_t... Iseq> struct gen_seq<0, Iseq...> { using type = seq<Iseq...>; };

namespace detail
{
  /* template likelihood multidimensional array class:
     s is the index array so that |s| is the number of
     indices (rank) and s[i], i=0,...,N is the size
     of each dimension;
     t is the vector containing the s[0] x ... x s[N] 
     components of the multidimensional array
     s[0] is the number of observations;
     s[1],...,s[N] the number of state factors */
  template <typename T, typename S>
  class likelihood;
 
  template <typename T, std::size_t... Iseq>
  class likelihood<T, seq<Iseq...>>
  {
  public:
    likelihood()
        : s{}
    {
      t = NULL;
    }

    likelihood(decltype(Iseq)... size)
        : s{{ size... }}
    {
      t = new T[mult(s)];
      memset(t, 0, mult(s)*sizeof(T));
    }

    ~likelihood()
    {
      if (t)
        delete [] t;
    }

    T& operator()(decltype(Iseq)... i)
    {
      //std::cout << "index = " << index({{ i... }}) << std::endl;
      return t[index({{ i... }})];
    }

    const T& operator()(decltype(Iseq)... i) const
    {
      return t[index({{ i... }})];
    }

    const T& operator[](int i)
    {
      return t[i];
    }

    void setValue(T value, decltype(Iseq)... i)
    {
      t[index({{ i... }})] = value;
    }

    void Zeros()
    {
      for (std::size_t i = 0; i < mult(s); ++i)
        t[i] = 0.0;
    }

    /* two dimentional identity array */
    void Eye()
    {
      assert(s.size()==2);
      assert(s[0] == s[1]);

      for (std::size_t i = 0; i < mult(s); ++i)
        t[i] = 0.0;

      for (std::size_t i = 0; i < s[0]; ++i)
        t[s[0]*i+i]=1;
    }

    /* add a constant value to all elements */
    void Addp0()
    {
      for (std::size_t i = 0; i < mult(s); ++i)
        t[i] += p0;
    }

    /* normalisation (columns) */
    void Norm()
    {
      std::size_t range = s[1];
      for (std::size_t i = 2; i < s.size(); ++i)
      {
        range = range * s[i];
      }

#ifdef _OPENMP
      #pragma omp parallel for
#endif
      for (std::size_t j = 0; j < range; ++j)
      {
        T sum = 0.0;
        auto a = &t[j];
#ifdef _OPENMP
        #pragma omp parallel for reduction (+:sum)
#endif
        for (std::size_t k = 0; k < s[0]; ++k)
	{
	  sum += a[k*range];
	}

        if (sum > 0)
#ifdef _OPENMP
          #pragma omp parallel for
#endif
          for (std::size_t k = 0; k < s[0]; ++k)
          {
  	    a[k*range] /= sum;
	  }
	else
#ifdef _OPENMP
          #pragma omp parallel for
#endif
          for (std::size_t k = 0; k < s[0]; ++k)
          {
  	    a[k*range] /= s[0];
	  }
      }
    }

    /* sum of this object and likelihood object given as a parameter */
    void sum(const likelihood<T,seq<Iseq...>>& b)
    {
#ifdef _OPENMP
      #pragma omp parallel for
#endif
      for (std::size_t i = 0; i < mult(s); ++i)
        t[i] += b.t[i];
    }

    /* return order (rank) */
    const std::size_t get_order()
    {
      return s.size();
    }

    /* return size of first dimension */
    const std::size_t get_firstdimension()
    {
      return s[0];
    }

    /* return array with sizes for each dimension */
    std::size_t *get_dimensions()
    {
      std::size_t* res = new std::size_t[s.size()];
      for (std::size_t i = 0; i < s.size(); ++i)
        res[i] = s[i];
      return res;
    }

    /* return total number of elements */
    const std::size_t get_tnc()
    {
      return mult(s);
    }

    /* index of first dimension with maximum value in t(:,a[0],...,a[Nf-1]) */
    int MaxIndex(const std::vector<std::size_t>& a) const
    {
      assert(a.size()==s.size()-1);

#ifdef _OPENMP
      typedef struct { T val; int loc; char pad[128]; } tvals;
      int nt; 
      T max = -INFINITY;
      int maxind = -1;

      #pragma omp parallel
      {
        #pragma omp single
        nt = omp_get_num_threads();
      }

      tvals *maxinfo = new tvals[nt];

      #pragma omp parallel shared(maxinfo)
      {
        int id = omp_get_thread_num();
        maxinfo[id].val = -INFINITY;
        maxinfo[id].loc = -1;

        #pragma omp for 
        for (std::size_t k = 0; k < s[0]; ++k)
        {
          std::size_t ind = k;

          for (std::size_t i = 0; i < a.size(); ++i)
            ind = ind * s[i+1] + a[i];

          if (t[ind] > maxinfo[id].val)
          {
            maxinfo[id].val = t[ind];
            maxinfo[id].loc = k;
	  }
        }
        #pragma omp flush (maxinfo)
        #pragma omp master
        {
          max = maxinfo[0].val;
          maxind = maxinfo[0].loc;
 
          for (int i = 1; i < nt; i++) {
            if (maxinfo[i].val > max) {
              max = maxinfo[i].val;
              maxind = maxinfo[i].loc;
            }
          }
        }
      }

      delete [] maxinfo;
#else
      T max = -INFINITY;
      int maxind = -1;

      for (std::size_t k = 0; k < s[0]; ++k)
      {
        std::size_t ind = k;

        for (std::size_t i = 0; i < a.size(); ++i)
          ind = ind * s[i+1] + a[i];

	if (t[ind] > max)
	{
	  max = t[ind];
	  maxind = k;
	}
      }
#endif

      return maxind;
    }

    /* return index array */
    const std::array<std::size_t, sizeof...(Iseq)> GetIndexArray()
    {
      return s;
    }

    /* constructor by passing a matrix */
    likelihood(std::vector<std::vector<T>> const &matrix)
         : likelihood<T,seq<Iseq...>>(matrix.size(),matrix[0].size())
    {
      std::size_t i = 0;
      for (std::vector<T> row: matrix)
        for (T val: row)
          t[i++] = val;
    }

    /* copy constructor by passing the index array */
    likelihood(const std::array<std::size_t, sizeof...(Iseq)>& ia)
        : s(ia)
    {
      t = new T[mult(ia)];
    }

    /* return a new object obtained by multiplying each element of the
    array by the logarithm of itself*/
    likelihood AlogA()
    {
      likelihood a(s); 

#ifdef _OPENMP
      #pragma omp parallel for
#endif
      for (std::size_t i = 0; i < mult(s); ++i)
        a.t[i] = t[i] * _log(t[i]);

      return a;
    }

    /* multidimensional dot (inner) product
    extract the array elements corresponding
    to the index tuple sq along dimension f */
    T **Dot(std::vector<int> sq, std::size_t f)
    {
      T **_Ag = 0;

      _Ag=new T*[s[0]];
      for (std::size_t k = 0; k < s[0]; ++k)
        _Ag[k]=new T[s[f+1]];

#ifdef _OPENMP
      #pragma omp parallel for
#endif
      for (std::size_t k = 0; k < s[0]; ++k)
      {
        auto _a = _Ag[k];

        for (std::size_t i = 0; i < s[f+1]; ++i)
	{
          std::size_t ind = k;

          for (std::size_t j = 0; j < sq.size(); ++j)
	  {
	    if (j != f)
	      ind = ind * s[j+1] + sq[j];
	    else
	      ind = ind * s[j+1] + i;
	  }

  	  _a[i] =  t[ind];
          //std::cout << "_Ag[" << k << "][" << i << "] = t[" << ind << "]=" << t[ind] << std::endl;
	}
      }

      return _Ag;
    }

    /* multidimensional dot (inner) product
    inner product obtained by summing the products of
    the likelihood and the vectors xt[i], i=0,...,Nf-1,
    along leading dimension of the likelihood and
    epistemic value */
    T *HDot(T **xt, likelihood& l, T *H)
    {
      T *_q = 0;

      _q=new T[s[0]];

      T *x = cross(xt, s);

      std::size_t offset = mult(s)/s[0];

      T sum_H = 0;

#ifdef _OPENMP
      #pragma omp parallel for reduction (+:sum_H)
#endif
      for (std::size_t k = 0; k < s[0]; ++k)
      {
        std::size_t _offset = offset * k;

        auto sum_k = T{};
#if defined _OPENMP && _OPENMP >= 201307
        auto const*const __restrict a = &t[_offset];
        auto const*const __restrict b = &l.t[_offset];
        auto sum = T{};
        #pragma omp simd reduction (+:sum,sum_k)
        for(std::size_t j = 0; j < offset; ++j)
        {
          T xval = x[j];

          sum += a[j] * xval;

          sum_k += b[j] * xval;
        }

        _q[k] = sum;
#else
        _q[k] = opt_dot<T>(offset, &t[_offset], &l.t[_offset], &x[0], &sum_k);
#endif

        sum_H += sum_k;
      }

      *H = sum_H;

      delete [] x;

      return _q;
    }

    /* multidimensional dot (inner) product
    inner product obtained by summing the products of
    the likelihood and the vectors xt[i], i=0,...,Nf-1,
    along leading dimension of the likelihood and
    epistemic value of the whole likelihood */
    T *HDot(T **xt, T *H)
    {
      T *_q = 0;

      _q=new T[s[0]];

      T *x = cross(xt, s);

      std::size_t offset = mult(s)/s[0];

      T sum_H = 0;

#ifdef _OPENMP
      #pragma omp parallel for reduction (+:sum_H)
#endif
      for (std::size_t k = 0; k < s[0]; ++k)
      {
        std::size_t _offset = offset * k;

        auto sum_k = T{};
#if defined _OPENMP && _OPENMP >= 201307
        auto const*const __restrict a = &t[_offset];
        auto sum = T{};
        #pragma omp simd reduction (+:sum,sum_k)
        for(std::size_t j = 0; j < offset; ++j)
        {
          T xval = x[j];

          sum += a[j] * xval;

          sum_k += a[j] * _log(a[j]) * xval;
        }

        _q[k] = sum;
#else
        _q[k] = opt_dot<T>(offset, &t[_offset], &x[0], &sum_k);
#endif

        sum_H += sum_k;
      }

      *H = sum_H;

      delete [] x;

      return _q;
    }

    /* epistemic value */
    T HDot(T **xt)
    {
      T H = 0;

      T *x = cross(xt, s);

      std::size_t offset = mult(s)/s[0];

#ifdef _OPENMP
      #pragma omp parallel for reduction (+:H)
#endif
      for (std::size_t k = 0; k < s[0]; ++k)
      {
        std::size_t _offset = offset * k;

        auto sum = T{};
#if defined _OPENMP && _OPENMP >= 201307
        auto const*const __restrict a = &t[_offset];
        #pragma omp simd reduction (+:sum)
        for (std::size_t j = 0; j < offset; ++j)
          sum += a[j] * _log(a[j]) * x[j];
#else
        sum = opt_hdot<T>(offset, &t[_offset], &x[0]);
#endif

        H += sum;
      }

      delete [] x;

      return H;
    }

    /* find the likelihood elements t(:,sq[0],...,sq[Nf-1])
    and store them in the vector p */
    void find(std::vector<int> sq, std::vector<T> &p)
    {
#ifdef _OPENMP
      #pragma omp parallel for
#endif
      for (std::size_t k = 0; k < s[0]; ++k)
      {
        std::size_t ind = k;

        for (std::size_t j = 0; j < sq.size(); ++j)
	  ind = ind * s[j+1] + sq[j];

        p.at(k) = t[ind];
      }
    }
 
  private:
    std::size_t index(const std::array<std::size_t, sizeof...(Iseq)>& a) const
    {
      std::size_t ind = a[0];
      for (std::size_t i = 1; i < a.size(); ++i)
      {
        ind = ind * s[i] + a[i];
      }
      return ind;
    }
 
    std::size_t mult(const std::array<std::size_t, sizeof...(Iseq)>& a)
    {
      return std::accumulate(begin(a), end(a), 1, std::multiplies<std::size_t>{});
    }
 
    T *t;
    const std::array<std::size_t, sizeof...(Iseq)> s;
  };
}
#endif

//template <typename T, std::size_t N>
//using likelihood = detail::likelihood<T, typename gen_seq<N>::type>;
