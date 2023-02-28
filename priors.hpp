#ifndef PRIORS_HPP
#define PRIORS_HPP
#include <iostream>
#include <cstdlib>
#include <stdlib.h>
#include <cmath>
#include "util.hpp"
#include "constants.h"

/* Priors array (of size Ns) class */
template <typename T>
class Priors {
protected:
  unsigned int Ns;
private:
  T *value;

public:
  Priors()
  {
    this->Ns = 0;

    value = NULL;
  }

  Priors(unsigned int Ns_)
  {
    this->Ns = Ns_;

    value = new T[Ns_];
  }

  void setValue(T val, unsigned int i)
  {
    this->value[i] = val;
  }

  T getValue(unsigned int i)
  {
    return this->value[i];
  }

  void Zeros()
  {
    for(unsigned int i = 0; i < this->Ns; i++)
      value[i] = 0.0;
  }

  /* logarithmic transformation (after normalisation) */
  void NormLog()
  {
    T sum = 0.0;
#ifdef _OPENMP
    #pragma omp parallel for reduction (+:sum)
#endif
    for(unsigned int i = 0; i < this->Ns; i++)
      sum += value[i];
    if (sum > 0)
#ifdef _OPENMP
      #pragma omp parallel for
#endif
      for(unsigned int i = 0; i < this->Ns; i++)
      {
        T norm_v = value[i] / sum;

        value[i] = _log(norm_v);
      }
    else
#ifdef _OPENMP
      #pragma omp parallel for
#endif
      for(unsigned int i = 0; i < this->Ns; i++)
      {
        T norm_v = value[i] / this->Ns;

        value[i] = _log(norm_v);
      }
  }

  unsigned int get_size()
  {
    return Ns;
  }

  /* constructor by passing a vector */
  Priors(std::vector<T> const &v)
       : Priors<T>(v.size())
  {
    std::size_t i = 0;
    for (T val: v)
      value[i++] = val;
  }

  /* copy constructor by passing the object */
  Priors(const Priors<T> &p)
  {
    this->Ns = p.Ns;

    value = new T[this->Ns];
    for(unsigned int i = 0; i < this->Ns; i++)
      value[i] = p.value[i];
  }

  ~Priors() {
    delete [] value;
  }
};
#endif
