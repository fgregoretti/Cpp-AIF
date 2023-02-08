#ifndef BELIEFS_HPP
#define BELIEFS_HPP
#include <iostream>
#include <cstdlib>
#include <stdlib.h>
#include <cmath>
#include "util.hpp"
#include "constants.h"

/* Beliefs array (Ns by T) class */
template <typename Ty>
class Beliefs {
protected:
  unsigned int Ns;
  unsigned int T;
public:
  Ty *value;

public:
  Beliefs()
  {
    this->Ns = 0;
    this->T = 0;

    value = NULL;
  }

  Beliefs(unsigned int Ns_, unsigned int T_ = 1)
  {
    this->Ns = Ns_;
    this->T = T_;

    value = new Ty[T_*Ns_];
  }

  void Zeros()
  {
    for(std::size_t i = 0; i < this->T*this->Ns; i++)
      value[i] = 0.0;
  }

  void Ones()
  {
    for(std::size_t i = 0; i < this->T*this->Ns; i++)
      value[i] = 1.0;
  }

  /* normalisation (columns) */
  void Norm()
  {
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for(std::size_t j = 0; j < this->T; j++)
    {
      std::size_t sttidx = j*this->Ns;
      Ty sum = 0.0;

#ifdef _OPENMP
      #pragma omp parallel for reduction (+:sum)
#endif
      for(std::size_t i = 0; i < this->Ns; i++)
        sum += value[sttidx+i];

      if (sum > 0)
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for(std::size_t i = 0; i < this->Ns; i++)
          value[sttidx+i] /= sum;
      else
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for(std::size_t i = 0; i < this->Ns; i++)
          value[sttidx+i] /= this->Ns;
    }
  }

  /* logarithmic transformation */
  void Log()
  {
    for(std::size_t i = 0; i < this->T*this->Ns; i++)
      value[i] = _log(value[i]);
  }

  /* logarithmic transformation (after normalisation) */
  void NormLog()
  {
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for(std::size_t j = 0; j < this->T; j++)
    {
      std::size_t sttidx = j*this->Ns;
      Ty sum = 0.0;

#ifdef _OPENMP
      #pragma omp parallel for reduction (+:sum)
#endif
      for(std::size_t i = 0; i < this->Ns; i++)
        sum += value[sttidx+i];

      if (sum > 0)
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for(std::size_t i = 0; i < this->Ns; i++)
        {
          value[sttidx+i] /= sum;

          value[sttidx+i] = _log(value[sttidx+i]);
        }
      else
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for(std::size_t i = 0; i < this->Ns; i++)
        {
          value[sttidx+i] /= this->Ns;

          value[sttidx+i] = _log(value[sttidx+i]);
        }
    }
  }

  unsigned int get_size()
  {
    return Ns;
  }

  /* constructor by passing a vector */
  Beliefs(std::vector<Ty> D)
       : Beliefs<Ty>(D.size())
  {
    std::size_t i = 0;
    for (Ty val: D)
      value[i++] = val;
  }

  /* copy constructor by passing the object */
  Beliefs(const Beliefs<Ty> &b)
  {
    this->Ns = b.Ns;
    this->T = b.T;

    value = new Ty[this->T*this->Ns];

    for(std::size_t i = 0; i < this->T*this->Ns; i++)
      value[i] = b.value[i];
  }

  ~Beliefs() {
    delete [] value;
  }
};
#endif
