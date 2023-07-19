// BSD 3-Clause License

// Copyright (c) 2022, Francesco Gregoretti

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
private:
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

  void setValue(Ty val, unsigned int i)
  {
    this->value[i] = val;
  }

  void setValue(Ty val, unsigned int i, unsigned int t_)
  {
    this->value[t_*Ns+i] = val;
  }

  Ty getValue(unsigned int i)
  {
    return this->value[i];
  }

  Ty getValue(unsigned int i, unsigned int t_)
  {
    return this->value[t_*Ns+i];
  }

  Ty *getArray(unsigned int t_)
  {
    return &(this->value[t_*Ns]);
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
