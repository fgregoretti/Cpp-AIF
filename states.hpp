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

#ifndef STATES_HPP
#define STATES_HPP
#include <iostream>
#include <cstdlib>
#include <stdlib.h>

/* States integer array (of size T) class */
class States {
protected:
  unsigned int T;
private:
  unsigned int *id;

public:
  States(unsigned int T_)
  {
    this->T = T_;

    id = new unsigned int [T_];
  }

  void Zeros()
  {
    for(std::size_t i = 0; i < this->T; i++)
      id[i] = 0;
  }

  void Set(unsigned int val, unsigned int t)
  {
    id[t] = val;
  }

  void Set(unsigned int val)
  {
    id[0] = val;
  }

  unsigned int Get(unsigned int t)
  {
    return id[t];
  }

  unsigned int Get()
  {
    return id[0];
  }

  inline unsigned int StateFind(unsigned int t)
  {
    return Get(t);
  }

  inline unsigned int StateFind()
  {
    return Get();
  }

  /* copy constructor by passing the object */
  States(const States &s)
  {
    this->T = s.T;

    id = new unsigned int [this->T];
    for(std::size_t i = 0; i < this->T; i++)
      id[i] = s.id[i];
  }

  ~States() {
      delete [] id;
  }
};
#endif
