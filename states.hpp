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
