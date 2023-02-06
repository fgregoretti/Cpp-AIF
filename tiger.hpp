#include <vector>
#include "beliefs.hpp"
#include "transitions.hpp"
#include "likelihood.hpp"
#include "priors.hpp"
#include "mdp.hpp"
#include "kron.hpp"

template <typename T>
class _Beliefs : public Beliefs<T>
{
public:
  using Beliefs<T>::Beliefs;

  _Beliefs(std::vector<T> D1, std::vector<T> D2)
       : Beliefs<T>(D1.size()*D2.size())
  {
    kron(D1, D2, &this->value[0]);
  }
};

//template <typename Ty, std::size_t M>
//class _MDP : public MDP<Ty,M>
//{
//public:
//  using MDP<Ty,M>::MDP;
//};
