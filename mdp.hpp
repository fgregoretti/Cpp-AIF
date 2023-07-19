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

#ifndef MDP_HPP
#define MDP_HPP
#include <iostream>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <stdlib.h>
#include <random>
#include "states.hpp"
#include "beliefs.hpp"
#include "transitions.hpp"
#include "likelihood.hpp"
#include "priors.hpp"
#include "constants.h"
#include "util.hpp"
#include "construct_policies.hpp"
#include "common.h"
//#ifdef _OPENMP
//#include "omp.h"
//#endif

template <typename T, std::size_t N>
using likelihood = detail::likelihood<T, typename gen_seq<N>::type>;

template <typename T>
void delete_pointed_to(T* const ptr) { delete ptr; }

template <typename T>
void set_pointed_to_null(T* ptr) { ptr = NULL; }

template <typename Ty, std::size_t M>
class MDP {
protected:
  unsigned int T; /* temporal horizon */
  Ty alpha; /* gamma hyperparameter */
  Ty beta; /* gamma hyperparameter */
  Ty lambda; /* precision update rate */
  Ty gamma;
  unsigned int N; /* number of variational iterations */
#ifndef FULL
  unsigned int policy_len;
#endif
  unsigned int seed;
  std::vector<unsigned int> Ns;
  unsigned int Nu; /* number of hidden controls */
  std::vector<unsigned int> No; /* number of outcomes */
  std::vector<std::vector<int>> _V; /* policies */
  std::vector<Beliefs<Ty>*> _lnD;
  std::vector<States*> _S; /* real state in the world */
  std::vector<std::vector<Transitions<Ty>*> > _B;
  std::vector<std::vector<likelihood<Ty,M>*> > _A;
#ifdef WITH_GP
  std::vector<std::vector<likelihood<Ty,M>*> > _AA;
#endif
#ifndef NO_PRECOMPUTE_ALOGA
  std::vector<std::vector<likelihood<Ty,M>*> > _AlogA;
#endif
  std::vector<Priors<Ty>*> _lnC;
  std::vector<likelihood<Ty,M>*> Au;
  std::vector<Beliefs<Ty>*> _X;
  std::vector<States*> _O; /* states observed */
  std::vector<int> U; /* action selected each time */
  std::mt19937 generator;
  std::vector<std::vector<Ty>> _ut; /* policy expectations */
  std::vector<std::vector<Ty>> _P; /* posterior beliefs about control */
  std::vector<Ty> _W; /* posterior precision */
#ifdef FULL
  std::vector<unsigned int> _wt; /* indices of allowable policies */
#endif
#ifdef LEARNING
  std::vector<std::vector<std::vector<std::vector<Ty>>>> _xt;
#endif

public:
  unsigned int Nf;
  unsigned int Ng;
  unsigned int Np;
  std::vector<std::vector<int>> _st;
  std::vector<std::vector<int>> _ot;

  MDP(std::vector<Beliefs<Ty>*>& __D, /* initial state probabilities */
      std::vector<States*>& __S, /* true initial state */
      std::vector<std::vector<Transitions<Ty>*>>& __B, /* transition probabilities */
      std::vector<std::vector<likelihood<Ty,M>*>>& __A, /* observation model */
#ifdef WITH_GP
      std::vector<std::vector<likelihood<Ty,M>*>>& __AA, /* observation process */
#endif
      std::vector<Priors<Ty>*>& __C, /* terminal cost probabilities */
      std::vector<std::vector<int>>& __V, /* policies */
      unsigned int T_ = 10, Ty alpha_ = 8, Ty beta_ = 4,
      Ty lambda_ = 0, Ty gamma_ = 1, unsigned int N_ = 4,
#ifndef FULL
      unsigned int policy_len_ = 1,
#endif
      unsigned int seed_ = 0);

  virtual int get_st(unsigned int f, unsigned int t, int action);
  virtual void logBtimesX(unsigned int f, unsigned int t, std::vector<Ty> &v);
  virtual void marginal_likelihood(unsigned int f, unsigned int t, std::vector<int>& sq, std::vector<Ty>& v);
  void infer_states(unsigned int t);
  std::vector<Ty> infer_policies(unsigned int t);
  int sample_action(unsigned int t);
  void sample_state(unsigned int t, int action);
  void sample_observation(unsigned int t, int action);
  virtual void active_inference();
  std::vector<std::vector<likelihood<Ty,M>*>>& update_A(
                std::vector<std::vector<likelihood<Ty,M>*>>& _a,
                Ty eta, unsigned int tt);
  std::vector<std::vector<Transitions<Ty>*>>& update_B(
                std::vector<std::vector<Transitions<Ty>*>>& _b,
                Ty eta, unsigned int tt);
  std::vector<Priors<Ty>*>& update_C(std::vector<Priors<Ty>*>& _c,
                Ty eta, unsigned int tt);
  std::vector<Beliefs<Ty>*>& update_D(std::vector<Beliefs<Ty>*>& _d,
                Ty eta, unsigned int tt);
  int getU(unsigned int t) { return this->U[t]; }

  virtual ~MDP() {
    std::for_each(_lnD.begin(), _lnD.end(), delete_pointed_to<Beliefs<Ty>>);
    std::for_each(_lnC.begin(), _lnC.end(), delete_pointed_to<Priors<Ty>>);
    std::for_each(_X.begin(), _X.end(), delete_pointed_to<Beliefs<Ty>>);
    if (Au.size() != 0)
      for (unsigned int g = 0; g < Ng; g++)
        if (_A[g].size() > 1)
          delete Au[g];
    std::for_each(_O.begin(), _O.end(), delete_pointed_to<States>);
#ifndef NO_PRECOMPUTE_ALOGA
    for (unsigned int g = 0; g < Ng; g++)
      std::for_each(_AlogA[g].begin(), _AlogA[g].end(), delete_pointed_to<likelihood<Ty,M>>);
#endif
  }

  Ty generateRand()
  {
    std::uniform_real_distribution<Ty> distribution(0, 1);
    return distribution(generator);
  }

  unsigned int generateRandAcT()
  {
    std::uniform_int_distribution<int> distribution(0, Nu-1);
    return distribution(generator);
  }

  unsigned int generateRandAcT(unsigned int size)
  {
    std::uniform_int_distribution<int> distribution(0, size-1);
    return distribution(generator);
  }
};

template <typename Ty, std::size_t M>
MDP<Ty,M>::MDP(std::vector<Beliefs<Ty>*>& __D,
  std::vector<States*>& __S,
  std::vector<std::vector<Transitions<Ty>*>>& __B,
  std::vector<std::vector<likelihood<Ty,M>*>>& __A,
#ifdef WITH_GP
  std::vector<std::vector<likelihood<Ty,M>*>>& __AA,
#endif
  std::vector<Priors<Ty>*>& __C,
  std::vector<std::vector<int>>& __V,
  unsigned int T_, Ty alpha_, Ty beta_,
  Ty lambda_, Ty gamma_, unsigned int N_,
#ifndef FULL
  unsigned int policy_len_,
#endif
  unsigned int seed_) :
  T(T_),
  alpha(alpha_),
  beta(beta_),
  lambda(lambda_),
  gamma(gamma_),
  N(N_),
#ifndef FULL
  policy_len(policy_len_),
#endif
  seed(seed_) {
  Ng = __A.size(); /* number of outcome factors */
  Nf = __S.size(); /* number of hidden-states factors */
  Nu = __B[0].size(); /* number of hidden controls */

  if (__V.size() != 0)
  {
    _V = __V;
  }
  else
#ifdef FULL
    _V = construct_policies(Nu, T);
#else
    _V = construct_policies(Nu, (int) policy_len);
#endif

  Np = _V[0].size();

#ifdef DEBUG
  std::cout << "MDP: Nf=" << Nf << " Ng=" << Ng << " Nu=" << Nu << std::endl;
#ifdef FULL
  for (unsigned int j = 0; j < T; j++)
#else
  for (unsigned int j = 0; j < policy_len; j++)
#endif
  {
    std::cout << "MDP: V[" << j << "] = ";
    for (int val: _V[j]) {
      std::cout << val << " ";
    }
    std::cout << std::endl;
  }
#endif

  if (Nf != __B.size())
  {
    std::cerr << "true initial state __S and transition probabilities __B are not consistent" << std::endl;
    exit(-1);
  }

  if (__D.size() == 0)
    for (unsigned int i = 0; i < Nf; i++) {
      Beliefs<Ty> *Initial_D = new Beliefs<Ty>(__B[i][0]->get_size());
      Initial_D->Ones();
      Initial_D->Norm();
      __D.push_back(Initial_D);
    }

  if (Nf != __D.size())
  {
    std::cerr << "true initial state __S and initial state probabilities __D are not consistent" << std::endl;
    exit(-1);
  }

  if (Ng != __C.size())
  {
    std::cerr << "__C not correctly specified" << std::endl;
    exit(-1);
  }

  std::vector<std::size_t> s;

  for (unsigned int i = 0; i < Nf; i++) {
    /* initial beliefs */
    _lnD.push_back(new Beliefs<Ty>(*__D[i]));

    _lnD[i]->NormLog();

    Ns.push_back(_lnD[i]->get_size());

#ifdef DEBUG
    std::cout << "MDP: _lnD[" << i << "] = ";
    for (unsigned int e = 0; e < Ns[i]; e++)
      std::cout << _lnD[i]->getValue(e) << " ";
    std::cout << std::endl;
#endif

    /* initial states */
    //_S.push_back(new States(*__S[i]));
    _S.push_back(__S[i]);

    /* states sampled */
    s.push_back(_S[i]->StateFind());

    /* transition probabilities (priors) */
    std::vector<Transitions<Ty>*> b1;
    for (unsigned int j = 0; j < __B[i].size(); j++)
    {
      if (Ns[i] != __B[i][j]->get_size())
      {
        std::cerr << "__B not correctly specified" << std::endl;
        exit(-1);
      }

      __B[i][j]->Norm();

      //b1.push_back(new Transitions<Ty>(*__B[i][j]));
      b1.push_back(__B[i][j]);
    }

    _B.push_back(b1);

#ifdef DEBUG
    for (unsigned int j = 0; j < _B[i].size(); j++)
    {
      std::cout << "MDP: _B[" << i << "][" << j << "] = ";
      _B[i][j]->Print();
    }
#endif

    /* expectations of hidden states */
    _X.push_back(new Beliefs<Ty>(Ns[i],T));
    _X[i]->Zeros();
  }

  for (unsigned int g = 0; g < Ng; g++) {
    std::vector<likelihood<Ty,M>*> a1;
#ifndef NO_PRECOMPUTE_ALOGA
    std::vector<likelihood<Ty,M>*> a2;
#endif
#ifdef WITH_GP
    std::vector<likelihood<Ty,M>*> aa1;
#endif

    /* future outcomes probabilities (priors) */
    _lnC.push_back(new Priors<Ty>(*__C[g]));

    _lnC[g]->NormLog();

    if ( (__A[g].size() > 1) && (__A[g].size() != Nu) )
    {
      std::cerr << "__A not correctly specified" << std::endl;
      exit(-1);
    }
#ifdef WITH_GP
    if ( __AA[g].size() != __A[g].size() )
    {
      std::cerr << "__AA not correctly specified" << std::endl;
      exit(-1);
    }
#endif

    for (unsigned int j = 0; j < __A[g].size(); j++)
    {
      if (__A[g][j]->get_order() != Ns.size()+1 )
      {
        std::cerr << "__A not correctly specified" << std::endl;
        exit(-1);
      }
#ifdef WITH_GP
      if (__AA[g][j]->get_order() != Ns.size()+1 )
      {
        std::cerr << "__AA not correctly specified" << std::endl;
        exit(-1);
      }
#endif

#ifdef CHECK_CONSISTENCY_VERBOSE
      auto dims = __A[g][j]->get_dimensions();

      if (dims[0] != _lnC[g]->get_size())
      {
        std::cerr << "__A and __C are not consistently defined" << std::endl;
        exit(-1);
      }

      for (unsigned int i = 0; i < Nf; i++)
        if (Ns[i] != dims[i+1]) {
          std::cerr << "__A and __D are not consistent" << std::endl;
          exit(-1);
        }
      delete [] dims;
#endif
 
      /* likelihood model */
      //__A[g][j]->Addp0();
      __A[g][j]->Norm();

      //a1.push_back(new likelihood<Ty,M>(*__A[g][j]));
      a1.push_back(__A[g][j]);
#ifndef NO_PRECOMPUTE_ALOGA
      a2.push_back(new likelihood<Ty,M>(__A[g][j]->AlogA()));
#endif
#ifdef WITH_GP
      __AA[g][j]->Norm();
      aa1.push_back(__AA[g][j]);
#endif
    }

    _A.push_back(a1);
#ifndef NO_PRECOMPUTE_ALOGA
    _AlogA.push_back(a2);
#endif
#ifdef WITH_GP
    _AA.push_back(aa1);
#endif

#ifdef DEBUG
    for (unsigned int j = 0; j < _A[g].size(); j++)
    {
      std::cout << "MDP: _A[" << g << "][" << j << "] = ";
      for (unsigned int e = 0; e < _A[g][j]->get_tnc(); e++)
        std::cout << (*_A[g][j])[e] << " ";
      std::cout << std::endl;
    }
#endif

    //No.push_back(__A[g][0]->get_firstdimension());
    No.push_back(_lnC[g]->get_size());

#ifdef DEBUG
    std::cout << "MDP: _lnC[" << g << "] = ";
    for (unsigned int e = 0; e < No[g]; e++)
      std::cout << _lnC[g]->getValue(e) << " ";
    std::cout << std::endl;
#endif

    if (_A[g].size() > 1)
    {
      Au.push_back(new likelihood<Ty,M>(__A[g][0]->GetIndexArray()));
      Au[g]->Zeros();
      /* could it be also Au.push_back(new likelihood<Ty,M>(*__A[i][0]));
       starting j-cycle from 1 */

      /* summing likelihoods parameters over actions, for each A factor */
      for(unsigned int j = 0; j < __A[g].size(); j++)
        Au[g]->sum(*__A[g][j]);

      Au[g]->Norm();
    }
    else
    {
      likelihood<Ty,M>* _au1 = _A[g][0];
      Au.push_back(_au1);
    }

#ifdef DEBUG
    std::cout << "MDP: Au[" << g << "] = ";
    for (unsigned int e = 0; e < Au[g]->get_tnc(); e++)
      std::cout << (*Au[g])[e] << " ";
    std::cout << std::endl;

    std::cout << "MDP: s = ";
    for (std::size_t val: s) {
      std::cout << val << " ";
    }
    std::cout << std::endl;
#endif

    /* index observation with max probability */
    int q = Au[g]->MaxIndex(s);
#ifdef DEBUG
    std::cout << "MDP: q=" << q << std::endl;
#endif

    /* initial outcomes */
    _O.push_back(new States(T));
    _O[g]->Set(q);
  }

  U.resize(T, -1);

  _st.resize(T, std::vector<int>(Nf, -1));
  _ot.resize(T, std::vector<int>(Ng, -1));
  _ut.resize(T, std::vector<Ty>(Np, 0));
  _P.resize(T, std::vector<Ty>(Nu, 0));
  _W.resize(T, 0);
#ifdef LEARNING
  _xt.resize(T);
  for (unsigned int i = 0; i < T; i++)
  {
    _xt[i].resize(Np);
    for (unsigned int j = 0; j < Np; j++)
      _xt[i][j].resize(Nf);
  }
#endif

  for (unsigned int i = 0; i < Nf; i++)
    _st[0][i] = _S[i]->Get();

  for (unsigned int g = 0; g < Ng; g++)
    _ot[0][g] = _O[g]->Get();

  generator.seed(seed);
}

template <typename Ty, std::size_t M>
int MDP<Ty,M>::get_st(unsigned int f, unsigned int t, int action)
{
#ifdef DEBUG
  std::cout << "get_st: t=" << t << " action=" << action << " _S[" << f << "]->StateFind(" << t << ")=" << _S[f]->StateFind(t) << std::endl;
#endif

#ifdef SAMPLE_AS_MAX
  int act_u = _B[f].size() == 1 ? 0 : action;
  return _B[f][act_u]->MaxIndex(_S[f]->StateFind(t));
#else
  std::vector<Ty> ps(Ns[f], 0.0);

  int act_u = _B[f].size() == 1 ? 0 : action;
  _B[f][act_u]->extract_column(_S[f]->StateFind(t),ps);

#ifdef DEBUG
  std::cout << "get_st: ps = ";
  for (Ty val: ps) {
    std::cout << val << " ";
  }
  std::cout << std::endl;
#endif

  return CDFs<Ty>(ps, generateRand());
#endif
}

template <typename Ty, std::size_t M>
void MDP<Ty,M>::logBtimesX(unsigned int f, unsigned int t, std::vector<Ty>& v)
{
  int act_ut = _B[f].size() == 1 ? 0 : U[t-1];
  _B[f][act_ut]->logTxv(_X[f]->getArray(t-1), v);
}

template <typename Ty, std::size_t M>
void MDP<Ty,M>::marginal_likelihood(unsigned int f, unsigned int tt, std::vector<int>& sq, std::vector<Ty>& v)
{
  for (unsigned int g = 0; g < Ng; g++)
  {
    Ty **Ag = NULL;

    if (tt > 0)
    {
      int act_ut = _A[g].size() == 1 ? 0 : U[tt-1];
      Ag = _A[g][act_ut]->Dot(sq,f);
    }
    else
      Ag = Au[g]->Dot(sq,f);

#ifdef DEBUG
    std::cout << "marginal_likelihood: tt=" << tt << " f=" << f << " g=" << g << " Ag = ";
    for (std::size_t k = 0; k < _A[g][0]->get_firstdimension(); k++)
    {
      for(unsigned int ii = 0; ii < Ns[f]; ii++)
        std::cout << Ag[k][ii] << " ";
      std::cout << std::endl;
    }
#endif

    for(unsigned int ii = 0; ii < Ns[f]; ii++)
      v[ii] += _log(Ag[_O[g]->StateFind(tt)][ii]);

    for (std::size_t k = 0; k < _A[g][0]->get_firstdimension(); k++)
      delete [] Ag[k];
    delete [] Ag;
  }
}

template <typename Ty, std::size_t M>
void MDP<Ty,M>::infer_states(unsigned int tt)
{
  std::vector<int> sq;

  for (unsigned int i = 0; i < Nf; i++)
    sq.push_back(_S[i]->StateFind(tt));

#ifdef DEBUG
  std::cout << "infer_states: tt=" << tt << " sq = ";
  for (unsigned int val: sq) {
    std::cout << val << " ";
  }
  std::cout << std::endl;
#endif

  if (tt > 0)
  {
#ifdef FULL
    /* retain allowable policies (that are consistent with last action) */
    std::vector<unsigned int> __wt(_wt);

    _wt.clear();

    for(unsigned int jj = 0; jj < __wt.size(); jj++)
      if (_V[tt-1][__wt[jj]] == U[tt-1])
        _wt.push_back(__wt[jj]);
#endif

    /* update policy expectations */
    Ty _ut_sum = 0.0;

#ifdef FULL
    for (unsigned int val: _wt)
#else
    for (unsigned int val = 0; val < Np; val++)
#endif
      _ut_sum += _ut[tt-1][val];
#ifdef FULL
    for (unsigned int val: _wt)
#else
    for (unsigned int val = 0; val < Np; val++)
#endif
      _ut[tt][val] = _ut[tt-1][val] / _ut_sum;
  }
  else
  {
    /* initialise policy expectations */
    for(unsigned int jj = 0; jj < Np; jj++)
    {
#ifdef FULL
      _wt.push_back(jj);
#endif
      _ut[tt][jj] = 1. / Np;
    }
  }
#ifdef DEBUG
#ifdef FULL
  std::cout << "infer_states: _wt = ";
  for (unsigned int val: _wt) {
    std::cout << val << " ";
  }
  std::cout << std::endl;
#endif
  for (unsigned int j = 0; j < T; j++)
  {
    std::cout << "infer_states: _ut[" << j << "] = ";
    for (Ty val: _ut[j]) {
      std::cout << val << " ";
    }
    std::cout << std::endl;
  }
#endif

//#ifdef _OPENMP
//  #pragma omp parallel for
//#endif
  /* expectations of allowable policies and current state */
  for (unsigned int i = 0; i < Nf; i++) {
    std::vector<Ty> v(Ns[i], 0.0);

    /* marginal likelihood over outcome factors */
    marginal_likelihood(i, tt, sq, v);

    if (tt > 0)
    {
      /* update current state expectations */
      logBtimesX(i, tt, v);
    }
    else
    {
      /* initialise current state expectations */
      for(unsigned int ii = 0; ii < Ns[i]; ii++)
#ifdef WITHOUT_TRUE_INITIAL_STATE
        v[ii] = _lnD[i]->getValue(ii);
#else
        v[ii] += _lnD[i]->getValue(ii);
#endif
    }

    softmax<Ty>(v);

#ifdef DEBUG
    std::cout << "infer_states: v[" << i << "] = ";
    for (Ty val: v) {
      std::cout << val << " ";
    }
    std::cout << std::endl;
#endif

    for (std::size_t j = 0; j != Ns[i]; ++j)
      _X[i]->setValue(v[j],j,tt);
#ifdef DEBUG
    std::cout << "infer_states: _X[" << i << "] = ";
    for (std::size_t j = 0; j != Ns[i]; ++j)
      std::cout << _X[i]->getValue(j,tt) << " ";
    std::cout << std::endl;
#endif
  }
}

template <typename Ty, std::size_t M>
std::vector<Ty> MDP<Ty,M>::infer_policies(unsigned int tt)
{
#ifdef FULL
  unsigned int Np_t = _wt.size();
#else
  unsigned int Np_t = Np;
#endif

  std::vector<Ty> G(Np_t, 0.0); 

#ifdef _OPENMP
  #pragma omp parallel for
#endif
  for (unsigned int k = 0; k < Np_t; k++)
  {
    /* path integral of expected free energy */
    Ty **x = new Ty*[Nf];
    for (unsigned int i = 0; i < Nf; i++)
      x[i] = new Ty[Ns[i]];

    for (unsigned int i = 0; i < Nf; i++)
      for (std::size_t j = 0; j != Ns[i]; ++j)
        x[i][j] = _X[i]->getValue(j,tt);

#ifdef FULL
    for (unsigned int j = tt; j < T; j++)
#else
    for (unsigned int j = 0; j < policy_len; j++)
#endif
    {
#ifdef DEBUG
      std::cout << "infer_policies: tt=" << tt << " k=" << k << " j=" << j << std::endl;
#endif
      /* transition probability from current state */
      for (unsigned int i = 0; i < Nf; i++)
      {
        /* hidden state belief expected according to the k-th policy */
#ifdef FULL
        int act_u = _B[i].size() == 1 ? 0 : _V[j][_wt[k]];
        _B[i][act_u]->Txv(x[i], x[i]);
#else
        int act_u = _B[i].size() == 1 ? 0 : _V[j][k];
        _B[i][act_u]->Txv(x[i], x[i]);
#endif
#ifdef DEBUG
        std::cout << "infer_policies: x[" << i << "] = ";
        for (std::size_t jj = 0; jj != Ns[i]; ++jj)
          std::cout << x[i][jj] << " ";
        std::cout << std::endl;
#endif
#ifdef LEARNING
#ifdef FULL
        if (j == tt)
#else
        if (j == 0)
#endif
	{
        for (std::size_t jj = 0; jj != Ns[i]; ++jj)
#ifdef FULL
	  _xt[tt][_wt[k]][i].push_back(x[i][jj]);
#else
	  _xt[tt][k][i].push_back(x[i][jj]);
#endif
        }
#endif
      }

      /* predicted entropy and divergence */
      for (unsigned int g = 0; g < Ng; g++)
      {
#ifdef FULL
        int act_t = (_A[g].size() == 1) ? 0 : _V[j][_wt[k]];
#else
        int act_t = (_A[g].size() == 1) ? 0 : _V[j][k];
#endif
        Ty H = 0.0;

#ifdef NO_PRECOMPUTE_ALOGA
        auto qo = _A[g][act_t]->HDot(x, &H);
#else
        auto qo = _A[g][act_t]->HDot(x, *_AlogA[g][act_t], &H);
#endif
#ifdef DEBUG
        std::cout << "infer_policies: g=" << g << " H=" << H << " qo = ";
        for (unsigned int kk = 0; kk < No[g]; kk++)
          std::cout << qo[kk] << " ";
        std::cout << std::endl;
#endif

        G[k] += H;

        for (unsigned int kk = 0; kk < No[g]; kk++)
          if (qo[kk] != 0.0)
            G[k] += (_lnC[g]->getValue(kk) - log(qo[kk]))*qo[kk]; /* extrinsic value */
#ifdef DEBUG
        std::cout << "infer_policies: g=" << g << " G=" << G[k] << std::endl;
#endif

        delete [] qo;
      }
    }
#ifdef DEBUG
    std::cout << "infer_policies: G[" << k << "]=" << G[k] << std::endl;
#endif

    for (unsigned int i = 0; i < Nf; i++)
      delete [] x[i];
    delete [] x;
  }

  Ty b = alpha / gamma; /* expected rate parameter */

  /* Variational iterations (assuming precise inference about past action) */
  for (unsigned int it = 0; it < N; it++)
  {
    std::vector<Ty> __ut(Np_t, 0.0);

    /* policy */
    for (unsigned int i = 0; i < Np_t; i++)
      __ut[i] = _W[tt]*G[i];

    softmax<Ty>(__ut);

    for (unsigned int i = 0; i < Np_t; i++)
#ifdef FULL
      _ut[tt][_wt[i]] = __ut[i];
#else
      _ut[tt][i] = __ut[i];
#endif

    /* precision */
    b = lambda*b + (1 - lambda)*(beta - std::inner_product(std::begin(__ut), std::end(__ut), std::begin(G), 0.0));
    _W[tt] = alpha / b;
  }
#ifdef DEBUG
  std::cout << "infer_policies: _ut[" << tt << "] = ";
  for (Ty val: _ut[tt]) {
    std::cout << val << " ";
  }
  std::cout << std::endl;
#endif

  /* posterior expectations (control) */
#ifdef FULL
  for (unsigned int k = tt; k < T; k++)
#else
  unsigned int _T = (tt+policy_len < T) ? tt+policy_len : T;
  for (unsigned int k = tt; k < _T; k++)
#endif
    for (unsigned int j = 0; j < Nu; j++)
    {
      _P[k][j] = 0.0;

      for (unsigned int i = 0; i < Np_t; i++)
#ifdef FULL
        if (_V[k][_wt[i]] == (int) j)
          _P[k][j] += _ut[tt][_wt[i]];
#else
        if (_V[k-tt][i] == (int) j)
          _P[k][j] += _ut[tt][i];
#endif
    }
#ifdef DEBUG
  for (unsigned int k = 0; k < T; k++)
  {
    std::cout << "infer_policies: _P[" << k << "] = ";
    for (Ty val: _P[k]) {
      std::cout << val << " ";
    }
    std::cout << std::endl;
  }
#endif

  return G;
}

template <typename Ty, std::size_t M>
int MDP<Ty,M>::sample_action(unsigned int tt)
{
#ifdef BEST_AS_CDFS
  std::vector<Ty> _P_t(_P[tt].begin(), _P[tt].end());
  int a = CDFs<Ty>(_P_t, generateRand());
#elif BEST_AS_MAX
  std::vector<Ty> _P_t(_P[tt].begin(), _P[tt].end());
  std::vector<int> maxima = findMaxima(_P_t);
  int a = maxima[generateRandAcT(maxima.size())];
  //int a = std::max_element(_P[tt].begin(),_P[tt].end()) - _P[tt].begin();
#endif
#ifdef DEBUG
  std::cout << "sample_action: tt=" << tt << " a=" << a << std::endl;
#endif

  U[tt] = a;

  return a;
}

template <typename Ty, std::size_t M>
void MDP<Ty,M>::sample_state(unsigned int tt, int action)
{
  for (unsigned int i = 0; i < Nf; i++)
  {
    _st[tt][i] = get_st(i, tt-1, action);
#ifdef DEBUG
    std::cout << "sample_state: _st[" << tt << "][" << i << "]=" << _st[tt][i] << std::endl;
#endif

    _S[i]->Set(_st[tt][i],tt);
  }
}

template <typename Ty, std::size_t M>
void MDP<Ty,M>::sample_observation(unsigned int tt, int action)
{
  for (unsigned int g = 0; g < Ng; g++) {
    std::vector<Ty> po(No[g], 0.0);

#ifdef WITH_GP
    int act_t = (_AA[g].size() == 1) ? 0 : action;
    _AA[g][act_t]->find(_st[tt], po);
#else
    int act_t = (_A[g].size() == 1) ? 0 : action;
    _A[g][act_t]->find(_st[tt], po);
#endif
#ifdef DEBUG
    std::cout << "sample_observation: po = ";
    for (Ty val: po) {
      std::cout << val << " ";
    }
    std::cout << std::endl;
#endif

    _ot[tt][g] = CDFs<Ty>(po, generateRand());
#ifdef DEBUG
    std::cout << "sample_observation: _ot[" << tt << "][" << g << "]=" << _ot[tt][g] << std::endl;
#endif

    _O[g]->Set(_ot[tt][g],tt);
  }
}

template <typename Ty, std::size_t M>
void MDP<Ty,M>::active_inference()
{
  unsigned int tt = 0;

  while (tt < T)
  {
#ifdef PRINT
    std::cout << "active_inference: tt=" << tt << std::endl;
#endif

    infer_states(tt);

    /* value of policies (G) */
    std::vector<Ty> G = infer_policies(tt);

    /* next action (the action that minimises expected free energy) */
    int a = sample_action(tt);

    /* sampling of next state (outcome) */
    if (tt < T-1)
    {
      /* next sampled state */
      sample_state(tt+1, a);

      /* next observed state */
      sample_observation(tt+1, a);
    }

    tt += 1;
  }
}

#ifdef LEARNING
/* Mapping from hidden states to outcomes: _a */
template <typename Ty, std::size_t M>
std::vector<std::vector<likelihood<Ty,M>*>>& MDP<Ty,M>::update_A(
                std::vector<std::vector<likelihood<Ty,M>*>>& _a,
                Ty eta, unsigned int tt)
{
  for (unsigned int g = 0; g < Ng; g++) {
    likelihood<Ty,M> *_da = new likelihood<Ty,M>(_a[g][0]->GetIndexArray());
    _da->cross(_O[g]->StateFind(tt), tt, _X);

    unsigned int act_Nu = _a[g].size() == 1 ? 1 : Nu;

    for (unsigned int j = 0; j < act_Nu; j++)
    {
      likelihood<Ty,M> *_dau = new likelihood<Ty,M>(_a[g][0]->GetIndexArray());
      _dau->multiplies(*_da, *_a[g][j]);

      _a[g][j]->sum(*_dau, eta);

      delete _dau;
    }

    delete _da;

#ifdef DEBUG
    for (unsigned int j = 0; j < _A[g].size(); j++)
    {
      std::cout << "update_A: _a[" << g << "][" << j << "] = ";
      for (unsigned int e = 0; e < _a[g][j]->get_tnc(); e++)
        std::cout << (*_a[g][j])[e] << " ";
      std::cout << std::endl;
    }
#endif
  }

  return _a;
}

/* Mapping from past hidden states to current hidden states
   modulated by the posteriors of the policies: b(u) */
template <typename Ty, std::size_t M>
std::vector<std::vector<Transitions<Ty>*>>& MDP<Ty,M>::update_B(
                std::vector<std::vector<Transitions<Ty>*>>& _b,
                Ty eta, unsigned int tt)
{
#ifdef FULL
  unsigned int Np_t = _wt.size();
#else
  unsigned int Np_t = Np;
#endif

  for (unsigned int i = 0; i < Nf; i++)
  {
    for (unsigned int k = 0; k < Np_t; k++)
    {
#ifdef FULL
      int v = _b[i].size() == 1 ? 0 : _V[tt-1][_wt[k]];
      int p = _wt[k];
      Ty _ut_tk = _ut[tt][_wt[k]];
#else
      int v = _b[i].size() == 1 ? 0 : _V[(tt-1)%policy_len][k];
      int p = k;
      Ty _ut_tk = _ut[tt][k];
#endif
 
      std::vector<std::vector<Ty>> db(Ns[i], std::vector<Ty>(Ns[i], 0));

      for (std::size_t jj = 0; jj != Ns[i]; ++jj)
        for (std::size_t kk = 0; kk != Ns[i]; ++kk)
          db[jj][kk] = _ut_tk * _xt[tt][p][i][jj] * _xt[tt-1][p][i][kk];

      db = _b[i][v]->multiplies(db);

      std::vector<std::vector<Ty>> _b_iu(Ns[i], std::vector<Ty>(Ns[i], 0));
      _b[i][v]->add(db, _b_iu, eta);

      Transitions<Ty> *__b_iu = new Transitions<FLOAT_TYPE>(_b_iu);
      delete _b[i][v];
      _b[i][v] = __b_iu;
    }

#ifdef DEBUG
    for (unsigned int j = 0; j < _b[i].size(); j++)
    {
      std::cout << "update_B: _b[" << i << "][" << j << "] = ";
      _b[i][j]->Print();
    }
#endif
  }

  return _b;
}

/* Accumulation of prior preferences: c */
template <typename Ty, std::size_t M>
std::vector<Priors<Ty>*>& MDP<Ty,M>::update_C(
                std::vector<Priors<Ty>*>& _c,
                Ty eta, unsigned int tt)
{
  for (unsigned int g = 0; g < Ng; g++)
  {
    unsigned int _dc = _O[g]->StateFind(tt);

    if (_c[g]->getValue(_dc) > 0)
      _c[g]->setValue(_c[g]->getValue(_dc) + 1*eta, _dc);

#ifdef DEBUG
    std::cout << "update_C: _c[" << g << "] = ";
    for (unsigned int e = 0; e < No[g]; e++)
      std::cout << _c[g]->getValue(e) << " ";
    std::cout << std::endl;
#endif
  }

  return _c;
}

/* Initial hidden states: d */
template <typename Ty, std::size_t M>
std::vector<Beliefs<Ty>*>& MDP<Ty,M>::update_D(
                std::vector<Beliefs<Ty>*>& _d,
                Ty eta, unsigned int tt)
{
  for (unsigned int i = 0; i < Nf; i++)
  {
    for (std::size_t j = 0; j != Ns[i]; ++j)
      if (_d[i]->getValue(j) > 0)
        _d[i]->setValue(_d[i]->getValue(j)+_X[i]->getValue(j,tt)*eta, j);

#ifdef DEBUG
    std::cout << "update_D: _d[" << i << "] = ";
    for (unsigned int e = 0; e < Ns[i]; e++)
      std::cout << _d[i]->getValue(e) << " ";
    std::cout << std::endl;
#endif
  }

  return _d;
}
#endif
#endif
