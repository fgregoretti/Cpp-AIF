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
  Ty alpha; /* gamma hyperparametera */
  Ty beta; /* gamma hyperparameter */
  Ty lambda; /* precision update rate */
  Ty gamma;
  unsigned int N; /* number of variational iterations */
  unsigned int seed;
  std::vector<unsigned int> Ns;
  unsigned int Nu;
  std::vector<unsigned int> No; /* number of outcomes */
  std::vector<std::vector<int>> _V;
  std::vector<Beliefs<Ty>*> _lnD;
  std::vector<States*> _S;
  std::vector<std::vector<Transitions<Ty>*> > _B;
  std::vector<std::vector<likelihood<Ty,M>*> > _A;
#ifndef NO_PRECOMPUTE_ALOGA
  std::vector<std::vector<likelihood<Ty,M>*> > _AlogA;
#endif
  std::vector<Priors<Ty>*> _lnC;
  std::vector<likelihood<Ty,M>*> Au;
  std::vector<Beliefs<Ty>*> _X;
  std::vector<States*> _O;
  std::vector<int> U;
  std::mt19937 generator;
  std::vector<std::vector<Ty>> _ut;
  std::vector<std::vector<Ty>> _P;
  std::vector<Ty> _W;
  std::vector<unsigned int> _wt;

public:
  unsigned int Nf;
  unsigned int Ng;
  //unsigned int tt;
  unsigned int Np;
  std::vector<std::vector<int>> _st;
  std::vector<std::vector<int>> _ot;

  MDP(std::vector<Beliefs<Ty>*>& __D, /* initial state probabilities */
      std::vector<States*>& __S, /* true initial state */
      std::vector<std::vector<Transitions<Ty>*>>& __B, /* transition probabilities */
      std::vector<std::vector<likelihood<Ty,M>*>>& __A, /* observation model */
      std::vector<Priors<Ty>*>& __C, /* terminal cost probabilities */
      std::vector<std::vector<int>>& __V, /* policies */
      unsigned int T_ = 10, Ty alpha_ = 8, Ty beta_ = 4,
      Ty lambda_ = 0, Ty gamma_ = 1, unsigned int N_ = 4,
      unsigned int seed_ = 0);

  virtual int get_st(unsigned int f, unsigned int t, int action);
  virtual void logBtimesX(unsigned int f, unsigned int t, std::vector<Ty> &v);
  virtual void marginal_likelihood(unsigned int f, unsigned int t, std::vector<int>& sq, std::vector<Ty>& v);
  void infer_states(unsigned int t);
  std::vector<Ty> infer_policies(unsigned int t);
  int sample_action(unsigned int t);
  void sample_state(unsigned int t, int action);
  void sample_observation(unsigned int t, int action);
  void active_inference();
  int getU(unsigned int t) { return this->U[t]; }

  virtual ~MDP() {
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
  std::vector<Priors<Ty>*>& __C,
  std::vector<std::vector<int>>& __V,
  unsigned int T_, Ty alpha_, Ty beta_,
  Ty lambda_, Ty gamma_, unsigned int N_,
  unsigned int seed_) :
  T(T_),
  alpha(alpha_),
  beta(beta_),
  lambda(lambda_),
  gamma(gamma_),
  N(N_),
  seed(seed_) {
  Ng = __A.size(); /* number of outcome factors */
  Nf = __S.size(); /* number of hidden-states factors */
  Nu = __B[0].size(); /* number of hidden controls */

  if (__V.size() != 0)
  {
    _V = __V;
  }
  Np = __V[0].size();

#ifdef DEBUG
  std::cout << "MDP: Nf=" << Nf << " Ng=" << Ng << " Nu=" << Nu << std::endl;
  for (unsigned int j = 0; j < T; j++)
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
    __D[i]->NormLog();

    //_lnD.push_back(new Beliefs<Ty>(__D[i]));
    _lnD.push_back(__D[i]);

    Ns.push_back(_lnD[i]->get_size());

#ifdef DEBUG
    std::cout << "MDP: _lnD[" << i << "] = ";
    for (unsigned int e = 0; e < Ns[i]; e++)
      std::cout << _lnD[i]->value[e] << " ";
    std::cout << std::endl;
#endif

    /* real state in the world */
    //_S.push_back(new States(*__S[i]));
    _S.push_back(__S[i]);

    s.push_back(_S[i]->StateFind());

    /* transition probabilities (priors) */
    std::vector<Transitions<Ty>*> b1;
    for (unsigned int j = 0; j < Nu; j++)
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
    for (unsigned int j = 0; j < Nu; j++)
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

    __C[g]->NormLog();

    //_lnC.push_back(new Priors<Ty>(*__C[g]));
    _lnC.push_back(__C[g]);

    if ( (__A[g].size() > 1) && (__A[g].size() != Nu) )
    {
      std::cerr << "__A not correctly specified" << std::endl;
      exit(-1);
    }

    for (unsigned int j = 0; j < __A[g].size(); j++)
    {
      if (__A[g][j]->get_order() != Ns.size()+1 )
      {
        std::cerr << "__A not correctly specified" << std::endl;
        exit(-1);
      }

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
 
      //__A[g][j]->Addp0();
      __A[g][j]->Norm();

      //a1.push_back(new likelihood<Ty,M>(*__A[g][j]));
      a1.push_back(__A[g][j]);
#ifndef NO_PRECOMPUTE_ALOGA
      a2.push_back(new likelihood<Ty,M>(__A[g][j]->AlogA()));
#endif
    }

    _A.push_back(a1);
#ifndef NO_PRECOMPUTE_ALOGA
    _AlogA.push_back(a2);
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
      std::cout << _lnC[g]->value[e] << " ";
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

    /* states observed */
    _O.push_back(new States(T));
    _O[g]->StateSet(q);
  }

  U.resize(T, -1);

  _st.resize(T, std::vector<int>(Nf, -1));
  _ot.resize(T, std::vector<int>(Ng, -1));
  _ut.resize(T, std::vector<Ty>(Np, 0));
  _P.resize(T, std::vector<Ty>(Nu, 0));
  _W.resize(T, 0);

  //tt = 0;

  for (unsigned int i = 0; i < Nf; i++)
    _st[0][i] = _S[i]->id[0];

  for (unsigned int g = 0; g < Ng; g++)
    _ot[0][g] = _O[g]->id[0];

  generator.seed(seed);
}

template <typename Ty, std::size_t M>
int MDP<Ty,M>::get_st(unsigned int f, unsigned int t, int action)
{                                                                                                                 
  std::vector<Ty> ps(Ns[f], 0.0);

#ifdef DEBUG
  std::cout << "get_st: t=" << t << " _S[" << f << "]->StateFind(" << t << ")=" << _S[f]->StateFind(t) << std::endl;
#endif
  _B[f][action]->extract_column(_S[f]->StateFind(t),ps);

#ifdef DEBUG
  std::cout << "get_st: ps = ";
  for (Ty val: ps) {
    std::cout << val << " ";
  }
  std::cout << std::endl;
#endif

  return CDFs<Ty>(ps, generateRand());
}

template <typename Ty, std::size_t M>
void MDP<Ty,M>::logBtimesX(unsigned int f, unsigned int t, std::vector<Ty>& v)
{
  _B[f][U[t-1]]->logTxv(&_X[f]->value[(t-1)*Ns[f]], v);
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
    std::cout << "infer_states: g=" << g << " Ag = ";
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
  std::cout << "infer_states: sq = ";
  for (unsigned int val: sq) {
    std::cout << val << " ";
  }
  std::cout << std::endl;
#endif

  //std::vector<unsigned int> _wt;

  if (tt > 0)
  {
    /* retain allowable policies (that are consistent with last action) */
    std::vector<unsigned int> __wt(_wt);

    _wt.clear();

    for(unsigned int jj = 0; jj < __wt.size(); jj++)
      if (_V[tt-1][__wt[jj]] == U[tt-1])
        _wt.push_back(__wt[jj]);

    /* update policy expectations */
    Ty _ut_sum = 0.0;
    for (unsigned int val: _wt)
      _ut_sum += _ut[tt-1][val];
    for (unsigned int val: _wt)
      _ut[tt][val] = _ut[tt-1][val] / _ut_sum;
  }
  else
  {
    /* initialise policy expectations */
    for(unsigned int jj = 0; jj < Np; jj++)
    {
      _wt.push_back(jj);
      _ut[tt][jj] = 1. / Np;
    }
  }
#ifdef DEBUG
  std::cout << "infer_states: _wt = ";
  for (unsigned int val: _wt) {
    std::cout << val << " ";
  }
  std::cout << std::endl;
  for (unsigned int j = 0; j < T; j++)
  {
    std::cout << "infer_states: _ut[" << j << "] = ";
    for (Ty val: _ut[j]) {
      std::cout << val << " ";
    }
    std::cout << std::endl;
  }
#endif

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
        v[ii] = _lnD[i]->value[ii];
#else
        v[ii] += _lnD[i]->value[ii];
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
      _X[i]->value[tt*Ns[i]+j] = v[j];
#ifdef DEBUG
    std::cout << "infer_states: _X[" << i << "] = ";
    for (std::size_t j = 0; j != Ns[i]; ++j)
      std::cout << _X[i]->value[(tt)*Ns[i]+j] << " ";
    std::cout << std::endl;
#endif
  }

  //return _wt;
}

template <typename Ty, std::size_t M>
std::vector<Ty> MDP<Ty,M>::infer_policies(unsigned int tt)
{
  unsigned int Np_t = _wt.size();                                                                               
  std::vector<Ty> G(Np_t, 0.0); 

  for (unsigned int k = 0; k < Np_t; k++)
  {
    Ty **x = new Ty*[Nf];
    for (unsigned int i = 0; i < Nf; i++)
      x[i] = new Ty[Ns[i]];

    for (unsigned int i = 0; i < Nf; i++)
      for (std::size_t j = 0; j != Ns[i]; ++j)
        x[i][j] = _X[i]->value[tt*Ns[i]+j];

    for (unsigned int j = tt; j < T; j++)
    {
      std::cout << "infer_policies: tt=" << tt << " k=" << k << " j=" << j << std::endl;
      /* transition probability from current state */
      for (unsigned int i = 0; i < Nf; i++)
      {
        _B[i][_V[j][_wt[k]]]->Txv(&x[i][0], &x[i][0]);
#ifdef DEBUG                                                                                                      
        std::cout << "infer_policies: x[" << i << "] = ";
        for (std::size_t jj = 0; jj != Ns[i]; ++jj)
          std::cout << x[i][jj] << " ";
        std::cout << std::endl;
#endif
      }

      /* predicted entropy and divergence */
      for (unsigned int g = 0; g < Ng; g++)
      {
        int act_t = (_A[g].size() == 1) ? 0 : _V[j][_wt[k]];
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
            G[k] += (_lnC[g]->value[kk] - log(qo[kk]))*qo[kk]; /* extrinsic value */
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

  Ty b = alpha / gamma;

  /* Variational iterations (assuming precise inference about past action) */
  for (unsigned int it = 0; it < N; it++)
  {
    std::vector<Ty> __ut(Np_t, 0.0);

    /* policy */
    for (unsigned int i = 0; i < Np_t; i++)
      __ut[i] = _W[tt]*G[i];

    softmax<Ty>(__ut);

    for (unsigned int i = 0; i < Np_t; i++)
      _ut[tt][_wt[i]] = __ut[i];

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

  std::cout << std::endl;
  /* posterior expectations (control) */
  for (unsigned int k = tt; k < T; k++)
    for (unsigned int j = 0; j < Nu; j++)
    {
      _P[k][j] = 0.0;

      for (unsigned int i = 0; i < _wt.size(); i++)
       if (_V[k][_wt[i]] == (int) j)
          _P[k][j] += _ut[tt][_wt[i]];
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
  std::cout << std::endl;
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
  int a = std::max_element(_P[tt].begin(),_P[tt].end()) - _P[tt].begin();
#endif
#ifdef DEBUG
  std::cout << "sample_action: a=" << a << std::endl;
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

    _S[i]->id[tt] = _st[tt][i];
  }
}

template <typename Ty, std::size_t M>
void MDP<Ty,M>::sample_observation(unsigned int tt, int action)
{
  for (unsigned int g = 0; g < Ng; g++) {
    std::vector<Ty> po(No[g], 0.0);

    int act_t = (_A[g].size() == 1) ? 0 : action;
    _A[g][act_t]->find(_st[tt], po);

    _ot[tt][g] = CDFs<Ty>(po, generateRand());
#ifdef DEBUG
    std::cout << "sample_observation: _ot[" << tt << "][" << g << "]=" << _ot[tt][g] << std::endl;
#endif

    _O[g]->id[tt] = _ot[tt][g];
  }
}

template <typename Ty, std::size_t M>
void MDP<Ty,M>::active_inference()
{
  unsigned int tt = 0;

  while (tt < T)
  {
#ifdef PRINT
    std::cout << "tt=" << tt << std::endl;
#endif

    infer_states(tt);

    std::vector<Ty> G = infer_policies(tt);

    int a = sample_action(tt);

    if (tt < T-1)
    {
      sample_state(tt+1, a);

      sample_observation(tt+1, a);
    }

    tt += 1;
  }
}
#endif
