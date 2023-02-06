#include <iostream>
#include <cstdlib>
#include <vector>
#include <stdlib.h>
#include <ctime>
#include <iomanip>
#include "common.h"
#include "mdp.hpp"
#include "tiger.hpp"
#include "kron.hpp"

int main(int argc,char *argv[])
{
  if ( (argc > 1) && ((std::string(argv[1]) == "-h") || (std::string(argv[1]) == "--help")) )
  {
    std::cerr << "Usage: " << argv[0] << " <seed> <context>" << std::endl
              << "context: <0> tiger at the left, <1> tiger at the right"
	      << std::endl;

    return 0;
  }

  int seed = 0;
  if (argc > 1)
    seed = atoi(argv[1]);

  /* context
     l = 1;              tiger at the left
     r = 1-l;            tiger at the right */
  unsigned int context = 0;
  if (argc > 2)
    context = atoi(argv[2]);
  std::cout << "context=" << context << std::endl;

  unsigned int T = 3;

  /* prior beliefs about initial state */
  
  /* two factors */
  std::vector<FLOAT_TYPE> D1 = {1., 0., 0., 0.}; /* hidden location states */
  std::vector<FLOAT_TYPE> D2 = {1./2, 1./2}; /* % cue left, cue right */

  std::vector<FLOAT_TYPE> D = kron(D1,D2);
  std::vector<Beliefs<FLOAT_TYPE>*> __D;
  //_Beliefs<FLOAT_TYPE> *Initial_D = new _Beliefs<FLOAT_TYPE>(D1, D2);
  Beliefs<FLOAT_TYPE> *Initial_D = new Beliefs<FLOAT_TYPE>(D);
  //std::cout << "Initial_D size=" << Initial_D->get_size() << std::endl;
  //for (std::size_t i = 0; i < Initial_D->get_size(); i++)
  //  std::cout << "Initial_D->value[" << i << "]=" << Initial_D->value[i] << std::endl;
  //__D.push_back((Beliefs<FLOAT_TYPE> *) Initial_D);
  __D.push_back(Initial_D);

  /* true initial state */
  std::vector<States*> __S;
  States *Initial_S = new States(T);
  Initial_S->Zeros();
  Initial_S->StateSet(context);
  __S.push_back(Initial_S);

  /* controlled transitions: Bu
     ----------------------------------------------------------
      we specify the probabilistic transitions of hidden states
      for each factor. Here, there are four actions taking the
      agent directly to each of the four locations */

  std::vector<std::vector<Transitions<FLOAT_TYPE>*>> __B;

  std::vector<std::vector<FLOAT_TYPE>> eye {
              { 1., 0. },
              { 0., 1. }
          };

  const FLOAT_TYPE h = .9;
  const FLOAT_TYPE k = (1.-h)/3.;

  std::vector<std::vector<FLOAT_TYPE>> B1 {
              { h, k, k, h },
              { k, h, k, k },
              { k, k, h, k },
              { k, k, k, k }
          };

  std::vector<std::vector<FLOAT_TYPE>> K1 = kron(B1,eye);
  //for (std::vector<FLOAT_TYPE> row: K1)
  //{
  //  for (FLOAT_TYPE val: row) {
  //    std::cout << val << " ";
  //  }
  //  std::cout << std::endl;
  //}
  std::vector<Transitions<FLOAT_TYPE>*> _b1;
  Transitions<FLOAT_TYPE> *__b1 = new Transitions<FLOAT_TYPE>(K1);
  _b1.push_back(__b1);

  std::vector<std::vector<FLOAT_TYPE>> B2 {
              { k, k, k, k },
              { h, h, k, h },
              { k, k, h, k },
              { k, k, k, k },
          };

  std::vector<std::vector<FLOAT_TYPE>> K2 = kron(B2,eye);
  //for (std::vector<FLOAT_TYPE> row: K2)
  //{
  //  for (FLOAT_TYPE val: row) {
  //    std::cout << val << " ";
  //  }
  //  std::cout << std::endl;
  //}
  Transitions<FLOAT_TYPE> *__b2 = new Transitions<FLOAT_TYPE>(K2);
  _b1.push_back(__b2);

  std::vector<std::vector<FLOAT_TYPE>> B3 {
              { k, k, k, k },
              { k, h, k, k },
              { h, k, h, h },
              { k, k, k, k },
          };

  std::vector<std::vector<FLOAT_TYPE>> K3 = kron(B3,eye);
  Transitions<FLOAT_TYPE> *__b3 = new Transitions<FLOAT_TYPE>(K3);
  _b1.push_back(__b3);

  std::vector<std::vector<FLOAT_TYPE>> B4 {
              { k, k, k, k },
              { k, h, k, k },
              { k, k, h, k },
              { h, k, k, h },
          };

  std::vector<std::vector<FLOAT_TYPE>> K4 = kron(B4,eye);
  Transitions<FLOAT_TYPE> *__b4 = new Transitions<FLOAT_TYPE>(K4);
  _b1.push_back(__b4);
  __B.push_back(_b1);

  /* outcome probabilities: A
     ------------------------------------------------------
      probabilistic mapping from hidden states to outcomes;
      where outcome can be exteroceptive or interoceptive:
      The exteroceptive outcomes A1 provide cues about
      location and context,
      while interoceptive outcome A2 denotes different
      levels of reward */
  std::vector<std::vector<likelihood<FLOAT_TYPE,2>*>> __A;

  std::vector<std::vector<FLOAT_TYPE>> A1 {
              { 1., 1., 0., 0., 0., 0., 0., 0. },
              { 0., 0., 1., 1., 0., 0., 0., 0. },
              { 0., 0., 0., 0., 1., 1., 0., 0. },
              { 0., 0., 0., 0., 0., 0., 1., 1. }
          };
  std::vector<likelihood<FLOAT_TYPE,2>*> _a1;
  likelihood<FLOAT_TYPE,2> *__a1 = new likelihood<FLOAT_TYPE,2>(A1);
  //for (std::size_t i = 0; i < __a1->get_tnc(); i++)
  //  std::cout << "_a1[" << i << "]=" << (*__a1)[i] << std::endl;
  _a1.push_back(__a1);
  __A.push_back(_a1);

  const FLOAT_TYPE a = .9;
  const FLOAT_TYPE b = 1.-a;

  const FLOAT_TYPE d = 1.;
  const FLOAT_TYPE e = 1.-d; 

  std::vector<std::vector<FLOAT_TYPE>> A2 {
              { .5, .5, 0., 0., 0., 0., d,  e  },
              { .5, .5, 0., 0., 0., 0., e,  d  },
              { 0., 0., b,  a,  a,  b,  0., 0. },
              { 0., 0., a,  b,  b,  a,  0., 0. }
          };
  std::vector<likelihood<FLOAT_TYPE,2>*> _a2;
  likelihood<FLOAT_TYPE,2> *__a2 = new likelihood<FLOAT_TYPE,2>(A2);
  //for (std::size_t i = 0; i < __a2->get_tnc(); i++)
  //  std::cout << "_a2[" << i << "]=" << (*__a2)[i] << std::endl;
  _a2.push_back(__a2);
  __A.push_back(_a2);

  /* priors: (utility) C
     -----------------------------------------------------
      we specify the prior preferences in terms of log
      probabilities over outcomes. Here, the agent prefers
      rewards to losses - and does not like to be exposed */

  std::vector<Priors<FLOAT_TYPE>*> __C;

  std::vector<FLOAT_TYPE> C1 = {1., 1., 1., 1.};
  softmax<FLOAT_TYPE>(C1);
  Priors<FLOAT_TYPE>* Initial_C1 = new Priors<FLOAT_TYPE>(C1);
  __C.push_back(Initial_C1);

  const FLOAT_TYPE c = 2.;
  std::vector<FLOAT_TYPE> C2 = {0., 0., c, -c};
  softmax<FLOAT_TYPE>(C2);
  Priors<FLOAT_TYPE>* Initial_C2 = new Priors<FLOAT_TYPE>(C2);
  __C.push_back(Initial_C2);

  std::vector<std::vector<int>> V {
    { 0,  0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3 },
    { 0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3 },
    { 0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3 }
  };

  time_t start, end;

  time(&start);

  //_MDP<FLOAT_TYPE,2> *mdp = new _MDP<FLOAT_TYPE,2>(__D,__S,__B,__A,__C,V,T,64,4,1./4,1,4,seed);
  MDP<FLOAT_TYPE,2> *mdp = new MDP<FLOAT_TYPE,2>(__D,__S,__B,__A,__C,V,T,64,4,1./4,1,4,seed);

  time(&end);

  std::cout << "Time taken by MDP constructor is : "
            << difftime(end,start) << " sec " << std::endl; 

  time_t start_ats, end_ats;

  time(&start_ats);

  mdp->active_inference();

  time(&end_ats);

  std::cout << "Time taken by active inference is : "
            << difftime(end_ats,start_ats) << " sec "
	    << std::endl; 
  std::cout << "Total Time is : " << difftime(end_ats,start)
            << " sec " << std::endl; 

  std::cout << "=========" << std::endl;

  for (std::size_t i = 0; i < mdp->_st.size(); i++)
    std::cout << i+1 << "," << mdp->getU(i) << ","
              << mdp->_st[i][0] << std::endl;

  delete mdp;

  delete __D[0];
  delete __A[0][0];
  delete __A[1][0];
  delete __B[0][0];
  delete __B[0][1];
  delete __B[0][2];
  delete __B[0][3];
  delete __C[0];
  delete __C[1];
  delete __S[0];
}
