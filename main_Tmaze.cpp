#include <iostream>
#include <cstdlib>
#include <vector>
#include <stdlib.h>
#include <ctime>
#include <iomanip>
#include "common.h"
#include "mdp.hpp"

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

  std::vector<Beliefs<FLOAT_TYPE>*> __D;
  Beliefs<FLOAT_TYPE> *d1 = new Beliefs<FLOAT_TYPE>(D1);
  __D.push_back(d1);
  Beliefs<FLOAT_TYPE> *d2 = new Beliefs<FLOAT_TYPE>(D2);
  __D.push_back(d2);

  /* true initial state */
  std::vector<States*> __S;
  States *s1 = new States(T);
  s1->Zeros();
  s1->StateSet(0);
  __S.push_back(s1);
  States *s2 = new States(T);
  s2->Zeros();
  s2->StateSet(context);
  __S.push_back(s2);

  /* controlled transitions: __B
     ----------------------------------------------------------
      we specify the probabilistic transitions of hidden states
      for each factor. */

  std::vector<std::vector<Transitions<FLOAT_TYPE>*>> __B;

  /* Here, there are four actions taking the
  agent directly to each of the four locations */
  std::vector<Transitions<FLOAT_TYPE>*> _b1;

  //const FLOAT_TYPE h = .9;
  //const FLOAT_TYPE k = (1.-h)/3.;
  const FLOAT_TYPE h = 1.;
  const FLOAT_TYPE k = 0.;

  std::vector<std::vector<FLOAT_TYPE>> B1 {
              { h, k, k, h },
              { k, h, k, k },
              { k, k, h, k },
              { k, k, k, k }
          };

  Transitions<FLOAT_TYPE> *__b1 = new Transitions<FLOAT_TYPE>(B1);
  _b1.push_back(__b1);

  std::vector<std::vector<FLOAT_TYPE>> B2 {
              { k, k, k, k },
              { h, h, k, h },
              { k, k, h, k },
              { k, k, k, k },
          };

  Transitions<FLOAT_TYPE> *__b2 = new Transitions<FLOAT_TYPE>(B2);
  _b1.push_back(__b2);

  std::vector<std::vector<FLOAT_TYPE>> B3 {
              { k, k, k, k },
              { k, h, k, k },
              { h, k, h, h },
              { k, k, k, k },
          };

  Transitions<FLOAT_TYPE> *__b3 = new Transitions<FLOAT_TYPE>(B3);
  _b1.push_back(__b3);

  std::vector<std::vector<FLOAT_TYPE>> B4 {
              { k, k, k, k },
              { k, h, k, k },
              { k, k, h, k },
              { h, k, k, h },
          };

  Transitions<FLOAT_TYPE> *__b4 = new Transitions<FLOAT_TYPE>(B4);
  _b1.push_back(__b4);

  std::vector<Transitions<FLOAT_TYPE>*> _b2;

  std::vector<std::vector<FLOAT_TYPE>> eye {
              { 1., 0. },
              { 0., 1. }
          };
  /* context, which cannot be changed by action */
  for (unsigned int j = 0; j < 4; j++) {
    Transitions<FLOAT_TYPE> *__b2 = new Transitions<FLOAT_TYPE>(eye);
    _b2.push_back(__b2);
  }

  __B.push_back(_b1);
  __B.push_back(_b2);

  /* outcome probabilities: __A
     ------------------------------------------------------
      probabilistic mapping from hidden states to outcomes;
      where outcome can be exteroceptive or interoceptive:
      The exteroceptive outcomes _a1 provide cues about
      location and context,
      while interoceptive outcome _a2 denotes different
      levels of reward */
  std::vector<std::vector<likelihood<FLOAT_TYPE,3>*>> __A;

  std::vector<likelihood<FLOAT_TYPE,3>*> _a1;
  likelihood<FLOAT_TYPE,3> __a1(4,4,2);
  __a1.Zeros();
  /* cue start cue left cue right cue down */
  __a1(0,0,0)=1; __a1(1,1,0)=1; __a1(2,2,0)=1; __a1(3,3,0)=1;
  /* cue start cue left cue right cue down */
  __a1(0,0,1)=1; __a1(1,1,1)=1; __a1(2,2,1)=1; __a1(3,3,1)=1;
  _a1.push_back(&__a1);
  __A.push_back(_a1);

  const FLOAT_TYPE a = .9;
  const FLOAT_TYPE b = 1.-a;

  const FLOAT_TYPE d = 1.;
  const FLOAT_TYPE e = 1.-d; 

  std::vector<likelihood<FLOAT_TYPE,3>*> _a2;
  likelihood<FLOAT_TYPE,3> __a2(4,4,2);
  __a2.Zeros();
  /* CS left CS right reward positive reward negative */
  __a2(0,0,0)=0.5; __a2(0,3,0)=d; __a2(1,0,0)=0.5; __a2(1,3,0)=e;
  __a2(2,1,0)=a;   __a2(2,2,0)=b; __a2(3,1,0)=b;   __a2(3,2,0)=a;
  /* CS left CS right reward positive reward negative */
  __a2(0,0,1)=0.5; __a2(0,3,1)=e; __a2(1,0,1)=0.5; __a2(1,3,1)=d;
  __a2(2,1,1)=b;   __a2(2,2,1)=a; __a2(3,1,1)=a;   __a2(3,2,1)=b;
  _a2.push_back(&__a2);
  __A.push_back(_a2);

  /* priors: (utility) __C
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

  /* policies */
  std::vector<std::vector<int>> V {
    { 0,  0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3 },
    { 0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3 },
    { 0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3 }
  };

  time_t start, end;

  time(&start);

  MDP<FLOAT_TYPE,3> *mdp = new MDP<FLOAT_TYPE,3>(__D,__S,__B,__A,__C,V,T,64,4,1./4,1,4,seed);

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
              << mdp->_st[i][0] << "," <<
	      mdp->_st[i][1] << std::endl;

  delete mdp;

  delete __D[0];
  delete __D[1];
  delete __B[0][0];
  delete __B[0][1];
  delete __B[0][2];
  delete __B[0][3];
  delete __B[1][0];
  delete __B[1][1];
  delete __B[1][2];
  delete __B[1][3];
  delete __C[0];
  delete __C[1];
  delete __S[0];
  delete __S[1];
}
