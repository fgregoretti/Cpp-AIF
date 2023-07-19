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

#include <iostream>
#include <cstdlib>
#include <vector>
#include <stdlib.h>
#include <cstring>
#include <iomanip>
#include "mdp.hpp"

const std::string LocationString[] = { "Center", "Left", "Right", "Bottom" };
const std::string RewardString[] = { "Cue Left", "Cue Right", "Reward!", "No Reward" };
const std::string ActionString[] = { "Move to Center", "Move to Left", "Move to Right", "Move to Bottom" };

int main(int argc,char *argv[])
{
  if ( (argc > 1) && ((std::string(argv[1]) == "-h") || (std::string(argv[1]) == "--help")) )
  {
    std::cerr << "Usage: " << argv[0] << " <seed> <context>" << std::endl
              << "context: <0> reward at the left, <1> reward at the right"
	      << std::endl;

    return 0;
  }

  int seed = 0;
  if (argc > 1)
    seed = atoi(argv[1]);

  /* context
     l = 1;              reward at the left
     r = 1-l;            reward at the right */
  unsigned int context = 0;
  if (argc > 2)
    context = atoi(argv[2]);
  std::cout << "context=" << context << std::endl;

  unsigned int T = 3;

  /* prior beliefs about initial state */
  
  /* two factors */
  std::vector<FLOAT_TYPE> D0 = {1., 0., 0., 0.}; /* hidden location states */
  std::vector<FLOAT_TYPE> D1 = {1./2, 1./2}; /* cue left, cue right */

  std::vector<Beliefs<FLOAT_TYPE>*> __D;
  Beliefs<FLOAT_TYPE> *d0 = new Beliefs<FLOAT_TYPE>(D0);
  __D.push_back(d0);
  Beliefs<FLOAT_TYPE> *d1 = new Beliefs<FLOAT_TYPE>(D1);
  __D.push_back(d1);

  /* true initial state */
  std::vector<States*> __S;
  States *s0 = new States(T);
  s0->Zeros();
  s0->Set(0);
  __S.push_back(s0);
  States *s1 = new States(T);
  s1->Zeros();
  s1->Set(context);
  __S.push_back(s1);

  /* controlled transitions: __B
     ----------------------------------------------------------
      we specify the probabilistic transitions of hidden states
      for each factor. */

  std::vector<std::vector<Transitions<FLOAT_TYPE>*>> __B;

  /* Here, there are four actions taking the
  agent directly to each of the four locations */
  std::vector<Transitions<FLOAT_TYPE>*> _b0;

  //const FLOAT_TYPE h = .9;
  //const FLOAT_TYPE k = (1.-h)/3.;
  const FLOAT_TYPE h = 1.;
  const FLOAT_TYPE k = 0.;

  std::vector<std::vector<FLOAT_TYPE>> B0_0 {
              { h, k, k, h },
              { k, h, k, k },
              { k, k, h, k },
              { k, k, k, k }
          };

  Transitions<FLOAT_TYPE> *__b0 = new Transitions<FLOAT_TYPE>(B0_0);
  _b0.push_back(__b0);

  std::vector<std::vector<FLOAT_TYPE>> B0_1 {
              { k, k, k, k },
              { h, h, k, h },
              { k, k, h, k },
              { k, k, k, k },
          };

  Transitions<FLOAT_TYPE> *__b1 = new Transitions<FLOAT_TYPE>(B0_1);
  _b0.push_back(__b1);

  std::vector<std::vector<FLOAT_TYPE>> B0_2 {
              { k, k, k, k },
              { k, h, k, k },
              { h, k, h, h },
              { k, k, k, k },
          };

  Transitions<FLOAT_TYPE> *__b2 = new Transitions<FLOAT_TYPE>(B0_2);
  _b0.push_back(__b2);

  std::vector<std::vector<FLOAT_TYPE>> B0_3 {
              { k, k, k, k },
              { k, h, k, k },
              { k, k, h, k },
              { h, k, k, h },
          };

  Transitions<FLOAT_TYPE> *__b3 = new Transitions<FLOAT_TYPE>(B0_3);
  _b0.push_back(__b3);

  std::vector<Transitions<FLOAT_TYPE>*> _b1;

  std::vector<std::vector<FLOAT_TYPE>> eye {
              { 1., 0. },
              { 0., 1. }
          };
  /* context, which cannot be changed by action */
  //for (unsigned int j = 0; j < 4; j++) {
    Transitions<FLOAT_TYPE> *__b = new Transitions<FLOAT_TYPE>(eye);
    _b1.push_back(__b);
  //}

  __B.push_back(_b0);
  __B.push_back(_b1);

  /* outcome probabilities: __A
     ------------------------------------------------------
      probabilistic mapping from hidden states to outcomes;
      where outcome can be exteroceptive or interoceptive:
      The exteroceptive outcomes _a0 provide cues about
      location and context,
      while interoceptive outcome _a1 denotes different
      levels of reward */
  std::vector<std::vector<likelihood<FLOAT_TYPE,3>*>> __A;

  std::vector<likelihood<FLOAT_TYPE,3>*> _a0;
  likelihood<FLOAT_TYPE,3> __a0(4,4,2);

  std::vector<likelihood<FLOAT_TYPE,3>*> _a1;
  likelihood<FLOAT_TYPE,3> __a1(4,4,2);

  __a0.Zeros();
  /* cue start cue left cue right cue down */
  __a0(0,0,0)=1; __a0(1,1,0)=1; __a0(2,2,0)=1; __a0(3,3,0)=1;
  /* cue start cue left cue right cue down */
  __a0(0,0,1)=1; __a0(1,1,1)=1; __a0(2,2,1)=1; __a0(3,3,1)=1;
  _a0.push_back(&__a0);
  __A.push_back(_a0);

  const FLOAT_TYPE a = .9;
  const FLOAT_TYPE b = 1.-a;

  const FLOAT_TYPE d = 1.;
  const FLOAT_TYPE e = 1.-d; 

  __a1.Zeros();
  /* CS left CS right reward positive reward negative */
  __a1(0,0,0)=0.5; __a1(0,3,0)=d; __a1(1,0,0)=0.5; __a1(1,3,0)=e;
  __a1(2,1,0)=a;   __a1(2,2,0)=b; __a1(3,1,0)=b;   __a1(3,2,0)=a;
  /* CS left CS right reward positive reward negative */
  __a1(0,0,1)=0.5; __a1(0,3,1)=e; __a1(1,0,1)=0.5; __a1(1,3,1)=d;
  __a1(2,1,1)=b;   __a1(2,2,1)=a; __a1(3,1,1)=a;   __a1(3,2,1)=b;
  _a1.push_back(&__a1);
  __A.push_back(_a1);

  /* priors: (utility) __C
     -----------------------------------------------------
      we specify the prior preferences in terms of log
      probabilities over outcomes. Here, the agent prefers
      rewards to losses - and does not like to be exposed */

  std::vector<Priors<FLOAT_TYPE>*> __C;

  std::vector<FLOAT_TYPE> C0 = {1., 1., 1., 1.};
  softmax<FLOAT_TYPE>(C0);
  Priors<FLOAT_TYPE>* Initial_C0 = new Priors<FLOAT_TYPE>(C0);
  __C.push_back(Initial_C0);

  const FLOAT_TYPE c = 2.;
  std::vector<FLOAT_TYPE> C1 = {0., 0., c, -c};
  softmax<FLOAT_TYPE>(C1);
  Priors<FLOAT_TYPE>* Initial_C1 = new Priors<FLOAT_TYPE>(C1);
  __C.push_back(Initial_C1);

  /* policies */
  std::vector<std::vector<int>> V {
    { 0,  0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3 },
    { 0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3 },
    { 0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3 }
  };
  //std::vector<std::vector<int>> V;

#ifdef FULL
  MDP<FLOAT_TYPE,3> *mdp = new MDP<FLOAT_TYPE,3>(__D,__S,__B,__A,__C,V,T,64,4,1./4,1,4,seed);
#else
  MDP<FLOAT_TYPE,3> *mdp = new MDP<FLOAT_TYPE,3>(__D,__S,__B,__A,__C,V,T,64,4,1./4,1,4,1,seed);
#endif

  mdp->active_inference();

  for (std::size_t i = 0; i < mdp->_st.size(); i++)
    std::cout << "T=" << i+1
              << " Location: [" << LocationString[mdp->_st[i][0]] << "] "
              << "Observation: [" << RewardString[mdp->_ot[i][1]] << "]"
	      << " Action: [" << ActionString[mdp->getU(i)] << "]"
              << std::endl;

  delete mdp;

  delete __D[0];
  delete __D[1];
  delete __B[0][0];
  delete __B[0][1];
  delete __B[0][2];
  delete __B[0][3];
  delete __B[1][0];
  delete __C[0];
  delete __C[1];
  delete __S[0];
  delete __S[1];
}
