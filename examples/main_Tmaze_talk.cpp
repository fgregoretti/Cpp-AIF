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

#include "mdp.hpp"

const std::string LocationString[] = { "Center", "Left", "Right", "Bottom" };         
const std::string RewardString[] = { "Cue Left", "Cue Right", "Reward!", "No Reward" };                                                                                     
const std::string ActionString[] = { "Move to Center", "Move to Left", "Move to Right", "Move to Bottom" };

int main()
{
  int T = 3;

  std::vector<States*> S;

  States s0(T);
  s0.Zeros();
  s0.Set(0,0);
  S.push_back(&s0);

  States s1(T);
  s1.Zeros();
  s1.Set(1,0);
  S.push_back(&s1);

  std::vector<std::vector<likelihood<double,3>*>> A;

  likelihood<double,3> a0(4,4,2);
  likelihood<double,3> a1(4,4,2);

  a0.Zeros();
  a0(0,0,0)=1; a0(1,1,0)=1; a0(2,2,0)=1; a0(3,3,0)=1;
  a0(0,0,1)=1; a0(1,1,1)=1; a0(2,2,1)=1; a0(3,3,1)=1;

  const double a = .9;
  const double b = 1.-a;

  const double d = 1.;
  const double e = 1.-d; 

  a1.Zeros();
  a1(0,0,0)=0.5; a1(0,3,0)=d; a1(1,0,0)=0.5; a1(1,3,0)=e;
  a1(2,1,0)=a;   a1(2,2,0)=b; a1(3,1,0)=b;   a1(3,2,0)=a;
  a1(0,0,1)=0.5; a1(0,3,1)=e; a1(1,0,1)=0.5; a1(1,3,1)=d;
  a1(2,1,1)=b;   a1(2,2,1)=a; a1(3,1,1)=a;   a1(3,2,1)=b;

  std::vector<likelihood<double,3>*> A0;
  A0.push_back(&a0);
  A.push_back(A0);

  std::vector<likelihood<double,3>*> A1;
  A1.push_back(&a1);
  A.push_back(A1);

  std::vector<std::vector<FLOAT_TYPE>> B0_0 {
              { 1, 0, 0, 1 },
              { 0, 1, 0, 0 },
              { 0, 0, 1, 0 },
              { 0, 0, 0, 0 }
          };

  std::vector<std::vector<FLOAT_TYPE>> B0_1 {
              { 0, 0, 0, 0 },
              { 1, 1, 0, 1 },
              { 0, 0, 1, 0 },
              { 0, 0, 0, 0 },
          };

  std::vector<std::vector<FLOAT_TYPE>> B0_2 {
              { 0, 0, 0, 0 },
              { 0, 1, 0, 0 },
              { 1, 0, 1, 1 },
              { 0, 0, 0, 0 },
          };
          
  std::vector<std::vector<FLOAT_TYPE>> B0_3 {
              { 0, 0, 0, 0 },
              { 0, 1, 0, 0 },
              { 0, 0, 1, 0 },
              { 1, 0, 0, 1 },
          }; 

  Transitions<double> b0(B0_0);
  Transitions<double> b1(B0_1);
  Transitions<double> b2(B0_2);
  Transitions<double> b3(B0_3);

  std::vector<std::vector<double>> eye {
              { 1., 0. },
              { 0., 1. }
          };
  Transitions<double> bb(eye);

  std::vector<std::vector<Transitions<double>*>> B;

  std::vector<Transitions<double>*> B0;
  B0.push_back(&b0);
  B0.push_back(&b1);
  B0.push_back(&b2);
  B0.push_back(&b3);
  B.push_back(B0);

  std::vector<Transitions<double>*> B1;
  B1.push_back(&bb);
  B.push_back(B1);

  std::vector<double> D0 = {1., 0., 0., 0.}; /* hidden location states */
  std::vector<double> D1 = {1./2, 1./2}; /* cue left, cue right */

  Beliefs<double> d0(D0);
  Beliefs<double> d1(D1);

  std::vector<Beliefs<double>*> D;
  D.push_back(&d0);
  D.push_back(&d1);

  std::vector<double> C0 = {1., 1., 1., 1.};
  softmax<double>(C0);
  Priors<double> c0(C0);

  std::vector<double> C1 = {0., 0., 2, -2};
  softmax<double>(C1);
  Priors<double> c1(C1);

  std::vector<Priors<double>*> C;
  C.push_back(&c0);
  C.push_back(&c1);

  std::vector<std::vector<int>> V;

  int seed = 0;
  MDP<double,3> mdp(D,S,B,A,C,V,T,64,4,1./4,1,4,seed);

  mdp.active_inference();

  for (int i = 0; i < T; i++)
    std::cout << "T=" << i+1
              << " Location: [" << LocationString[mdp._st[i][0]] << "] "
              << "Observation: [" << RewardString[mdp._ot[i][1]] << "]"
	      << " Action: [" << ActionString[mdp.getU(i)] << "]"
              << std::endl;
}
