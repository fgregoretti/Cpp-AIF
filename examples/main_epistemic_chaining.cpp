#include <iostream>
#include <cstdlib>
#include <vector>
#include <stdlib.h>
#include <ctime>
#include <iomanip>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "common.h"
#include "mdp.hpp"
#include "epistemic_chaining.hpp"

void Init_7_5(Grid<int>& grid_, Coord& cue1_pos_,
              std::vector<Coord>& cue2_pos_, Coord& start_pos_,
	      std::vector<Coord>& reward_pos_, unsigned int reward)
{
  std::cout << "epistemic_chaining(7, 5)" << std::endl;
  cue1_pos_ = Coord(0, 2);
  cue2_pos_ = { Coord(2, 4), Coord(3, 3), Coord(3, 1), Coord(2, 0) };
  start_pos_ = Coord(0, 4);
  reward_pos_ = { Coord(5, 3), Coord(5, 1) };
  grid_.SetAllValues(-1);
  grid_(cue1_pos_) = 1;
  for (unsigned int i = 0; i < cue2_pos_.size(); ++i) {
    grid_(cue2_pos_[i]) = 2+i;
  }
  grid_(reward_pos_[reward]) = 100;
  grid_(reward_pos_[1-reward]) = -100;
}

int main(int argc,char *argv[])
{
  if ( (argc > 1) && ((std::string(argv[1]) == "-h") || (std::string(argv[1]) == "--help")) )
  {
    std::cerr << "Usage: " << argv[0] << " <sizex> <sizey> <seed> <temporal horizon> <cue2> <reward>" << std::endl
              << "sizex: map size x; sizey: map size y; temporal horizon: number of total timesteps; "
              << "cue2: <0> cue2 at L1, <1> cue2 at L2, <2> cue2 at L3, <3> cue2 at L4; "
              << "reward: <0> reward on first location, <1> reward on secondo location"
	      << std::endl;

    return 0;
  }

  unsigned int size_x = 0;
  if (argc > 1)
    size_x = atoi(argv[1]);
  std::cout << "size_x=" << size_x << std::endl;

  unsigned int size_y = 0;
  if (argc > 2)
    size_y = atoi(argv[2]);
  std::cout << "size_y=" << size_y << std::endl;

  int seed = 0;
  if (argc > 3)
    seed = atoi(argv[3]);
  std::cout << "seed=" << seed << std::endl;

  unsigned int T = 10;
  if (argc > 4)
    T = atoi(argv[4]);
  std::cout << "T=" << T << std::endl;

  unsigned int cue2 = 0;
  if (argc > 5)
    cue2 = atoi(argv[5]);
  std::cout << "cue2=" << cue2 << std::endl;

  unsigned int reward = 0;
  if (argc > 6)
    reward = atoi(argv[6]);
  std::cout << "reward=" << reward << std::endl;

  Grid<int> grid_(size_x, size_y);
  Coord cue1_pos_;
  std::vector<Coord> cue2_pos_;
  Coord start_position_;
  std::vector<Coord> reward_pos_;

  if (size_x == 7 && size_y == 5) {
    Init_7_5(grid_, cue1_pos_, cue2_pos_, start_position_, reward_pos_, reward);
  } else {
    std::cerr << "grid world(" << size_x << "," << size_y << ") not implemented yet" << std::endl;
    exit(0);
  }

  unsigned int Nu = 4;
  std::vector<int> Ns = NumStates(size_x, size_y, cue2_pos_.size());
  unsigned int Nf = 3;
  unsigned int Ng = 4;
  std::cout << "Ns=[ ";
  for (unsigned int i = 0; i < Ns.size(); ++i)
    std::cout << Ns[i] << " ";
  std::cout << "]" << std::endl;
#ifdef _OPENMP
  #pragma omp parallel
  {
    #pragma omp single
    printf("Num_threads=%d\n", omp_get_num_threads());
  }
#endif

  /* prior beliefs about initial state */
  std::vector<Beliefs<FLOAT_TYPE>*> __D;

  _Beliefs<FLOAT_TYPE> *_d1 = new _Beliefs<FLOAT_TYPE>(Ns[0]);
  _d1->epistemic_chaining_init(start_position_, grid_);
  __D.push_back((Beliefs<FLOAT_TYPE> *) _d1);

  _Beliefs<FLOAT_TYPE> *_d2 = new _Beliefs<FLOAT_TYPE>(Ns[1]);
  _d2->Ones();
  __D.push_back((Beliefs<FLOAT_TYPE> *) _d2);

  _Beliefs<FLOAT_TYPE> *_d3 = new _Beliefs<FLOAT_TYPE>(Ns[2]);
  _d3->Ones();
  __D.push_back((Beliefs<FLOAT_TYPE> *) _d3);
  
  /* true initial state */
  std::vector<States*> __S;

  States *_s1 = new States(T);
  _s1->Zeros();
  int start_state = grid_.CoordToIndex(start_position_);
  _s1->Set(start_state);
  __S.push_back(_s1);

  States *_s2 = new States(T);
  _s2->Zeros();
  _s2->Set(cue2);
  __S.push_back(_s2);

  States *_s3 = new States(T);
  _s3->Zeros();
  _s3->Set(reward);
  __S.push_back(_s3);

  /* transition */
  std::vector<std::vector<Transitions<FLOAT_TYPE>*>> __B;

  std::vector<Transitions<FLOAT_TYPE>*> _b1;
  for (unsigned int a = 0; a < Nu; a++) {
    _Transitions<FLOAT_TYPE> *__b1 = new _Transitions<FLOAT_TYPE>(Ns[0], Ns[0]);
    __b1->epistemic_chaining_init(a, grid_);
    _b1.push_back((Transitions<FLOAT_TYPE> *) __b1);
  }
  __B.push_back(_b1);

  std::vector<Transitions<FLOAT_TYPE>*> _b2;
  Transitions<FLOAT_TYPE> *__b2 = new Transitions<FLOAT_TYPE>(Ns[1], Ns[1]);
  __b2->Eye();
  _b2.push_back(__b2);
  __B.push_back(_b2);

  std::vector<Transitions<FLOAT_TYPE>*> _b3;
  Transitions<FLOAT_TYPE> *__b3 = new Transitions<FLOAT_TYPE>(Ns[2], Ns[2]);
  __b3->Eye();
  _b3.push_back(__b3);
  __B.push_back(_b3);

  /* likelihood */
  std::vector<std::vector<likelihood<FLOAT_TYPE,4>*>> __A;

  std::vector<likelihood<FLOAT_TYPE,4>*> _a1;
  _likelihood<FLOAT_TYPE,4> *__a1 = new _likelihood<FLOAT_TYPE,4>(Ns[0],Ns[0],Ns[1],Ns[2]);
  __a1->Observe(Ns);
  _a1.push_back((likelihood<FLOAT_TYPE,4> *) __a1);
  __A.push_back(_a1);

  std::vector<likelihood<FLOAT_TYPE,4>*> _a2;
  _likelihood<FLOAT_TYPE,4> *__a2 = new _likelihood<FLOAT_TYPE,4>(5,Ns[0],Ns[1],Ns[2]);
  __a2->Observe(Ns, grid_, cue1_pos_, cue2_pos_);
  _a2.push_back((likelihood<FLOAT_TYPE,4> *) __a2);
  __A.push_back(_a2);

  std::vector<likelihood<FLOAT_TYPE,4>*> _a3;
  _likelihood<FLOAT_TYPE,4> *__a3 = new _likelihood<FLOAT_TYPE,4>(3,Ns[0],Ns[1],Ns[2]);
  __a3->Observe(Ns, grid_, cue2_pos_);
  _a3.push_back((likelihood<FLOAT_TYPE,4> *) __a3);
  __A.push_back(_a3);

  std::vector<likelihood<FLOAT_TYPE,4>*> _a4;
  _likelihood<FLOAT_TYPE,4> *__a4 = new _likelihood<FLOAT_TYPE,4>(3,Ns[0],Ns[1],Ns[2]);
  __a4->Observe(Ns, grid_, reward_pos_, 0.9);
  _a4.push_back((likelihood<FLOAT_TYPE,4> *) __a4);
  __A.push_back(_a4);

  /* priors */
  std::vector<Priors<FLOAT_TYPE>*> __C;

  std::vector<FLOAT_TYPE> C1(Ns[0]);
  std::fill(C1.begin(), C1.end(), 0); //1./Ns[0]);
  softmax<FLOAT_TYPE>(C1);
  Priors<FLOAT_TYPE>* _C1 = new Priors<FLOAT_TYPE>(C1);
  __C.push_back(_C1);

  std::vector<FLOAT_TYPE> C2(5);
  std::fill(C2.begin(), C2.end(), 0); //1./5);
  softmax<FLOAT_TYPE>(C2);
  Priors<FLOAT_TYPE>* _C2 = new Priors<FLOAT_TYPE>(C2);
  __C.push_back(_C2);

  std::vector<FLOAT_TYPE> C3(3);
  std::fill(C3.begin(), C3.end(), 0); //1./3);
  softmax<FLOAT_TYPE>(C3);
  Priors<FLOAT_TYPE>* _C3 = new Priors<FLOAT_TYPE>(C3);
  __C.push_back(_C3);

  std::vector<FLOAT_TYPE> C4(3);
  std::fill(C4.begin(), C4.end(), 0);
  C4[1] = 2.0; /* make the agent want to encounter the "Cheese" observation level */
  C4[2] = -4.0; /* make the agent not want to encounter the "Shock" observation level */
  softmax<FLOAT_TYPE>(C4);
  Priors<FLOAT_TYPE>* _C4 = new Priors<FLOAT_TYPE>(C4);
  __C.push_back(_C4);

  std::vector<std::vector<int>> V;

  unsigned int N = 4;
  unsigned int policyDepth = 4;

  time_t start, end;

  time(&start);

  _MDP<FLOAT_TYPE,4> *mdp = new _MDP<FLOAT_TYPE,4>(__D,__S,__B,__A,__C,V,grid_,T,64,4,1./4,1,N,policyDepth,seed);

  time(&end);

  std::cout << "Time taken by MDP constructor is : "
            << difftime(end,start) << " sec " << std::endl; 

  PrintState(start_state, grid_);

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

  delete mdp;

  for (unsigned int f = 0; f < Nf; f++)
    delete __D[f];
  for (unsigned int f = 0; f < Nf; f++)
    delete __S[f];
  for (unsigned int f = 0; f < Nf; f++)
    for (unsigned int a = 0; a < __B[f].size(); a++)
      delete __B[f][a];
  for (unsigned int g = 0; g < Ng; g++)
    for (unsigned int a = 0; a < __A[g].size(); a++)
      delete __A[g][a];
  for (unsigned int g = 0; g < Ng; g++)
    delete __C[g];
}
