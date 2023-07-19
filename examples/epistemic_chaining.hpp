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

#ifndef EPISTEMIC_CHAINING_HPP
#define EPISTEMIC_CHAINING_HPP
#include <string>
#include <cstring>
#include <cmath>
#include "beliefs.hpp"
#include "transitions.hpp"
#include "likelihood.hpp"
#include "priors.hpp"
#include "mdp.hpp"
#include "grid.hpp"

enum {
  NORTH, EAST, SOUTH, WEST
};

const Coord MoveTo[4] { Coord(0, 1), Coord(1, 0),
                        Coord(0, -1), Coord(-1, 0)
};

const std::string MoveString[] = { "North", "East", "South", "West" };
const std::string Cue2String[] = { "L1", "L2", "L3", "L4" };
const std::string Cue1ObsString[] = { "Null", "L1", "L2", "L3", "L4" };
const std::string Cue2ObsString[] = { "Null", "Reward on first location", "Reward on second location" };
const std::string RewardString[] = { "Null", "Cheese", "Shock" };

/* list of dimensionalities of the hidden states */
std::vector<int> NumStates(int size_x, int size_y, int cue2_num_)
{
  std::vector<int> num_states(3);

  num_states[0] = size_x * size_y;
  num_states[1] = cue2_num_;
  num_states[2] = 2;

  return num_states;
}

/* return the state obtained performing action 'action'
from state 'state' */
int NextState(unsigned int state, unsigned int action,
              Grid<int> grid_)
{
  Coord current_pos = grid_.IndexToCoord(state);

  current_pos += MoveTo[action];

  if (grid_.Inside(current_pos)) {
    int next_pos = grid_.CoordToIndex(current_pos);

    return next_pos;
  }
  else
  {
    //std::cout << "current_pos.x= " << current_pos.x << " current_pos.y= " << current_pos.y << std::endl;
    return state;
  }
}

void PrintAction(unsigned int action)
{
  std::cout << "\nAction: ";
  std::cout << MoveString[action] << std::endl;
}

void PrintState(unsigned int state, Grid<int> grid_)
{
  std::cout << "State: " << std::endl;
  for (int x = 0; x < grid_.dimx() + 2; x++)
    std::cout << "# ";
  std::cout << std::endl;

  for (int y = grid_.dimy() - 1; y >= 0; y--) {
    std::cout << "# ";

    for (int x = 0; x < grid_.dimx(); x++) {
      Coord pos(x, y);
      int value = grid_(pos);

      if (grid_.GetCoord(state) == Coord(x, y))
        std::cout << "M ";
      else if (value == 1)
        std::cout << "C1";
      else if (value >= 2 && value < 100)
        std::cout << "L" << value-1;
      else if (value == 100)
        std::cout << "C!";
      else if (value == -100)
        std::cout << "S!";
      else
        std::cout << ". ";
    }

    std::cout << "#" << std::endl;
  }

  for (int x = 0; x < grid_.dimx() + 2; x++)
    std::cout << "# ";
  std::cout << std::endl;
}

template <typename T>
class _Beliefs : public Beliefs<T>
{
public:
  using Beliefs<T>::Beliefs;

  void epistemic_chaining_init(Coord start_pos_,
                               Grid<int> grid_)
  {
    this->Zeros();

    int pos = grid_.CoordToIndex(start_pos_);

    this->setValue(1.0,pos);
  }
};

template <typename Ty>
class _Transitions : public Transitions<Ty>
{
public:
  using Transitions<Ty>::Transitions;

  void epistemic_chaining_init(int action, Grid<int> grid_)
  {
    for(unsigned int i = 0; i < this->Ns; i++)
    {
      this->SetCol(0,i);
      this->SetRowPtr(0,i);
      this->SetData(0.0,i);
    }
    this->SetRowPtr(0,this->Ns);

    unsigned int row[this->Ns];
    memset(row, 0, this->Ns*sizeof(unsigned int));
    unsigned int col_ptr[this->Ns+1];
    memset(col_ptr, 0, (this->Ns+1)*sizeof(unsigned int));

    for (unsigned int s = 0; s < this->Ns; s++)
    {
      col_ptr[s] = s;
      row[s] = NextState(s, action, grid_);
      this->SetData(1,s);
    }
    col_ptr[this->Ns] = this->Ns;

    this->csc_tocsr(col_ptr, row);
  }
};

template <typename T, std::size_t N>
using likelihood = detail::likelihood<T, typename gen_seq<N>::type>;

template <typename T, std::size_t N>
class _likelihood : public detail::likelihood<T, typename gen_seq<N>::type>
{
public:
  using likelihood<T,N>::likelihood;

  /* location observation */
  void Observe(std::vector<int> num_states);
  /* cue1 observation */
  void Observe(std::vector<int> num_states,
               Grid<int> grid_, Coord cue1_location,
               std::vector<Coord> cue2_location);
  /* cue2 observation */
  void Observe(std::vector<int> num_states, Grid<int> grid_,
               std::vector<Coord> cue2_location);
  /* reward observation  */
  void Observe(std::vector<int> num_states, Grid<int> grid_,
               std::vector<Coord> reward_location, T a);
};

/* location observation */
template <typename T, std::size_t N>
void _likelihood<T,N>::Observe(std::vector<int> num_states)
{
  this->Zeros();

  /* make the location observation only depend on the location
     state  */
  for (int s = 0; s < num_states[0]; s++)
    for (int j = 0; j < num_states[1]; ++j)
      for (int k = 0; k < num_states[2]; ++k)
        this->setValue(1,s,s,j,k);
}

/* cue1 observation */
template <typename T, std::size_t N>
void _likelihood<T,N>::Observe(std::vector<int> num_states,
               Grid<int> grid_, Coord cue1_location,
               std::vector<Coord> cue2_location) 
{
  this->Zeros();

  /* make Null the most likely observation everywhere */
  for (int i = 0; i < num_states[0]; ++i)
    for (int j = 0; j < num_states[1]; ++j)
      for (int k = 0; k < num_states[2]; ++k)
        this->setValue(1,0,i,j,k);


  /* make the cue1 signal to be contingent upon both the agent's 
     presence at the cue 1 location and the location of cue2 */
  for (unsigned int i = 0; i < cue2_location.size(); ++i)
    for (int k = 0; k < num_states[2]; ++k)
    {
      this->setValue(0,0,grid_.CoordToIndex(cue1_location),i,k);
      this->setValue(1,i+1,grid_.CoordToIndex(cue1_location),i,k);
    }
}

/* cue2 observation */
template <typename T, std::size_t N>
void _likelihood<T,N>::Observe(std::vector<int> num_states, Grid<int> grid_,
               std::vector<Coord> cue2_location) 
{
  this->Zeros();

  /* make Null the most likely observation everywhere */
  for (int i = 0; i < num_states[0]; ++i)
    for (int j = 0; j < num_states[1]; ++j)
      for (int k = 0; k < num_states[2]; ++k)
        this->setValue(1,0,i,j,k);

  /* if the agent is located at the cue2 location, provide 
       a signal indicating the location of the reward */
  for (unsigned int i = 0; i < cue2_location.size(); ++i)
  {
    int loc_index = grid_.CoordToIndex(cue2_location[i]);

    for (int k = 0; k < num_states[2]; ++k)
      this->setValue(0,0,loc_index,i,k);
    this->setValue(1,1,loc_index,i,0);
    this->setValue(1,2,loc_index,i,1);
  }
}

/* reward observation  */
template <typename T, std::size_t N>
void _likelihood<T,N>::Observe(std::vector<int> num_states, Grid<int> grid_,
               std::vector<Coord> reward_location, T a)
{
  this->Zeros();

  /* make Null the most likely observation everywhere */
  for (int i = 0; i < num_states[0]; ++i)
    for (int j = 0; j < num_states[1]; ++j)
      for (int k = 0; k < num_states[2]; ++k)
        this->setValue(1,0,i,j,k);

  int reward_first_index = grid_.CoordToIndex(reward_location[0]);
  int reward_second_index = grid_.CoordToIndex(reward_location[1]);

  /* fill out the contingences arising when the agent is located
     in the reward location identified as 'first' */
  for (int j = 0; j < num_states[1]; ++j)
  {
    this->setValue(a,1,reward_first_index,j,0);
    this->setValue((1-a)/2,1,reward_first_index,j,1);
    this->setValue(a,2,reward_first_index,j,1);
    this->setValue((1-a)/2,2,reward_first_index,j,0);
    for (int k = 0; k < num_states[2]; ++k)
      this->setValue((1-a)/2,0,reward_first_index,j,k);
  }

  /* fill out the contingences arising when the agent is located
     in the reward location identified as 'second' */
  for (int j = 0; j < num_states[1]; ++j)
  {
    this->setValue(a,1,reward_second_index,j,1);
    this->setValue((1-a)/2,1,reward_second_index,j,0);
    this->setValue(a,2,reward_second_index,j,0);
    this->setValue((1-a)/2,2,reward_second_index,j,1);
    for (int k = 0; k < num_states[2]; ++k)
      this->setValue((1-a)/2,0,reward_second_index,j,k);
  }
}

template <typename Ty, std::size_t M>
class _MDP : public MDP<Ty,M>
{
public:
  _MDP(std::vector<Beliefs<Ty>*>& __D, /* initial state probabilities */
       std::vector<States*>& __S, /* true initial state */
       std::vector<std::vector<Transitions<Ty>*>>& __B, /* transition probabilities */
       std::vector<std::vector<likelihood<Ty,M>*>>& __A, /* observation model */
       std::vector<Priors<Ty>*>& __C, /* terminal cost probabilities */
       std::vector<std::vector<int>>& __V, /* policies */
       Grid<int>& grid_,
       unsigned int T_ = 10, Ty alpha_ = 8, Ty beta_ = 4,
       Ty lambda_ = 0, Ty gamma_ = 1, unsigned int N_ = 4,
#ifndef FULL
       unsigned int policy_len_ = 1,
#endif
       unsigned int seed_ = 0)
       : MDP<Ty,M>(__D, __S, __B, __A, __C, __V, T_,
#ifndef FULL
       alpha_, beta_, lambda_, gamma_, N_, policy_len_, seed_),
#else
       alpha_, beta_, lambda_, gamma_, N_, seed_),
#endif
       grid(grid_) {
    };

  Grid<int> grid;

  void active_inference() override
  {
    unsigned int tt = 0;

    while (tt < this->T)
    {
#ifdef PRINT
      std::cout << "active_inference: tt=" << tt << std::endl;
#endif

      this->infer_states(tt);

      /* value of policies (G) */
      std::vector<Ty> G = this->infer_policies(tt);

      /* next action (the action that minimises expected free energy) */
      int a = this->sample_action(tt);
#ifdef PRINT
      PrintAction(a);
#endif

      /* sampling of next state (outcome) */
      if (tt < this->T-1)
      {
        /* next sampled state */
        this->sample_state(tt+1, a);

        /* next observed state */
        this->sample_observation(tt+1, a);

#ifdef PRINT
        PrintState(this->_S[0]->Get(tt+1), grid);
        std::cout << "Reward: " << RewardString[this->_O[3]->Get(tt+1)] << std::endl;
#endif
        if (grid(grid.GetCoord(this->_S[0]->Get(tt+1))) == 100 ||
	    grid(grid.GetCoord(this->_S[0]->Get(tt+1))) == -100)
          break;
      }

      tt += 1;
    }
  }
};
#endif
