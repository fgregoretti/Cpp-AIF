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

#ifndef CONSTRUCT_POLICIES_HPP
#define CONSTRUCT_POLICIES_HPP
#include <iostream>
#include <vector>

void generatePolicies(int t, int n, std::vector<int>& combination,
                      std::vector<std::vector<int>>& _policies)
{
  if (t == 0) {
    _policies.push_back(combination);
    return;
  }

  for (int i = 0; i < n; i++) {
    combination.push_back(i);
    generatePolicies(t - 1, n, combination, _policies);
    combination.pop_back();
  }
}

std::vector<std::vector<int>> construct_policies(int Nu, int policy_len)
{
  std::vector<std::vector<int>> policies(policy_len);

  std::vector<int> combination;
  std::vector<std::vector<int>> _policies;

  generatePolicies(policy_len, Nu, combination, _policies);

  for (int t = 0; t < policy_len; t++)
    for (std::size_t i = 0; i < _policies.size(); i++)
      policies[t].push_back(_policies[i][t]);

  return policies;
}
#endif
