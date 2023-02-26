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
