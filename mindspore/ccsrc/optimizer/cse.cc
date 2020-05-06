/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "optimizer/cse.h"
#include <vector>
#include <set>
#include <unordered_map>
#include "./common.h"

namespace mindspore {
/* namespace to support opt */
namespace opt {
using mindspore::abstract::AbstractBase;
using mindspore::abstract::AbstractFunction;
using mindspore::abstract::AbstractFunctionPtr;

BasePtr AbsOf(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto node_abs = node->abstract();
  // in testcase: TestOptOpt.CSE, node->abstract() is null;
  if (node_abs == nullptr) {
    return kAnyValue;
  }

  return node_abs;
}

bool CSE::BuildOrderGroupAndDoReplace(const FuncGraphManagerPtr manager) const {
  bool changed = false;
  for (FuncGraphPtr fg : manager->func_graphs()) {
    MS_EXCEPTION_IF_NULL(fg);
    std::vector<std::size_t> order_group;
    std::unordered_map<std::size_t, std::vector<AnfNodePtr>> groups;
    std::unordered_map<AnfNodePtr, std::size_t> hashes;

    std::vector<AnfNodePtr> toposet = TopoSort(fg->get_return());
    for (auto node : toposet) {
      MS_EXCEPTION_IF_NULL(node);
      if (hashes.find(node) != hashes.end()) {
        continue;
      }

      std::size_t h = 0;
      if (node->isa<ValueNode>()) {
        ValueNodePtr value_node = node->cast<ValueNodePtr>();
        auto value = value_node->value();
        MS_EXCEPTION_IF_NULL(value);
        h = hash_combine(value->hash(), (AbsOf(value_node)->hash()));
      } else if (node->isa<CNode>()) {
        auto cnode = node->cast<CNodePtr>();
        auto &inputs = cnode->inputs();
        size_t init = 0;
        h = std::accumulate(inputs.begin(), inputs.end(), init, [&hashes](std::size_t hash, const AnfNodePtr &node_in) {
          return hash_combine(hash, hashes[node_in]);
        });
      } else if (node->isa<Parameter>()) {
        h = node->hash();
      } else {
        MS_LOG(ERROR) << "Unknow node type";
      }

      hashes[node] = h;
      if (groups.find(h) == groups.end()) {
        std::vector<AnfNodePtr> innervec({node});
        groups[h] = innervec;
        order_group.emplace_back(h);
      } else {
        groups[h].push_back(node);
      }
    }

    changed = DoReplace(manager, order_group, &groups) || changed;
  }

  return changed;
}

bool CSE::CheckReplace(const AnfNodePtr &main, const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(main);
  MS_EXCEPTION_IF_NULL(node);

  bool replace = false;
  if (main->isa<ValueNode>() && node->isa<ValueNode>()) {
    auto main_value = GetValueNode(main);
    auto node_value = GetValueNode(node);
    replace = (AbsOf(main) == AbsOf(node)) && (*main_value == *node_value);
  } else if (main->isa<CNode>() && node->isa<CNode>()) {
    auto c_main = main->cast<CNodePtr>();
    auto c_node = node->cast<CNodePtr>();
    const auto &inp1 = c_main->inputs();
    const auto &inp2 = c_node->inputs();
    if (inp1.size() == inp2.size()) {
      bool appsame = true;
      for (size_t j = 0; j < inp1.size(); j++) {
        MS_EXCEPTION_IF_NULL(inp1[j]);
        MS_EXCEPTION_IF_NULL(inp2[j]);
        if (!(*inp1[j] == *inp2[j])) {
          // Handle the case of two different Tensor, but with the same value
          if (IsValueNode<tensor::Tensor>(inp1[j]) && IsValueNode<tensor::Tensor>(inp2[j])) {
            auto tensor1 = GetValueNode<tensor::TensorPtr>(inp1[j]);
            auto tensor2 = GetValueNode<tensor::TensorPtr>(inp2[j]);
            if (tensor1->ValueEqual(*tensor2)) {
              continue;
            }
          }
          appsame = false;
          break;
        }
      }
      replace = appsame;
    }
  }
  return replace;
}

bool CSE::DoReplace(const FuncGraphManagerPtr manager, const std::vector<std::size_t> &order_group,
                    std::unordered_map<std::size_t, std::vector<AnfNodePtr>> *groups) const {
  bool changes = false;
  std::set<size_t> clear_set;
  for (auto &h : order_group) {
    std::vector<AnfNodePtr> &group = (*groups)[h];
    // If there are more than 2 node in that group, they may be same common expression can be eliminated.
    if (group.size() > 1) {
      for (size_t k = 0; k < group.size() - 1; k++) {
        AnfNodePtr main = group[k];
        MS_EXCEPTION_IF_NULL(main);

        // When all node in group has been replaced
        // or a valuenode node, skip compare in group
        if ((k + 1 + clear_set.size() == group.size()) || (k > 0 && main->isa<ValueNode>())) {
          break;
        }

        // skip node has been replaced
        if (clear_set.find(k) != clear_set.end()) {
          continue;
        }

        // Compare with rest elements in this group.
        for (size_t i = k + 1; i < group.size(); i++) {
          auto node = group[i];
          MS_EXCEPTION_IF_NULL(node);

          if (clear_set.find(i) != clear_set.end()) {
            continue;
          }
          if (main->func_graph() != node->func_graph()) {
            continue;
          }
          if (CheckReplace(node, main)) {
            changes = true;
            (void)manager->Replace(node, main);
            (void)clear_set.insert(i);
          }
        }
      }
      clear_set.clear();
    }
  }

  return changes;
}

bool CSE::Cse(const FuncGraphPtr root, const FuncGraphManagerPtr manager) const {
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(root);

  return BuildOrderGroupAndDoReplace(manager);
}
}  // namespace opt
}  // namespace mindspore
