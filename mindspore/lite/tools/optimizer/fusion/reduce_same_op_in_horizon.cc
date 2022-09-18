/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#define USE_DEPRECATED_API
#include "tools/optimizer/fusion/reduce_same_op_in_horizon.h"
#include <algorithm>
#include <map>
#include <set>
#include <utility>
#include <vector>
#include "tools/optimizer/common/gllo_utils.h"
#include "include/errorcode.h"

namespace mindspore {
namespace opt {
namespace {
bool CheckValueIsEqual(const ValuePtr &left, const ValuePtr &right) {
  if (left == nullptr || right == nullptr) {
    return false;
  }
  if (left.get() == right.get()) {
    return true;
  }
  if (utils::isa<tensor::Tensor>(left) && utils::isa<tensor::Tensor>(right)) {
    auto left_tensor = left->cast<tensor::TensorPtr>();
    auto right_tensor = right->cast<tensor::TensorPtr>();
    return left_tensor->tensor::MetaTensor::operator==(*right_tensor) &&
           left_tensor->data_ptr()->equals(*right_tensor->data_ptr());
  }
  return *left == *right;
}
}  // namespace

bool ReduceSameOpInHorizon::CheckCNodeIsEqual(const CNodePtr &left, const CNodePtr &right) {
  if (left == nullptr || right == nullptr) {
    return false;
  }
  if (left.get() == right.get()) {
    return true;
  }
  if (left->size() != right->size()) {
    return false;
  }
  for (size_t i = 0; i < left->size(); ++i) {
    auto left_input = left->input(i);
    auto right_input = right->input(i);
    if (left_input.get() == right_input.get()) {
      continue;
    }
    ValuePtr left_value{nullptr};
    ValuePtr right_value{nullptr};
    if (utils::isa<Parameter>(left_input) && utils::isa<Parameter>(right_input)) {
      if (param_->train_model) {
        return false;
      }
      left_value = left_input->cast<ParameterPtr>()->default_param();
      right_value = right_input->cast<ParameterPtr>()->default_param();
    }
    if (utils::isa<ValueNode>(left_input) && utils::isa<ValueNode>(right_input)) {
      left_value = left_input->cast<ValueNodePtr>()->value();
      right_value = right_input->cast<ValueNodePtr>()->value();
    }
    if (!CheckValueIsEqual(left_value, right_value)) {
      return false;
    }
  }
  return true;
}

int ReduceSameOpInHorizon::Process(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    MS_LOG(ERROR) << "funcGraph's manager is a nullptr.";
    return lite::RET_NULL_PTR;
  }
  auto node_list = TopoSort(func_graph->get_return());
  std::map<AnfNodePtr, size_t> node_with_index;
  for (size_t i = 0; i < node_list.size(); ++i) {
    node_with_index[node_list[i]] = i;
  }
  auto graph_inputs = func_graph->get_inputs();
  std::set<AnfNodePtr> has_replaced;
  for (auto &node : node_list) {
    if ((std::find(graph_inputs.begin(), graph_inputs.end(), node) == graph_inputs.end() &&
         !utils::isa<CNodePtr>(node)) ||
        has_replaced.find(node) != has_replaced.end()) {
      continue;
    }
    auto node_users = manager->node_users()[node];
    std::vector<CNodePtr> post_cnodes_unordered;
    (void)std::transform(node_users.begin(), node_users.end(), std::back_inserter(post_cnodes_unordered),
                         [&node_list](const std::pair<AnfNodePtr, int> &node_info) {
                           return (node_info.first == nullptr ||
                                   std::find(node_list.begin(), node_list.end(), node_info.first) == node_list.end())
                                    ? nullptr
                                    : node_info.first->cast<CNodePtr>();
                         });
    std::map<size_t, CNodePtr> post_cnodes_inordered;
    for (auto &cnode : post_cnodes_unordered) {
      if (cnode == nullptr) {
        continue;
      }
      post_cnodes_inordered[node_with_index[cnode]] = cnode;
    }
    std::vector<CNodePtr> post_cnodes;
    (void)std::transform(post_cnodes_inordered.begin(), post_cnodes_inordered.end(), std::back_inserter(post_cnodes),
                         [](const std::pair<size_t, CNodePtr> &node_pair) { return node_pair.second; });
    std::vector<int> flags(post_cnodes.size(), 1);
    for (size_t i = 0; i < post_cnodes.size(); ++i) {
      if (!flags[i]) {
        continue;
      }
      auto left = post_cnodes[i];
      for (size_t j = i + 1; j < post_cnodes.size(); ++j) {
        if (!flags[j] || left == post_cnodes[j]) {
          continue;
        }
        auto right = post_cnodes[j];
        if (!CheckCNodeIsEqual(left, right)) {
          continue;
        }
        (void)manager->Replace(right, left);
        has_replaced.insert(right);
        flags[j] = 0;
      }
    }
  }
  return lite::RET_OK;
}

bool ReduceSameOpInHorizon::Run(const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    return false;
  }
  if (param_ == nullptr) {
    MS_LOG(ERROR) << "ConverterPara must be supplied.";
    return false;
  }
  UpdateManager(func_graph);
  auto status = Process(func_graph);
  return status == lite::RET_OK;
}
}  // namespace opt
}  // namespace mindspore
