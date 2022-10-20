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
#include "tools/optimizer/fusion/transpose_gather_fusion.h"
#include <algorithm>
#include <set>
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/optimizer/common/format_utils.h"
#include "tools/lite_exporter/fetch_content.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kPermMaxSize = 20;
}
bool TransposeGatherFusion::Run(const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "func_graph is a nullptr, cannot do TransposeGatherFusion.";
    return false;
  }
  auto node_list = TopoSort(func_graph->get_return());
  std::set<AnfNodePtr> has_visited;
  for (auto &node : node_list) {
    if (has_visited.find(node) != has_visited.end()) {
      continue;
    }
    if (!utils::isa<CNode>(node) || !CheckPrimitiveType(node, prim::kPrimTranspose)) {
      continue;
    }
    has_visited.insert(node);
    auto transpose = node->cast<CNodePtr>();
    if (Process(func_graph, transpose, &has_visited) != lite::RET_OK) {
      MS_LOG(ERROR) << "Do TransposeGatherFusion failed.";
      return false;
    }
  }
  return true;
}

int TransposeGatherFusion::Process(const FuncGraphPtr &func_graph, const CNodePtr &transpose,
                                   std::set<AnfNodePtr> *has_visited) {
  MS_ASSERT(func_graph != nullptr && transpose != nullptr && has_visited != nullptr);
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_MSG(manager != nullptr, lite::RET_NULL_PTR, "manager is a nullptr.");
  FindNodes(func_graph, transpose);
  if (gather_nodes_.empty()) {
    return lite::RET_OK;
  }
  if (!CheckCanFused(transpose)) {
    return lite::RET_OK;
  }
  for (size_t i = 0; i < gather_nodes_.size(); ++i) {
    manager->SetEdge(gather_nodes_[i], 1, transpose->input(1));
    *gather_axes_data_ptr_[i] = gather_updated_axes_[i];
    for (const auto &post_transpose : transpose_nodes_[i]) {
      if (!manager->Replace(post_transpose, gather_nodes_[i])) {
        MS_LOG(ERROR) << "do Manager-Replace failed.";
        return lite::RET_ERROR;
      }
      has_visited->insert(post_transpose);
    }
  }
  return lite::RET_OK;
}

void TransposeGatherFusion::FindNodes(const FuncGraphPtr &func_graph, const CNodePtr &transpose) {
  gather_nodes_.clear();
  transpose_nodes_.clear();
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    return;
  }
  auto &node_users = manager->node_users();
  auto FindPostNodes = [&node_users](std::vector<CNodePtr> *post_nodes, const CNodePtr &node,
                                     const PrimitivePtr &required) {
    auto post_node_users = node_users[node];
    for (const auto &post_node_user : post_node_users) {
      auto post_node = post_node_user.first;
      if (!utils::isa<CNode>(post_node) || !CheckPrimitiveType(post_node, required)) {
        return false;
      }
      if (post_node_user.second != 1) {
        return false;
      }
      post_nodes->push_back(post_node->cast<CNodePtr>());
    }
    return true;
  };
  std::vector<CNodePtr> gather_nodes;
  if (!FindPostNodes(&gather_nodes, transpose, prim::kPrimGather) || gather_nodes.empty()) {
    return;
  }
  std::vector<std::vector<CNodePtr>> transpose_nodes(gather_nodes.size());
  for (size_t i = 0; i < gather_nodes.size(); ++i) {
    if (!FindPostNodes(&transpose_nodes[i], gather_nodes[i], prim::kPrimTranspose) || transpose_nodes[i].empty()) {
      return;
    }
  }
  gather_nodes_ = gather_nodes;
  transpose_nodes_ = transpose_nodes;
}

bool TransposeGatherFusion::CheckCanFused(const CNodePtr &transpose) {
  if (!CheckCommonAttr(transpose)) {
    return false;
  }
  std::vector<int> pre_perm;
  if (GetTransposePerm(transpose, &pre_perm) != lite::RET_OK || pre_perm.empty()) {
    return false;
  }
  gather_updated_axes_.clear();
  gather_axes_data_ptr_.clear();
  for (size_t i = 0; i < gather_nodes_.size(); ++i) {
    if (gather_nodes_[i]->size() < kInputSizeFour || gather_nodes_[i]->input(kInputIndexThree) == nullptr ||
        utils::isa<CNode>(gather_nodes_[i]->input(kInputIndexThree))) {
      return false;
    }
    lite::DataInfo data_info;
    if (lite::FetchConstData(gather_nodes_[i], kInputIndexThree, converter::kFmkTypeMs, &data_info, false) !=
        lite::RET_OK) {
      return false;
    }
    if (data_info.data_ptr_ == nullptr ||
        (data_info.data_type_ != kNumberTypeInt && data_info.data_type_ != kNumberTypeInt32)) {
      return false;
    }
    gather_axes_data_ptr_.push_back(static_cast<int *>(data_info.data_ptr_));
    int axis = *(static_cast<int *>(data_info.data_ptr_));
    std::vector<int> post_perm;
    for (const auto &post_transpose : transpose_nodes_[i]) {
      if (GetTransposePerm(post_transpose, &post_perm) != lite::RET_OK || post_perm.empty()) {
        return false;
      }
      if (!CheckIsMatch(pre_perm, post_perm, axis)) {
        return false;
      }
    }
  }
  return true;
}

bool TransposeGatherFusion::CheckCommonAttr(const CNodePtr &transpose) {
  if (IsMarkedTrainOp(transpose)) {
    return false;
  }
  auto prim = GetCNodePrimitive(transpose);
  MS_CHECK_TRUE_RET(prim != nullptr, false);
  if (IsQuantParameterNode(prim)) {
    return false;
  }
  for (auto &gather : gather_nodes_) {
    if (IsMarkedTrainOp(gather)) {
      return false;
    }
    prim = GetCNodePrimitive(gather);
    MS_CHECK_TRUE_RET(prim != nullptr, false);
    if (IsQuantParameterNode(prim)) {
      return false;
    }
  }
  for (const auto &transpose_nodes : transpose_nodes_) {
    for (const auto &transpose_node : transpose_nodes) {
      if (IsMarkedTrainOp(transpose_node)) {
        return false;
      }
      prim = GetCNodePrimitive(transpose_node);
      MS_CHECK_TRUE_RET(prim != nullptr, false);
      if (IsQuantParameterNode(prim)) {
        return false;
      }
    }
  }
  return true;
}

bool TransposeGatherFusion::CheckIsMatch(const std::vector<int> &pre_perm, const std::vector<int> &post_perm,
                                         int axis) {
  if (pre_perm.size() > kPermMaxSize || post_perm.size() > kPermMaxSize) {
    MS_LOG(ERROR) << "Transpose's perm has exceeded limit.";
    return false;
  }
  int in_rank = static_cast<int>(pre_perm.size());
  int out_rank = static_cast<int>(post_perm.size());
  if (out_rank < in_rank - 1) {
    MS_LOG(ERROR) << "Gather may be invalid.";
    return false;
  }
  std::vector<int> flags;
  for (int i = 0; i < in_rank; ++i) {
    flags.push_back(i);
  }
  std::vector<int> first_transform;
  (void)std::transform(pre_perm.begin(), pre_perm.end(), std::back_inserter(first_transform), [&flags](const int dim) {
    return dim >= 0 && dim < static_cast<int>(flags.size()) ? flags[dim] : -1;
  });
  if (std::any_of(first_transform.begin(), first_transform.end(), [](const int val) { return val == -1; })) {
    return false;
  }
  axis = axis < 0 ? axis + in_rank : axis;
  if (axis < 0 || axis >= in_rank) {
    MS_LOG(ERROR) << "Gather's axis is out of range.";
    return false;
  }
  int axis_point = first_transform[axis];
  std::vector<int> radiation;
  if (in_rank == out_rank) {
    radiation.push_back(axis_point);
  } else if (in_rank < out_rank) {
    for (int i = 0; i < out_rank - in_rank; ++i) {
      radiation.push_back(in_rank + i);
    }
  }
  std::vector<int> second_transform(first_transform.begin(), first_transform.begin() + axis);
  (void)second_transform.insert(second_transform.end(), radiation.begin(), radiation.end());
  (void)second_transform.insert(second_transform.end(), first_transform.begin() + axis + 1, first_transform.end());
  if (second_transform.size() != post_perm.size()) {
    return false;
  }
  std::vector<int> third_transform;
  (void)std::transform(
    post_perm.begin(), post_perm.end(), std::back_inserter(third_transform), [&second_transform](const int dim) {
      return dim >= 0 && dim < static_cast<int>(second_transform.size()) ? second_transform[dim] : -1;
    });
  if (std::any_of(third_transform.begin(), third_transform.end(), [](const int val) { return val == -1; })) {
    return false;
  }
  std::vector<int> align_transform(flags.begin(), flags.begin() + axis_point);
  (void)align_transform.insert(align_transform.end(), radiation.begin(), radiation.end());
  (void)align_transform.insert(align_transform.end(), flags.begin() + axis_point + 1, flags.end());
  gather_updated_axes_.push_back(axis_point);
  return third_transform == align_transform;
}
}  // namespace opt
}  // namespace mindspore
