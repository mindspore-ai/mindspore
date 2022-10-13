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
#include "tools/optimizer/fusion/strided_slice_fusion.h"
#include <memory>
#include <vector>
#include "tools/optimizer/fusion/strided_slice_checker.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/lite_exporter/fetch_content.h"
#include "ir/func_graph.h"
#include "nnacl/op_base.h"
#include "ops/strided_slice.h"
#include "ops/op_name.h"

namespace mindspore {
namespace opt {
namespace {
bool CheckContinuity(const std::vector<CNodePtr> &nodes, int axis) {
  MS_ASSERT(!nodes.empty());
  for (const auto &node : nodes) {
    if (!StridedSliceChecker::CheckCommonInfo(node)) {
      return false;
    }
  }
  std::vector<int> first_begin;
  if (StridedSliceChecker::GetBegin(nodes.front(), &first_begin) != lite::RET_OK) {
    return false;
  }
  std::vector<int> first_end;
  if (StridedSliceChecker::GetEnd(nodes.front(), &first_end) != lite::RET_OK) {
    return false;
  }
  MS_CHECK_TRUE_RET(first_begin.size() == first_end.size(), false);
  if (axis >= static_cast<int>(first_begin.size())) {
    return false;
  }
  for (size_t i = 1; i < nodes.size(); ++i) {
    std::vector<int> second_begin;
    if (StridedSliceChecker::GetBegin(nodes[i], &second_begin) != lite::RET_OK) {
      return false;
    }
    std::vector<int> second_end;
    if (StridedSliceChecker::GetEnd(nodes[i], &second_end) != lite::RET_OK) {
      return false;
    }
    MS_CHECK_TRUE_RET(second_begin.size() == second_end.size(), false);
    if (second_begin.size() != first_begin.size()) {
      return false;
    }
    for (int j = 0; j < static_cast<int>(first_begin.size()); ++j) {
      if (j == axis) {
        continue;
      }
      if (second_begin[j] != first_begin[j] || second_end[j] != first_end[j]) {
        return false;
      }
    }
    if (second_begin[axis] != first_end[axis]) {
      return false;
    }
    first_begin = second_begin;
    first_end = second_end;
  }
  return true;
}
}  // namespace

bool StridedSliceFusion::Run(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, false, "FuncGraph is a nullptr.");
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    MS_LOG(ERROR) << "The manager of this graph is a nullptr.";
    return false;
  }
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    MS_CHECK_TRUE_RET(node != nullptr, false);
    if (!utils::isa<CNode>(node)) {
      continue;
    }
    if (!CheckPrimitiveType(node, prim::kPrimConcat)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (IsMarkedTrainOp(cnode)) {
      continue;
    }
    auto prim = GetCNodePrimitive(cnode);
    MS_CHECK_TRUE_MSG(prim != nullptr, false, "Concat's prim is a nullptr.");
    axis_ = prim->GetAttr(ops::kAxis) == nullptr ? 0 : static_cast<int>(GetValue<int64_t>(prim->GetAttr(ops::kAxis)));
    if (axis_ < 0) {
      continue;
    }
    if (Process(func_graph, cnode) != lite::RET_OK) {
      MS_LOG(ERROR) << "Do StridedSliceFusion failed.";
      return false;
    }
  }
  UpdateManager(func_graph);
  return true;
}

int StridedSliceFusion::Process(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  FindStridedSliceOp(func_graph, cnode);
  if (!CheckCanFusion()) {
    return lite::RET_OK;
  }
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  for (const auto &nodes : strided_slice_ops_) {
    auto first_node = nodes.front();
    auto end_node = nodes.back();
    manager->SetEdge(first_node, kInputIndexThree, end_node->input(kInputIndexThree));
    for (size_t i = 1; i < nodes.size(); ++i) {
      if (!manager->Replace(nodes[i], NewValueNode(std::make_shared<UMonad>()))) {
        MS_LOG(ERROR) << "Manager Replace strided_slice op with Mond failed.";
        return lite::RET_ERROR;
      }
    }
    auto first_prim = GetCNodePrimitive(first_node);
    MS_ASSERT(first_prim != nullptr);
    auto end_prim = GetCNodePrimitive(end_node);
    MS_ASSERT(end_prim != nullptr);
    first_prim->set_attr(ops::kEndMask, end_prim->GetAttr(ops::kEndMask));
  }
  auto inputs = cnode->inputs();
  std::vector<AnfNodePtr> new_inputs;
  for (const auto &input : inputs) {
    if (utils::isa<ValueNode>(input) && utils::isa<Monad>(input->cast<ValueNodePtr>()->value())) {
      continue;
    }
    new_inputs.push_back(input);
  }
  cnode->set_inputs(new_inputs);
  return lite::RET_OK;
}

void StridedSliceFusion::FindStridedSliceOp(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  strided_slice_ops_.clear();
  AnfNodePtr input{nullptr};
  size_t index = 0;
  for (size_t i = 1; i < cnode->size(); ++i) {
    if (!utils::isa<CNode>(cnode->input(i)) || !CheckPrimitiveType(cnode->input(i), prim::kPrimStridedSlice)) {
      continue;
    }
    auto pre_cnode = cnode->input(i)->cast<CNodePtr>();
    if (pre_cnode->size() != kInputSizeFive) {
      strided_slice_ops_.clear();
      return;
    }
    if (IsMultiOutputTensors(func_graph, pre_cnode)) {
      continue;
    }
    auto input_cur = pre_cnode->input(1);
    if (input_cur == nullptr) {
      strided_slice_ops_.clear();
      return;
    }
    if (input_cur == input && i - index == 1) {
      strided_slice_ops_[strided_slice_ops_.size() - 1].push_back(pre_cnode);
    } else {
      strided_slice_ops_.push_back({pre_cnode});
      input = input_cur;
    }
    index = i;
  }
}

bool StridedSliceFusion::CheckCanFusion() {
  std::vector<std::vector<CNodePtr>> strided_slice_ops = strided_slice_ops_;
  strided_slice_ops_.clear();
  for (auto &nodes : strided_slice_ops) {
    if (nodes.size() <= 1) {
      continue;
    }
    if (CheckContinuity(nodes, axis_)) {
      strided_slice_ops_.push_back(nodes);
    }
  }
  if (strided_slice_ops_.empty()) {
    return false;
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
