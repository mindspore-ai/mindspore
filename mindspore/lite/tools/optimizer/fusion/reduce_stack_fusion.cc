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
#include "tools/optimizer/fusion/reduce_stack_fusion.h"
#include <functional>
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/lite_exporter/fetch_content.h"
#include "ops/op_name.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace opt {
bool ReduceStackFusion::Run(const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "func_graph is a nullptr, cannot do ReduceStackFusion.";
    return false;
  }
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNode>(node) || !CheckPrimitiveType(node, prim::kPrimStack)) {
      continue;
    }
    auto stack_cnode = node->cast<CNodePtr>();
    if (Process(func_graph, stack_cnode) != lite::RET_OK) {
      MS_LOG(ERROR) << "Do ReduceStackFusion failed.";
      return false;
    }
  }
  return true;
}

int ReduceStackFusion::Process(const FuncGraphPtr &func_graph, const CNodePtr &stack) {
  MS_ASSERT(func_graph != nullptr && stack != nullptr);
  if (!CheckCanFusion(func_graph, stack)) {
    return lite::RET_OK;
  }
  reduce_prim_->AddAttr(ops::kKeepDims, MakeValue(true));
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    MS_LOG(ERROR) << "Manager is a nullptr.";
    return lite::RET_NULL_PTR;
  }
  if (!manager->Replace(stack, stack->input(1))) {
    MS_LOG(ERROR) << "do Manager-Replace failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

bool ReduceStackFusion::CheckCanFusion(const FuncGraphPtr &func_graph, const CNodePtr &stack) {
  MS_ASSERT(func_graph != nullptr && stack != nullptr);
  if (IsMarkedTrainOp(stack)) {
    return false;
  }
  if (stack->size() != kInputSizeTwo) {
    return false;
  }
  auto prim = GetCNodePrimitive(stack);
  MS_CHECK_TRUE_RET(prim != nullptr, false);
  if (IsQuantParameterNode(prim)) {
    return false;
  }
  auto axis = prim->GetAttr(ops::kAxis) == nullptr ? 0 : GetValue<int64_t>(prim->GetAttr(ops::kAxis));
  auto input_node = stack->input(1);
  if (!utils::isa<CNode>(input_node) || !CheckPrimitiveType(input_node, prim::kPrimReduceFusion)) {
    return false;
  }
  auto reduce = input_node->cast<CNodePtr>();
  return CheckReduce(func_graph, reduce, axis);
}

bool ReduceStackFusion::CheckReduce(const FuncGraphPtr &func_graph, const CNodePtr &reduce, int stack_axis) {
  MS_ASSERT(func_graph != nullptr && reduce != nullptr);
  if (IsMarkedTrainOp(reduce)) {
    return false;
  }
  if (IsMultiOutputTensors(func_graph, reduce)) {
    return false;
  }
  if (reduce->size() < kInputSizeThree || reduce->input(ops::kInputIndex2) == nullptr ||
      utils::isa<CNode>(reduce->input(ops::kInputIndex2))) {
    return false;
  }
  reduce_prim_ = GetCNodePrimitive(reduce);
  MS_CHECK_TRUE_RET(reduce_prim_ != nullptr, false);
  if (IsQuantParameterNode(reduce_prim_)) {
    return false;
  }
  bool keep_dim =
    reduce_prim_->GetAttr(ops::kKeepDims) == nullptr ? false : GetValue<bool>(reduce_prim_->GetAttr(ops::kKeepDims));
  if (keep_dim) {
    return false;
  }
  lite::DataInfo data_info;
  if (lite::FetchConstData(reduce, ops::kInputIndex2, converter::kFmkTypeMs, &data_info, false) != lite::RET_OK) {
    return false;
  }
  if ((data_info.data_type_ != kNumberTypeInt && data_info.data_type_ != kNumberTypeInt32) ||
      data_info.data_ptr_ == nullptr) {
    return false;
  }
  auto num = std::accumulate(data_info.shape_.begin(), data_info.shape_.end(), 1, std::multiplies<>());
  if (num > 1) {
    return false;
  }
  return *(static_cast<int *>(data_info.data_ptr_)) == stack_axis;
}
}  // namespace opt
}  // namespace mindspore
