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
#include "tools/optimizer/fusion/remove_transitivity_op.h"
#include <vector>
#include "tools/optimizer/fusion/strided_slice_checker.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "ops/op_name.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace opt {
bool RemoveTransitivityOp::Run(const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "func_graph is a nullptr, cannot do RemoveTransitivityOp.";
    return false;
  }
  auto ret = preprocessor_.Run(func_graph);
  if (ret != lite::RET_OK && ret != lite::RET_NOT_SUPPORT) {
    MS_LOG(ERROR) << "Do dynamic-shape infer failed.";
    return false;
  }
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNode>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (IsMarkedTrainOp(cnode)) {
      continue;
    }
    int status = lite::RET_OK;
    if (CheckPrimitiveType(cnode, prim::kPrimStridedSlice)) {
      status = HandleStridedSlice(func_graph, cnode);
    } else if (CheckPrimitiveType(cnode, prim::kPrimConcat)) {
      status = HandleConcat(func_graph, cnode);
    } else if (CheckPrimitiveType(cnode, prim::kPrimReduceFusion)) {
      status = HandleReduce(func_graph, cnode);
    }
    if (status != lite::RET_OK) {
      MS_LOG(ERROR) << "Do RemoveTransitivityOp failed, node is " << node->fullname_with_scope();
      return false;
    }
  }
  return true;
}

int RemoveTransitivityOp::HandleStridedSlice(const FuncGraphPtr &func_graph, const CNodePtr &strided_slice) {
  MS_ASSERT(func_graph != nullptr && strided_slice != nullptr);
  if (!StridedSliceChecker::CheckCommonInfo(strided_slice)) {
    return lite::RET_OK;
  }
  auto prim = GetCNodePrimitive(strided_slice);
  MS_CHECK_TRUE_MSG(prim != nullptr, lite::RET_NULL_PTR, "StridedSlice's prim is a nullptr.");
  if (IsQuantParameterNode(prim)) {
    return lite::RET_OK;
  }
  std::vector<int> begin;
  auto ret = StridedSliceChecker::GetBegin(strided_slice, &begin);
  if (ret == lite::RET_NOT_SUPPORT) {
    return lite::RET_OK;
  }
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Get Strided_slice's begin failed, node is " << strided_slice->fullname_with_scope();
    return ret;
  }
  std::vector<int> end;
  ret = StridedSliceChecker::GetEnd(strided_slice, &end);
  if (ret == lite::RET_NOT_SUPPORT) {
    return lite::RET_OK;
  }
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Get Strided_slice's end failed, node is " << strided_slice->fullname_with_scope();
    return ret;
  }
  MS_CHECK_TRUE_MSG(begin.size() == end.size(), lite::RET_ERROR, "Strided_slice begin-size is not equal end-size");
  for (size_t i = 0; i < begin.size(); ++i) {
    if (begin[i] != 0 || end[i] != INT_MAX) {
      return lite::RET_OK;
    }
  }
  return DoReplace(func_graph, strided_slice);
}

int RemoveTransitivityOp::HandleConcat(const FuncGraphPtr &func_graph, const CNodePtr &concat) {
  MS_ASSERT(func_graph != nullptr && concat != nullptr);
  auto prim = GetCNodePrimitive(concat);
  MS_CHECK_TRUE_MSG(prim != nullptr, lite::RET_NULL_PTR, "Concat's prim is a nullptr.");
  if (IsQuantParameterNode(prim)) {
    return lite::RET_OK;
  }
  if (concat->size() != kInputSizeTwo || CheckPrimitiveType(concat->input(1), prim::kPrimMakeTuple) ||
      CheckPrimitiveType(concat->input(1), kPrimMakeTupleV2) ||
      CheckPrimitiveType(concat->input(1), prim::kPrimSplit)) {
    return lite::RET_OK;
  }
  return DoReplace(func_graph, concat);
}

int RemoveTransitivityOp::HandleReduce(const FuncGraphPtr &func_graph, const CNodePtr &reduce) {
  MS_ASSERT(func_graph != nullptr && reduce != nullptr);
  auto &shape_container = preprocessor_.GetShapeContainer();
  if (shape_container.find(reduce) == shape_container.end()) {
    return lite::RET_OK;
  }
  auto prim = GetCNodePrimitive(reduce);
  MS_CHECK_TRUE_MSG(prim != nullptr, lite::RET_NULL_PTR, "Reduce's prim is a nullptr.");
  if (!IsReduceModeMeetOutEqualIn(prim)) {
    return lite::RET_OK;
  }
  if (IsQuantParameterNode(prim)) {
    return lite::RET_OK;
  }
  auto attr = prim->GetAttr(ops::kCoeff);
  if (attr != nullptr && fabs(GetValue<float>(attr) - 1.f) > FLT_EPSILON) {
    return lite::RET_OK;
  }
  auto &in_shapes = shape_container.at(reduce).first;
  auto &out_shapes = shape_container.at(reduce).second;
  if (in_shapes.empty() || out_shapes.empty()) {
    return lite::RET_OK;
  }
  if (in_shapes.front() != out_shapes.front()) {
    return lite::RET_OK;
  }
  return DoReplace(func_graph, reduce);
}

int RemoveTransitivityOp::DoReplace(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_MSG(manager != nullptr, lite::RET_NULL_PTR, "Manager is a nullptr.");
  if (!manager->Replace(cnode, cnode->input(1))) {
    MS_LOG(ERROR) << "Do manager-Replace failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}
}  // namespace opt
}  // namespace mindspore
