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
#include "tools/optimizer/fusion/reshape_shape_fusion.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace opt {
bool ReshapeShapeFusion::Run(const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "func_graph is a nullptr, cannot do ReshapeShapeFusion.";
    return false;
  }
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNode>(node) || !CheckPrimitiveType(node, prim::kPrimReshape)) {
      continue;
    }
    auto reshape_cnode = node->cast<CNodePtr>();
    if (Process(func_graph, reshape_cnode) != lite::RET_OK) {
      MS_LOG(ERROR) << "Do ReduceStackFusion failed.";
      return false;
    }
  }
  return true;
}

int ReshapeShapeFusion::Process(const FuncGraphPtr &func_graph, const CNodePtr &reshape) {
  MS_ASSERT(func_graph != nullptr && reshape != nullptr);
  if (IsMarkedTrainOp(reshape)) {
    return lite::RET_OK;
  }
  auto prim = GetCNodePrimitive(reshape);
  MS_CHECK_TRUE_MSG(prim != nullptr, lite::RET_NULL_PTR, "Reshape has no primitive.");
  if (IsQuantParameterNode(prim)) {
    return lite::RET_OK;
  }
  if (reshape->size() < kInputSizeThree) {
    return lite::RET_OK;
  }
  auto first_input = reshape->input(1);
  auto second_input = reshape->input(kInputIndexTwo);
  MS_CHECK_TRUE_MSG(second_input != nullptr, lite::RET_NULL_PTR, "Reshape's second-input is a nullptr.");
  if (!utils::isa<CNode>(second_input) || !CheckPrimitiveType(second_input, prim::kPrimShape)) {
    return lite::RET_OK;
  }
  auto shape = second_input->cast<CNodePtr>();
  if (shape->size() < kInputSizeTwo) {
    return lite::RET_OK;
  }
  if (IsMarkedTrainOp(shape)) {
    return lite::RET_OK;
  }
  prim = GetCNodePrimitive(shape);
  MS_CHECK_TRUE_MSG(prim != nullptr, lite::RET_NULL_PTR, "Shape has no primitive.");
  if (IsQuantParameterNode(prim)) {
    return lite::RET_OK;
  }
  auto shape_input = shape->input(1);
  if (first_input != shape_input) {
    return lite::RET_OK;
  }
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_MSG(manager != nullptr, lite::RET_NULL_PTR, "FuncGraph has no manager.");
  if (!manager->Replace(reshape, first_input)) {
    MS_LOG(ERROR) << "ReshapeShapeFusion: do replace failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}
}  // namespace opt
}  // namespace mindspore
