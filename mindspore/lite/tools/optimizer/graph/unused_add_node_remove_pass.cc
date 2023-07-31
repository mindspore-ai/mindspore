/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/graph/unused_add_node_remove_pass.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/lite_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "ops/constant_of_shape.h"
#include "ops/fusion/add_fusion.h"
#include "ops/fusion/mul_fusion.h"
#include "ops/op_utils.h"
#include "src/common/utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kAddInputSize = 3;
constexpr size_t kAddInputIndex2 = 2;
}  // namespace
bool RemoveUnusedAddNodePass::Run(const FuncGraphPtr &func_graph) {
  MS_LOG(INFO) << "RemoveUnusedAddNodePass run.";
  auto node_list = TopoSort(func_graph->get_return());
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return false;
  }
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    if (!CheckPrimitiveType(node, prim::kPrimAddFusion)) {
      MS_LOG(DEBUG) << "node is not AddFusion.";
      continue;
    }
    auto add_inputs = node->cast<CNodePtr>()->inputs();
    if (add_inputs.size() != kAddInputSize) {
      MS_LOG(DEBUG) << "node input size is wrong.";
      continue;
    }
    for (size_t i = 1; i < add_inputs.size(); i++) {
      auto add_input = add_inputs[i];
      if (!utils::isa<CNodePtr>(add_input)) {
        MS_LOG(DEBUG) << "node is not cnode";
        continue;
      }
      if (!CheckPrimitiveType(add_input, prim::kPrimMul) && !CheckPrimitiveType(add_input, prim::kPrimMulFusion)) {
        MS_LOG(DEBUG) << "node is not mul or mulfusion";
        continue;
      }
      auto mul_cnode = add_input->cast<CNodePtr>();
      auto mul_inputs = mul_cnode->inputs();
      for (auto mul_input : mul_inputs) {
        if (!utils::isa<CNodePtr>(mul_input)) {
          MS_LOG(DEBUG) << "node is not Cnode";
          continue;
        }
        if (!CheckPrimitiveType(mul_input, prim::kPrimConstantOfShape)) {
          MS_LOG(DEBUG) << "node is not kPrimConstantOfShape";
          continue;
        }
        auto const_of_shape_cnode = mul_input->cast<CNodePtr>();
        if (const_of_shape_cnode->inputs().empty()) {
          MS_LOG(ERROR) << "inputs is empty.";
          return false;
        }
        auto prim = ops::GetOperator<mindspore::ops::ConstantOfShape>(const_of_shape_cnode->input(0));
        if (prim == nullptr) {
          MS_LOG(ERROR) << "remove add 0 failed.";
          return false;
        }
        auto value = prim->get_value();
        if (value.size() != 1 || value[0] != 0) {
          MS_LOG(DEBUG) << "value is wrong.";
          continue;
        }
        auto node_index = i == 1 ? kAddInputIndex2 : 1;
        MS_LOG(INFO) << "remove node name: " << node->fullname_with_scope()
                     << "remove node input index: " << node_index;
        manager->Replace(node, node->cast<CNodePtr>()->input(node_index));
        MS_LOG(INFO) << "RemoveUnusedAddNodePass success, node name: " << node->fullname_with_scope();
      }
    }
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
