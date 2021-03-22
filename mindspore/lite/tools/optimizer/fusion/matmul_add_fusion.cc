/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "tools/optimizer/fusion/matmul_add_fusion.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t AddInputSize = 3;
constexpr size_t MatMulInputSize = 3;
bool CheckAndGetMatMulIndex(const CNodePtr &cnode, size_t *index) {
  MS_ASSERT(cnode != nullptr);
  MS_ASSERT(index != nullptr);
  if (cnode->size() != AddInputSize) {
    return false;
  }
  size_t matmul_index = 0;
  for (size_t i = 1; i < cnode->size(); ++i) {
    if (CheckPrimitiveType(cnode->input(i), prim::kPrimMatMul)) {
      auto matmul_cnode = cnode->input(i)->cast<CNodePtr>();
      if (matmul_cnode->size() > MatMulInputSize) {
        continue;
      }
      matmul_index = i;
      break;
    }
  }
  if (matmul_index == 0) {
    return false;
  }
  *index = matmul_index;
  return true;
}
}  // namespace

bool MatMulAddFusion::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nulltr);
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNode>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!CheckPrimitiveType(node, prim::kPrimAddFusion) && !CheckPrimitiveType(node, prim::kPrimBiasAdd)) {
      continue;
    }
    size_t index = 0;
    if (!CheckAndGetMatMulIndex(cnode, &index)) {
      continue;
    }
    auto matmul_cnode = cnode->input(index)->cast<CNodePtr>();
    auto bias_node = cnode->input(AddInputSize - index);
    if (!utils::isa<Parameter>(bias_node) || !bias_node->cast<ParameterPtr>()->default_param()) {
      continue;
    }
    matmul_cnode->add_input(bias_node);
    auto manager = func_graph->manager();
    MS_ASSERT(manager != nullptr);
    matmul_cnode->set_fullname_with_scope(node->fullname_with_scope());
    manager->Replace(node, matmul_cnode);
  }
  return false;
}
}  // namespace opt
}  // namespace mindspore
