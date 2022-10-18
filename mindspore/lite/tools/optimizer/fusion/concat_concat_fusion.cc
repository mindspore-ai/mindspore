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
#include "tools/optimizer/fusion/concat_concat_fusion.h"
#include <vector>
#include "ir/func_graph.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "ops/op_name.h"

namespace mindspore {
namespace opt {
bool ConcatConcatFusion::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNode>(node) || !CheckPrimitiveType(node, prim::kPrimConcat)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (Process(func_graph, cnode) != lite::RET_OK) {
      MS_LOG(ERROR) << "Do ConcatConcatFusion failed, node name is " << node->fullname_with_scope();
      return false;
    }
  }
  UpdateManager(func_graph);
  return true;
}

int ConcatConcatFusion::Process(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  auto prim = GetCNodePrimitive(cnode);
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Concat's prim is a nullptr, node name is " << cnode->fullname_with_scope();
    return lite::RET_NULL_PTR;
  }
  if (IsQuantParameterNode(prim)) {
    return lite::RET_OK;
  }
  auto axis = prim->GetAttr(ops::kAxis) != nullptr ? GetValue<int64_t>(prim->GetAttr(ops::kAxis)) : 0;
  auto &inputs = cnode->inputs();
  std::vector<AnfNodePtr> new_inputs;
  for (const auto &node : inputs) {
    if (!utils::isa<CNode>(node) || !CheckPrimitiveType(node, prim::kPrimConcat)) {
      new_inputs.push_back(node);
      continue;
    }
    auto pre_concat = node->cast<CNodePtr>();
    if (IsMultiOutputTensors(func_graph, pre_concat)) {
      new_inputs.push_back(node);
      continue;
    }
    auto pre_prim = GetCNodePrimitive(pre_concat);
    if (pre_prim == nullptr) {
      MS_LOG(ERROR) << "Concat's prim is a nullptr, node name is " << pre_concat->fullname_with_scope();
      return lite::RET_NULL_PTR;
    }
    if (IsQuantParameterNode(pre_prim)) {
      new_inputs.push_back(node);
      continue;
    }
    auto pre_axis = pre_prim->GetAttr(ops::kAxis) != nullptr ? GetValue<int64_t>(pre_prim->GetAttr(ops::kAxis)) : 0;
    if (pre_axis != axis) {
      new_inputs.push_back(node);
      continue;
    }
    auto pre_inputs = pre_concat->inputs();
    new_inputs.insert(new_inputs.end(), pre_inputs.begin() + 1, pre_inputs.end());
  }
  cnode->set_inputs(new_inputs);
  return lite::RET_OK;
}
}  // namespace opt
}  // namespace mindspore
