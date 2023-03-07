/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/optimizer/format_type/remove_internal_output.h"
#include <memory>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
bool UsedForOutputOnly(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto &node_users = manager->node_users();
  auto iter = node_users.find(node);
  if (iter == node_users.end()) {
    return false;
  }
  const auto &node_set = iter->second;
  for (const auto &node_index : node_set) {
    if (!common::AnfAlgo::CheckPrimitiveType(node_index.first, prim::kPrimMakeTuple)) {
      return false;
    }
  }
  return true;
}
}  // namespace
const BaseRef RemoveInternalOutputTransOp::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  auto prim = std::make_shared<Primitive>(kTransDataOpName);
  return VectorRef({prim, X});
}

const BaseRef RemoveInternalOutputCast::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  return VectorRef({prim::kPrimCast, X});
}

const AnfNodePtr RemoveInternalOutput::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                               const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimCast) &&
      !common::AnfAlgo::GetBooleanAttr(node, kIsBackendCast)) {
    return nullptr;
  }
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  if (kernel_graph == nullptr) {
    return nullptr;
  }
  if (!kernel_graph->IsUniqueTargetInternalOutput(node, 0)) {
    return nullptr;
  }
  if (!UsedForOutputOnly(func_graph, node)) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  CheckCNodeInputSize(cnode, kTransOpInputTensorNum);
  auto input_node = cnode->input(1);
  if (!common::AnfAlgo::CheckPrimitiveType(input_node, prim::kPrimTupleGetItem)) {
    kernel_graph->ReplaceInternalOutput(node, input_node);
  } else {
    auto tuple_getitem = input_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(tuple_getitem);
    size_t idx = common::AnfAlgo::GetTupleGetItemOutIndex(tuple_getitem);
    AnfNodePtr real_input_node = common::AnfAlgo::GetTupleGetItemRealInput(tuple_getitem);
    kernel_graph->ReplaceInternalOutput(node, real_input_node, 0, idx);
  }
  return input_node;
}
}  // namespace opt
}  // namespace mindspore
