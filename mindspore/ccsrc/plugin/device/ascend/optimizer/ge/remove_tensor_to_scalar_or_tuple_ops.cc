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

#include "plugin/device/ascend/optimizer/ge/remove_tensor_to_scalar_or_tuple_ops.h"
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <tuple>
#include <algorithm>
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/comm_manager.h"
#include "include/backend/optimizer/helper.h"
#include "ops/array_ops.h"
#include "ops/other_ops.h"
#include "ops/arithmetic_ops.h"
#include "include/backend/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
namespace {
bool IsTensorConverter(const BaseRef &ref) {
  if (utils::isa<AnfNodePtr>(ref)) {
    AnfNodePtr node = utils::cast<AnfNodePtr>(ref);
    MS_EXCEPTION_IF_NULL(node);
    if (IsPrimitive(node, prim::kPrimTensorToTuple) || IsPrimitive(node, prim::kPrimTensorToScalar)) {
      return true;
    }
  }
  return false;
}
}  // namespace

const BaseRef RemoveTensorToScalarOrTupleOps::DefinePattern() const {
  VarPtr resize = std::make_shared<CondVar>(IsTensorConverter);
  VarPtr inputs = std::make_shared<SeqVar>();
  return VectorRef({resize, inputs});
}

const AnfNodePtr RemoveTensorToScalarOrTupleOps::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                         const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  if (kernel_graph->is_dynamic_shape() || !kernel_graph->is_graph_run_mode()) {
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(node);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto tensor_converter = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(tensor_converter);
  auto x = tensor_converter->input(kIndex1);
  MS_EXCEPTION_IF_NULL(x);
  auto node_users = manager->node_users();
  if (node_users.find(tensor_converter) == node_users.end()) {
    return nullptr;
  }
  auto &child_node_users = node_users[tensor_converter];
  for (auto &child_node_user : child_node_users) {
    auto child_node = child_node_user.first->cast<CNodePtr>();
    manager->SetEdge(child_node, GetInputNodeIndex(tensor_converter, child_node) + kSizeOne, x);
  }
  return x;
}
}  // namespace opt
}  // namespace mindspore
