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

#include "plugin/device/ascend/optimizer/enhancer/getnext_tensor_move_elimination.h"
#include <memory>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "frontend/optimizer/opt.h"

namespace mindspore::opt {
namespace {
constexpr size_t kTensorMoveNextNodeInputSize = 2;
}  // namespace

const BaseRef GetnextTensorMoveElimination::DefinePattern() const {
  auto prim_tensor_move = std::make_shared<Primitive>(kTensorMoveOpName);
  VarPtr x = std::make_shared<SeqVar>();
  VectorRef tensor_move({prim_tensor_move, x});
  return tensor_move;
}

const AnfNodePtr GetnextTensorMoveElimination::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                       const EquivPtr &equiv) const {
  if (graph == nullptr || node == nullptr || equiv == nullptr) {
    return nullptr;
  }
  auto tensor_move_node = node->cast<CNodePtr>();
  if (tensor_move_node == nullptr) {
    return nullptr;
  }

  // 1. tensor move has attr kAttrLabelForInsertStreamActive
  if (!common::AnfAlgo::HasNodeAttr(kAttrLabelForInsertStreamActive, tensor_move_node)) {
    MS_LOG(DEBUG) << "node has no label_for_insert_stream_active attr";
    return nullptr;
  }

  // 2. tensor move's output has only one user next_node
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (manager->node_users().find(tensor_move_node) == manager->node_users().end()) {
    MS_LOG(EXCEPTION) << "tensor move has no output in manager";
  }
  auto next_nodes = manager->node_users()[tensor_move_node];
  if (next_nodes.size() > 1) {
    MS_LOG(DEBUG) << "node's output has more than one users";
    return nullptr;
  }

  // 3. next_node is not nop node, not communicaiton node, not graph output and it has only one input which is tensor
  // move's output
  for (auto &item : next_nodes) {
    MS_EXCEPTION_IF_NULL(item.first);
    auto next_node = item.first->cast<CNodePtr>();
    if (common::AnfAlgo::IsNopNode(next_node)) {
      return nullptr;
    }

    if (common::AnfAlgo::IsCommunicationOp(next_node)) {
      return nullptr;
    }

    auto graph_outputs = common::AnfAlgo::GetAllOutput(graph->output(), {prim::kPrimTupleGetItem});
    auto iter = std::find(graph_outputs.begin(), graph_outputs.end(), next_node);
    if (iter != graph_outputs.end()) {
      return nullptr;
    }

    if (next_node->inputs().size() != kTensorMoveNextNodeInputSize) {
      MS_LOG(DEBUG) << "next node has more than one input";
      return nullptr;
    }
    // add attr label_for_insert_stream_active for next_node
    common::AnfAlgo::SetNodeAttr(kAttrLabelForInsertStreamActive, MakeValue(true), next_node);
  }

  return tensor_move_node->input(1);
}
}  // namespace mindspore::opt
