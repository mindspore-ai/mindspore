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

#include "backend/common/pass/insert_tensor_move_for_communication.h"
#include "include/common/utils/anfalgo.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace opt {
constexpr auto kSingleOutput = 1;

bool InsertTensorMoveForCommunication::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(graph);
  MS_EXCEPTION_IF_NULL(kernel_graph);

  std::vector<AnfNodePtr> node_list = TopoSort(graph->get_return());
  for (auto &node : node_list) {
    if (node == nullptr || !common::AnfAlgo::IsFusedCommunicationOp(node)) {
      continue;
    }
    auto communication_op = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(communication_op);
    auto input_num = common::AnfAlgo::GetInputNum(communication_op);
    for (size_t i = 0; i < input_num; ++i) {
      auto input = common::AnfAlgo::GetInputNode(communication_op, i);
      // Need to insert TensorMove in these cases:
      // 1. (Parameter/ValueNode) -> CommunicationOp.
      // 2. (Parameter/ValueNode) -> NopNode -> CommunicationOp.
      // 3. (Parameter/ValueNode) -> RefNode -> CommunicationOp.
      auto real_input_with_index = common::AnfAlgo::VisitKernelWithReturnType(input, 0, true);
      MS_EXCEPTION_IF_NULL(real_input_with_index.first);
      // Skip UMonad op.
      if (HasAbstractMonad(real_input_with_index.first)) {
        continue;
      }
      if (real_input_with_index.first->isa<Parameter>() || real_input_with_index.first->isa<ValueNode>() ||
          kernel_graph->IsInRefOutputMap(real_input_with_index)) {
        auto tensor_move = CreateTensorMoveOp(graph, input);
        FuncGraphManagerPtr manager = graph->manager();
        MS_EXCEPTION_IF_NULL(manager);
        manager->SetEdge(communication_op, SizeToInt(i) + 1, tensor_move);
        MS_LOG(DEBUG) << "Insert Input TensorMove for op " << communication_op->fullname_with_scope();
      }
    }
  }

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->get_param<int>(MS_CTX_MEMORY_OPTIMIZE_LEVEL) == kOptimizeO0) {
    // not use somas
    return true;
  }

  // Need to insert TensorMove if the output of FusedCommunicationOp is GraphOutput
  std::set<AnfNodePtr> candidate_set;
  auto outputs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
  for (const auto &output : outputs) {
    const auto &output_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
    if (!common::AnfAlgo::IsFusedCommunicationOp(output_with_index.first) ||
        common::AnfAlgo::GetOutputTensorNum(output_with_index.first) == kSingleOutput) {
      continue;
    }
    candidate_set.insert(output_with_index.first);
    break;
  }

  for (const auto &node : candidate_set) {
    (void)InsertTensorMoveForGraphOutput(graph, node);
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
