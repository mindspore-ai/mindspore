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
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace opt {
namespace {
bool IsNeedInsertForInput(const AnfNodePtr &communication_op, const AnfNodePtr &input_node, size_t input_index,
                          const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(communication_op);
  MS_EXCEPTION_IF_NULL(input_node);
  MS_EXCEPTION_IF_NULL(kernel_graph);

  // Skip UMonad op.
  if (HasAbstractMonad(input_node)) {
    return false;
  }

  // Need to insert TensorMove in these cases:
  // 1. (Parameter/ValueNode) -> CommunicationOp.
  // 2. (Parameter/ValueNode) -> NopNode -> CommunicationOp.
  // 3. (Parameter/ValueNode) -> RefNode -> CommunicationOp.
  // 4. Backoff node -> CommunicationOp(or Node -> Backoff communicationOp).
  return (input_node->isa<Parameter>() || input_node->isa<ValueNode>() ||
          kernel_graph->IsInRefOutputMap({input_node, input_index}) ||
          (AnfAlgo::FetchDeviceTarget(input_node, kernel_graph.get()) !=
           AnfAlgo::FetchDeviceTarget(communication_op, kernel_graph.get())));
}
}  // namespace

constexpr auto kSingleNum = 1;

bool InsertTensorMoveForCommunication::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(graph);
  MS_EXCEPTION_IF_NULL(kernel_graph);

  std::vector<AnfNodePtr> node_list = TopoSort(graph->get_return());
  std::vector<CNodePtr> communication_op_list;
  for (auto &node : node_list) {
    if (node == nullptr || !common::AnfAlgo::IsCommunicationOp(node)) {
      continue;
    }
    auto communication_op = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(communication_op);
    auto input_num = common::AnfAlgo::GetInputTensorNum(communication_op);
    if (input_num <= kSingleNum) {
      continue;
    }
    (void)communication_op_list.emplace_back(communication_op);
    for (size_t i = 0; i < input_num; ++i) {
      auto input = common::AnfAlgo::GetInputNode(communication_op, i);
      // Need to insert TensorMove in these cases:
      // 1. (Parameter/ValueNode) -> CommunicationOp.
      // 2. (Parameter/ValueNode) -> NopNode -> CommunicationOp.
      // 3. (Parameter/ValueNode) -> RefNode -> CommunicationOp.
      // 4. Backoff node -> CommunicationOp(or Node -> Backoff communicationOp).
      auto real_input_with_index = common::AnfAlgo::VisitKernelWithReturnType(input, 0, true);
      if (IsNeedInsertForInput(communication_op, real_input_with_index.first, real_input_with_index.second,
                               kernel_graph)) {
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

  // Need to insert TensorMove if the output of CommunicationOp is RefNode
  std::set<AnfNodePtr> ref_origin_set;
  for (auto &kv : kernel_graph->GetRefMap()) {
    (void)ref_origin_set.insert(kv.second.first);
  }
  for (auto &communication_op : communication_op_list) {
    auto used_node_list = GetRealNodeUsedList(graph, communication_op);
    for (auto &used_node : (*used_node_list)) {
      if (ref_origin_set.find(used_node.first) == ref_origin_set.end()) {
        continue;
      }
      auto tensor_move = CreateTensorMoveOp(graph, used_node.first);
      FuncGraphManagerPtr manager = graph->manager();
      MS_EXCEPTION_IF_NULL(manager);
      (void)manager->Replace(used_node.first, tensor_move);
    }
  }

  // Need to insert TensorMove if the output of FusedCommunicationOp is GraphOutput
  std::set<AnfNodePtr> candidate_set;
  auto outputs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
  for (const auto &output : outputs) {
    const auto &output_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
    if (!common::AnfAlgo::IsCommunicationOp(output_with_index.first) ||
        AnfAlgo::GetOutputTensorNum(output_with_index.first) <= kSingleNum) {
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
