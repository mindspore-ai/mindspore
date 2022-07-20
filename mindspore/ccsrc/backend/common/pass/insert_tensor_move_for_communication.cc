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

#include <vector>
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
// Insert TensorMove between Parameter/ValueNode and communication operator.
//  Parameter             Parameter
//    |          =>           |
//  AllReduce             TensorMove
//                            |
//                        AllReduce
bool InsertTensorMoveForCommunication::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
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
      auto real_input = common::AnfAlgo::VisitKernel(input, 0).first;
      MS_EXCEPTION_IF_NULL(real_input);
      if (real_input->isa<Parameter>() || real_input->isa<ValueNode>()) {
        auto tensor_move = CreateTensorMoveOp(graph, input);
        FuncGraphManagerPtr manager = graph->manager();
        MS_EXCEPTION_IF_NULL(manager);
        manager->SetEdge(communication_op, static_cast<int>(i + 1), tensor_move);
        MS_LOG(DEBUG) << "Insert TensorMove for op " << communication_op->fullname_with_scope();
      }
    }
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
