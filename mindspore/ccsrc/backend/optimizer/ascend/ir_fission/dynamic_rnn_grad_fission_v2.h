/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_DYNAMIC_RNN_GRAD_FISSION_V2_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_DYNAMIC_RNN_GRAD_FISSION_V2_H_

#include <vector>
#include "backend/optimizer/common/optimizer.h"

namespace mindspore {
namespace opt {
class DynamicRnnGradFissionV2 : public PatternProcessPass {
 public:
  explicit DynamicRnnGradFissionV2(bool multigraph = true)
      : PatternProcessPass("dynamic_rnn_grad_fission_v2", multigraph) {}
  ~DynamicRnnGradFissionV2() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  void CreateTLoopNode(const FuncGraphPtr &func_graph, const CNodePtr &dynamic_rnn_grad_cnode,
                       std::vector<std::vector<AnfNodePtr>> *result_nodes) const;
  AnfNodePtr CreateLSTMSPlitV(const FuncGraphPtr &func_graph, const AnfNodePtr &input,
                              const std::vector<std::vector<size_t>> &split_shapes,
                              const std::vector<TypeId> &split_types, const std::vector<int64_t> &size_split,
                              size_t num_split_x) const;
  void CreateTLoopNodeWithEdge(const FuncGraphPtr &func_graph, const CNodePtr &dynamic_rnn_grad_cnode,
                               const std::vector<std::vector<AnfNodePtr>> &result_nodes, size_t num_split_x,
                               std::vector<std::vector<AnfNodePtr>> *loop_node_outputs) const;
  AnfNodePtr AddLSTMInputGradNode(const FuncGraphPtr &func_graph, const CNodePtr &dynamic_rnn_grad_cnode,
                                  std::vector<AnfNodePtr> *outputs) const;
  AnfNodePtr CreateSplitV(const FuncGraphPtr &func_graph, const CNodePtr &dynamic_rnn_grad_cnode) const;
  AnfNodePtr CreateHConcat(const FuncGraphPtr &func_graph, const CNodePtr &dynamic_rnn_grad_cnode,
                           const AnfNodePtr &splitv) const;
  AnfNodePtr CreateConcat(const FuncGraphPtr &func_graph, const CNodePtr &dynamic_rnn_grad_cnode,
                          const AnfNodePtr &h_concat) const;
  AnfNodePtr CreateConcatNodeT1(const FuncGraphPtr &func_graph, const CNodePtr &dynamic_rnn_grad_cnode) const;
  AnfNodePtr CreateBatchMatMul(const FuncGraphPtr &func_graph, const AnfNodePtr &lstm_input_grad,
                               const AnfNodePtr &concat) const;
  AnfNodePtr CreateBatchMatMul2(const FuncGraphPtr &func_graph, const AnfNodePtr &lstm_input_grad,
                                const AnfNodePtr &node) const;
  AnfNodePtr CreateDwReduceSum(const FuncGraphPtr &func_graph, const CNodePtr &dynamic_rnn_grad_cnode,
                               const AnfNodePtr &batch_matmul) const;
  AnfNodePtr CreateDwReshape(const FuncGraphPtr &func_graph, const CNodePtr &dynamic_rnn_grad_cnode,
                             const AnfNodePtr &batch_matmul) const;
  AnfNodePtr CreateValueNode(const FuncGraphPtr &func_graph, const CNodePtr &dynamic_rnn_grad_cnode) const;
  AnfNodePtr CreateDbReduceSum(const FuncGraphPtr &func_graph, const CNodePtr &, const AnfNodePtr &lstm_input_grad,
                               const AnfNodePtr &value_node) const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_DYNAMIC_RNN_GRAD_FISSION_V2_H_
