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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_DYNAMIC_GRU_V2_GRAD_FISSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_DYNAMIC_GRU_V2_GRAD_FISSION_H_

#include <vector>
#include <string>
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class DynamicGRUV2GradFission : public PatternProcessPass {
 public:
  explicit DynamicGRUV2GradFission(bool multigraph = true)
      : PatternProcessPass("dynamic_gru_v2_grad_fission", multigraph) {}
  ~DynamicGRUV2GradFission() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  AnfNodePtr CreateGRUV2HiddenGradCellNode(const FuncGraphPtr &func_graph, const CNodePtr &dynamic_gru_v2_grad_cnode,
                                           const AnfNodePtr &last_gru_hidden_grad_node,
                                           const AnfNodePtr &last_matmul_node, const std::string &gate_order,
                                           const size_t cur_t) const;
  void AddTLoopNode(const FuncGraphPtr &func_graph, const CNodePtr &dynamic_gru_v2_grad_cnode,
                    std::vector<std::vector<AnfNodePtr>> *result_nodes) const;
  AnfNodePtr AddTConcatNode(const FuncGraphPtr &func_graph, const std::vector<AnfNodePtr> &gru_hidden_grad_nodes,
                            size_t concat_output_index) const;
  std::vector<AnfNodePtr> AddGRUHiddenGradNode(const FuncGraphPtr &func_graph,
                                               const CNodePtr &dynamic_gru_v2_grad_cnode) const;
  AnfNodePtr AddHSplitNode(const FuncGraphPtr &func_graph, const CNodePtr &dynamic_gru_v2_grad_cnode) const;
  AnfNodePtr CreateHReshape(const FuncGraphPtr &graph, const AnfNodePtr &node) const;
  AnfNodePtr AddHConcatNode(const FuncGraphPtr &func_graph, const CNodePtr &dynamic_gru_v2_grad_cnode,
                            const AnfNodePtr &splitv) const;
  AnfNodePtr AddDwhMatmulNode(const FuncGraphPtr &func_graph, const AnfNodePtr &dgate_h, const AnfNodePtr &node) const;
  AnfNodePtr CreateDgateHSplitVDNode(const FuncGraphPtr &func_graph, const AnfNodePtr &dgate_h) const;
  AnfNodePtr CreateDgateXConcatDNode(const FuncGraphPtr &func_graph, const AnfNodePtr &split,
                                     const AnfNodePtr &dnt_x) const;
  AnfNodePtr CreateDwxBatchMatMul(const FuncGraphPtr &graph, const AnfNodePtr &node1, const AnfNodePtr &node2) const;
  AnfNodePtr CreateDxtBatchMatMul(const FuncGraphPtr &func_graph, const AnfNodePtr &dgate_concat,
                                  const AnfNodePtr &weight_input, const AnfNodePtr &dx) const;
  AnfNodePtr CreateWBroadcastToDNode(const FuncGraphPtr &graph, const AnfNodePtr &node) const;
  AnfNodePtr CreateDwReduceSumDNode(const FuncGraphPtr &graph, const AnfNodePtr &matmul,
                                    const AnfNodePtr &gru_grad) const;
  AnfNodePtr CreateDbReduceSumDNode(const FuncGraphPtr &graph, const AnfNodePtr &node, const AnfNodePtr &node2) const;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_DYNAMIC_GRU_V2_GRAD_FISSION_H_
