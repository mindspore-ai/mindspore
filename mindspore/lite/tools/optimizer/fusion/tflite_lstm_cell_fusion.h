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
#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_TFLITE_LSTM_CELL_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_TFLITE_LSTM_CELL_FUSION_H_

#include <vector>
#include <memory>
#include <string>
#include "tools/optimizer/common/pattern_process_pass_extends.h"
#include "include/common/utils/utils.h"
#include "include/errorcode.h"

namespace mindspore {
namespace opt {
class TfliteLstmCellFusion : public LitePatternProcessPass {
 public:
  explicit TfliteLstmCellFusion(const std::string &name = "TfliteLstmCellFusion", bool multigraph = true,
                                int input_length = 0, int var_num = 0, int cond_nodes_num = 0, int cond_cnodes_num = 0,
                                int body_nodes_num = 0, int body_cnodes_num = 0);

  ~TfliteLstmCellFusion() override = default;

  static EquivPtr MatchGraph(const FuncGraphPtr &func_graph, const PrimitiveVarMapPtr &primitive_vars,
                             const AnfNodePtr &pattern);

  static EquivPtr CheckSubGraph(const AnfNodePtr &pattern, const PrimitiveVarMapPtr &primitive_vars,
                                const AnfNodePtr &anf_sub_graph, size_t cnode_num, size_t all_node_num);

  static lite::STATUS SetAbstractTuple(const CNodePtr &cnode, int output_num);

  static CNodePtr CreateOutputGetItem(const FuncGraphPtr &func_graph, const CNodePtr &node, int item_index);

  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

  const BaseRef DefinePattern() const override;

 protected:
  bool Init() const;

  static lite::STATUS GetFloatScalarFromTensorInfo(const AnfNodePtr &tensor_info, float *v);

  static CNodePtr CreateSqueezeNode(const FuncGraphPtr &func_graph, const CNodePtr &input_node,
                                    const std::vector<int> &axis);

  static lite::STATUS AdjustOtherGetItems(const FuncGraphPtr &func_graph, const CNodePtr &while_cnode,
                                          const CNodePtr &lstm_cnode, const CNodePtr &output_get_item);

  static AnfNodePtr GetCondGraphPattern(const PrimitiveVarMapPtr &primitive_vars);

  virtual AnfNodePtr GetBodyGraphPattern(const PrimitiveVarMapPtr &primitive_vars) const;

  virtual CNodePtr CreateLSTMNode(const FuncGraphPtr &func_graph, const EquivPtr &equiv, const EquivPtr &body_equiv,
                                  const std::string &base_name, float zoneout_cell, float zoneout_hidden) const;

 private:
  CNodePtr GetWhileCnode(const AnfNodePtr &cnode) const;
  bool CheckBodyGraph(const EquivPtr &equiv, float *zoneout_cell, float *zoneout_hidden) const;

  static bool CheckReferencedOutputs(const FuncGraphPtr &func_graph, const CNodePtr &while_cnode);

  static lite::STATUS GetConcatedParam(const std::vector<AnfNodePtr> &params, const ParameterPtr &new_param,
                                       bool is_bias);

 protected:
  mutable VarPtr cell_zoneout_old_ = nullptr;
  mutable VarPtr cell_zoneout_new_ = nullptr;
  mutable VarPtr hidden_zoneout_old_ = nullptr;
  mutable VarPtr hidden_zoneout_new_ = nullptr;
  mutable std::vector<VarPtr> while_input_vars_;

 private:
  size_t while_input_var_num_ = 0;
  size_t while_inputs_num_ = 0;
  size_t cond_nodes_num_ = 0;
  size_t cond_cnodes_num_ = 0;
  size_t body_nodes_num_ = 0;
  size_t body_cnodes_num_ = 0;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_TFLITE_LSTM_CELL_FUSION_H_
