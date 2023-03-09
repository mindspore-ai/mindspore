/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_FUSED_BATCH_NORM_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_FUSED_BATCH_NORM_FUSION_H_

#include <vector>
#include <memory>
#include <string>
#include "include/backend/optimizer/optimizer.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace opt {
class FusedBatchNormFusion : public PatternProcessPass {
 public:
  explicit FusedBatchNormFusion(const std::string &name = "fused_batch_norm_fusion", bool multigraph = true)
      : PatternProcessPass(name, multigraph) {
    data_input0_var_ = std::make_shared<Var>();
    data_input1_var_ = std::make_shared<Var>();
    data_input2_var_ = std::make_shared<Var>();
    variable_input0_var_ = std::make_shared<Var>();
    variable_input1_var_ = std::make_shared<Var>();
    constant_input0_var_ = std::make_shared<Var>();
    constant_input1_var_ = std::make_shared<Var>();
    batch_norm_var_ = std::make_shared<Var>(std::make_shared<Primitive>(prim::kPrimBatchNorm->name()));
    assign_sub0_var_ = std::make_shared<Var>(std::make_shared<Primitive>(prim::kPrimAssignSub->name()));
    assign_sub1_var_ = std::make_shared<Var>(std::make_shared<Primitive>(prim::kPrimAssignSub->name()));
    monad0_var_ = std::make_shared<Var>();
    monad1_var_ = std::make_shared<Var>();
  }
  ~FusedBatchNormFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &equiv) const override;

 protected:
  AnfNodePtr CreateBNTrainingReduce(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                    const EquivPtr &equiv) const;
  void GetBNTrainingUpdateInputs(const EquivPtr &equiv, const std::vector<AnfNodePtr> &bn_training_reduce_outputs,
                                 std::vector<AnfNodePtr> *const bn_training_update_inputs) const;
  void GetBNTrainingUpdateAbstractList(const EquivPtr &equiv, const AnfNodePtr &bn,
                                       std::vector<AbstractBasePtr> *const abstract_list) const;
  AnfNodePtr CreateBNTrainingUpdate(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &equiv,
                                    const std::vector<AnfNodePtr> &bn_training_reduce_outputs) const;
  ValuePtr GetFactor(const EquivPtr &equiv) const;
  void EliminateMonadNodes(const FuncGraphPtr &func_graph, const EquivPtr &equiv) const;

  VarPtr data_input0_var_;
  VarPtr data_input1_var_;
  VarPtr data_input2_var_;
  VarPtr variable_input0_var_;
  VarPtr variable_input1_var_;
  VarPtr constant_input0_var_;
  VarPtr constant_input1_var_;
  VarPtr batch_norm_var_;
  VarPtr assign_sub0_var_;
  VarPtr assign_sub1_var_;
  VarPtr monad0_var_;
  VarPtr monad1_var_;
};

class FusedBatchNormMixPrecisionFusion0 : public FusedBatchNormFusion {
 public:
  explicit FusedBatchNormMixPrecisionFusion0(bool multigraph = true)
      : FusedBatchNormFusion("fused_batch_norm_mix_precision_fusion", multigraph) {}

  ~FusedBatchNormMixPrecisionFusion0() override = default;
  const BaseRef DefinePattern() const override;
};

class FusedBatchNormMixPrecisionFusion1 : public FusedBatchNormFusion {
 public:
  explicit FusedBatchNormMixPrecisionFusion1(bool multigraph = true)
      : FusedBatchNormFusion("fused_batch_norm_mix_precision_fusion", multigraph) {}

  ~FusedBatchNormMixPrecisionFusion1() override = default;
  const BaseRef DefinePattern() const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_FUSED_BATCH_NORM_FUSION_H_
