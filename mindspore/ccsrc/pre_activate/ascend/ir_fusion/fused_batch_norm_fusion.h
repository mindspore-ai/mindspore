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
#ifndef MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_IR_FUSION_FUSED_BATCH_NORM_FUSION_H_
#define MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_IR_FUSION_FUSED_BATCH_NORM_FUSION_H_

#include <vector>
#include <memory>
#include "pre_activate/common/optimizer.h"

namespace mindspore {
namespace opt {
class FusedBatchNormFusion : public PatternProcessPass {
 public:
  explicit FusedBatchNormFusion(bool multigraph = true)
      : PatternProcessPass("fused_batch_norm_fusion", multigraph),
        data_input_var0_(std::make_shared<Var>()),
        data_input_var1_(std::make_shared<Var>()),
        data_input_var2_(std::make_shared<Var>()),
        variable_input_var0_(std::make_shared<Var>()),
        variable_input_var1_(std::make_shared<Var>()),
        constant_input_var0_(std::make_shared<Var>()),
        constant_input_var1_(std::make_shared<Var>()) {}
  ~FusedBatchNormFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  abstract::AbstractTuplePtr CreateAbstractOfFusedBatchNorm(const EquivPtr &equiv, const AnfNodePtr &bn) const;

  ValuePtr GetFactor(const EquivPtr &equiv) const;

  VarPtr data_input_var0_;
  VarPtr data_input_var1_;
  VarPtr data_input_var2_;
  VarPtr variable_input_var0_;
  VarPtr variable_input_var1_;
  VarPtr constant_input_var0_;
  VarPtr constant_input_var1_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_IR_FUSION_FUSED_BATCH_NORM_FUSION_H_
