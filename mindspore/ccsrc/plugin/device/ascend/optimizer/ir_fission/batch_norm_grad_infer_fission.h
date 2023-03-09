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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_BATCH_NORM_GRAD_INFER_FISSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_BATCH_NORM_GRAD_INFER_FISSION_H_

#include <memory>
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class BatchNormGradInferFission : public PatternProcessPass {
 public:
  explicit BatchNormGradInferFission(bool multigraph = true)
      : PatternProcessPass("batch_norm_grad_infer_fission", multigraph),
        input0_var_(std::make_shared<Var>()),
        input1_var_(std::make_shared<Var>()),
        input2_var_(std::make_shared<Var>()),
        input3_var_(std::make_shared<Var>()),
        input4_var_(std::make_shared<Var>()) {}
  ~BatchNormGradInferFission() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                           const EquivPtr &equiv) const override;

 private:
  AnfNodePtr CreateBNInferGrad(const FuncGraphPtr &func_graph, const AnfNodePtr &bn_grad, const EquivPtr &equiv) const;
  AnfNodePtr CreateBNTrainingUpdateGrad(const FuncGraphPtr &func_graph, const AnfNodePtr &bn_grad,
                                        const EquivPtr &equiv) const;

  VarPtr input0_var_;
  VarPtr input1_var_;
  VarPtr input2_var_;
  VarPtr input3_var_;
  VarPtr input4_var_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_BATCH_NORM_GRAD_INFER_FISSION_H_
