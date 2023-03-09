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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GPU_BATCH_NORM_RELU_GRAD_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GPU_BATCH_NORM_RELU_GRAD_FUSION_H_

#include <memory>
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class BatchNormReluGradFusion : public PatternProcessPass {
 public:
  explicit BatchNormReluGradFusion(bool multigraph = true)
      : PatternProcessPass("batch_norm_relu_grad_fusion", multigraph) {
    dy_ = std::make_shared<Var>();
    y_ = std::make_shared<Var>();
    x_ = std::make_shared<Var>();
    scale_ = std::make_shared<Var>();
    save_mean_ = std::make_shared<Var>();
    save_var_ = std::make_shared<Var>();
    reserve_ = std::make_shared<Var>();
  }
  ~BatchNormReluGradFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  VarPtr dy_;
  VarPtr y_;
  VarPtr x_;
  VarPtr scale_;
  VarPtr save_mean_;
  VarPtr save_var_;
  VarPtr reserve_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GPU_BATCH_NORM_RELU_GRAD_FUSION_H_
