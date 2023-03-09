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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_SOFTMAX_GRAD_EXT_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_SOFTMAX_GRAD_EXT_FUSION_H_

#include <memory>
#include <string>
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class SoftmaxGradExtFusion : public PatternProcessPass {
 public:
  explicit SoftmaxGradExtFusion(const std::string &name = "softmax_grad_ext_fusion", bool multigraph = true)
      : PatternProcessPass(name, multigraph) {
    input0_ = std::make_shared<Var>();
    input1_ = std::make_shared<Var>();
    input2_ = std::make_shared<Var>();
    sum_var_ = std::make_shared<Var>(std::make_shared<Primitive>(prim::kPrimReduceSumD->name()));
  }
  ~SoftmaxGradExtFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 protected:
  VarPtr input0_;
  VarPtr input1_;
  VarPtr input2_;
  VarPtr sum_var_;
};

class SoftmaxGradExtFusionV2 : public SoftmaxGradExtFusion {
 public:
  explicit SoftmaxGradExtFusionV2(bool multigraph = true)
      : SoftmaxGradExtFusion("softmax_grad_ext_fusion_v2", multigraph) {}
  ~SoftmaxGradExtFusionV2() override = default;
  const BaseRef DefinePattern() const override;
};

class SoftmaxGradExtFusionV3 : public SoftmaxGradExtFusion {
 public:
  explicit SoftmaxGradExtFusionV3(bool multigraph = true)
      : SoftmaxGradExtFusion("softmax_grad_ext_fusion_v3", multigraph) {}
  ~SoftmaxGradExtFusionV3() override = default;
  const BaseRef DefinePattern() const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_SOFTMAX_GRAD_EXT_FUSION_H_
