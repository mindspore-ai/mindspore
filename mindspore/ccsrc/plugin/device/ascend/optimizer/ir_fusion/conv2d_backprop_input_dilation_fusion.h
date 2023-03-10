/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_CONV2D_BACKPROP_INPUT_DILATION_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_CONV2D_BACKPROP_INPUT_DILATION_FUSION_H_

#include <memory>
#include <vector>
#include "include/backend/optimizer/optimizer.h"
#include "plugin/device/ascend/hal/common/platform_info_util.h"

namespace mindspore {
namespace opt {
class Conv2dBackpropInputDilationFusion : public PatternProcessPass {
 public:
  explicit Conv2dBackpropInputDilationFusion(bool multigraph = true)
      : PatternProcessPass("conv2d_backprop_input_dilation_fusion", multigraph) {
    x0_ = std::make_shared<Var>();
    x1_ = std::make_shared<Var>();
    conv2d_bp_input_var_ = std::make_shared<Var>(std::make_shared<Primitive>(prim::kPrimConv2DBackpropInputD->name()));
  }
  ~Conv2dBackpropInputDilationFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 protected:
  AnfNodePtr CreateConv2DbpInput(const FuncGraphPtr &, const AnfNodePtr &, const AnfNodePtr &, const EquivPtr &) const;
  AnfNodePtr CreateDilation(const FuncGraphPtr &, const AnfNodePtr &, const AnfNodePtr &) const;
  AnfNodePtr CreatePad(const FuncGraphPtr &, const AnfNodePtr &, const AnfNodePtr &,
                       const std::vector<AnfNodePtr> &) const;

 private:
  VarPtr x0_;
  VarPtr x1_;
  VarPtr conv2d_bp_input_var_;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_CONV2D_BACKPROP_INPUT_DILATION_FUSION_H_
