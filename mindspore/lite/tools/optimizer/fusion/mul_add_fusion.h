/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_PASS_FUSION_MUL_ADD_FUSION_H_
#define MINDSPORE_LITE_SRC_PASS_FUSION_MUL_ADD_FUSION_H_

#include <string>
#include "backend/optimizer/common/optimizer.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace opt {
class MulAddFusion : public PatternProcessPass {
 public:
  explicit MulAddFusion(bool multigraph = true, const std::string &name = "MulAddFusion")
      : PatternProcessPass(name, multigraph) {}
  ~MulAddFusion() override = default;

 private:
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
  bool CheckMulNode(const FuncGraphPtr &func_graph) const;
  bool CheckAddNode() const;
  bool GetMulInputShape() const;
  bool ScaleInputShapeValid() const;

 private:
  mutable AnfNodePtr mul_anode_ = nullptr;
  mutable AnfNodePtr mul_input_anode_ = nullptr;
  mutable AnfNodePtr mul_const_anode_ = nullptr;
  mutable ShapeVector mul_input_shape_;
  mutable AnfNodePtr add_anode_ = nullptr;
  mutable AnfNodePtr add_const_anode_ = nullptr;
  mutable tensor::TensorPtr scale_tensor_ = nullptr;
  mutable tensor::TensorPtr bias_tensor_ = nullptr;
  mutable ActivationType scale_act_type_ = ActivationType::NO_ACTIVATION;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_PASS_FUSION_CONV_ACTIVATION_FUSION_H_
