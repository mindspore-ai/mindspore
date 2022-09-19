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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_SCALE_SCALE_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_SCALE_SCALE_FUSION_H_

#include <vector>
#include <string>
#include "tools/optimizer/common/pattern_process_pass_extends.h"

namespace mindspore::opt {
class ScaleScaleFusion : public LitePatternProcessPass {
 public:
  explicit ScaleScaleFusion(bool multigraph = true, const std::string &name = "ScaleScaleFusion")
      : LitePatternProcessPass(name, multigraph) {}
  ~ScaleScaleFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  bool CheckScaleNode(const CNodePtr &scale_cnode) const;
  int GetInputParamsAndTensors(const CNodePtr &up_scale_cnode, const CNodePtr &down_scale_cnode) const;
  ParameterPtr GenerateNewWeightNode(const FuncGraphPtr &func_graph, const std::string &name) const;
  ParameterPtr GenerateNewBiasNode(const FuncGraphPtr &func_graph, const std::string &name) const;
  tensor::TensorPtr GetMultiplyResultTensorInfo(const tensor::TensorPtr &left_tensor,
                                                const tensor::TensorPtr &right_tensor) const;

 private:
  mutable std::vector<int64_t> scale_input_shape_;
  mutable std::vector<int64_t> expand_shape_;
  mutable tensor::TensorPtr up_weight_tensor_;
  mutable tensor::TensorPtr up_bias_tensor_;
  mutable tensor::TensorPtr down_weight_tensor_;
  mutable tensor::TensorPtr down_bias_tensor_;
  mutable size_t up_scale_axis_;
  mutable size_t down_scale_axis_;
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_SCALE_SCALE_FUSION_H_
