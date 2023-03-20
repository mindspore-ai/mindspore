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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_MUL_ADD_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_MUL_ADD_FUSION_H_

#include <memory>
#include <string>
#include <unordered_map>
#include "include/backend/optimizer/optimizer.h"
#include "tools/converter/quantizer/quant_param_holder.h"
#include "tools/optimizer/common/multiple_pattern_process_pass.h"
#include "ops/fusion/scale_fusion.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace opt {
class MulAddFusion : public MultiplePatternProcessPass {
 public:
  explicit MulAddFusion(const std::string &name = "MulAddFusion", bool multigraph = true)
      : MultiplePatternProcessPass(name, multigraph) {}

  ~MulAddFusion() override = default;

  AnfNodePtr Process(const std::string &pattern_name, const FuncGraphPtr &, const AnfNodePtr &,
                     const EquivPtr &) const override;
  std::unordered_map<std::string, VectorRef> DefinePatterns() const override;

 private:
  VectorRef DefineMulFirstPattern() const;
  VectorRef DefineMulSecondPattern() const;

  bool CheckAddNode(const mindspore::CNodePtr &cnode) const;
  bool CheckMulNode(const mindspore::FuncGraphPtr &func_graph, const mindspore::CNodePtr &cnode) const;
  bool ScaleInputShapeValid(size_t *axis_offset) const;
  bool MulInputAnodeIsInferred(const AnfNodePtr &mul_input_anode) const;
  bool AdjustScaleBiasTensorShape(size_t *axis_offset) const;
  bool CopyNodeFormat(CNodePtr node, mindspore::ops::PrimitiveCPtr prim) const;

 private:
  mutable ShapeVector mul_input_shape_;
  mutable AnfNodePtr mul_const_anode_ = nullptr;
  mutable AnfNodePtr add_const_anode_ = nullptr;
  mutable tensor::TensorPtr scale_tensor_ = nullptr;
  mutable tensor::TensorPtr bias_tensor_ = nullptr;
  mutable ActivationType scale_act_type_ = ActivationType::NO_ACTIVATION;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_PASS_FUSION_CONV_ACTIVATION_FUSION_H_
