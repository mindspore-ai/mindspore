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

#include "tools/optimizer/fusion/onnx_gelu_fusion.h"

namespace mindspore {
namespace opt {
namespace {
constexpr float DIFF_THRESHOLD = 0.0001;
constexpr float DIV_Y = 1.41421;
constexpr float ADD_Y = 1.0;
constexpr float MUL1_y = 0.5;
}  // namespace

// gelu(x) = 1/2 * x * [1 + erf(x / sqrt(2))]
const BaseRef OnnxGeLUFusion::DefinePattern() const {
  VectorRef div_ref({std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimDivFusion>), input_, div_y_});
  VectorRef erf_ref({std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimErf>), div_ref});
  VectorRef add_ref({std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>), erf_ref, add_y_});
  VectorRef mul1_ref({std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>), input_, mul1_y_});
  VectorRef mul2_ref({std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>), mul1_ref, add_ref});
  return mul2_ref;
}

bool OnnxGeLUFusion::CheckPattern(const EquivPtr &equiv) const {
  MS_ASSERT(equiv != nullptr);
  float div_y = GetParameterValue(equiv, div_y_);
  if (div_y < 0 || fabs(div_y - DIV_Y) > DIFF_THRESHOLD) {
    return false;
  }
  float add_y = GetParameterValue(equiv, add_y_);
  if (add_y < 0 || fabs(add_y - ADD_Y) > DIFF_THRESHOLD) {
    return false;
  }
  float mul1_y = GetParameterValue(equiv, mul1_y_);
  if (mul1_y < 0 || fabs(mul1_y - MUL1_y) > DIFF_THRESHOLD) {
    return false;
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
