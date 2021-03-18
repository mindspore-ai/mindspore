/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/fusion/onnx_layer_norm_fusion.h"
#include <memory>
#include "ops/rsqrt.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore {
namespace opt {
const BaseRef OnnxLayerNormFusion::DefinePattern() const {
  VectorRef mean1_ref = VectorRef({mean1_, input_, mean1_axes_});
  VectorRef sub1_ref = VectorRef({std::make_shared<CondVar>(IsSubNode), input_, mean1_ref});
  VectorRef sub2_ref = VectorRef({std::make_shared<CondVar>(IsSubNode), input_, mean1_ref});
  VectorRef pow_ref = VectorRef({std::make_shared<CondVar>(IsPowNode), sub2_ref, std::make_shared<Var>()});
  VectorRef mean2_ref = VectorRef({mean2_, pow_ref, mean2_axes_});
  VectorRef add1_ref = VectorRef({std::make_shared<CondVar>(IsAddNode), mean2_ref, epsilon_});
  VectorRef sqrt_ref = VectorRef({std::make_shared<CondVar>(IsSqrtNode), add1_ref});
  VectorRef div_ref = VectorRef({std::make_shared<CondVar>(IsDivNode), sub1_ref, sqrt_ref});
  VectorRef mul_ref = VectorRef({std::make_shared<CondVar>(IsMulNode), gamma_, div_ref});
  VectorRef add2_ref = VectorRef({std::make_shared<CondVar>(IsAddNode), mul_ref, beta_});
  return add2_ref;
}
}  // namespace opt
}  // namespace mindspore
