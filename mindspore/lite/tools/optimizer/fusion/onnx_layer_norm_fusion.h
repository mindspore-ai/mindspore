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
#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_ONNX_LAYER_NORM_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_ONNX_LAYER_NORM_FUSION_H_

#include <vector>
#include <memory>
#include <string>
#include "tools/optimizer/fusion/tf_norm_fusion.h"
#include "backend/optimizer/common/optimizer.h"
#include "utils/utils.h"

namespace mindspore {
namespace opt {

class OnnxLayerNormFusion : public TfNormFusion {
 public:
  explicit OnnxLayerNormFusion(const std::string &name = "onnx_layer_norm_fusion", bool multigraph = true)
      : TfNormFusion(name, multigraph) {}

  ~OnnxLayerNormFusion() override = default;
  const BaseRef DefinePattern() const override;
};

inline bool IsPowNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    return CheckPrimitiveType(utils::cast<AnfNodePtr>(n), prim::kPrimPowFusion);
  }
  return false;
}
inline bool IsSqrtNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    return CheckPrimitiveType(utils::cast<AnfNodePtr>(n), prim::kPrimSqrt);
  }
  return false;
}
inline bool IsDivNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    return CheckPrimitiveType(utils::cast<AnfNodePtr>(n), prim::kPrimDiv) ||
           CheckPrimitiveType(utils::cast<AnfNodePtr>(n), std::make_shared<Primitive>("DivFusion"));
  }
  return false;
}
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_ONNX_LAYER_NORM_FUSION_H_
