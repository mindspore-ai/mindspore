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

#ifndef MINDSPORE_CORE_OPS_LAYER_NORM_FUSION_H_
#define MINDSPORE_CORE_OPS_LAYER_NORM_FUSION_H_
#include <vector>
#include <memory>

#include "ops/layer_norm.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameLayerNormFusion = "LayerNormFusion";
class LayerNormFusion : public LayerNorm {
 public:
  LayerNormFusion() : LayerNorm(kNameLayerNormFusion) {}
  ~LayerNormFusion() = default;
  MS_DECLARE_PARENT(LayerNormFusion, LayerNorm);
  void Init(const int64_t begin_norm_axis = 1, const int64_t begin_params_axis = 1, const float epsilon = 1e-7,
            const bool elementwise_affine = false);
  void set_elementwise_affine(const bool elementwise_affine);
  bool get_elementwise_affine() const;
};

AbstractBasePtr LayerNormFusionInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args);
using PrimLayerNormFusionPtr = std::shared_ptr<LayerNormFusion>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_LAYER_NORM_FUSION_H_
