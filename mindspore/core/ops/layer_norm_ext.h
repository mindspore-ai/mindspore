/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_LAYER_NORM_EXT_H_
#define MINDSPORE_CORE_OPS_LAYER_NORM_EXT_H_
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameLayerNormExt = "LayerNormExt";
/// \brief Applies the Layer Normalization to the input tensor.
class MIND_API LayerNormExt : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(LayerNormExt);
  /// \brief Constructor.
  LayerNormExt() : BaseOperator(kNameLayerNormExt) {
    InitIOName({"input", "normalized_shape", "weight", "bias", "eps"}, {"output", "mean_out", "rstd_out"});
  }
  void Init() const {}
};

using PrimLayerNormExtPtr = std::shared_ptr<LayerNormExt>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_LAYER_NORM_EXT_H_
