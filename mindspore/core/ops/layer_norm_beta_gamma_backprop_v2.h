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

#ifndef MINDSPORE_CORE_OPS_LAYERNORMBETAGAMMABACKPROPV2_H_
#define MINDSPORE_CORE_OPS_LAYERNORMBETAGAMMABACKPROPV2_H_
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
class LayerNormBetaGammaBackpropV2 : public PrimitiveC {
 public:
  LayerNormBetaGammaBackpropV2() : PrimitiveC(prim::kPrimLayerNormBetaGammaBackpropV2->name()) {}
  ~LayerNormBetaGammaBackpropV2() = default;
  MS_DECLARE_PARENT(LayerNormBetaGammaBackpropV2, PrimitiveC);
  void Init(const std::vector<int64_t> &shape_gamma);
  void set_shape_gamma(const std::vector<int64_t> &shape_gamma);
  std::vector<int64_t> get_shape_gamma() const;
};

AbstractBasePtr LayerNormBetaGammaBackpropV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_LAYERNORMBETAGAMMABACKPROPV2_H_
