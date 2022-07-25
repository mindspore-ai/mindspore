/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_ADAPTIVEMAXPOOL2D_H_
#define MINDSPORE_CORE_OPS_ADAPTIVEMAXPOOL2D_H_

#include <vector>
#include <string>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kAdaptiveMaxPool2D = "AdaptiveMaxPool2D";
constexpr size_t kFormatCHWShapeSize = 3;
constexpr size_t kFormatNCHWShapeSize = 4;
constexpr size_t kOutputSizeAttrSize = 2;
constexpr int64_t kPyValueNone = -1;
constexpr int64_t kDynamicRankValue = -2;

// AdaptiveMaxPool2D operation. Refer to Python API @ref mindspore.nn.AdaptiveMaxPool2d for more details.
class MIND_API AdaptiveMaxPool2D : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(AdaptiveMaxPool2D);
  AdaptiveMaxPool2D() : BaseOperator(kAdaptiveMaxPool2D) { InitIOName({"input_x"}, {"output"}); }
  std::vector<int64_t> output_size() const;
  bool return_indices() const;
};

abstract::AbstractBasePtr AdaptiveMaxPool2DInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                 const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ADAPTIVEMAXPOOL2D_H_
