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

#ifndef MINDSPORE_CORE_OPS_FRACTIONAL_MAX_POOL_WITH_FIXED_KSIZE_H_
#define MINDSPORE_CORE_OPS_FRACTIONAL_MAX_POOL_WITH_FIXED_KSIZE_H_

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameFractionalMaxPoolWithFixedKsize = "FractionalMaxPoolWithFixedKsize";
/// \brief Fractional max pooling operation.
/// Refer to Python API @ref mindspore.ops.FractionalMaxPoolWithFixedKsize for more details.
class MIND_API FractionalMaxPoolWithFixedKsize : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(FractionalMaxPoolWithFixedKsize);
  /// \brief Constructor.
  FractionalMaxPoolWithFixedKsize() : BaseOperator(kNameFractionalMaxPoolWithFixedKsize) {
    InitIOName({"input_x", "random_samples"}, {"y", "argmax"});
  }
};

abstract::AbstractBasePtr FractionalMaxPoolWithFixedKsizeInfer(
  const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
  const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_FRACTIONAL_MAX_POOL_WITH_FIXED_KSIZE_H_
