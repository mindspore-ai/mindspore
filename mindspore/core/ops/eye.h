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

#ifndef MINDSPORE_CORE_OPS_EYE_H_
#define MINDSPORE_CORE_OPS_EYE_H_
#include <vector>
#include <memory>
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameEye = "Eye";
/// \brief Returns a Tensor whose value is evenly spaced in the interval start and end (including start and end).
/// Refer to Python API @ref mindspore.ops.Eye for more details.
class MIND_API Eye : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Eye);
  /// \brief Constructor.
  Eye() : BaseOperator(kNameEye) { InitIOName({"n", "m", "t"}, {"y"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.FloorDiv for the inputs.
  void Init() const {}
};

abstract::AbstractBasePtr EyeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_EYE_H_
