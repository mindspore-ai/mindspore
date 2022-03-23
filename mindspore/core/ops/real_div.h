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

#ifndef MINDSPORE_CORE_OPS_REAL_DIV_H_
#define MINDSPORE_CORE_OPS_REAL_DIV_H_
#include <string>
#include <vector>
#include <memory>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameRealDiv = "RealDiv";
/// \brief Divides the first input tensor by the second input tensor in floating-point type element-wise.
/// Refer to Python API @ref mindspore.ops.RealDiv for more details.
class MIND_API RealDiv : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(RealDiv);
  /// \brief Constructor.
  RealDiv() : BaseOperator(kNameRealDiv) { InitIOName({"x", "y"}, {"output"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.RealDiv for the inputs.
  void Init() const {}
};

abstract::AbstractBasePtr RealDivInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_REAL_DIV_H_
