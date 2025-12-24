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

#ifndef MINDSPORE_CORE_OPS_COMPLEXABS_H_
#define MINDSPORE_CORE_OPS_COMPLEXABS_H_
#include <memory>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameComplexAbs = "ComplexAbs";

/// \brief Returns a Tensor that is the complex_abs part of the input.
/// Refer to Python API @ref mindspore.ops.ComplexAbs for more details.
class OPS_API ComplexAbs : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ComplexAbs);
  /// \brief Constructor.
  ComplexAbs() : BaseOperator(kNameComplexAbs) { InitIOName({"x"}, {"y"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.ComplexAbs for the inputs.
  void Init() const {}
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_COMPLEXABS_H_
