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

#ifndef MINDSPORE_CORE_OPS_DIV_H_
#define MINDSPORE_CORE_OPS_DIV_H_
#include <string>
#include <vector>
#include <memory>
#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameDiv = "Div";
/// \brief Computes the quotient of dividing the first input tensor by the second input tensor element-wise.
/// Refer to Python API @ref mindspore.ops.Div for more details.
class MIND_API Div : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Div);
  /// \brief Constructor.
  Div() : BaseOperator(kNameDiv) { InitIOName({"x", "y"}, {"output"}); }
  explicit Div(const std::string k_name) : BaseOperator(k_name) { InitIOName({"x", "y"}, {"output"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.Div for the inputs.
  void Init() const {}
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_DIV_H_
