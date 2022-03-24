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

#ifndef MINDSPORE_CORE_OPS_SUB_H_
#define MINDSPORE_CORE_OPS_SUB_H_
#include <string>
#include <vector>
#include <memory>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSub = "Sub";
/// \brief Subtracts the second input tensor from the first input tensor element-wise.
/// Refer to Python API @ref mindspore.ops.Sub for more details.
class MIND_API Sub : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Sub);
  /// \brief Constructor.
  Sub() : BaseOperator(kNameSub) { InitIOName({"x", "y"}, {"output"}); }
  explicit Sub(const std::string k_name) : BaseOperator(k_name) { InitIOName({"x", "y"}, {"output"}); }
  /// \brief Init.
  void Init() const {}
};

abstract::AbstractBasePtr SubInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SUB_H_
