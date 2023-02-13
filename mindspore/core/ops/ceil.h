/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_CEIL_H_
#define MINDSPORE_CORE_OPS_CEIL_H_
#include <vector>
#include <memory>

#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameCeil = "Ceil";
/// \brief Rounds a tensor up to the closest integer element-wise.
/// Refer to Python API @ref mindspore.ops.Ceil for more details.
class MIND_API Ceil : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Ceil);
  /// \brief Constructor.
  Ceil() : BaseOperator(kNameCeil) { InitIOName({"x"}, {"y"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.Ceil for the inputs.
  void Init() const {}
};
MIND_API abstract::AbstractBasePtr CeilInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_CEIL_H_
