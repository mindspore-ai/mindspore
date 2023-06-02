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

#ifndef MINDSPORE_CORE_OPS_SOFTSHRINK_GRAD_H_
#define MINDSPORE_CORE_OPS_SOFTSHRINK_GRAD_H_
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSoftShrinkGrad = "SoftShrinkGrad";
class MIND_API SoftShrinkGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SoftShrinkGrad);
  SoftShrinkGrad() : BaseOperator(kNameSoftShrinkGrad) { InitIOName({"input_grad", "input_x"}, {"output"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.HShrink for the inputs.
  void Init(const float &lambd = 0.5);
  /// \brief Method to set lambd. The Default value is 0.5.
  void set_lambd(const float &lambd = 0.5);
  /// \brief Method to get lambd.
  float get_lambd() const;
};
MIND_API abstract::AbstractBasePtr SoftShrinkGradInfer(const abstract::AnalysisEnginePtr &,
                                                       const PrimitivePtr &primitive,
                                                       const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_SOFTSHRINK_GRAD_H_
