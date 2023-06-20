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

#ifndef MINDSPORE_CORE_OPS_MAXIMUM_GRAD_H_
#define MINDSPORE_CORE_OPS_MAXIMUM_GRAD_H_
#include <memory>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMaximumGrad = "MaximumGrad";
class MIND_API MaximumGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(MaximumGrad);
  MaximumGrad() : BaseOperator(kNameMaximumGrad) { InitIOName({"x1", "x2", "grads"}, {"y1", "y2"}); }
  void Init(const bool grad_x = true, const bool grad_y = true);
  void set_grad_x(const bool grad_x);
  void set_grad_y(const bool grad_y);
  bool get_grad_x() const;
  bool get_grad_y() const;
};
MIND_API abstract::AbstractBasePtr MaximumGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                    const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimMaximumGradPtr = std::shared_ptr<MaximumGrad>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_MAXIMUM_GRAD_H_
