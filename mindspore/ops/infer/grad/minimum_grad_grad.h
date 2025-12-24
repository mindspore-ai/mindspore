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

#ifndef MINDSPORE_CORE_OPS_GRAD_MINIMUM_GRAD_GRAD_H_
#define MINDSPORE_CORE_OPS_GRAD_MINIMUM_GRAD_GRAD_H_

#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMinimumGradGrad = "MinimumGradGrad";
class OPS_API MinimumGradGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(MinimumGradGrad);
  MinimumGradGrad() : BaseOperator(kNameMinimumGradGrad) {
    InitIOName({"x1", "x2", "grad_y1", "grad_y2"}, {"sopd_x1", "sopd_x2", "sopd_grad"});
  }
  void Init(const bool grad_x = true, const bool grad_y = true);
  void set_grad_x(const bool grad_x);
  void set_grad_y(const bool grad_y);
  bool get_grad_x() const;
  bool get_grad_y() const;
};
OPS_API abstract::AbstractBasePtr MinimumGradGradInfer(const abstract::AnalysisEnginePtr &,
                                                       const PrimitivePtr &primitive,
                                                       const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_GRAD_MINIMUM_GRAD_GRAD_H_
