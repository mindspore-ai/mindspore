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

#ifndef MINDSPORE_CORE_OPS_MULTILABEL_MARGIN_LOSS_GRAD_H_
#define MINDSPORE_CORE_OPS_MULTILABEL_MARGIN_LOSS_GRAD_H_

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMultilabelMarginLossGrad = "MultilabelMarginLossGrad";
class OPS_API MultilabelMarginLossGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(MultilabelMarginLossGrad);
  MultilabelMarginLossGrad() : BaseOperator(kNameMultilabelMarginLossGrad) {
    InitIOName({"y_grad", "x", "target", "is_target"}, {"x_grad"});
  }
  int64_t get_reduction() const;
};

OPS_API abstract::AbstractBasePtr MultilabelMarginLossGradInfer(
  const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
  const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimMultilabelMarginLossGradPtr = std::shared_ptr<MultilabelMarginLossGrad>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_MULTILABEL_MARGIN_LOSS_GRAD_H_
