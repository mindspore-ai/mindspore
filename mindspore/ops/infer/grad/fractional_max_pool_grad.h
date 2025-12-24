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

#ifndef MINDSPORE_CORE_OPS_FRACTIONAL_MAX_POOL_GRAD_H_
#define MINDSPORE_CORE_OPS_FRACTIONAL_MAX_POOL_GRAD_H_
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameFractionalMaxPoolGrad = "FractionalMaxPoolGrad";
class OPS_API FractionalMaxPoolGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(FractionalMaxPoolGrad);
  FractionalMaxPoolGrad() : BaseOperator(kNameFractionalMaxPoolGrad) {
    InitIOName({"orig_input", "orig_output", "out_backprop", "row_pooling_sequence", "col_pooling_sequence"}, {"y"});
  }
  bool get_overlapping() const;
  /// \brief Method to get overlapping attributes.
  ///
  /// \return overlapping attributes.
};
OPS_API abstract::AbstractBasePtr FractionalMaxPoolGradInfer(const abstract::AnalysisEnginePtr &,
                                                             const PrimitivePtr &primitive,
                                                             const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimFractionalMaxPoolGrad = std::shared_ptr<FractionalMaxPoolGrad>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_FRACTIONAL_MAX_POOL_GRAD_H_
