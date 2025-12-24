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

#ifndef MINDSPORE_CORE_OPS_INSTANCE_NORM_V2_GRAD_H_
#define MINDSPORE_CORE_OPS_INSTANCE_NORM_V2_GRAD_H_
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameInstanceNormV2Grad = "InstanceNormV2Grad";
/// \brief InstanceNormV2Grad defined the InstanceNormV2Grad operator prototype.
class OPS_API InstanceNormV2Grad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(InstanceNormV2Grad);
  /// \brief Constructor.
  InstanceNormV2Grad() : BaseOperator(kNameInstanceNormV2Grad) {
    InitIOName({"dy", "x", "gamma", "mean", "variance", "save_mean", "save_variance"}, {"pd_x", "pd_gamma", "pd_beta"});
  }
};

OPS_API abstract::AbstractBasePtr InstanceNormV2GradInfer(const abstract::AnalysisEnginePtr &,
                                                          const PrimitivePtr &primitive,
                                                          const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimInstanceNormV2GradPtr = std::shared_ptr<InstanceNormV2Grad>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_INSTANCE_NORM_V2_GRAD_H_
