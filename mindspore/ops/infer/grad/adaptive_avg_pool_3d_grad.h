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

#ifndef MINDSPORE_CORE_OPS_ADAPTIVEAVGPOOL3DGRAD_H_
#define MINDSPORE_CORE_OPS_ADAPTIVEAVGPOOL3DGRAD_H_
#include <memory>
#include <vector>
#include "abstract/abstract_value.h"
#include "mindapi/base/types.h"
#include "ops/base_operator.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAdaptiveAvgPool3DGrad = "AdaptiveAvgPool3DGrad";
class OPS_API AdaptiveAvgPool3DGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(AdaptiveAvgPool3DGrad);
  AdaptiveAvgPool3DGrad() : BaseOperator(kNameAdaptiveAvgPool3DGrad) {
    InitIOName({"input_grad", "orig_input_shape"}, {"output_grad"});
  }
};

OPS_API abstract::AbstractBasePtr AdaptiveAvgPool3DGradInfer(const abstract::AnalysisEnginePtr &,
                                                             const PrimitivePtr &primitive,
                                                             const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimAdaptiveAvgPool3DGradPtr = std::shared_ptr<AdaptiveAvgPool3DGrad>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ADAPTIVEAVGPOOL3DGRAD_H_
