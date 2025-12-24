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

#ifndef MINDSPORE_P_S_R_O_I_POOLING_GRAD_H
#define MINDSPORE_P_S_R_O_I_POOLING_GRAD_H

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNamePSROIPoolingGrad = "PSROIPoolingGrad";

class OPS_API PSROIPoolingGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(PSROIPoolingGrad);
  PSROIPoolingGrad() : BaseOperator(kNamePSROIPoolingGrad) {}
};

OPS_API abstract::AbstractBasePtr PSROIPoolingGradInfer(const abstract::AnalysisEnginePtr &,
                                                        const PrimitivePtr &primitive,
                                                        const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_P_S_R_O_I_POOLING_GRAD_H
