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

#ifndef MINDSPORE_CORE_OPS_BATCH_NORM_GRAD_H_
#define MINDSPORE_CORE_OPS_BATCH_NORM_GRAD_H_
#include <vector>
#include <memory>
#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameBatchNormGrad = "BatchNormGrad";
class MIND_API BatchNormGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(BatchNormGrad);
  BatchNormGrad() : BaseOperator(kNameBatchNormGrad) {}
  void Init(const bool is_training = false, const float epsilon = 1e-05);
  void set_is_training(const bool is_training);
  void set_epsilon(const float epsilon);
  bool get_is_training() const;
  float get_epsilon() const;
};

abstract::AbstractBasePtr BatchNormGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_BATCH_NORM_GRAD_H_
