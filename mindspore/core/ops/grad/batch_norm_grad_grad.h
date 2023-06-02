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

#ifndef MINDSPORE_CORE_OPS_BATCH_NORM_GRAD_GRAD_H_
#define MINDSPORE_CORE_OPS_BATCH_NORM_GRAD_GRAD_H_

#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameBatchNormGradGrad = "BatchNormGradGrad";
class MIND_API BatchNormGradGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(BatchNormGradGrad);
  BatchNormGradGrad() : BaseOperator(kNameBatchNormGradGrad) {
    InitIOName({"x", "dy", "scale", "mean", "variance", "dout_dx", "dout_dscale", "dout_dbias"},
               {"dx", "ddy", "dscale"});
  }
  void Init(bool is_training = false, float epsilon = 1e-05, const std::string &format = "NCHW");
  void set_is_training(bool is_training);
  bool get_is_training() const;
  void set_epsilon(float epsilon);
  float get_epsilon() const;
  void set_format(const std::string &format);
  std::string get_format() const;
};

MIND_API abstract::AbstractBasePtr BatchNormGradGradInfer(const abstract::AnalysisEnginePtr &,
                                                          const PrimitivePtr &primitive,
                                                          const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_BATCH_NORM_GRAD_GRAD_H_
