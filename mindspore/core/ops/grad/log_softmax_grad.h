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

#ifndef MINDSPORE_CORE_OPS_GRAD_LOG_SOFTMAX_GRAD_H_
#define MINDSPORE_CORE_OPS_GRAD_LOG_SOFTMAX_GRAD_H_
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameLogSoftmaxGrad = "LogSoftmaxGrad";
class MIND_API LogSoftmaxGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(LogSoftmaxGrad);
  LogSoftmaxGrad() : BaseOperator(kNameLogSoftmaxGrad) { InitIOName({"x", "grad"}, {"y"}); }
  explicit LogSoftmaxGrad(const std::string k_name) : BaseOperator(k_name) { InitIOName({"x", "grad"}, {"y"}); }
  void Init(const int64_t axis = -1);
  void set_axis(const int64_t epsilon);
  int64_t get_axis() const;
};

MIND_API abstract::AbstractBasePtr LogSoftmaxGradInfer(const abstract::AnalysisEnginePtr &,
                                                       const PrimitivePtr &primitive,
                                                       const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_GRAD_LOG_SOFTMAX_GRAD_H_
