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

#ifndef MINDSPORE_CORE_OPS_BN_INFER_GRAD_H_
#define MINDSPORE_CORE_OPS_BN_INFER_GRAD_H_
#include <memory>
#include <string>
#include "ops/base_operator.h"
#include "mindapi/base/types.h"
#include "mindapi/base/format.h"

namespace mindspore {
namespace ops {
constexpr auto kNameBNInferGrad = "BNInferGrad";
class MIND_API BNInferGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(BNInferGrad);
  BNInferGrad() : BaseOperator(kNameBNInferGrad) {}
  explicit BNInferGrad(const std::string kernel_name) : BaseOperator(kernel_name) {}
  void Init(const float epsilon = 1e-05, const std::string &inplace_algo = "cover");
  void set_epsilon(const float epsilon);
  float get_epsilon() const;
  std::string get_inplace_algo() const;
  void set_inplace_algo(const std::string &inplace_algo);
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_BN_INFER_GRAD_H_
