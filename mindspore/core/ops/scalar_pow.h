/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_SCALAR_POW_H_
#define MINDSPORE_CORE_OPS_SCALAR_POW_H_
#include "mindspore/core/ops/arithmetic_ops.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
/// \brief ScalarDiv op is used to div between variable scalar.
class MIND_API ScalarPow : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ScalarPow);
  /// \brief Constructor.
  ScalarPow() : BaseOperator(kScalarPowOpName) { InitIOName({"x", "y"}, {"output"}); }
  /// \brief Init.
  void Init() const {}
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SCALAR_POW_H_
