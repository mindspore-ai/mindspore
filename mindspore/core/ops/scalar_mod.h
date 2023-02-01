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

#ifndef MINDSPORE_CORE_OPS_SCALAR_MOD_H_
#define MINDSPORE_CORE_OPS_SCALAR_MOD_H_
#include "ops/base_operator.h"
#include "mindspore/core/ops/core_ops.h"

namespace mindspore {
namespace ops {
/// \brief ScalarMod op is used to mod between variable scalar.
class MIND_API ScalarMod : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ScalarMod);
  /// \brief Constructor.
  ScalarMod() : BaseOperator(prim::kScalarMod) {}
  /// \brief Init.
  void Init() const {}
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SCALAR_MOD_H_
