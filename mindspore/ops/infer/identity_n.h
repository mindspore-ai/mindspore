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

#ifndef MINDSPORE_CORE_OPS_IDENTITY_N_H_
#define MINDSPORE_CORE_OPS_IDENTITY_N_H_
#include <memory>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameIdentityN = "IdentityN";
/// \brief Returns Tensor list with the same shape and contents as input.
/// Refer to Python API @ref mindspore.ops.IdentityN for more details.
class OPS_API IdentityN : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(IdentityN);

  /// \brief Constructor.
  IdentityN() : BaseOperator(kNameIdentityN) { InitIOName({"x"}, {"y"}); }

  /// \brief Init.
  void Init() const {}
};
using PrimIdentityNPtr = std::shared_ptr<IdentityN>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_IDENTITY_N_H_
