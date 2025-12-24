/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_PACK_H_
#define MINDSPORE_CORE_OPS_PACK_H_

#include <algorithm>
#include <map>
#include <memory>
#include <string>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNamePack = "Pack";
/// \brief Stacks a list of tensors in specified axis.
/// Refer to Python API @ref mindspore.ops.Stack for more details.
class OPS_API Pack : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Pack);
  /// \brief Constructor.
  Pack() : BaseOperator(kNamePack) {}
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.Stack for the inputs.
  void Init(const int64_t &axis = 0);
  /// \brief Set axis.
  void set_axis(const int64_t &axis);
  /// \brief Get axis.
  ///
  /// \return axis.
  int64_t get_axis() const;
};
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_PACK_H_
