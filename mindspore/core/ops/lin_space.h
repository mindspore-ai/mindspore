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

#ifndef MINDSPORE_CORE_OPS_LIN_SPACE_H_
#define MINDSPORE_CORE_OPS_LIN_SPACE_H_
#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameLinSpace = "LinSpace";
/// \brief Returns a Tensor whose value is evenly spaced in the interval start and stop (including start and stop).
/// Refer to Python API @ref mindspore.ops.LinSpace for more details.
class MIND_API LinSpace : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(LinSpace);
  /// \brief Constructor.
  LinSpace() : BaseOperator(kNameLinSpace) { InitIOName({"start", "stop", "num"}, {"output"}); }
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_LIN_SPACE_H_
