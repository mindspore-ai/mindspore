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

#ifndef MINDSPORE_CORE_OPS_SQUARE_SUM_ALL_H_
#define MINDSPORE_CORE_OPS_SQUARE_SUM_ALL_H_
#include <set>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSquareSumAll = "SquareSumAll";

/// \brief Returns the square sum of a tensor element-wise. Refer to Python API @ref mindspore.ops.SquareSumAll for more
/// details.
class OPS_API SquareSumAll : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SquareSumAll);
  /// \brief Constructor.
  SquareSumAll() : BaseOperator(kNameSquareSumAll) { InitIOName({"x", "y"}, {"output_x", "output_y"}); }
  /// \brief Init.
  void Init() const {}
};

OPS_API abstract::AbstractBasePtr SquareSumAllInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                    const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SQUARE_SUM_ALL_H_
