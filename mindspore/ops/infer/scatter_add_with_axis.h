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

#ifndef MINDSPORE_CORE_OPS_SCATTER_ADD_WITH_AXIS_H_
#define MINDSPORE_CORE_OPS_SCATTER_ADD_WITH_AXIS_H_
#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameScatterAddWithAxis = "ScatterAddWithAxis";
/// \brief Updates tensor values by using input indices and value.
/// Refer to Python API @ref mindspore.ops.ScatterAddWithAxis for more details.
class OPS_API ScatterAddWithAxis : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ScatterAddWithAxis);
  /// \brief Constructor.
  ScatterAddWithAxis() : BaseOperator(kNameScatterAddWithAxis) { InitIOName({"input_x", "indices", "update"}, {"y"}); }
  /// \brief Init. Refer to the parameters of Python API @ref
  /// mindspore.ops.ScatterAddWithAxis for the inputs.
  void Init(const int64_t axis = 0);
  /// \brief Set axis.
  void set_axis(const int64_t axis);
  /// \brief Get axis.
  int64_t get_axis() const;
};
OPS_API abstract::AbstractBasePtr ScatterAddWithAxisInfer(const abstract::AnalysisEnginePtr &,
                                                          const PrimitivePtr &primitive,
                                                          const std::vector<abstract::AbstractBasePtr> &input_args);
using kPrimScatterAddWithAxisPtr = std::shared_ptr<ScatterAddWithAxis>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SCATTER_ADD_WITH_AXIS_H_
