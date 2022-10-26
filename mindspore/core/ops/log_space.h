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

#ifndef MINDSPORE_CORE_OPS_LOG_SPACE_H_
#define MINDSPORE_CORE_OPS_LOG_SPACE_H_
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameLogSpace = "LogSpace";
/// \brief Returns a Tensor whose value is evenly spaced in the interval start and end (including start and end).
/// Refer to Python API @ref mindspore.ops.LogSpace for more details.
class MIND_API LogSpace : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(LogSpace);
  /// \brief Constructor.
  LogSpace() : BaseOperator(kNameLogSpace) { InitIOName({"start", "end"}, {"y"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.LogSpace for the inputs.
  void Init() const {}

  void Init(int64_t steps, int64_t base);
  /// \brief Set steps.
  void set_steps(int64_t steps);
  /// \brief Set base.
  void set_base(int64_t base);

  /// \return base.
  int64_t get_base() const;

  /// \return steps.
  int64_t get_steps() const;
};
abstract::AbstractBasePtr LogSpaceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_LOG_SPACE_H_
