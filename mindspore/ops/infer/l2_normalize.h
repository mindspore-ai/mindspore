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

#ifndef MINDSPORE_CORE_OPS_L2_NORMALIZE_H_
#define MINDSPORE_CORE_OPS_L2_NORMALIZE_H_
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameL2Normalize = "L2Normalize";
/// \brief L2 Normalization Operator. Refer to Python API @ref mindspore.ops.L2Normalize for more details.
class OPS_API L2Normalize : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(L2Normalize);
  /// \brief Constructor.
  explicit L2Normalize(const std::string &name = kNameL2Normalize) : BaseOperator(name) {
    InitIOName({"input_x"}, {"output_y"});
  }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.L2Normalize for the inputs.
  void Init(const std::vector<int64_t> &axis, const float epsilon = 1e-4);
  /// \brief Set axis.
  void set_axis(const std::vector<int64_t> &axis);
  /// \brief Set epsilon.
  void set_epsilon(const float epsilon);
  /// \brief Get axis.
  ///
  /// \return axis.
  std::vector<int64_t> get_axis() const;
  /// \brief Get epsilon.
  ///
  /// \return epsilon.
  float get_epsilon() const;
};
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_L2_NORMALIZE_H_
