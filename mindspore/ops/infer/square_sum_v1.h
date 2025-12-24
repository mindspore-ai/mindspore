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

#ifndef MINDSPORE_CORE_OPS_SQUARE_SUM_V1_H_
#define MINDSPORE_CORE_OPS_SQUARE_SUM_V1_H_
#include <set>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameKeepDims = "keep_dims";
constexpr auto kNameAxis = "axis";
constexpr auto kNameSquareSumV1 = "SquareSumV1";

/// \brief Returns the square sum of a tensor element-wise. Refer to Python API @ref mindspore.ops.SquareSumV1 for more
/// details.
class OPS_API SquareSumV1 : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SquareSumV1);
  /// \brief Constructor.
  SquareSumV1() : BaseOperator(kNameSquareSumV1) { InitIOName({"x"}, {"output"}); }

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] axis Define axis.
  /// \param[in] keep_dims Define keep dims.
  void Init(int axis, bool keep_dims = true);

  /// \brief Method to set mode attributes.
  ///
  /// \param[in] axis Define axis.
  void set_axis(int64_t axis);

  /// \brief Method to get axis attributes.
  ///
  /// \return mode attributes.
  int64_t get_axis() const;

  /// \brief Method to set mode attributes.
  ///
  /// \param[in] keep_dims Define keep dims.
  void set_keep_dims(bool keep_dims);

  /// \brief Method to get mode attributes.
  ///
  /// \return keep dims attributes.
  bool get_keep_dims() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SQUARE_SUM_V1_H_
