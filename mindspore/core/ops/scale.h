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

#ifndef MINDSPORE_CORE_OPS_SCALE_H_
#define MINDSPORE_CORE_OPS_SCALE_H_
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "ops/op_utils.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameScale = "Scale";
/// \brief Scale defined Scale operator prototype of lite.
class MS_CORE_API Scale : public PrimitiveC {
 public:
  /// \brief Constructor.
  Scale() : PrimitiveC(kNameScale) {}

  /// \brief Constructor.
  explicit Scale(const std::string k_name) : PrimitiveC(k_name) {}

  /// \brief Destructor.
  ~Scale() = default;

  MS_DECLARE_PARENT(Scale, PrimitiveC);

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] axis Define the first axis of input[0] along which to apply input[1], can be negative to index from
  ///            the end. Default -1.
  void Init(const int64_t axis = -1);

  /// \brief Method to set axis attribute.
  ///
  /// \param[in] axis Define the first axis of input[0] along which to apply input[1], can be negative to index from
  ///            the end. Default -1.
  void set_axis(const int64_t axis);

  /// \brief Method to get axis attribute.
  ///
  /// \return axis attribute.
  int64_t get_axis() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SCALE_H_
