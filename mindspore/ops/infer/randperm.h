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

#ifndef MINDSPORE_CORE_OPS_RANDPERM_H_
#define MINDSPORE_CORE_OPS_RANDPERM_H_
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/type_id.h"
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameRandperm = "Randperm";
/// \brief Parametric Rectified Linear Unit activation function.
/// Refer to Python API @ref mindspore.ops.Randperm for more details.
class OPS_API Randperm : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Randperm);
  /// \brief Constructor.
  Randperm() : BaseOperator(kNameRandperm) { InitIOName({"x", "weight"}, {"output"}); }
  explicit Randperm(const std::string k_name) : BaseOperator(k_name) { InitIOName({"x", "weight"}, {"output"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.Randperm for the inputs.
  void Init(const int64_t max_length = 1, const int64_t pad = -1, const TypeId dtype = kNumberTypeInt32) {
    this->set_max_length(max_length);
    this->set_pad(pad);
    this->set_dtype(dtype);
  }

  // def __init__(self, max_length = 1, pad = -1, dtype = mstype.int32)

  /// \brief Set max_length.
  void set_max_length(const int64_t max_length);

  /// \brief Set pad.
  void set_pad(const int64_t pad);

  /// \brief Set dtype.
  void set_dtype(const TypeId dtype);

  /// \brief Get max_length.
  ///
  /// \return max_length.
  int64_t get_max_length() const;

  /// \brief Get pad.
  ///
  /// \return pad.
  int64_t get_pad() const;

  /// \brief Get dtype.
  ///
  /// \return dtype.
  TypeId get_dtype() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_RANDPERM_H_
