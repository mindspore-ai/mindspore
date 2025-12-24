/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_ARGMAX_FUSION_H_
#define MINDSPORE_CORE_OPS_ARGMAX_FUSION_H_
#include <memory>
#include <vector>
#include "mindapi/base/type_id.h"
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameArgMaxFusion = "ArgMaxFusion";
/// \brief ArgMaxFusion defined ArgMax operator prototype of lite.
class OPS_API ArgMaxFusion : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ArgMaxFusion);
  /// \brief Constructor.
  ArgMaxFusion() : BaseOperator(kNameArgMaxFusion) { InitIOName({"x"}, {"output"}); }

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] keep_dims Define a boolean value to indicate the dimension of output is equal to that of input or not.
  /// \param[in] out_max_value Define a boolean value to indicate whether to output maximum value.
  /// \param[in] top_k Define the number of maximum value along with axis.
  /// \param[in] axis Define where the argmax operation applies to.
  void Init(const bool keep_dims, const bool out_max_value, const int64_t top_k, const int64_t axis = -1);

  /// \brief Method to set keep_dims attribute.
  ///
  /// \param[in] keep_dims Define a boolean value to indicate the dimension of output is equal to that of input or not.
  void set_keep_dims(const bool keep_dims);

  /// \brief Method to set out_max_value attribute.
  ///
  /// \param[in] out_max_value Define a boolean value to indicate whether to output maximum value.
  void set_out_max_value(const bool out_max_value);

  /// \brief Method to set top_k attribute, default is 1.
  ///
  /// \param[in] top_k Define the number of maximum value along with axis.
  void set_top_k(const int64_t top_k);

  /// \brief Method to get keep_dims attribute.
  ///
  /// \return a boolean value to indicate the dimension of output is equal to that of input or not.
  bool get_keep_dims() const;

  /// \brief Method to get out_max_value attribute.
  ///
  /// \return a boolean value to indicate whether to output maximum value.
  bool get_out_max_value() const;

  /// \brief Method to get top_k attribute.
  ///
  /// \return the number of maximum value along with axis.
  int64_t get_top_k() const;

  /// \brief Set axis.
  void set_axis(const int64_t axis);

  /// \brief Set output_type.
  void set_output_type(const TypeId output_type);

  /// \brief Get axis.
  ///
  /// \return axis.
  int64_t get_axis() const;

  /// \brief Get output_type.
  ///
  /// \return output_type.
  TypeId get_output_type() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ARGMAX_FUSION_H_
