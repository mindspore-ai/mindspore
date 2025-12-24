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

#ifndef MINDSPORE_CORE_OPS_FORMAT_TRANSPOSE_H_
#define MINDSPORE_CORE_OPS_FORMAT_TRANSPOSE_H_
#include <memory>
#include "mindapi/base/format.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameFormatTranspose = "FormatTranspose";
/// \brief Transpose tensor from specific source format to dest format.
class OPS_API FormatTranspose : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(FormatTranspose);
  /// \brief Constructor.
  FormatTranspose() : BaseOperator(kNameFormatTranspose) {}

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] src_format Define the source format of the input.
  /// \param[in] dst_format Define the dest format of the output.
  void Init(const Format &src_format, const Format &dst_format);

  /// \brief Method to set source format attribute.
  ///
  /// \param[in] src_format Define the source format of the input.
  void set_src_format(const Format &src_format);

  /// \brief Method to set dest format attribute.
  ///
  /// \param[in] src_format Define the dest format of the output.
  void set_dst_format(const Format &dst_format);

  /// \brief Method to get source format attribute.
  ///
  /// \return the source format of the input.
  Format get_src_format() const;

  /// \brief Method to get dest format attribute.
  ///
  /// \return the dest format of the output.
  Format get_dst_format() const;
};
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_FORMAT_TRANSPOSE_H_
