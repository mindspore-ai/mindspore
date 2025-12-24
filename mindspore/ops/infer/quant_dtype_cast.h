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

#ifndef MINDSPORE_CORE_OPS_QUANTD_TYPE_CAST_H_
#define MINDSPORE_CORE_OPS_QUANTD_TYPE_CAST_H_

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameQuantDTypeCast = "QuantDTypeCast";
/// \brief QuantDTypeCast QuantDTypeCast the QuantDTypeCast operator prototype.
class OPS_API QuantDTypeCast : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(QuantDTypeCast);
  /// \brief Constructor.
  QuantDTypeCast() : BaseOperator(kNameQuantDTypeCast) {}

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] src_t Define the data type of input.
  /// \param[in] dst_t Define the data type of output.
  void Init(const int64_t src_t, const int64_t dst_t);

  /// \brief Method to set src_t attribute.
  ///
  /// \param[in] src_t Define the data type of input.
  void set_src_t(const int64_t src_t);

  /// \brief Method to get src_t attribute.
  ///
  /// \return the data type of input.
  int64_t get_src_t() const;

  /// \brief Method to set dst_t attribute.
  ///
  /// \param[in] dst_t Define the data type of output.
  void set_dst_t(const int64_t dst_t);

  /// \brief Method to get dst_t attribute.
  ///
  /// \return the data type of output.
  int64_t get_dst_t() const;

  /// \brief Method to set axis attribute.
  ///
  /// \param[in] axis Define per-channel prefer dim.
  void set_axis(const int64_t axis);

  /// \brief Method to get axis attribute.
  ///
  /// \return the per-channel prefer dim.
  int64_t get_axis() const;
};
OPS_API abstract::AbstractBasePtr QuantDTypeCastInfer(const abstract::AnalysisEnginePtr &,
                                                      const PrimitivePtr &primitive,
                                                      const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_QUANTD_TYPE_CAST_H_
