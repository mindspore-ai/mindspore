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

#ifndef MINDSPORE_CORE_OPS_DYNAMIC_QUANT_H_
#define MINDSPORE_CORE_OPS_DYNAMIC_QUANT_H_

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameDynamicQuant = "DynamicQuant";
/// \brief the DynamicQuant operator prototype.
class MIND_API DynamicQuant : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(DynamicQuant);
  /// \brief Constructor.
  DynamicQuant() : BaseOperator(kNameDynamicQuant) {}

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] symmetric Define whether symmetric quantization.
  /// \param[in] dst_type Define the data type of output.
  void Init(const bool symmetric, const int64_t dst_type);

  /// \brief Method to set symmetric attribute.
  ///
  /// \param[in] symmetric Define whether symmetric quantization.
  void set_symmetric(const bool symmetric);

  /// \brief Method to get symmetric attribute.
  ///
  /// \return Whether symmetric quantization.
  bool get_symmetric() const;

  /// \brief Method to set dst_type attribute.
  ///
  /// \param[in] dst_type Define the data type of output.
  void set_dst_type(const int64_t dst_type);

  /// \brief Method to get dst_type attribute.
  ///
  /// \return the data type of output.
  int64_t get_dst_type() const;
};
MIND_API abstract::AbstractBasePtr DynamicQuantInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                     const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_QUANTD_TYPE_CAST_H_
