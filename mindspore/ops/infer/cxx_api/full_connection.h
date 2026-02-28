/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_FULL_CONNECTION_FUSION_H_
#define MINDSPORE_CORE_OPS_FULL_CONNECTION_FUSION_H_
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameFullConnection = "FullConnection";
/// \brief FullConnection defined FullConnection operator prototype of lite.
class OPS_API FullConnection : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(FullConnection);
  /// \brief Constructor.
  FullConnection() : BaseOperator(kNameFullConnection) { InitIOName({"x1", "x2", "b"}, {"output"}); }

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] has_bias Define a boolean value to indicate the op has bias or not.
  /// \param[in] axis Define a axis that the inner product is done along with
  /// \param[in] use_axis Define a boolean value to indicate the op uses a axis or not.
  /// \param[in] activation_type Define the activation type.
  void Init(const bool has_bias, const int64_t axis, const bool use_axis, const ActivationType &activation_type);

  /// \brief Method to set has_axis attribute.
  ///
  /// \param[in] has_bias Define a boolean value to indicate the op has bias or not.
  void set_has_bias(const bool has_bias);

  /// \brief Method to set axis attribute.
  ///
  /// \param[in] axis Define a axis the inner product must be along with
  void set_axis(const int64_t axis);

  /// \brief Method to set use_axis attribute.
  ///
  /// \param[in] use_axis Define a boolean value to indicate the op uses a axis or not.
  void set_use_axis(const bool use_axis);

  /// \brief Method to set activation type.
  ///
  /// \param[in] activation_type Define the activation type.
  void set_activation_type(const ActivationType &activation_type);

  /// \brief Method to get has_bias attribute.
  ///
  /// \return a boolean value
  bool get_has_bias() const;

  /// \brief Method to get axis attribute.
  ///
  /// \return axis
  int64_t get_axis() const;

  /// \brief Method to get use_axis attribute.
  ///
  /// \return a boolean value.
  bool get_use_axis() const;

  /// \brief Method to get activation type.
  ///
  /// \return activation type.
  ActivationType get_activation_type() const;
};
abstract::AbstractBasePtr FullConnectionInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_FULL_CONNECTION_FUSION_H_
