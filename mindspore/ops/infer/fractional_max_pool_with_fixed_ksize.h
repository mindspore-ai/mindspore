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

#ifndef MINDSPORE_CORE_OPS_FRACTIONAL_MAX_POOL_WITH_FIXED_KSIZE_H_
#define MINDSPORE_CORE_OPS_FRACTIONAL_MAX_POOL_WITH_FIXED_KSIZE_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameFractionalMaxPoolWithFixedKsize = "FractionalMaxPoolWithFixedKsize";
/// \brief Fractional max pooling operation.
/// Refer to Python API @ref mindspore.ops.FractionalMaxPoolWithFixedKsize for more details.
class OPS_API FractionalMaxPoolWithFixedKsize : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(FractionalMaxPoolWithFixedKsize);
  /// \brief Constructor.
  FractionalMaxPoolWithFixedKsize() : BaseOperator(kNameFractionalMaxPoolWithFixedKsize) {
    InitIOName({"input_x", "random_samples"}, {"y", "argmax"});
  }
  void InputCheck(const PrimitivePtr &primitive, const std::vector<int64_t> &x_shape,
                  const std::vector<int64_t> &random_samples_shape);
  void Init(const std::vector<int64_t> ksize, const std::vector<int64_t> output_shape, const std::string data_format);
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.FractionalMaxPoolWithFixedKsize for the
  /// inputs.
  void set_ksize(const std::vector<int64_t> ksize);
  /// \brief Set ksize.
  void set_output_shape(const std::vector<int64_t> output_shape);
  /// \brief Set output shape.
  void set_data_format(const std::string data_format);
  /// \brief Set data format.
  std::vector<int64_t> get_ksize() const;
  /// \brief Method to get ksize attributes.
  ///
  /// \return ksize attributes.
  std::vector<int64_t> get_output_shape() const;
  /// \brief Method to get output shape attributes.
  ///
  /// \return output shape attributes.
  std::string get_data_format() const;
  /// \brief Method to get data format attributes.
  ///
  /// \return data format attributes.
};

OPS_API abstract::AbstractBasePtr FractionalMaxPoolWithFixedKsizeInfer(
  const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
  const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_FRACTIONAL_MAX_POOL_WITH_FIXED_KSIZE_H_
