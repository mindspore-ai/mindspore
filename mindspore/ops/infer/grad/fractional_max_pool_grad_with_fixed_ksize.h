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

#ifndef MINDSPORE_CORE_OPS_FRACTIONAL_MAX_POOL_GRAD_WITH_FIXED_KSIZE_H_
#define MINDSPORE_CORE_OPS_FRACTIONAL_MAX_POOL_GRAD_WITH_FIXED_KSIZE_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameFractionalMaxPoolGradWithFixedKsize = "FractionalMaxPoolGradWithFixedKsize";
class OPS_API FractionalMaxPoolGradWithFixedKsize : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(FractionalMaxPoolGradWithFixedKsize);
  FractionalMaxPoolGradWithFixedKsize() : BaseOperator(kNameFractionalMaxPoolGradWithFixedKsize) {
    InitIOName({"origin_input", "out_backprop", "argmax"}, {"y"});
  }
  std::vector<bool> InputDynamic(const std::vector<int64_t> &out_backprop_shape_,
                                 const std::vector<int64_t> &argmax_shape_,
                                 const std::vector<int64_t> &origin_input_shape_, bool out_backprop_shape_dy_,
                                 bool argmax_shape_dy_, bool origin_input_shape_dy_);
  void Init(const std::string data_format);
  /// \brief Init. Refer to the parameters of Python API @ref
  /// mindspore.ops.operations._grad_ops.FractionalMaxPoolWithFixedKsize for the inputs.
  void set_data_format(const std::string data_format);
  /// \brief Set data format.
  std::string get_data_format() const;
  /// \brief Method to get data format attributes.
  ///
  /// \return data format attributes.
};

OPS_API abstract::AbstractBasePtr FractionalMaxPoolGradWithFixedKsizeInfer(
  const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
  const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_FRACTIONAL_MAX_POOL_GRAD_WITH_FIXED_KSIZE_H_
