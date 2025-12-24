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

#ifndef MINDSPORE_CORE_OPS_MAX_POOL_GRAD_GRAD_H_
#define MINDSPORE_CORE_OPS_MAX_POOL_GRAD_GRAD_H_
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ir/anf.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMaxPoolGradGrad = "MaxPoolGradGrad";
class OPS_API MaxPoolGradGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(MaxPoolGradGrad);
  /// \brief Constructor.
  MaxPoolGradGrad() : BaseOperator(kNameMaxPoolGradGrad) {
    InitIOName({"orig_input", "orig_output", "grad"}, {"output"});
  }

  /// \brief Constructor.
  explicit MaxPoolGradGrad(const std::string k_name) : BaseOperator(k_name) {
    InitIOName({"orig_input", "orig_output", "grad"}, {"output"});
  }

  /// \brief Method to set kernel_size attribute.
  ///
  /// \param[in] kernel_size Define the size of kernel used to take the maximum value. The length of kernel_size is the
  /// same as the dimension of the input.
  void set_kernel_size(const std::vector<int64_t> &kernel_size);

  /// \brief Method to get kernel_size attribute.
  ///
  /// \return The kernel_size vector.
  std::vector<int64_t> get_kernel_size() const;

  /// \brief Method to set strides attribute.
  ///
  /// \param[in] strides Define the distance of kernel moving. The length of strides is the same as the dimension of the
  /// input.
  void set_strides(const std::vector<int64_t> &strides);

  /// \brief Method to get strides attribute.
  ///
  /// \return The strides vector.
  std::vector<int64_t> get_strides() const;

  /// \brief Method to set pad_mode attribute.
  ///
  /// \param[in] pad_mode: The optional value for pad mode, is SAME or VALID.
  void set_pad_mode(const PadMode &pad_mode);

  /// \brief Method to get pad_mode attribute.
  ///
  /// \return The pad mode of max pool operator.
  PadMode get_pad_mode() const;
};

OPS_API abstract::AbstractBasePtr MaxPoolGradGradInfer(const abstract::AnalysisEnginePtr &,
                                                       const PrimitivePtr &primitive,
                                                       const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_MAX_POOL_GRAD_GRAD_H_
