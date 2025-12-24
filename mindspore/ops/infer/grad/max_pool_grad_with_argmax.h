/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_GRAD_MAX_POOL_GRAD_WITH_ARGMAX_H_
#define MINDSPORE_CORE_OPS_GRAD_MAX_POOL_GRAD_WITH_ARGMAX_H_

#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/format.h"
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMaxPoolGradWithArgmax = "MaxPoolGradWithArgmax";
/// \brief Max pooling operation.
class OPS_API MaxPoolGradWithArgmax : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(MaxPoolGradWithArgmax);
  /// \brief Constructor.
  MaxPoolGradWithArgmax() : BaseOperator(kNameMaxPoolGradWithArgmax) {
    InitIOName({"input", "grad", "argmax"}, {"output"});
  }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.MaxPoolWithArgmax for the inputs.
  void Init(const std::vector<int64_t> &kernel_size = {1}, const std::vector<int64_t> &stride = {1},
            const PadMode &pad_mode = VALID, const Format &format = NCHW);
  /// \brief Set pad_mode.
  void set_pad_mode(const PadMode &pad_mode);
  /// \brief Set kernel_size.
  void set_kernel_size(const std::vector<int64_t> &kernel_size);
  /// \brief Set strides.
  void set_strides(const std::vector<int64_t> &strides);

  /// \return kernel_size.
  std::vector<int64_t> get_kernel_size() const;

  /// \return pad_mode
  PadMode get_pad_mode() const;

  /// \return strides.
  std::vector<int64_t> get_strides() const;
};

OPS_API abstract::AbstractBasePtr MaxPoolGradWithArgmaxInfer(const abstract::AnalysisEnginePtr &,
                                                             const PrimitivePtr &primitive,
                                                             const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_MAX_POOL_GRAD_WITH_ARGMAX_H_
