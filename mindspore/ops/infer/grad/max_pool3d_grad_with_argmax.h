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

#ifndef MINDSPORE_CORE_OPS_MAX_POOL3D_GRAD_WITH_ARGMAX_H_
#define MINDSPORE_CORE_OPS_MAX_POOL3D_GRAD_WITH_ARGMAX_H_

#include <memory>
#include <set>
#include <string>
#include <vector>
#include "ops/base_operator.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMaxPool3DGradWithArgmax = "MaxPool3DGradWithArgmax";
class OPS_API MaxPool3DGradWithArgmax : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(MaxPool3DGradWithArgmax);
  MaxPool3DGradWithArgmax() : BaseOperator(kNameMaxPool3DGradWithArgmax) {
    InitIOName({"x", "grads", "argmax"}, {"y"});
  }

  void Init(const std::vector<int64_t> &kernel_size, const std::vector<int64_t> &strides,
            const std::vector<int64_t> &pads, const std::vector<int64_t> &dialtion = {1, 1, 1}, bool ceil_mode = false,
            const Format &format = NCDHW);

  void set_kernel_size(const std::vector<int64_t> &kernel_size);
  void set_strides(const std::vector<int64_t> &strides);
  void set_pads(const std::vector<int64_t> &pads);
  void set_dilation(const std::vector<int64_t> &dialtion);
  void set_ceil_mode(bool ceil_mode);
  void set_format(const Format &format);

  std::vector<int64_t> get_kernel_size() const;
  std::vector<int64_t> get_strides() const;
  std::vector<int64_t> get_pads() const;
  std::vector<int64_t> get_dilation() const;
  bool get_ceil_mode() const;
  Format get_format() const;
};

OPS_API abstract::AbstractBasePtr MaxPool3DGradWithArgmaxInfer(
  const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
  const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimMaxPool3DGradWithArgmax = std::shared_ptr<MaxPool3DGradWithArgmax>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_MAX_POOL3D_GRAD_WITH_ARGMAX_H_
