/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_AVG_POOL_3D_GRAD_H_
#define MINDSPORE_CORE_OPS_AVG_POOL_3D_GRAD_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/format.h"
#include "mindapi/base/types.h"
#include "infer/grad/pool_grad.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAvgPool3DGrad = "AvgPool3DGrad";

class OPS_API AvgPool3DGrad : public PoolGrad {
 public:
  MIND_API_BASE_MEMBER(AvgPool3DGrad);
  AvgPool3DGrad() : PoolGrad(kNameAvgPool3DGrad) { InitIOName({"origin_input_size", "grad"}, {"output"}); }

  void Init(const std::vector<int64_t> &kernel_size = {1, 1, 1, 1, 1},
            const std::vector<int64_t> &strides = {1, 1, 1, 1, 1}, const PadMode &pad_mode = VALID,
            const std::vector<int64_t> &pad_list = {0, 0, 0, 0, 0, 0}, bool ceil_mode = false,
            bool count_include_pad = true, int64_t divisor_override = 0, const Format &format = NCHW);

  void set_pad_list(const std::vector<int64_t> &pad_list);
  void set_ceil_mode(bool ceil_mode);
  void set_count_include_pad(bool count_include_pad);
  void set_divisor_override(int64_t divisor_override);

  std::vector<int64_t> get_pad_list() const;
  bool get_ceil_mode() const;
  bool get_count_include_pad() const;
  int64_t get_divisor_override() const;
};

OPS_API abstract::AbstractBasePtr AvgPool3DGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                     const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_AVG_POOL_3D_GRAD_H_
