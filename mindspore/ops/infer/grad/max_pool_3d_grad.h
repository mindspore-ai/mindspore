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

#ifndef MINDSPORE_CORE_OPS_MAX_POOL_3D_GRAD_H_
#define MINDSPORE_CORE_OPS_MAX_POOL_3D_GRAD_H_
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "infer/grad/pool_grad.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMaxPool3DGrad = "MaxPool3DGrad";
class OPS_API MaxPool3DGrad : public PoolGrad {
 public:
  MIND_API_BASE_MEMBER(MaxPool3DGrad);
  MaxPool3DGrad() : PoolGrad(kNameMaxPool3DGrad) { InitIOName({"x_origin", "out_origin", "grad"}, {"output"}); }

  void Init(const std::vector<int64_t> &kernel_size = {1, 1, 1, 1, 1},
            const std::vector<int64_t> &strides = {1, 1, 1, 1, 1}, const PadMode &pad_mode = VALID,
            const std::vector<int64_t> &pad_list = {0, 0, 0, 0, 0, 0}, const Format &format = NCHW);
  void set_pad_list(const std::vector<int64_t> &pad_list);
  std::vector<int64_t> get_pad_list() const;
};

OPS_API abstract::AbstractBasePtr MaxPool3DGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                     const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimMaxPool3DGradPtr = std::shared_ptr<MaxPool3DGrad>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_MAX_POOL_3D_GRAD_H_
