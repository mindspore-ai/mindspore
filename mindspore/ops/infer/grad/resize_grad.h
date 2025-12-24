/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_GRAD_RESIZE_GRAD_H_
#define MINDSPORE_CORE_OPS_GRAD_RESIZE_GRAD_H_
#include <memory>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameResizeGrad = "ResizeGrad";
class OPS_API ResizeGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ResizeGrad);
  ResizeGrad() : BaseOperator(kNameResizeGrad) {}
  void Init(const ResizeMethod method, const bool align_corners);
  void set_method(const ResizeMethod method);
  void set_align_corners(const bool align_corners);
  ResizeMethod get_method() const;
  bool get_align_corners() const;
};

OPS_API abstract::AbstractBasePtr ResizeGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                  const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_GRAD_RESIZE_GRAD_H_
