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

#include "ops/grad/resize_grad.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(ResizeGrad, BaseOperator);
void ResizeGrad::Init(const ResizeMethod method, const bool align_corners) {
  this->set_method(method);
  this->set_align_corners(align_corners);
}

void ResizeGrad::set_method(const ResizeMethod method) {
  auto swi = static_cast<int64_t>(method);
  (void)this->AddAttr(kMethod, api::MakeValue(swi));
}

void ResizeGrad::set_align_corners(const bool align_corners) {
  (void)this->AddAttr(kAlignCorners, api::MakeValue(align_corners));
}

ResizeMethod ResizeGrad::get_method() const {
  auto value_ptr = GetAttr(kMethod);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return ResizeMethod(GetValue<int64_t>(value_ptr));
}

bool ResizeGrad::get_align_corners() const {
  auto value_ptr = GetAttr(kAlignCorners);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<bool>(value_ptr);
}

REGISTER_PRIMITIVE_C(kNameResizeGrad, ResizeGrad);
}  // namespace ops
}  // namespace mindspore
