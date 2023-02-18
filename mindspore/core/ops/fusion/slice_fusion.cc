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

#include "ops/fusion/slice_fusion.h"
#include <string>
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(SliceFusion, BaseOperator);
void SliceFusion::Init(const std::vector<int64_t> &axes) { this->set_axes(axes); }

void SliceFusion::set_axes(const std::vector<int64_t> &axes) { (void)this->AddAttr(kAxes, api::MakeValue(axes)); }

std::vector<int64_t> SliceFusion::get_axes() const {
  auto value_ptr = GetAttr(kAxes);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

REGISTER_PRIMITIVE_C(kNameSliceFusion, SliceFusion);
}  // namespace ops
}  // namespace mindspore
