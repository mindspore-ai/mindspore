/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "c_ops/arg_min.h"

namespace mindspore {
void ArgMin::Init(bool keep_dims, int64_t axis) {
  set_axis(axis);
  set_keep_dims(keep_dims);
}

void ArgMin::set_axis(int64_t axis) { this->AddAttr(kAxis, MakeValue(axis)); }
void ArgMin::set_keep_dims(bool keep_dims) { this->AddAttr(kOutputType, MakeValue(keep_dims)); }

int64_t ArgMin::get_axis() {
  auto value_ptr = GetAttr(kAxis);
  return GetValue<int64_t>(value_ptr);
}

bool ArgMin::get_keep_dims() {
  auto value_ptr = GetAttr(kKeepDims);
  return GetValue<bool>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNameArgMin, ArgMin);
}  // namespace mindspore
