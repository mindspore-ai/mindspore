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
#include "ops/glu.h"
#include <algorithm>
#include "ir/dtype/tensor_type.h"

namespace mindspore {
namespace ops {
void GLU::Init(int64_t axis) { set_axis(axis); }

void GLU::set_axis(int64_t axis) { (void)AddAttr(kAxis, MakeValue(axis)); }

int64_t GLU::get_axis() const {
  auto value_ptr = GetAttr(kAxis);
  return GetValue<int64_t>(value_ptr);
}
}  // namespace ops
}  // namespace mindspore
