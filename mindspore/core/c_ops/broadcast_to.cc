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

#include "c_ops/broadcast_to.h"

namespace mindspore {
void BroadcastTo::Init(const std::vector<int64_t> &shape) { set_shape(shape); }

void BroadcastTo::set_shape(const std::vector<int64_t> &shape) {
  CheckAndConvertUtils::CheckInteger(kShapeSize, shape.size(), kGreaterThan, 0, name());
  CheckAndConvertUtils::CheckPositiveVector(kShape, shape, name(), false, true);
  AddAttr(kShape, MakeValue(shape));
}

std::vector<int64_t> BroadcastTo::get_shape() const {
  auto value_ptr = GetAttr(kShape);
  return GetValue<std::vector<int64_t>>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNameBroadcastTo, BroadcastTo);
}  // namespace mindspore
