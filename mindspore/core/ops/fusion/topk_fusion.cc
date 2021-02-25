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

#include "ops/fusion/topk_fusion.h"
#include <string>
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
void TopKFusion::Init(const bool sorted, const int64_t axis, const int64_t largest) {
  this->set_axis(axis);
  this->set_largest(largest);
  this->set_sorted(sorted);
}

void TopKFusion::set_axis(const int64_t axis) { this->AddAttr(kAxis, MakeValue(axis)); }

void TopKFusion::set_largest(const int64_t largest) { this->AddAttr(kLargest, MakeValue(largest)); }

int64_t TopKFusion::get_axis() const {
  auto value_ptr = GetAttr(kAxis);
  return GetValue<int64_t>(value_ptr);
}

int64_t TopKFusion::get_largest() const {
  auto value_ptr = GetAttr(kLargest);
  return GetValue<int64_t>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNameTopKFusion, TopKFusion);
}  // namespace ops
}  // namespace mindspore
