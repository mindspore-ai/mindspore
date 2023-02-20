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

#include "ops/fusion/topk_fusion.h"

#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/base_operator.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(TopKFusion, TopK);
void TopKFusion::Init(const bool sorted, const int64_t axis, const int64_t largest) {
  this->set_axis(axis);
  this->set_largest(largest);
  this->set_sorted(sorted);
}

void TopKFusion::set_axis(const int64_t axis) { (void)this->AddAttr(kAxis, api::MakeValue(axis)); }

void TopKFusion::set_largest(const int64_t largest) { (void)this->AddAttr(kLargest, api::MakeValue(largest)); }

int64_t TopKFusion::get_axis() const {
  auto value_ptr = GetAttr(kAxis);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

int64_t TopKFusion::get_largest() const {
  auto value_ptr = GetAttr(kLargest);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNameTopKFusion, TopKFusion);
}  // namespace ops
}  // namespace mindspore
