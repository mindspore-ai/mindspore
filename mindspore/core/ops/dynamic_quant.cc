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

#include "ops/dynamic_quant.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(DynamicQuant, BaseOperator);
void DynamicQuant::set_symmetric(const bool symmetric) { (void)AddAttr(kSymmetric, api::MakeValue(symmetric)); }
bool DynamicQuant::get_symmetric() const {
  auto value_ptr = this->GetAttr(kSymmetric);
  return GetValue<bool>(value_ptr);
}
void DynamicQuant::set_dst_type(const int64_t dst_type) { (void)AddAttr(kDstType, api::MakeValue(dst_type)); }
int64_t DynamicQuant::get_dst_type() const { return GetValue<int64_t>(GetAttr(kDstType)); }
void DynamicQuant::Init(const bool symmetric, const int64_t dst_type) {
  this->set_symmetric(symmetric);
  this->set_dst_type(dst_type);
}

REGISTER_PRIMITIVE_C(kNameDynamicQuant, DynamicQuant);
}  // namespace ops
}  // namespace mindspore
