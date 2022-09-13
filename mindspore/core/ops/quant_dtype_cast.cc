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

#include "ops/quant_dtype_cast.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(QuantDTypeCast, BaseOperator);
void QuantDTypeCast::set_src_t(const int64_t src_t) { (void)AddAttr(kSrcT, api::MakeValue(src_t)); }
int64_t QuantDTypeCast::get_src_t() const {
  auto value_ptr = this->GetAttr(kSrcT);
  return GetValue<int64_t>(value_ptr);
}
void QuantDTypeCast::set_dst_t(const int64_t dst_t) { (void)AddAttr(kDstT, api::MakeValue(dst_t)); }
int64_t QuantDTypeCast::get_dst_t() const { return GetValue<int64_t>(GetAttr(kDstT)); }
void QuantDTypeCast::set_axis(const int64_t axis) { (void)AddAttr(kAxis, api::MakeValue(axis)); }
int64_t QuantDTypeCast::get_axis() const { return GetValue<int64_t>(GetAttr(kAxis)); }

void QuantDTypeCast::Init(const int64_t src_t, const int64_t dst_t) {
  this->set_src_t(src_t);
  this->set_dst_t(dst_t);
}

REGISTER_PRIMITIVE_C(kNameQuantDTypeCast, QuantDTypeCast);
}  // namespace ops
}  // namespace mindspore
