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

#include "ops/fusion/arg_min_fusion.h"

#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(ArgMinFusion, ArgMin);
void ArgMinFusion::Init(bool keep_dims, bool out_max_value, int64_t top_k, int64_t axis) {
  set_axis(axis);
  set_keep_dims(keep_dims);
  set_out_max_value(out_max_value);
  set_top_k(top_k);
}

void ArgMinFusion::set_keep_dims(const bool keep_dims) { (void)this->AddAttr(kKeepDims, api::MakeValue(keep_dims)); }
void ArgMinFusion::set_out_max_value(bool out_max_value) { (void)AddAttr(kOutMaxValue, api::MakeValue(out_max_value)); }
void ArgMinFusion::set_top_k(int64_t top_k) { (void)this->AddAttr(kTopK, api::MakeValue(top_k)); }

bool ArgMinFusion::get_keep_dims() const {
  auto keep_dims = GetAttr(kKeepDims);
  MS_EXCEPTION_IF_NULL(keep_dims);
  return GetValue<bool>(keep_dims);
}

bool ArgMinFusion::get_out_max_value() const {
  auto value_ptr = GetAttr(kOutMaxValue);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<bool>(value_ptr);
}

int64_t ArgMinFusion::get_top_k() const {
  auto value_ptr = GetAttr(kTopK);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

REGISTER_PRIMITIVE_C(kNameArgMinFusion, ArgMinFusion);
}  // namespace ops
}  // namespace mindspore
