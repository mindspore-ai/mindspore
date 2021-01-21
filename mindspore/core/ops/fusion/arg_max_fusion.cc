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

#include "ops/fusion/arg_max_fusion.h"

namespace mindspore {
namespace ops {
void ArgMaxFusion::Init(const bool keep_dims, const bool out_max_value, const int64_t top_k, const int64_t axis) {
  set_axis(axis);
  set_keep_dims(keep_dims);
  set_out_max_value(out_max_value);
  set_top_k(top_k);
}

void ArgMaxFusion::set_keep_dims(const bool keep_dims) { this->AddAttr(kKeepDims, MakeValue(keep_dims)); }
void ArgMaxFusion::set_out_max_value(const bool out_max_value) {
  this->AddAttr(kOutMaxValue, MakeValue(out_max_value));
}
void ArgMaxFusion::set_top_k(const int64_t top_k) { this->AddAttr(kTopK, MakeValue(top_k)); }

bool ArgMaxFusion::get_keep_dims() const { return GetValue<bool>(GetAttr(kKeepDims)); }
bool ArgMaxFusion::get_out_max_value() const { return GetValue<bool>(GetAttr(kOutMaxValue)); }
int64_t ArgMaxFusion::get_top_k() const { return GetValue<int64_t>(GetAttr(kTopK)); }

REGISTER_PRIMITIVE_C(kNameArgMaxFusion, ArgMaxFusion);
}  // namespace ops
}  // namespace mindspore
