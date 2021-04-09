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

#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/fusion/reduce_fusion.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
void ReduceFusion::set_keep_dims(const bool keep_dims) { this->AddAttr(kKeepDims, MakeValue(keep_dims)); }

void ReduceFusion::set_mode(const ReduceMode mode) {
  int64_t swi;
  swi = mode;
  this->AddAttr(kMode, MakeValue(swi));
}

void ReduceFusion::set_reduce_to_end(const bool reduce_to_end) {
  this->AddAttr(kReduceToEnd, MakeValue(reduce_to_end));
}

void ReduceFusion::set_coeff(const float coeff) { this->AddAttr(kCoeff, MakeValue(coeff)); }

bool ReduceFusion::get_keep_dims() const {
  auto value_ptr = GetAttr(kKeepDims);
  return GetValue<bool>(value_ptr);
}

ReduceMode ReduceFusion::get_mode() const {
  auto value_ptr = GetAttr(kMode);
  return ReduceMode(GetValue<int64_t>(value_ptr));
}

bool ReduceFusion::get_reduce_to_end() const {
  auto value_ptr = GetAttr(kReduceToEnd);
  return GetValue<bool>(value_ptr);
}

float ReduceFusion::get_coeff() const {
  auto value_ptr = GetAttr(kCoeff);
  return GetValue<float>(value_ptr);
}

void ReduceFusion::Init(const bool keep_dims, const ReduceMode mode, const bool reduce_to_end, const float coeff) {
  this->set_keep_dims(keep_dims);
  this->set_mode(mode);
  this->set_reduce_to_end(reduce_to_end);
  this->set_coeff(coeff);
}
REGISTER_PRIMITIVE_C(kNameReduceFusion, ReduceFusion);
}  // namespace ops
}  // namespace mindspore
