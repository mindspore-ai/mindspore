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

#include "ops/split.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
void Split::Init(const std::vector<int64_t> &size_splits, const int64_t axis, const int64_t output_num) {
  this->set_axis(axis);
  this->set_output_num(output_num);
}

void Split::set_size_splits(const std::vector<int64_t> &size_splits) {
  this->AddAttr(kSizeSplits, MakeValue(size_splits));
}
void Split::set_axis(const int64_t axis) { this->AddAttr(kAxis, MakeValue(axis)); }
void Split::set_output_num(const int64_t output_num) { this->AddAttr(kOutputNum, MakeValue(output_num)); }

std::vector<int64_t> Split::get_size_splits() const {
  auto value_ptr = GetAttr(kSizeSplits);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

int64_t Split::get_axis() const {
  auto value_ptr = GetAttr(kAxis);
  return GetValue<int64_t>(value_ptr);
}

int64_t Split::get_output_num() const {
  auto value_ptr = GetAttr(kOutputNum);
  return GetValue<int64_t>(value_ptr);
}

REGISTER_PRIMITIVE_C(kNameSplit, Split);
}  // namespace ops
}  // namespace mindspore
