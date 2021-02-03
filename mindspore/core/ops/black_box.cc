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

#include "ops/black_box.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
void BlackBox::Init(const std::string &id, const int64_t size, const std::vector<int64_t> &address) {
  this->set_id(id);
  this->set_size(size);
  this->set_address(address);
}

void BlackBox::set_id(const std::string &id) { this->AddAttr(kId, MakeValue(id)); }

std::string BlackBox::get_id() const {
  auto value_ptr = this->GetAttr(kId);
  return GetValue<std::string>(value_ptr);
}

void BlackBox::set_size(const int64_t size) { this->AddAttr(kSize, MakeValue(size)); }

int64_t BlackBox::get_size() const {
  auto value_ptr = this->GetAttr(kSize);
  return GetValue<int64_t>(value_ptr);
}

void BlackBox::set_address(const std::vector<int64_t> &address) { this->AddAttr(kAddress, MakeValue(address)); }

std::vector<int64_t> BlackBox::get_address() const {
  auto value_ptr = this->GetAttr(kAddress);
  return GetValue<std::vector<int64_t>>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNameBlackBox, BlackBox);
}  // namespace ops
}  // namespace mindspore
