/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "ops/custom.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/common.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(Custom, BaseOperator);
void Custom::Init(const std::string &type, const std::map<std::string, std::vector<uint8_t>> &attrs) {
  this->set_type(type);
  this->set_attr(attrs);
}

void Custom::set_type(const std::string &type) { (void)this->AddAttr(kType, api::MakeValue(type)); }

std::string Custom::get_type() const {
  auto value_ptr = this->GetAttr(kType);
  return GetValue<std::string>(value_ptr);
}

void Custom::set_attr(const std::map<std::string, std::vector<uint8_t>> &attrs) {
  api::ValuePtrList value_ptr_list;
  for (const auto &attr : attrs) {
    (void)value_ptr_list.emplace_back(api::MakeValue<std::string>(attr.first));
    (void)value_ptr_list.emplace_back(api::MakeValue<std::vector<uint8_t>>(attr.second));
  }
  (void)this->AddAttr(kAttr, api::MakeValue(value_ptr_list));
}

std::map<std::string, std::vector<uint8_t>> Custom::get_attr() const {
  std::map<std::string, std::vector<uint8_t>> attrs;
  auto value_ptr_list = GetValue<api::ValuePtrList>(this->GetAttr(kAttr));
  for (size_t i = 0; i < value_ptr_list.size(); i += 2) {
    auto key = GetValue<std::string>(value_ptr_list[i]);
    auto value = GetValue<std::vector<uint8_t>>(value_ptr_list[i + 1]);
    attrs[key] = value;
  }
  return attrs;
}
REGISTER_PRIMITIVE_C(kNameCustom, Custom);
}  // namespace ops
}  // namespace mindspore
