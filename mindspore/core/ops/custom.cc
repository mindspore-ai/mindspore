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

#include "ops/custom.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
void Custom::Init(const std::vector<int64_t> &custom) { this->set_custom(custom); }

void Custom::set_custom(const std::vector<int64_t> &custom) { this->AddAttr(kCustom, MakeValue(custom)); }

std::vector<int64_t> Custom::get_custom() const {
  auto value_ptr = this->GetAttr(kCustom);
  return GetValue<std::vector<int64_t>>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNameCustom, Custom);
}  // namespace ops
}  // namespace mindspore
