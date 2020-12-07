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

#include "c_ops/adam.h"
#include "c_ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
void Adam::set_use_locking(const bool &use_locking) { this->AddAttr(kUseLocking, MakeValue(use_locking)); }

bool Adam::get_use_locking() const {
  auto value_ptr = GetAttr(kUseLocking);
  return GetValue<bool>(value_ptr);
}

void Adam::set_use_nesteroy(const bool &use_nesteroy) { this->AddAttr(kUseNesteroy, MakeValue(use_nesteroy)); }

bool Adam::get_use_nesteroy() const {
  auto value_ptr = GetAttr(kUseNesteroy);
  return GetValue<bool>(value_ptr);
}
void Adam::Init(const bool &use_locking, const bool &use_nesteroy) {
  this->set_use_locking(use_locking);
  this->set_use_nesteroy(use_nesteroy);
}
REGISTER_PRIMITIVE_C(kNameAdam, Adam);
}  // namespace mindspore
