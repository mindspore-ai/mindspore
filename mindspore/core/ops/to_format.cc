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

#include "ops/to_format.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
void ToFormat::set_src_t(const int64_t src_t) { this->AddAttr(kSrcT, MakeValue(src_t)); }
int64_t ToFormat::get_src_t() const {
  auto value_ptr = GetAttr(kSrcT);
  return GetValue<int64_t>(value_ptr);
}

void ToFormat::set_dst_t(const int64_t dst_t) { this->AddAttr(kDstT, MakeValue(dst_t)); }
int64_t ToFormat::get_dst_t() const {
  auto value_ptr = GetAttr(kDstT);
  return GetValue<int64_t>(value_ptr);
}

void ToFormat::Init(const int64_t src_t, const int64_t dst_t) {
  this->set_src_t(src_t);
  this->set_dst_t(dst_t);
}
REGISTER_PRIMITIVE_C(kNameToFormat, ToFormat);
}  // namespace ops
}  // namespace mindspore
