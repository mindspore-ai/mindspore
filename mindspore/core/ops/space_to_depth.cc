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

#include "ops/space_to_depth.h"

namespace mindspore {
namespace ops {
void SpaceToDepth::Init(const int64_t block_size, const Format &format) {
  this->set_block_size(block_size);
  this->set_format(format);
}

void SpaceToDepth::set_block_size(const int64_t block_size) {
  CheckAndConvertUtils::Check(kBlockSize, block_size, kGreaterEqual, "", 2, this->name());
  AddAttr(kBlockSize, MakeValue(block_size));
}

int64_t SpaceToDepth::get_block_size() const {
  auto value_ptr = GetAttr(kBlockSize);
  return GetValue<int64_t>(value_ptr);
}

void SpaceToDepth::set_format(const Format &format) {
  int64_t f = format;
  this->AddAttr(kFormat, MakeValue(f));
}

Format SpaceToDepth::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  return Format(GetValue<int64_t>(value_ptr));
}
REGISTER_PRIMITIVE_C(kNameSpaceToDepth, SpaceToDepth);
}  // namespace ops
}  // namespace mindspore
