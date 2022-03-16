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

#include <set>
#include "ops/embedding_lookup.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
void EmbeddingLookup::Init(const bool setattr_flag) { this->set_setattr_flag(setattr_flag); }

void EmbeddingLookup::set_setattr_flag(const bool setattr_flag) {
  (void)this->AddAttr(kSetattrFlag, MakeValue(setattr_flag));
}

bool EmbeddingLookup::get_setattr_flag() const {
  auto value_ptr = GetAttr(kSetattrFlag);
  return GetValue<bool>(value_ptr);
}

REGISTER_PRIMITIVE_C(kNameEmbeddingLookup, EmbeddingLookup);
}  // namespace ops
}  // namespace mindspore
