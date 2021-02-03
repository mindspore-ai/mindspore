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

#include "ops/fusion/embedding_lookup_fusion.h"
#include <string>
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
void EmbeddingLookupFusion::set_max_norm(const float max_norm) { this->AddAttr(kMaxNorm, MakeValue(max_norm)); }
float EmbeddingLookupFusion::get_max_norm() const {
  auto value_ptr = GetAttr(kMaxNorm);
  return GetValue<float>(value_ptr);
}
void EmbeddingLookupFusion::Init(const float max_norm) { this->set_max_norm(max_norm); }
REGISTER_PRIMITIVE_C(kNameEmbeddingLookupFusion, EmbeddingLookupFusion);
}  // namespace ops
}  // namespace mindspore
