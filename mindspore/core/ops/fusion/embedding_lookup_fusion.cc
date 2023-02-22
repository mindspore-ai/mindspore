/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(EmbeddingLookupFusion, BaseOperator);
void EmbeddingLookupFusion::set_max_norm(const float max_norm) {
  (void)this->AddAttr(kMaxNorm, api::MakeValue(max_norm));
}
float EmbeddingLookupFusion::get_max_norm() const {
  auto value_ptr = GetAttr(kMaxNorm);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<float>(value_ptr);
}
void EmbeddingLookupFusion::Init(const float max_norm) { this->set_max_norm(max_norm); }
REGISTER_PRIMITIVE_C(kNameEmbeddingLookupFusion, EmbeddingLookupFusion);
}  // namespace ops
}  // namespace mindspore
