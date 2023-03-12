/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "src/common/ops/operator_populate/operator_populate_register.h"
#include "nnacl/fp32/embedding_lookup_fp32.h"
#include "ops/embedding_lookup.h"
using mindspore::ops::kMaxNorm;
using mindspore::ops::kNameEmbeddingLookup;
using mindspore::schema::PrimitiveType_EmbeddingLookupFusion;

namespace mindspore {
namespace lite {
OpParameter *PopulateEmbeddingLookupOpParameter(const BaseOperatorPtr &base_operator) {
  auto param =
    reinterpret_cast<EmbeddingLookupParameter *>(PopulateOpParameter<EmbeddingLookupParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new EmbeddingLookupParameter failed.";
    return nullptr;
  }

  auto attr_max_nrom = base_operator->GetPrim()->GetAttr(kMaxNorm);
  if (attr_max_nrom == nullptr) {
    MS_LOG(ERROR) << "The attr(" << kMaxNorm << ") of operator(" << base_operator->name() << ") not exist";
    free(param);
    return nullptr;
  }
  auto max_nrom = GetValue<float>(attr_max_nrom);
  if (max_nrom < 0) {
    MS_LOG(ERROR) << "Embedding lookup max norm should be positive number, got " << max_nrom;
    free(param);
    return nullptr;
  }
  param->max_norm_ = max_nrom;
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameEmbeddingLookup, PrimitiveType_EmbeddingLookupFusion, PopulateEmbeddingLookupOpParameter)
}  // namespace lite
}  // namespace mindspore
