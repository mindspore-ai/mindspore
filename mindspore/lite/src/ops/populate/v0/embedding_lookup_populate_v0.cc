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

#include "schema/model_v0_generated.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/fp32/embedding_lookup_fp32.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateEmbeddingLookupParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto embedding_lookup_prim = primitive->value_as_EmbeddingLookup();
  EmbeddingLookupParameter *embedding_lookup_parameter =
    reinterpret_cast<EmbeddingLookupParameter *>(malloc(sizeof(EmbeddingLookupParameter)));
  if (embedding_lookup_parameter == nullptr) {
    MS_LOG(ERROR) << "malloc EmbeddingLookupParameter failed.";
    return nullptr;
  }
  memset(embedding_lookup_parameter, 0, sizeof(EmbeddingLookupParameter));
  embedding_lookup_parameter->op_parameter_.type_ = schema::PrimitiveType_EmbeddingLookupFusion;

  embedding_lookup_parameter->max_norm_ = embedding_lookup_prim->maxNorm();
  if (embedding_lookup_parameter->max_norm_ < 0) {
    MS_LOG(ERROR) << "Embedding lookup max norm should be positive number, got "
                  << embedding_lookup_parameter->max_norm_;
    free(embedding_lookup_parameter);
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(embedding_lookup_parameter);
}
}  // namespace

Registry g_embeddingLookupV0ParameterRegistry(schema::v0::PrimitiveType_EmbeddingLookup,
                                              PopulateEmbeddingLookupParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
