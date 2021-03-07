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
#include "nnacl/skip_gram_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateSkipGramParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto skip_gram_prim = primitive->value_as_SkipGram();
  SkipGramParameter *skipGramParameter = reinterpret_cast<SkipGramParameter *>(malloc(sizeof(SkipGramParameter)));
  if (skipGramParameter == nullptr) {
    MS_LOG(ERROR) << "malloc SkipGramParameter failed.";
    return nullptr;
  }
  memset(skipGramParameter, 0, sizeof(SkipGramParameter));
  skipGramParameter->op_parameter_.type_ = schema::PrimitiveType_SkipGram;

  skipGramParameter->ngram_size = skip_gram_prim->ngramSize();
  skipGramParameter->max_skip_size = skip_gram_prim->maxSkipSize();
  skipGramParameter->include_all_ngrams = skip_gram_prim->includeAllGrams();
  return reinterpret_cast<OpParameter *>(skipGramParameter);
}
}  // namespace

Registry g_skipGramV0ParameterRegistry(schema::v0::PrimitiveType_SkipGram, PopulateSkipGramParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
