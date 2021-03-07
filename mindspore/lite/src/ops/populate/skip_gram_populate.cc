/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "src/ops/populate/populate_register.h"
#include "nnacl/skip_gram_parameter.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateSkipGramParameter(const void *prim) {
  SkipGramParameter *skipGramParameter = reinterpret_cast<SkipGramParameter *>(malloc(sizeof(SkipGramParameter)));
  if (skipGramParameter == nullptr) {
    MS_LOG(ERROR) << "malloc SkipGramParameter failed.";
    return nullptr;
  }
  memset(skipGramParameter, 0, sizeof(SkipGramParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_SkipGram();
  skipGramParameter->op_parameter_.type_ = primitive->value_type();
  skipGramParameter->ngram_size = value->ngram_size();
  skipGramParameter->max_skip_size = value->max_skip_size();
  skipGramParameter->include_all_ngrams = value->include_all_grams();
  return reinterpret_cast<OpParameter *>(skipGramParameter);
}
Registry SkipGramParameterRegistry(schema::PrimitiveType_SkipGram, PopulateSkipGramParameter, SCHEMA_CUR);

}  // namespace lite
}  // namespace mindspore
