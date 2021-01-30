/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "src/ops/skip_gram.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "mindspore/lite/nnacl/skip_gram_parameter.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateSkipGramParameter(const mindspore::lite::PrimitiveC *primitive) {
  SkipGramParameter *skipGramParameter = reinterpret_cast<SkipGramParameter *>(malloc(sizeof(SkipGramParameter)));
  if (skipGramParameter == nullptr) {
    MS_LOG(ERROR) << "malloc SkipGramParameter failed.";
    return nullptr;
  }
  memset(skipGramParameter, 0, sizeof(SkipGramParameter));
  skipGramParameter->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::SkipGram *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  skipGramParameter->ngram_size = param->GetNgramSize();
  skipGramParameter->max_skip_size = param->GetMaxSkipSize();
  skipGramParameter->include_all_ngrams = param->GetIncludeAllNgrams();
  return reinterpret_cast<OpParameter *>(skipGramParameter);
}
Registry SkipGramParameterRegistry(schema::PrimitiveType_SkipGram, PopulateSkipGramParameter);

}  // namespace lite
}  // namespace mindspore
