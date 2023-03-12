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
#include "nnacl/skip_gram_parameter.h"
#include "ops/skip_gram.h"
using mindspore::ops::kNameSkipGram;
using mindspore::schema::PrimitiveType_SkipGram;
namespace mindspore {
namespace lite {
OpParameter *PopulateSkipGramOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<SkipGramParameter *>(PopulateOpParameter<SkipGramParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new SkipGramParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::SkipGram *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not SkipGram.";
    free(param);
    return nullptr;
  }
  param->ngram_size = static_cast<int>(op->get_ngram_size());
  param->max_skip_size = static_cast<int>(op->get_max_skip_size());
  param->include_all_ngrams = op->get_include_all_grams();
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameSkipGram, PrimitiveType_SkipGram, PopulateSkipGramOpParameter)
}  // namespace lite
}  // namespace mindspore
