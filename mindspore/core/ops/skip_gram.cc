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

#include "ops/skip_gram.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void SkipGram::set_include_all_grams(const bool include_all_grams) {
  (void)AddAttr(kIncludeALLGrams, api::MakeValue(include_all_grams));
}
bool SkipGram::get_include_all_grams() const {
  auto value_ptr = this->GetAttr(kIncludeALLGrams);
  return GetValue<bool>(value_ptr);
}
void SkipGram::set_max_skip_size(const int64_t max_skip_size) {
  (void)AddAttr(kMaxSkipSize, api::MakeValue(max_skip_size));
}
int64_t SkipGram::get_max_skip_size() const {
  auto value_ptr = this->GetAttr(kMaxSkipSize);
  return GetValue<int64_t>(value_ptr);
}
void SkipGram::set_ngram_size(const int64_t ngram_size) { (void)AddAttr(kNgramSize, api::MakeValue(ngram_size)); }
int64_t SkipGram::get_ngram_size() const {
  auto value_ptr = this->GetAttr(kNgramSize);
  return GetValue<int64_t>(value_ptr);
}
void SkipGram::Init(const bool include_all_grams, const int64_t max_skip_size, const int64_t ngram_size) {
  this->set_include_all_grams(include_all_grams);
  this->set_max_skip_size(max_skip_size);
  this->set_ngram_size(ngram_size);
}

MIND_API_OPERATOR_IMPL(SkipGram, BaseOperator);
REGISTER_PRIMITIVE_C(kNameSkipGram, SkipGram);
}  // namespace ops
}  // namespace mindspore
