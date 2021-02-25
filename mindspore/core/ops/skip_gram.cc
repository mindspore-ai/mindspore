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
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto SkipGram_prim = primitive->cast<PrimSkipGramPtr>();
  MS_EXCEPTION_IF_NULL(SkipGram_prim);
  auto prim_name = SkipGram_prim->name();
  if (input_args.size() != 1) {
    MS_LOG(ERROR) << "Skip Gram should have one input";
  }
  auto infer_value = input_args[0]->BuildValue();
  if (infer_value == nullptr) {
    MS_LOG(INFO) << "Do infer shape in runtime.";
  }
  auto in_shape = CheckAndConvertUtils::ConvertShapePtrToShape("in_shape", input_args[0]->BuildShape(), prim_name);
  return std::make_shared<abstract::Shape>(in_shape);
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = input_args[0]->BuildType();
  return infer_type;
}
}  // namespace

void SkipGram::set_include_all_grams(const bool include_all_grams) {
  AddAttr(kIncludeALLGrams, MakeValue(include_all_grams));
}
bool SkipGram::get_include_all_grams() const {
  auto value_ptr = this->GetAttr(kIncludeALLGrams);
  return GetValue<bool>(value_ptr);
}
void SkipGram::set_max_skip_size(const int64_t max_skip_size) { AddAttr(kMaxSkipSize, MakeValue(max_skip_size)); }
int64_t SkipGram::get_max_skip_size() const {
  auto value_ptr = this->GetAttr(kMaxSkipSize);
  return GetValue<int64_t>(value_ptr);
}
void SkipGram::set_ngram_size(const int64_t ngram_size) { AddAttr(kNgramSize, MakeValue(ngram_size)); }
int64_t SkipGram::get_ngram_size() const {
  auto value_ptr = this->GetAttr(kNgramSize);
  return GetValue<int64_t>(value_ptr);
}
void SkipGram::Init(const bool include_all_grams, const int64_t max_skip_size, const int64_t ngram_size) {
  this->set_include_all_grams(include_all_grams);
  this->set_max_skip_size(max_skip_size);
  this->set_ngram_size(ngram_size);
}

AbstractBasePtr SkipGramInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(InferType(primitive, input_args),
                                                    InferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNameSkipGram, SkipGram);
}  // namespace ops
}  // namespace mindspore
