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

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int SkipGram::GetNgramSize() const { return this->primitive_->value.AsSkipGram()->ngramSize; }
int SkipGram::GetMaxSkipSize() const { return this->primitive_->value.AsSkipGram()->maxSkipSize; }
bool SkipGram::GetIncludeAllNgrams() const { return this->primitive_->value.AsSkipGram()->includeAllGrams; }

void SkipGram::SetNgramSize(int ngram_size) { this->primitive_->value.AsSkipGram()->ngramSize = ngram_size; }
void SkipGram::SetMaxSkipSize(int max_skip_size) { this->primitive_->value.AsSkipGram()->maxSkipSize = max_skip_size; }
void SkipGram::SetIncludeAllNgrams(bool include_all_ngrams) {
  this->primitive_->value.AsSkipGram()->includeAllGrams = include_all_ngrams;
}

#else
int SkipGram::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);

  auto attr = primitive->value_as_SkipGram();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_SkipGram return nullptr";
    return RET_ERROR;
  }

  auto val_offset = schema::CreateSkipGram(*fbb, attr->includeAllGrams(), attr->maxSkipSize(), attr->ngramSize());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_SkipGram, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

int SkipGram::GetNgramSize() const { return this->primitive_->value_as_SkipGram()->ngramSize(); }
int SkipGram::GetMaxSkipSize() const { return this->primitive_->value_as_SkipGram()->maxSkipSize(); }
bool SkipGram::GetIncludeAllNgrams() const { return this->primitive_->value_as_SkipGram()->includeAllGrams(); }

PrimitiveC *SkipGramCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<SkipGram>(primitive);
}
Registry SkipGramRegistry(schema::PrimitiveType_SkipGram, SkipGramCreator);
#endif

int SkipGram::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  if (inputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "Skip Gram should have one input";
    return RET_INPUT_TENSOR_ERROR;
  }
  if (outputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "Skip Gram should have one outputs";
    return RET_INPUT_TENSOR_ERROR;
  }
  auto input = inputs_.front();
  auto output = outputs_.front();
  MS_ASSERT(input != nullptr);
  output->set_format(input->format());
  output->set_data_type(input->data_type());

  if (input->data_c() == nullptr) {
    MS_LOG(INFO) << "Do infer shape in runtime.";
    return RET_INFER_INVALID;
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
