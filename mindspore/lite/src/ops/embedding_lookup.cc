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

#include "src/ops/embedding_lookup.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
float EmbeddingLookup::GetMaxNorm() const { return this->primitive_->value.AsEmbeddingLookup()->maxNorm; }

void EmbeddingLookup::SetMaxNorm(float max_norm) { this->primitive_->value.AsEmbeddingLookup()->maxNorm = max_norm; }

#else
int EmbeddingLookup::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);

  auto attr = primitive->value_as_EmbeddingLookup();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_EmbeddingLookup return nullptr";
    return RET_ERROR;
  }

  auto val_offset = schema::CreateEmbeddingLookup(*fbb, attr->maxNorm());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_EmbeddingLookup, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
float EmbeddingLookup::GetMaxNorm() const { return this->primitive_->value_as_EmbeddingLookup()->maxNorm(); }

PrimitiveC *EmbeddingLookupCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<EmbeddingLookup>(primitive);
}
Registry EmbeddingLookupRegistry(schema::PrimitiveType_EmbeddingLookup, EmbeddingLookupCreator);
#endif

int EmbeddingLookup::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  if (inputs_.size() < kDoubleNum) {
    MS_LOG(ERROR) << "Embedding Lookup should have at least two inputs";
    return RET_INPUT_TENSOR_ERROR;
  }
  if (outputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "Embedding Lookup should have one outputs";
    return RET_INPUT_TENSOR_ERROR;
  }
  auto params_ = inputs_.front();
  MS_ASSERT(params_ != nullptr);
  auto ids = inputs_.back();
  MS_ASSERT(ids != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  output->set_format(params_->format());
  output->set_data_type(params_->data_type());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }

  auto embedding_shape = params_->shape();
  embedding_shape.erase(embedding_shape.begin());
  std::vector<int> output_shape(ids->shape());
  for (size_t i = 0; i < embedding_shape.size(); ++i) {
    output_shape.push_back(embedding_shape.at(i));
  }
  for (size_t i = 1; i < inputs_.size() - 1; ++i) {
    auto embedding_shape_t = inputs_.at(i)->shape();
    embedding_shape_t.erase(embedding_shape_t.begin());
    if (embedding_shape_t != embedding_shape) {
      MS_LOG(ERROR) << "The embedded layers should have the same shape";
      return RET_INPUT_TENSOR_ERROR;
    }
  }
  output->set_shape(output_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
