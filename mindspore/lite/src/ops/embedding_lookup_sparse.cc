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

#include "src/ops/embedding_lookup_sparse.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> EmbeddingLookupSparse::GetSpIds() const {
  return this->primitive_->value.AsEmbeddingLookupSparse()->spIds;
}
std::vector<float> EmbeddingLookupSparse::GetSpWeights() const {
  return this->primitive_->value.AsEmbeddingLookupSparse()->spWeights;
}
float EmbeddingLookupSparse::GetMaxNortm() const { return this->primitive_->value.AsEmbeddingLookupSparse()->maxNortm; }

void EmbeddingLookupSparse::SetSpIds(const std::vector<int> &sp_ids) {
  this->primitive_->value.AsEmbeddingLookupSparse()->spIds = sp_ids;
}
void EmbeddingLookupSparse::SetSpWeights(const std::vector<float> &sp_weights) {
  this->primitive_->value.AsEmbeddingLookupSparse()->spWeights = sp_weights;
}
void EmbeddingLookupSparse::SetMaxNortm(float max_nortm) {
  this->primitive_->value.AsEmbeddingLookupSparse()->maxNortm = max_nortm;
}

#else
int EmbeddingLookupSparse::UnPackToFlatBuilder(const schema::Primitive *primitive,
                                               flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_EmbeddingLookupSparse();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_EmbeddingLookupSparse return nullptr";
    return RET_ERROR;
  }
  std::vector<int32_t> spIds;
  if (attr->spIds() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->spIds()->size()); i++) {
      spIds.push_back(attr->spIds()->data()[i]);
    }
  }
  std::vector<float> spWeights;
  if (attr->spWeights() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->spWeights()->size()); i++) {
      spWeights.push_back(attr->spWeights()->data()[i]);
    }
  }
  auto val_offset = schema::CreateEmbeddingLookupSparseDirect(*fbb, &spIds, &spWeights);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_EmbeddingLookupSparse, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
std::vector<int> EmbeddingLookupSparse::GetSpIds() const {
  auto fb_vector = this->primitive_->value_as_EmbeddingLookupSparse()->spIds();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
std::vector<float> EmbeddingLookupSparse::GetSpWeights() const {
  auto fb_vector = this->primitive_->value_as_EmbeddingLookupSparse()->spWeights();
  return std::vector<float>(fb_vector->begin(), fb_vector->end());
}
float EmbeddingLookupSparse::GetMaxNortm() const {
  return this->primitive_->value_as_EmbeddingLookupSparse()->maxNortm();
}

#endif
}  // namespace lite
}  // namespace mindspore
