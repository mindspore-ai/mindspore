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

#include "c_ops/embedding_lookup_sparse.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> EmbeddingLookupSparse::GetSpIds() const {
  return this->primitive->value.AsEmbeddingLookupSparse()->spIds;
}
std::vector<float> EmbeddingLookupSparse::GetSpWeights() const {
  return this->primitive->value.AsEmbeddingLookupSparse()->spWeights;
}
float EmbeddingLookupSparse::GetMaxNortm() const { return this->primitive->value.AsEmbeddingLookupSparse()->maxNortm; }

void EmbeddingLookupSparse::SetSpIds(const std::vector<int> &sp_ids) {
  this->primitive->value.AsEmbeddingLookupSparse()->spIds = sp_ids;
}
void EmbeddingLookupSparse::SetSpWeights(const std::vector<float> &sp_weights) {
  this->primitive->value.AsEmbeddingLookupSparse()->spWeights = sp_weights;
}
void EmbeddingLookupSparse::SetMaxNortm(float max_nortm) {
  this->primitive->value.AsEmbeddingLookupSparse()->maxNortm = max_nortm;
}

#else

std::vector<int> EmbeddingLookupSparse::GetSpIds() const {
  auto fb_vector = this->primitive->value_as_EmbeddingLookupSparse()->spIds();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
std::vector<float> EmbeddingLookupSparse::GetSpWeights() const {
  auto fb_vector = this->primitive->value_as_EmbeddingLookupSparse()->spWeights();
  return std::vector<float>(fb_vector->begin(), fb_vector->end());
}
float EmbeddingLookupSparse::GetMaxNortm() const {
  return this->primitive->value_as_EmbeddingLookupSparse()->maxNortm();
}

void EmbeddingLookupSparse::SetSpIds(const std::vector<int> &sp_ids) {}
void EmbeddingLookupSparse::SetSpWeights(const std::vector<float> &sp_weights) {}
void EmbeddingLookupSparse::SetMaxNortm(float max_nortm) {}
#endif
}  // namespace mindspore
