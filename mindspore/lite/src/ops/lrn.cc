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

#include "src/ops/lrn.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
float Lrn::GetAlpha() const { return this->primitive_->value.AsLrn()->alpha; }
float Lrn::GetBeta() const { return this->primitive_->value.AsLrn()->beta; }
float Lrn::GetBias() const { return this->primitive_->value.AsLrn()->bias; }
int Lrn::GetSize() const { return this->primitive_->value.AsLrn()->size; }

void Lrn::SetAlpha(float alpha) { this->primitive_->value.AsLrn()->alpha = alpha; }
void Lrn::SetBeta(float beta) { this->primitive_->value.AsLrn()->beta = beta; }
void Lrn::SetBias(float bias) { this->primitive_->value.AsLrn()->bias = bias; }
void Lrn::SetSize(int size) { this->primitive_->value.AsLrn()->size = size; }

#else

float Lrn::GetAlpha() const { return this->primitive_->value_as_Lrn()->alpha(); }
float Lrn::GetBeta() const { return this->primitive_->value_as_Lrn()->beta(); }
float Lrn::GetBias() const { return this->primitive_->value_as_Lrn()->bias(); }
int Lrn::GetSize() const { return this->primitive_->value_as_Lrn()->size(); }

int Lrn::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Lrn();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Lrn return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateLrn(*fbb, attr->alpha(), attr->beta(), attr->bias(), attr->size());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Lrn, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
#endif
}  // namespace lite
}  // namespace mindspore
