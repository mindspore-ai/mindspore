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

#include "src/ops/clip.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
float Clip::GetMax() const { return this->primitive_->value.AsClip()->max; }
float Clip::GetMin() const { return this->primitive_->value.AsClip()->min; }

void Clip::SetMax(float max) { this->primitive_->value.AsClip()->max = max; }
void Clip::SetMin(float min) { this->primitive_->value.AsClip()->min = min; }

#else
int Clip::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Clip();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Clip return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateClip(*fbb, attr->max(), attr->min());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Clip, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
float Clip::GetMax() const { return this->primitive_->value_as_Clip()->max(); }
float Clip::GetMin() const { return this->primitive_->value_as_Clip()->min(); }

PrimitiveC *ClipCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Clip>(primitive); }
Registry ClipRegistry(schema::PrimitiveType_Clip, ClipCreator);
#endif

}  // namespace lite
}  // namespace mindspore
