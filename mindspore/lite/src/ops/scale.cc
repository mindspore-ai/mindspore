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

#include "src/ops/scale.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int Scale::GetAxis() const { return this->primitive_->value.AsScale()->axis; }

void Scale::SetAxis(int axis) { this->primitive_->value.AsScale()->axis = axis; }

#else

int Scale::GetAxis() const { return this->primitive_->value_as_Scale()->axis(); }
int Scale::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Scale();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Scale return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateScale(*fbb, attr->axis());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Scale, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
#endif
}  // namespace lite
}  // namespace mindspore
