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

#include "src/ops/not_equal.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
#else
int NotEqual::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto val_offset = schema::CreateNotEqual(*fbb);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_NotEqual, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
PrimitiveC *NotEqualCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<NotEqual>(primitive);
}
Registry NotEqualRegistry(schema::PrimitiveType_NotEqual, NotEqualCreator);

#endif

}  // namespace lite
}  // namespace mindspore
