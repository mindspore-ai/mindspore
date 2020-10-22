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

#include "src/ops/log_grad.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif
#include "src/ops/arithmetic_self.h"

namespace mindspore {
namespace lite {
#ifndef PRIMITIVE_WRITEABLE
int LogGrad::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(primitive != nullptr);
  MS_ASSERT(fbb != nullptr);
  auto attr = primitive->value_as_LogGrad();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_LogGrad return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateLogGrad(*fbb);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_LogGrad, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *LogGradCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<LogGrad>(primitive); }
Registry LogGradRegistry(schema::PrimitiveType_LogGrad, LogGradCreator);
#endif

}  // namespace lite
}  // namespace mindspore
