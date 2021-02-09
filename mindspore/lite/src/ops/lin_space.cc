/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/ops/lin_space.h"
#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifndef PRIMITIVE_WRITEABLE
int LinSpace::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto val_offset = schema::CreateLinSpace(*fbb);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_LinSpace, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
PrimitiveC *LinSpaceCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<LinSpace>(primitive);
}
Registry LinSpaceRegistry(schema::PrimitiveType_LinSpace, LinSpaceCreator);
#endif
int LinSpace::InferShape(std::vector<Tensor *> inputs, std::vector<Tensor *> outputs) {
  auto input = inputs.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs.front();
  MS_ASSERT(output != nullptr);
  output->set_data_type(input->data_type());
  output->set_format(input->format());
  auto num = inputs.at(2)->data_c();
  if (num == nullptr) {
    return RET_INFER_INVALID;
  }
  output->set_shape({reinterpret_cast<int *>(num)[0]});
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
