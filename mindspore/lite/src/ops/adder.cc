/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/ops/adder.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE

#else
int Adder::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Adder();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Adder return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateAdder(*fbb);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Adder, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *AdderCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Adder>(primitive); }
Registry AdderRegistry(schema::PrimitiveType_Adder, AdderCreator);
#endif

int Adder::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  MS_ASSERT(inputs_.size() == 2);
  auto input0 = inputs_.front();
  MS_ASSERT(input0 != nullptr);
  MS_ASSERT(input0->shape().size() == 2);
  auto input1 = inputs_.at(1);
  MS_ASSERT(input1 != nullptr);
  MS_ASSERT(input1->shape().size() == 2);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);

  output->set_data_type(input0->data_type());
  output->set_format(input0->format());
  if (!infer_flag()) {
    return RET_OK;
  }
  std::vector<int> in_shape;
  in_shape.push_back(input0->shape().at(0));
  in_shape.push_back(input1->shape().at(1));
  output->set_shape(in_shape);

  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
