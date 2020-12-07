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

#include "src/ops/unique.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifndef PRIMITIVE_WRITEABLE
int Unique::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Unique();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Unique return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateUnique(*fbb, attr->outType());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Unique, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *UniqueCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Unique>(primitive); }
Registry UniqueRegistry(schema::PrimitiveType_Unique, UniqueCreator);
#endif

int Unique::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  if (inputs_.size() != kSingleNum || outputs_.size() != kDoubleNum) {
    MS_LOG(ERROR) << "input size: " << inputs_.size() << ", output size: " << outputs_.size();
    return RET_INPUT_TENSOR_ERROR;
  }
  auto &input = inputs_.at(0);
  MS_ASSERT(input != nullptr);
  auto &output0 = outputs_.at(0);
  MS_ASSERT(output0 != nullptr);
  auto &output1 = outputs_.at(1);
  MS_ASSERT(output1 != nullptr);
  output0->set_data_type(input->data_type());
  output1->set_data_type(kNumberTypeInt32);
  output1->set_format(input->format());
  output0->set_format(input->format());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  output0->set_shape(input->shape());
  output1->set_shape(input->shape());
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
