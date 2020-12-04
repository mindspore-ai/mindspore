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

#include "src/ops/sparse_to_dense.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifndef PRIMITIVE_WRITEABLE
int SparseToDense::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_SparseToDense();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_SparseToDense return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateSparseToDense(*fbb);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_SparseToDense, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *SparseToDenseCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<SparseToDense>(primitive);
}
Registry SparseToDenseRegistry(schema::PrimitiveType_SparseToDense, SparseToDenseCreator);
#endif

int SparseToDense::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto output = outputs_.front();
  if (output == nullptr) {
    MS_LOG(ERROR) << "output null pointer dereferencing.";
    return RET_ERROR;
  }
  auto input2 = inputs_.at(2);
  outputs_.at(0)->set_data_type(input2->data_type());
  outputs_.at(0)->set_format(input2->format());

  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  if (this->primitive_ == nullptr) {
    return RET_NULL_PTR;
  }

  auto input1 = inputs_.at(1);
  int *input1_data = reinterpret_cast<int *>(input1->MutableData());
  std::vector<int> output_shape;
  for (int i = 0; i < input1->ElementsNum(); i++) {
    output_shape.push_back(input1_data[i]);
  }
  outputs_.at(0)->set_shape(output_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
