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

#include "src/ops/unstack.h"
#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int Unstack::GetAxis() const { return this->primitive_->value.AsUnstack()->axis; }

void Unstack::SetAxis(int axis) { this->primitive_->value.AsUnstack()->axis = axis; }

#else

int Unstack::GetAxis() const { return this->primitive_->value_as_Unstack()->axis(); }
int Unstack::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Unstack();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Unstack return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateUnstack(*fbb, attr->num(), attr->axis());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Unstack, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *UnstackCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Unstack>(primitive); }
Registry UnstackRegistry(schema::PrimitiveType_Unstack, UnstackCreator);
#endif

int Unstack::InferShape(std::vector<Tensor *> inputs, std::vector<Tensor *> outputs) {
  auto input = inputs.at(0);
  MS_ASSERT(input != nullptr);
  auto input_shape = input->shape();

  auto axis = GetAxis() < 0 ? GetAxis() + input_shape.size() : GetAxis();
  if (axis < 0 || axis >= input_shape.size()) {
    MS_LOG(ERROR) << "Invalid axis " << GetAxis();
    return RET_PARAM_INVALID;
  }
  for (auto &out : outputs) {
    MS_ASSERT(out != nullptr);
    out->set_data_type(input->data_type());
    out->set_format(input->format());
  }
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  std::vector<int> output_shape;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    if (i != axis) {
      output_shape.push_back(input_shape.at(i));
    }
  }
  for (auto &out : outputs) {
    MS_ASSERT(out != nullptr);
    out->set_shape(output_shape);
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
