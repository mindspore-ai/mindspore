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

#include "src/ops/assign_add.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int AssignAdd::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitive error";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_AssignAdd;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_AssignAdd) {
    MS_LOG(ERROR) << "PrimitiveType_AssignAdd primitive value type :  "
                  << schema::EnumNamePrimitiveType(primitive_->value.type) << "is  not equal"
                  << schema::EnumNamePrimitiveType(schema::PrimitiveType_AssignAdd);
    delete this->primitive_;
    this->primitive_ = nullptr;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    this->primitive_->value.value = new (std::nothrow) schema::AssignAddT();
    if (this->primitive_->value.value == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      delete this->primitive_;
      this->primitive_ = nullptr;
      return RET_ERROR;
    }
  }
  return RET_OK;
}
#else
int AssignAdd::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_AssignAdd();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_AssignAdd return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateAssignAdd(*fbb);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_AssignAdd, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *AssignAddCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<AssignAdd>(primitive);
}
Registry AssignAddRegistry(schema::PrimitiveType_AssignAdd, AssignAddCreator);
#endif

int AssignAdd::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  Tensor *x = inputs_.at(0);
  Tensor *y = inputs_.at(1);
  Tensor *out = outputs_.at(0);
  std::vector<int> x_shape = x->shape();
  if (x->data_type() != y->data_type()) {
    MS_LOG(ERROR) << "no matched shape of x and y";
    return RET_ERROR;
  }
  std::vector<int> output_shape(x_shape.size());
  for (size_t i = 0; i < x_shape.size(); i++) {
    output_shape[i] = x_shape[i];
  }
  out->set_shape(output_shape);
  out->set_format(x->format());
  out->set_data_type(x->data_type());
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
