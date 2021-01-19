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
#include "src/ops/select.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif
#include "src/tensorlist.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int Select::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_Select;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_Select) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::SelectT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }

    this->primitive_->value.value = attr;
    if (this->primitive_->value.value == nullptr) {
      MS_LOG(ERROR) << "primitive value is nullptr";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
#else
int Select::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Select();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Select return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateSelect(*fbb);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Select, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *SelectCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Select>(primitive); }
Registry SelectRegistry(schema::PrimitiveType_Select, SelectCreator);
#endif

int Select::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(inputs_.size() == 2 * outputs_.size() + 1);
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  for (size_t i = 0; i < outputs_.size(); i++) {
    auto *input = inputs_[i + 1];
    auto *output = outputs_[i];
    if (input == nullptr) {
      MS_LOG(ERROR) << "input tensor is nullptr";
      return RET_ERROR;
    }
    if (output == nullptr) {
      MS_LOG(ERROR) << "output tensor is nullptr";
      return RET_ERROR;
    }
    output->set_data_type(input->data_type());
    output->set_shape(input->shape());
    output->set_format(input->format());
    auto data_type = input->data_type();
    if (data_type == kObjectTypeTensorType) {
      auto input_tensorlist = reinterpret_cast<TensorList *>(input);
      auto output_tensorlist = reinterpret_cast<TensorList *>(output);
      output_tensorlist->set_element_shape(input_tensorlist->element_shape());
      output_tensorlist->set_max_elements_num(input_tensorlist->max_elements_num());
      output_tensorlist->set_tensors_data_type(input_tensorlist->tensors_data_type());
    }
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
