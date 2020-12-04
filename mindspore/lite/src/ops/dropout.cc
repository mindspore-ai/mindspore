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

#include "src/ops/dropout.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
float Dropout::GetRatio() const { return this->primitive_->value.AsDropout()->ratio; }

void Dropout::SetRatio(float ratio) { this->primitive_->value.AsDropout()->ratio = ratio; }

int Dropout::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_Dropout;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_Dropout) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::DropoutT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
    if (prim.GetAttr("keep_prob") != nullptr) {
      attr->ratio = GetValue<float>(prim.GetAttr("keep_prob"));
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
int Dropout::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Dropout();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Dropout return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateDropout(*fbb, attr->ratio());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Dropout, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
float Dropout::GetRatio() const { return this->primitive_->value_as_Dropout()->ratio(); }

PrimitiveC *DropoutCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Dropout>(primitive); }
Registry DropoutRegistry(schema::PrimitiveType_Dropout, DropoutCreator);
#endif
int Dropout::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output0 = outputs_.front();
  MS_ASSERT(output0 != nullptr);
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  output0->set_shape(input->shape());
  output0->set_data_type(input->data_type());
  output0->set_format(input->format());
  if (outputs_.size() > 1) {
    auto output1 = outputs_[1];
    MS_ASSERT(output1 != nullptr);
    output1->set_shape(input->shape());
    output1->set_data_type(input->data_type());
    output1->set_format(input->format());
  }
  return RET_OK;
}

}  // namespace lite
}  // namespace mindspore
