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

#include "src/ops/dropout_grad.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
float DropoutGrad::GetRatio() const { return this->primitive_->value.AsDropout()->ratio; }

void DropoutGrad::SetRatio(float ratio) { this->primitive_->value.AsDropout()->ratio = ratio; }

int DropoutGrad::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_DropoutGrad;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_DropoutGrad) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::DropoutGradT();
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
int DropoutGrad::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_DropoutGrad();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_DropoutGrad return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateDropoutGrad(*fbb, attr->ratio());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_DropoutGrad, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
float DropoutGrad::GetRatio() const { return this->primitive_->value_as_DropoutGrad()->ratio(); }

PrimitiveC *DropoutGradCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<DropoutGrad>(primitive);
}
Registry DropoutGradRegistry(schema::PrimitiveType_DropoutGrad, DropoutGradCreator);

#endif
int DropoutGrad::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  MS_ASSERT(inputs_.size() == 2);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  output->set_shape(input->shape());
  output->set_data_type(input->data_type());
  output->set_format(input->format());
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
