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

#include "src/ops/activation_grad.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int ActivationGrad::GetType() const { return this->primitive_->value.AsActivationGrad()->type; }
float ActivationGrad::GetAlpha() const { return this->primitive_->value.AsActivationGrad()->alpha; }
void ActivationGrad::SetType(int type) {
  this->primitive_->value.AsActivationGrad()->type = (schema::ActivationType)type;
}
void ActivationGrad::SetAlpha(float alpha) { this->primitive_->value.AsActivationGrad()->alpha = alpha; }
int ActivationGrad::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_ActivationGrad;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_ActivationGrad) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  auto attr = std::make_unique<schema::ActivationGradT>();
  if (prim.name() == "ReluGrad") {
    attr->type = schema::ActivationType_RELU;
  } else if (prim.name() == "SigmoidGrad") {
    attr->type = schema::ActivationType_SIGMOID;
  } else if (prim.name() == "ReLU6Grad") {
    attr->type = schema::ActivationType_RELU6;
  } else if (prim.name() == "HSigmoidGrad") {
    attr->type = schema::ActivationType_HSIGMOID;
  } else if (prim.name() == "HSwishGrad") {
    attr->type = schema::ActivationType_HSWISH;
  }
  attr->alpha = 0;  // alpha;
  this->primitive_->value.value = attr.release();
  if (this->primitive_->value.value == nullptr) {
    MS_LOG(ERROR) << "new primitiveT value failed";
    return RET_ERROR;
  }
  return RET_OK;
}
#else
int ActivationGrad::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_ActivationGrad();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_ActivationGrad return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateActivationGrad(*fbb, attr->type(), attr->alpha());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_ActivationGrad, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
int ActivationGrad::GetType() const { return this->primitive_->value_as_ActivationGrad()->type(); }
float ActivationGrad::GetAlpha() const { return this->primitive_->value_as_ActivationGrad()->alpha(); }

PrimitiveC *ActivationGradCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<ActivationGrad>(primitive);
}
Registry ActivationGradRegistry(schema::PrimitiveType_ActivationGrad, ActivationGradCreator);
#endif
}  // namespace lite
}  // namespace mindspore
