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

#include "src/ops/activation.h"
#include <memory>

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int Activation::GetType() const { return this->primitive_->value.AsActivation()->type; }
float Activation::GetAlpha() const { return this->primitive_->value.AsActivation()->alpha; }

void Activation::SetType(int type) { this->primitive_->value.AsActivation()->type = (schema::ActivationType)type; }
void Activation::SetAlpha(float alpha) { this->primitive_->value.AsActivation()->alpha = alpha; }

int Activation::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_Activation;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_Activation) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  auto attr = std::make_unique<schema::ActivationT>();
  if (prim.name() == "ReLU") {
    attr->type = schema::ActivationType_RELU;
  } else if (prim.name() == "Sigmoid") {
    attr->type = schema::ActivationType_SIGMOID;
  } else if (prim.name() == "ReLU6") {
    attr->type = schema::ActivationType_RELU6;
  }
  this->primitive_->value.value = attr.release();
  if (this->primitive_->value.value == nullptr) {
    MS_LOG(ERROR) << "new primitiveT value failed";
    return RET_ERROR;
  }
  return RET_OK;
}
#else
int Activation::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Activation();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Activation return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateActivation(*fbb, attr->type(), attr->alpha());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Activation, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
int Activation::GetType() const { return this->primitive_->value_as_Activation()->type(); }
float Activation::GetAlpha() const { return this->primitive_->value_as_Activation()->alpha(); }
#endif
}  // namespace lite
}  // namespace mindspore
