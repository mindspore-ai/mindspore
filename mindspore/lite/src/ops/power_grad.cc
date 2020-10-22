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

#include "src/ops/power_grad.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
float PowerGrad::GetPower() const { return this->primitive_->value.AsPowerGrad()->power; }
float PowerGrad::GetScale() const { return this->primitive_->value.AsPowerGrad()->scale; }
float PowerGrad::GetShift() const { return this->primitive_->value.AsPowerGrad()->shift; }

void PowerGrad::SetPower(float power) { this->primitive_->value.AsPowerGrad()->power = power; }
void PowerGrad::SetScale(float scale) { this->primitive_->value.AsPowerGrad()->scale = scale; }
void PowerGrad::SetShift(float shift) { this->primitive_->value.AsPowerGrad()->shift = shift; }
int PowerGrad::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_PowerGrad;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_PowerGrad) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::PowerGradT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
    attr->power = GetValue<float>(prim.GetAttr("power"));
    attr->scale = GetValue<float>(prim.GetAttr("scale"));
    attr->shift = GetValue<float>(prim.GetAttr("shift"));
    this->primitive_->value.value = attr;
    if (this->primitive_->value.value == nullptr) {
      MS_LOG(ERROR) << "primitive value is nullptr";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
#else

float PowerGrad::GetPower() const { return this->primitive_->value_as_PowerGrad()->power(); }
float PowerGrad::GetScale() const { return this->primitive_->value_as_PowerGrad()->scale(); }
float PowerGrad::GetShift() const { return this->primitive_->value_as_PowerGrad()->shift(); }

int PowerGrad::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);

  auto attr = primitive->value_as_PowerGrad();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_PowerGrad return nullptr";
    return RET_ERROR;
  }

  auto val_offset = schema::CreatePowerGrad(*fbb, attr->power(), attr->scale(), attr->shift());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_PowerGrad, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *PowerGradCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<PowerGrad>(primitive);
}
Registry PowerGradRegistry(schema::PrimitiveType_PowerGrad, PowerGradCreator);
#endif
}  // namespace lite
}  // namespace mindspore
