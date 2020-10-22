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

#include "src/ops/exp.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif
#include "src/ops/arithmetic_self.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
void Exp::SetBase(float base) { this->primitive_->value.AsExp()->base = base; }
void Exp::SetScale(float scale) { this->primitive_->value.AsExp()->scale = scale; }
void Exp::SetShift(float shift) { this->primitive_->value.AsExp()->shift = shift; }

float Exp::GetBase() const { return this->primitive_->value.AsExp()->base; }
float Exp::GetScale() const { return this->primitive_->value.AsExp()->scale; }
float Exp::GetShift() const { return this->primitive_->value.AsExp()->shift; }

int Exp::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_Exp;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_Exp) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    this->primitive_->value.value = new (std::nothrow) schema::ExpT();
    if (this->primitive_->value.value == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

#else

int Exp::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Exp();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Exp return nullptr";
    return RET_ERROR;
  }

  auto val_offset = schema::CreateExp(*fbb, attr->base(), attr->scale(), attr->shift());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Exp, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

float Exp::GetBase() const { return this->primitive_->value_as_Exp()->base(); }
float Exp::GetScale() const { return this->primitive_->value_as_Exp()->scale(); }
float Exp::GetShift() const { return this->primitive_->value_as_Exp()->shift(); }

PrimitiveC *ExpCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Exp>(primitive); }
Registry ExpRegistry(schema::PrimitiveType_Exp, ExpCreator);
#endif

}  // namespace lite
}  // namespace mindspore
