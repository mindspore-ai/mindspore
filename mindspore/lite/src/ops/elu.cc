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

#include "src/ops/elu.h"
#include <memory>

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
float Elu::GetAlpha() const { return this->primitive_->value.AsElu()->alpha; }

void Elu::SetAlpha(float alpha) { this->primitive_->value.AsElu()->alpha = alpha; }

int Elu::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_Elu;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_Elu) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  auto attr = std::make_unique<schema::EluT>();
  this->primitive_->value.value = attr.release();
  if (this->primitive_->value.value == nullptr) {
    MS_LOG(ERROR) << "new primitiveT value failed";
    return RET_ERROR;
  }
  return RET_OK;
}
#else
int Elu::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Elu();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Elu return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateElu(*fbb, attr->alpha());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Elu, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
float Elu::GetAlpha() const { return this->primitive_->value_as_Elu()->alpha(); }

PrimitiveC *EluCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Elu>(primitive); }
Registry EluRegistry(schema::PrimitiveType_Elu, EluCreator);
#endif

}  // namespace lite
}  // namespace mindspore
