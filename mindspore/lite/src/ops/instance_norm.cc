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

#include "src/ops/instance_norm.h"
#include <memory>

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
float InstanceNorm::GetEpsilon() const { return this->primitive_->value.AsInstanceNorm()->epsilon; }

void InstanceNorm::SetEpsilon(float epsilon) { this->primitive_->value.AsInstanceNorm()->epsilon = epsilon; }

int InstanceNorm::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_InstanceNorm;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_InstanceNorm) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::InstanceNormT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new InstanceNormT failed";
      delete this->primitive_;
      this->primitive_ = nullptr;
      return RET_ERROR;
    }
    attr->epsilon = GetValue<float>(prim.GetAttr("epsilon"));
    this->primitive_->value.value = attr;
  }
  return RET_OK;
}

#else
int InstanceNorm::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto val_offset = schema::CreateInstanceNorm(*fbb);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_InstanceNorm, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
float InstanceNorm::GetEpsilon() const { return this->primitive_->value_as_InstanceNorm()->epsilon(); }

PrimitiveC *InstanceNormCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<InstanceNorm>(primitive);
}
Registry InstanceNormRegistry(schema::PrimitiveType_InstanceNorm, InstanceNormCreator);
#endif

}  // namespace lite
}  // namespace mindspore
