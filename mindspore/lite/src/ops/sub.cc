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

#include "src/ops/sub.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int Sub::GetActivationType() const { return this->primitive_->value.AsSub()->activationType; }

void Sub::SetActivationType(int activation_type) {
  this->primitive_->value.AsSub()->activationType = (schema::ActivationType)activation_type;
}

int Sub::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_Sub;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_Sub) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    delete this->primitive_;
    this->primitive_ = nullptr;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::SubT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      delete this->primitive_;
      this->primitive_ = nullptr;
      return RET_ERROR;
    }
    attr->activationType = schema::ActivationType_NO_ACTIVATION;
    this->primitive_->value.value = attr;
  }
  return RET_OK;
}

#else

int Sub::GetActivationType() const { return this->primitive_->value_as_Sub()->activationType(); }
int Sub::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Sub();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Sub return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateSub(*fbb, attr->activationType());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Sub, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *SubCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Sub>(primitive); }
Registry SubRegistry(schema::PrimitiveType_Sub, SubCreator);

#endif

}  // namespace lite
}  // namespace mindspore
