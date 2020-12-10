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

#include "src/ops/partial.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE

int Partial::GetSubGraphIndex() const { return this->primitive_->value.AsPartial()->subGraphIndex; }

int Partial::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_Partial;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_Partial) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::PartialT();
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

int Partial::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Partial();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Partial return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreatePartial(*fbb, attr->subGraphIndex());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Partial, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

int Partial::GetSubGraphIndex() const { return this->primitive_->value_as_Partial()->subGraphIndex(); }

PrimitiveC *PartialCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Partial>(primitive); }
Registry PartialRegistry(schema::PrimitiveType_Partial, PartialCreator);

#endif

int Partial::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) { return RET_OK; }
}  // namespace lite
}  // namespace mindspore
