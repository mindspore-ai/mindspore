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

#include "src/ops/assign.h"
#include <memory>

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int Assign::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_Assign;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_Assign) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    this->primitive_->value.value = new (std::nothrow) schema::AssignT();
    if (this->primitive_->value.value == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
#else
int Assign::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Assign();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Assign return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateAssign(*fbb);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Assign, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *AssignCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Assign>(primitive); }
Registry AssignRegistry(schema::PrimitiveType_Assign, AssignCreator);
#endif

int Assign::InferShape(std::vector<lite::Tensor *> inputs, std::vector<lite::Tensor *> outputs) {
  if (2 != inputs.size()) {
    MS_LOG(ERROR) << "Assign should have at least 5 input tensors";
    return RET_ERROR;
  }

  if (inputs.at(0)->ElementsNum() != inputs.at(1)->ElementsNum()) {
    MS_LOG(ERROR) << "error input data size!";
    return RET_ERROR;
  }

  if (!outputs.empty()) {
    auto *out = outputs.front();
    MS_ASSERT(out != nullptr);
    out->set_data_type(inputs.at(0)->data_type());
    out->set_format(inputs.at(0)->format());
    out->set_shape({1});
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
