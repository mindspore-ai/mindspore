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

#include "src/ops/merge.h"
#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE

int Merge::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_Merge;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_Merge) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    this->primitive_->value.value = new (std::nothrow) schema::MergeT();
    if (this->primitive_->value.value == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
  }
  PopulaterQuantParam(prim, inputs);
  return RET_OK;
}

#else
int Merge::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Merge();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Merge return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateMerge(*fbb);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Merge, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *MergeCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Merge>(primitive); }
Registry MergeRegistry(schema::PrimitiveType_Merge, MergeCreator);
#endif

int Merge::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(inputs_.size() == 2 * outputs_.size());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  for (size_t i = 0; i < inputs_.size() / 2; i++) {
    outputs_[i]->set_data_type(inputs_[i]->data_type());
    outputs_[i]->set_shape(inputs_[i]->shape());
  }
  return RET_OK;
}

}  // namespace lite
}  // namespace mindspore
