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

#include "src/ops/while.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE

void While::SetCondSubgraphIndex(const int cond_subgraph_index) {
  this->primitive_->value.AsWhile()->condSubgraphIndex = cond_subgraph_index;
}
void While::SetBodySubgraphIndex(const int body_subgraph_index) {
  this->primitive_->value.AsWhile()->bodySubgraphIndex = body_subgraph_index;
}

int While::GetCondSubgraphIndex() const { return this->primitive_->value.AsWhile()->condSubgraphIndex; }
int While::GetBodySubgraphIndex() const { return this->primitive_->value.AsWhile()->bodySubgraphIndex; }

int While::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_While;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_While) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::WhileT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
    attr->bodySubgraphIndex = GetValue<bool>(prim.GetAttr("body_subgraph_index"));
    attr->condSubgraphIndex = GetValue<bool>(prim.GetAttr("cond_subgraph_index"));
    this->primitive_->value.value = attr;
  }
  return RET_OK;
}

#else

int While::GetCondSubgraphIndex() const { return this->primitive_->value_as_While()->condSubgraphIndex(); }
int While::GetBodySubgraphIndex() const { return this->primitive_->value_as_While()->bodySubgraphIndex(); }

int While::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_While();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_While return nullptr";
    return RET_ERROR;
  }
  auto cond_subgraph_index = attr->condSubgraphIndex();
  auto body_subgraph_index = attr->bodySubgraphIndex();
  auto val_offset = schema::CreateWhile(*fbb, body_subgraph_index, cond_subgraph_index);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_While, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *WhileCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<While>(primitive); }
Registry WhileRegistry(schema::PrimitiveType_While, WhileCreator);

#endif

int While::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  if (inputs_.size() != outputs_.size()) {
    MS_LOG(ERROR) << "The number of inputs and outputs varies";
    return RET_ERROR;
  }
  for (size_t i = 0; i < inputs_.size(); i++) {
    outputs_.at(i)->set_data_type(inputs_.at(i)->data_type());
    outputs_.at(i)->set_format(inputs_.at(i)->format());
    outputs_.at(i)->set_shape(inputs_.at(i)->shape());
  }

  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
