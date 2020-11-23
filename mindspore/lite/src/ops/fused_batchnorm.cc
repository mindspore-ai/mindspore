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

#include "src/ops/fused_batchnorm.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
float FusedBatchNorm::GetEpsilon() const { return this->primitive_->value.AsFusedBatchNorm()->epsilon; }
float FusedBatchNorm::GetMomentum() const { return this->primitive_->value.AsFusedBatchNorm()->momentum; }
int FusedBatchNorm::GetSpatial() const { return this->primitive_->value.AsFusedBatchNorm()->spatial; }

void FusedBatchNorm::SetEpsilon(float epsilon) { this->primitive_->value.AsFusedBatchNorm()->epsilon = epsilon; }
void FusedBatchNorm::SetMomentum(float momentum) { this->primitive_->value.AsFusedBatchNorm()->momentum = momentum; }
void FusedBatchNorm::SetSpatial(int spatial) { this->primitive_->value.AsFusedBatchNorm()->spatial = spatial; }

int FusedBatchNorm::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_FusedBatchNorm;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_FusedBatchNorm) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::FusedBatchNormT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new attr value failed";
      return RET_ERROR;
    }
    attr->epsilon = GetValue<float>(prim.GetAttr("epsilon"));
    attr->momentum = GetValue<float>(prim.GetAttr("momentum"));
    this->primitive_->value.value = attr;
  }
  return RET_OK;
}

#else
int FusedBatchNorm::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_FusedBatchNorm();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_FusedBatchNorm return nullptr";
    return RET_ERROR;
  }

  auto val_offset = schema::CreateFusedBatchNorm(*fbb, attr->epsilon(), attr->momentum(), attr->spatial());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_FusedBatchNorm, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
float FusedBatchNorm::GetEpsilon() const { return this->primitive_->value_as_FusedBatchNorm()->epsilon(); }
float FusedBatchNorm::GetMomentum() const { return this->primitive_->value_as_FusedBatchNorm()->momentum(); }
int FusedBatchNorm::GetSpatial() const { return this->primitive_->value_as_FusedBatchNorm()->spatial(); }

PrimitiveC *FusedBatchNormCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<FusedBatchNorm>(primitive);
}
Registry FusedBatchNormRegistry(schema::PrimitiveType_FusedBatchNorm, FusedBatchNormCreator);
#endif

int FusedBatchNorm::InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) {
  for (size_t i = 0; i < inputs_.size(); i++) {
    if (outputs_.size() <= i) {
      break;
    }
    outputs_.at(i)->set_shape(inputs_.at(i)->shape());
    outputs_.at(i)->set_data_type(inputs_.at(i)->data_type());
    outputs_.at(i)->set_format(inputs_.at(i)->format());
  }
  if (outputs_.size() > 5) {
    outputs_.at(5)->set_data_type(inputs_.at(0)->data_type());
    outputs_.at(5)->set_format(inputs_.at(0)->format());
    outputs_.at(5)->set_shape({1});
  }
  return 0;
}

}  // namespace lite
}  // namespace mindspore
