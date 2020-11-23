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

#include "src/ops/bn_grad.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
float BNGrad::GetEps() const { return this->primitive_->value.AsBNGrad()->eps; }
float BNGrad::GetMomentum() const { return this->primitive_->value.AsBNGrad()->momentum; }

void BNGrad::SetEps(float eps) { this->primitive_->value.AsBNGrad()->eps = eps; }
void BNGrad::SetMomentum(float momentum) { this->primitive_->value.AsBNGrad()->momentum = momentum; }
int BNGrad::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_BNGrad;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_BNGrad) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::BNGradT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
    attr->momentum = 0.1f;
    if (prim.GetAttr("momentum") != nullptr) {
      attr->momentum = GetValue<float>(prim.GetAttr("momentum"));
    }
    attr->eps = 1e-5;
    if (prim.GetAttr("epsilon") != nullptr) {
      attr->eps = GetValue<float>(prim.GetAttr("epsilon"));
    }
    this->primitive_->value.value = attr;
  }
  return RET_OK;
}
#else
int BNGrad::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_BNGrad();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_BNGradInput return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateBNGrad(*fbb, attr->eps(), attr->momentum());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_BNGrad, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *BNGradCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<BNGrad>(primitive); }
Registry BNGradRegistry(schema::PrimitiveType_BNGrad, BNGradCreator);

float BNGrad::GetEps() const { return this->primitive_->value_as_BNGrad()->eps(); }
float BNGrad::GetMomentum() const { return this->primitive_->value_as_BNGrad()->momentum(); }
#endif
int BNGrad::InferShape(std::vector<lite::Tensor *> inputs, std::vector<lite::Tensor *> outputs) {
  if (inputs.size() != 6) {
    MS_LOG(ERROR) << "BNGrad should have five inputs";
    return RET_ERROR;
  }
  if (outputs.size() != 3) {
    MS_LOG(ERROR) << "BNGrad should have three outputs";
    return RET_ERROR;
  }
  auto in = inputs[1];
  auto scale = inputs[2];

  if (in->shape().size() != 4) {
    MS_LOG(ERROR) << "Grad Fused batchnorm only support nhwc input!";
  }

  outputs[0]->set_shape(in->shape());
  outputs[1]->set_shape(scale->shape());
  outputs[2]->set_shape(scale->shape());
  outputs[0]->set_data_type(in->data_type());
  outputs[1]->set_data_type(scale->data_type());
  outputs[2]->set_data_type(scale->data_type());
  outputs[0]->set_format(in->format());
  outputs[1]->set_format(scale->format());
  outputs[2]->set_format(scale->format());
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
