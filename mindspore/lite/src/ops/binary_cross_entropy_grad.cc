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

#include <string>
#include "src/ops/binary_cross_entropy_grad.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE

int BinaryCrossEntropyGrad::GetReduction() const {
  return this->primitive_->value.AsBinaryCrossEntropyGrad()->reduction;
}

int BinaryCrossEntropyGrad::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitive error";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_BinaryCrossEntropyGrad;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_BinaryCrossEntropyGrad) {
    MS_LOG(ERROR) << "PrimitiveType_BinaryCrossEntropyGrad primitive value type :  "
                  << schema::EnumNamePrimitiveType(primitive_->value.type) << "is  not equal"
                  << schema::EnumNamePrimitiveType(schema::PrimitiveType_BinaryCrossEntropyGrad);
    delete this->primitive_;
    this->primitive_ = nullptr;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    schema::BinaryCrossEntropyGradT *attr = new (std::nothrow) schema::BinaryCrossEntropyGradT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new binary cross entropy attr failed!";
      delete this->primitive_;
      this->primitive_ = nullptr;
      return RET_ERROR;
    }
    // default is mean
    string reduction = "mean";
    if (prim.GetAttr("reduction") == nullptr) {
      MS_LOG(ERROR) << "get reduction failed!";
      delete this->primitive_;
      delete attr;
      this->primitive_ = nullptr;
      attr = nullptr;
      return RET_ERROR;
    } else {
      reduction = GetValue<string>(prim.GetAttr("reduction"));
    }

    if (reduction == "none") {
      attr->reduction = 0;
    } else if (reduction == "sum") {
      attr->reduction = 2;
    } else {
      // default is mean
      attr->reduction = 1;
    }
    this->primitive_->value.value = attr;
  }

  return RET_OK;
}
#else
int BinaryCrossEntropyGrad::UnPackToFlatBuilder(const schema::Primitive *primitive,
                                                flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_BinaryCrossEntropyGrad();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_BinaryCrossEntropyGrad return nullptr";
    return RET_ERROR;
  }
  int reduction = attr->reduction();
  auto val_offset = schema::CreateBinaryCrossEntropyGrad(*fbb, reduction);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_BinaryCrossEntropyGrad, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

int BinaryCrossEntropyGrad::GetReduction() const {
  return this->primitive_->value_as_BinaryCrossEntropyGrad()->reduction();
}

PrimitiveC *BinaryCrossEntropyGradCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<BinaryCrossEntropyGrad>(primitive);
}
Registry BinaryCrossEntropyGradRegistry(schema::PrimitiveType_BinaryCrossEntropyGrad, BinaryCrossEntropyGradCreator);
#endif
int BinaryCrossEntropyGrad::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  Tensor *x = inputs_[0];
  Tensor *out = outputs_[0];
  out->set_format(x->format());
  out->set_data_type(x->data_type());
  std::vector<int> x_shape = x->shape();
  std::vector<int> output_shape(x_shape.size());
  output_shape.assign(x_shape.begin(), x_shape.end());
  out->set_shape(output_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
