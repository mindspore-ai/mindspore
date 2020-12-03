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
#include "src/ops/binary_cross_entropy.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int BinaryCrossEntropy::GetReduction() const { return this->primitive_->value.AsBinaryCrossEntropy()->reduction; }

int BinaryCrossEntropy::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitive error";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_BinaryCrossEntropy;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_BinaryCrossEntropy) {
    MS_LOG(ERROR) << "PrimitiveType_BinaryCrossEntropy primitive value type :  "
                  << schema::EnumNamePrimitiveType(primitive_->value.type) << "is  not equal"
                  << schema::EnumNamePrimitiveType(schema::PrimitiveType_BinaryCrossEntropy);
    delete this->primitive_;
    this->primitive_ = nullptr;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    schema::BinaryCrossEntropyT *attr = new (std::nothrow) schema::BinaryCrossEntropyT();
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
int BinaryCrossEntropy::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_BinaryCrossEntropy();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_BinaryCrossEntropy return nullptr";
    return RET_ERROR;
  }
  int reduction = attr->reduction();
  auto val_offset = schema::CreateBinaryCrossEntropy(*fbb, reduction);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_BinaryCrossEntropy, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

int BinaryCrossEntropy::GetReduction() const { return this->primitive_->value_as_BinaryCrossEntropy()->reduction(); }

PrimitiveC *BinaryCrossEntropyCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<BinaryCrossEntropy>(primitive);
}
Registry BinaryCrossEntropyRegistry(schema::PrimitiveType_BinaryCrossEntropy, BinaryCrossEntropyCreator);
#endif
int BinaryCrossEntropy::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  Tensor *x = inputs_.at(0);
  Tensor *out = outputs_.at(0);
  out->set_format(x->format());
  out->set_data_type(x->data_type());
  int reduction = GetReduction();
  if (reduction == 1 || reduction == 2) {
    out->set_shape({1});
  } else {
    std::vector<int> x_shape = x->shape();
    std::vector<int> output_shape(x_shape.size());
    output_shape.assign(x_shape.begin(), x_shape.end());
    out->set_shape(output_shape);
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
