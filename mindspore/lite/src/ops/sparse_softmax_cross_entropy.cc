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

#include "src/ops/sparse_softmax_cross_entropy.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int SparseSoftmaxCrossEntropy::GetIsGrad() const {
  return this->primitive_->value.AsSparseSoftmaxCrossEntropy()->isGrad;
}

void SparseSoftmaxCrossEntropy::SetIsGrad(int isGrad) {
  this->primitive_->value.AsSparseSoftmaxCrossEntropy()->isGrad = isGrad;
}

int SparseSoftmaxCrossEntropy::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_SparseSoftmaxCrossEntropy;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_SparseSoftmaxCrossEntropy) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::SparseSoftmaxCrossEntropyT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }

    attr->isGrad = GetValue<bool>(prim.GetAttr("is_grad"));
    this->primitive_->value.value = attr;
    if (this->primitive_->value.value == nullptr) {
      MS_LOG(ERROR) << "primitive value is nullptr";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
#else

int SparseSoftmaxCrossEntropy::GetIsGrad() const {
  return this->primitive_->value_as_SparseSoftmaxCrossEntropy()->isGrad();
}
int SparseSoftmaxCrossEntropy::UnPackToFlatBuilder(const schema::Primitive *primitive,
                                                   flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_SparseSoftmaxCrossEntropy();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_SparseSoftmaxCrossEntropy return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateSparseSoftmaxCrossEntropy(*fbb, attr->isGrad());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_SparseSoftmaxCrossEntropy, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *SparseSoftmaxCrossEntropyCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<SparseSoftmaxCrossEntropy>(primitive);
}
Registry SparseSoftmaxCrossEntropyRegistry(schema::PrimitiveType_SparseSoftmaxCrossEntropy,
                                           SparseSoftmaxCrossEntropyCreator);
#endif

int SparseSoftmaxCrossEntropy::InferShape(std::vector<Tensor *> inputs, std::vector<Tensor *> outputs) {
  if (2 != inputs.size()) {
    MS_LOG(ERROR) << "SparseSoftmaxCrossEntropy should have at two inputs";
    return RET_ERROR;
  }

  if (1 != outputs.size()) {
    MS_LOG(ERROR) << "SparseSoftmaxCrossEntropy should have one output";
    return RET_ERROR;
  }
  auto *in0 = inputs.front();
  MS_ASSERT(in0 != nullptr);
  auto *out = outputs.front();
  MS_ASSERT(out != nullptr);

  if (GetIsGrad() != 0) {
    out->set_shape(in0->shape());
    out->set_data_type(in0->data_type());
    out->set_format(in0->format());
  } else {
    std::vector<int> outshape;
    outshape.push_back(1);
    out->set_shape(outshape);
    out->set_data_type(in0->data_type());
    out->set_format(in0->format());
  }

  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
