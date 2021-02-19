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

#include "src/ops/addn.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int AddN::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_AddN;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_AddN) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    this->primitive_->value.value = new (std::nothrow) schema::AddNT();
    if (this->primitive_->value.value == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

#else
int AddN::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto val_offset = schema::CreateAddN(*fbb, 0);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_AddN, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *AddNCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<AddN>(primitive); }
Registry AddNRegistry(schema::PrimitiveType_AddN, AddNCreator);
#endif

namespace {
constexpr int kLeastInputNum = 2;
}
int AddN::InferShape(std::vector<Tensor *> inputs, std::vector<Tensor *> outputs) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto input = inputs.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs.front();
  MS_ASSERT(output != nullptr);
  if (inputs.size() < kLeastInputNum) {
    MS_LOG(ERROR) << "input size" << inputs.size() << " is error!";
    return RET_INPUT_TENSOR_ERROR;
  }
  output->set_format(input->format());
  output->set_data_type(input->data_type());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }

  size_t max_dims = inputs.at(0)->shape().size();
  size_t max_dims_idx = 0;

  // determine max_dims
  for (size_t i = 1; i < inputs.size(); ++i) {
    if (inputs.at(i)->shape().size() > max_dims) {
      max_dims = inputs.at(i)->shape().size();
      max_dims_idx = i;
    }
  }

  output->set_shape(inputs.at(max_dims_idx)->shape());

  // make sure all elements have the same size or 1 (broadcasting) in all dimensions
  for (size_t i = 1; i < inputs.size(); ++i) {
    if ((inputs.at(i)->shape().size() != max_dims) &&
        (inputs.at(i)->ElementsNum() != inputs.at(max_dims_idx)->ElementsNum())) {
      MS_LOG(ERROR) << "AddN inputs shape is not equal!";
      return RET_INPUT_TENSOR_ERROR;
    }
    if (inputs.at(i)->data_type() != inputs.at(0)->data_type()) {
      MS_LOG(ERROR) << "AddN all input data type should be the same!";
      return RET_INPUT_TENSOR_ERROR;
    }
  }

  for (size_t d = 0; d < input->shape().size(); ++d) {
    size_t max_dim = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
      size_t shift = max_dims - inputs.at(i)->shape().size();
      size_t dim = (i < shift) ? 1 : inputs.at(i)->shape().at(d);
      if (dim > max_dim) {
        max_dim = dim;
      }
    }
    output->shape()[d] = max_dim;  // set the biggest dimension in the output tensor
  }

  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
