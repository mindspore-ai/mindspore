/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/ops/flatten_grad.h"
#include <memory>

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
int FlattenGrad::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto input = inputs_.front();
  auto output = outputs_.front();
  if (input == nullptr || output == nullptr) {
    MS_LOG(ERROR) << "FlattenGrad input or output is null!";
    return RET_ERROR;
  }
  if (inputs_.size() != kDoubleNum || outputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "input size: " << inputs_.size() << ", output size: " << outputs_.size();
    return RET_INPUT_TENSOR_ERROR;
  }

  output->set_data_type(input->data_type());
  output->set_format(input->format());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }

  auto output_size = inputs_.at(1)->shape().at(0);
  std::vector<int> output_shape(output_size);
  for (int i = 0; i < output_size; i++) {
    output_shape.at(i) = static_cast<int *>(inputs_.at(1)->data_c())[i];
  }
  output->set_shape(output_shape);
  return RET_OK;
}

#ifdef PRIMITIVE_WRITEABLE
int FlattenGrad::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_FlattenGrad;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_FlattenGrad) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::FlattenGradT();
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
int FlattenGrad::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto val_offset = schema::CreateFlattenGrad(*fbb);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_FlattenGrad, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
PrimitiveC *FlattenGradCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<FlattenGrad>(primitive);
}
Registry FlattenGradRegistry(schema::PrimitiveType_FlattenGrad, FlattenGradCreator);
#endif
}  // namespace lite
}  // namespace mindspore
