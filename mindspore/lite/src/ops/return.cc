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

#include "src/ops/return.h"
#include <memory>

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int Return::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_Return;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_Return) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::ReturnT();
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
PrimitiveC *ReturnCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Return>(primitive); }
Registry ReturnRegistry(schema::PrimitiveType_Return, ReturnCreator);
#endif

namespace {
constexpr size_t kInputSize = 1;
constexpr size_t kOutputSize = 1;
}  // namespace
int Return::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  if (inputs_.size() != kInputSize || outputs_.size() != kOutputSize) {
    return RET_ERROR;
  }
  auto input = inputs_.front();
  auto output = outputs_.front();
  if (input == nullptr || output == nullptr) {
    return RET_NULL_PTR;
  }
  output->set_data_type(input->data_type());
  output->set_format(input->format());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  if (this->primitive_ == nullptr) {
    return RET_NULL_PTR;
  }
  output->set_data_type(input->data_type());
  output->set_shape(input->shape());
  output->set_format(input->format());
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
