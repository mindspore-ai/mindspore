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

#include "src/ops/nonzero.h"
#include <algorithm>
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/tensor.h"
#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int NonZero::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_NonZero;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_NonZero) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    this->primitive_->value.value = new (std::nothrow) schema::NonZeroT();
    if (this->primitive_->value.value == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
  }
  PopulaterQuantParam(prim, inputs);
  return RET_OK;
}
#else
int NonZero::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_NonZero();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_NonZero return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateNonZero(*fbb);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_NonZero, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
PrimitiveC *NonZeroCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<NonZero>(primitive); }
Registry NonZeroRegistry(schema::PrimitiveType_NonZero, NonZeroCreator);
#endif
template <typename T>
void CalShape(const T *data, const std::vector<Tensor *> &inputs, std::vector<int> *out_shape) {
  int input_count = inputs[0]->ElementsNum();
  int input_dim_size = inputs[0]->shape().empty() ? 1 : inputs[0]->shape().size();
  out_shape->emplace_back(input_dim_size);
  int nonzero_size = 0;
  for (int i = 0; i < input_count; i++) {
    if (static_cast<int>(data[i]) != 0) {
      nonzero_size++;
    }
  }
  if (nonzero_size == 0) {
    return;
  } else {
    out_shape->emplace_back(nonzero_size);
  }
}
int NonZero::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  MS_ASSERT(inputs_.size() == 1);
  auto input_tensor = inputs_.front();
  MS_ASSERT(input_tensor != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  output->set_data_type(TypeId::kNumberTypeInt32);
  output->set_format(input_tensor->format());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }

  std::vector<int> out_shape;
  if (inputs_.size() == kSingleNum) {
    if (input_tensor->data_c() == nullptr) {
      MS_LOG(INFO) << "Do infer shape in runtime.";
      return RET_INFER_INVALID;
    }
    switch (input_tensor->data_type()) {
      case kNumberTypeBool: {
        auto data = reinterpret_cast<bool *>(input_tensor->MutableData());
        CalShape<bool>(data, inputs_, &out_shape);
      } break;
      default: {
        MS_LOG(ERROR) << "NonZero weight tensor has unsupported dataType: " << input_tensor->data_type();
        return RET_INFER_ERR;
      }
    }
  } else {
    MS_LOG(ERROR) << "inputs tensor size invalid.";
    return RET_INFER_ERR;
  }
  output->set_shape(out_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
