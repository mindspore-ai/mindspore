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

#include "src/ops/constant_of_shape.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/tensor.h"
#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore::lite {
namespace {
constexpr int kShapeInputNum = 1;
constexpr int kShapeOutputNum = 1;
}  // namespace
#ifdef PRIMITIVE_WRITEABLE
std::vector<float> ConstantOfShape::GetValue() const { return this->primitive_->value.AsConstantOfShape()->value; }

int ConstantOfShape::GetDataType() const { return this->primitive_->value.AsConstantOfShape()->dataType; }

#else
int ConstantOfShape::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_ConstantOfShape();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_ConstantOfShape return nullptr";
    return RET_ERROR;
  }
  std::vector<float> value;
  if (attr->value() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->value()->size()); i++) {
      value.push_back(attr->value()->data()[i]);
    }
  }
  auto val_offset = schema::CreateConstantOfShapeDirect(*fbb, attr->dataType(), &value);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_ConstantOfShape, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
std::vector<float> ConstantOfShape::GetValue() const {
  auto fb_vector = this->primitive_->value_as_ConstantOfShape()->value();
  return std::vector<float>(fb_vector->begin(), fb_vector->end());
}
int ConstantOfShape::GetDataType() const { return this->primitive_->value_as_ConstantOfShape()->dataType(); }

PrimitiveC *ConstantOfShapeCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<ConstantOfShape>(primitive);
}
Registry ConstantOfShapeRegistry(schema::PrimitiveType_ConstantOfShape, ConstantOfShapeCreator);

#endif

int ConstantOfShape::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  if (inputs_.size() != kShapeInputNum) {
    MS_LOG(ERROR) << "inputs to ConstantOfShape operator should be 1, but " << inputs_.size() << " is given.";
    return RET_ERROR;
  }
  if (inputs_.front() == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr!";
    return RET_PARAM_INVALID;
  }
  if (outputs_.size() != kShapeOutputNum) {
    MS_LOG(ERROR) << "outputs to ConstantOfShape operator should be 1, but " << outputs_.size() << " is given.";
    return RET_ERROR;
  }

  auto in_tensor = inputs_.front();
  auto out_tensor = outputs_.front();
  out_tensor->set_data_type(static_cast<TypeId>(GetDataType()));
  out_tensor->set_format(in_tensor->format());

  if (!infer_flag() || in_tensor->data_c() == nullptr) {
    return RET_INFER_INVALID;
  }

  int size = in_tensor->ElementsNum();
  std::vector<int> out_shape(size);

  switch (in_tensor->data_type()) {
    case kNumberTypeInt32: {
      int32_t *in_data = reinterpret_cast<int32_t *>(in_tensor->data_c());
      for (int i = 0; i < size; ++i) {
        out_shape[i] = in_data[i];
        MS_ASSERT(out_shape[i] > 0);
      }
      break;
    }
    case kNumberTypeInt64: {
      int64_t *in_data = reinterpret_cast<int64_t *>(in_tensor->data_c());
      for (int i = 0; i < size; ++i) {
        out_shape[i] = in_data[i];
        MS_ASSERT(out_shape[i] > 0);
      }
      break;
    }
    default:
      MS_LOG(INFO) << "Invalid input data type!";
      return RET_INFER_INVALID;
  }

  out_tensor->set_shape(out_shape);
  return RET_OK;
}
}  // namespace mindspore::lite
