/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/ops/crop_and_resize.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int CropAndResize::GetMethod() const { return this->primitive_->value.AsCropAndResize()->method; }
float CropAndResize::GetExtrapolationValue() const {
  return this->primitive_->value.AsCropAndResize()->extrapolation_value;
}

void CropAndResize::SetMethod(int method) {
  this->primitive_->value.AsCropAndResize()->method = (schema::ResizeMethod)method;
}
void CropAndResize::SetExtrapolationValue(float value) {
  this->primitive_->value.AsCropAndResize()->extrapolation_value = value;
}
#else

int CropAndResize::GetMethod() const { return this->primitive_->value_as_CropAndResize()->method(); }
float CropAndResize::GetExtrapolationValue() const {
  return this->primitive_->value_as_CropAndResize()->extrapolation_value();
}
int CropAndResize::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_CropAndResize();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_CropAndResize return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateCropAndResize(*fbb, attr->method(), attr->extrapolation_value());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_CropAndResize, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *CropAndResizeCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<CropAndResize>(primitive);
}
Registry CropAndResizeRegistry(schema::PrimitiveType_CropAndResize, CropAndResizeCreator);
#endif

namespace {
constexpr int kInputRank = 4;
}  // namespace
int CropAndResize::InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  if (inputs_.size() != 4) {
    MS_LOG(ERROR) << "Input tensor num should be 4 for crop_an_resize.";
    return RET_ERROR;
  }
  auto input = inputs_.front();
  if (input == nullptr) {
    return RET_ERROR;
  }
  if (!input->shape().empty() && input->shape().size() != kInputRank) {
    MS_LOG(ERROR) << "Size of input shape is wrong.";
    return RET_ERROR;
  }
  if (input->format() != schema::Format_NHWC) {
    MS_LOG(ERROR) << "Crop_an_resize op only support NHWC format.";
    return RET_ERROR;
  }

  auto output = outputs_.front();
  if (output == nullptr) {
    return RET_NULL_PTR;
  }
  output->set_data_type(input->data_type());
  output->set_format(input->format());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }

  std::vector<int> output_shape;
  if (inputs_[1]->data_c() != nullptr) {
    auto boxes_tensor = inputs_[1];
    output_shape.push_back(boxes_tensor->shape()[0]);
  } else {
    output_shape.push_back(input->Batch());
  }

  auto shape_tensor = inputs_[3];
  auto data = reinterpret_cast<int32_t *>(shape_tensor->data_c());
  if (data == nullptr) {
    MS_LOG(INFO) << "The data of 4th input tensor(shape tensor) for crop_an_resize op is nullptr.";
    return RET_INFER_INVALID;
  }
  output_shape.push_back(data[0]);
  output_shape.push_back(data[1]);
  output_shape.push_back(input->Channel());
  output->set_shape(output_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
