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

#include "src/ops/roi_pooling.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int ROIPooling::GetPooledH() const { return this->primitive_->value.AsROIPooling()->pooledH; }
int ROIPooling::GetPooledW() const { return this->primitive_->value.AsROIPooling()->pooledW; }
float ROIPooling::GetScale() const { return this->primitive_->value.AsROIPooling()->scale; }

void ROIPooling::SetPooledH(int pooled_h) { this->primitive_->value.AsROIPooling()->pooledH = pooled_h; }
void ROIPooling::SetPooledW(int pooled_w) { this->primitive_->value.AsROIPooling()->pooledW = pooled_w; }
void ROIPooling::SetScale(float scale) { this->primitive_->value.AsROIPooling()->scale = scale; }

#else

int ROIPooling::GetPooledH() const { return this->primitive_->value_as_ROIPooling()->pooledH(); }
int ROIPooling::GetPooledW() const { return this->primitive_->value_as_ROIPooling()->pooledW(); }
float ROIPooling::GetScale() const { return this->primitive_->value_as_ROIPooling()->scale(); }
int ROIPooling::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);

  auto attr = primitive->value_as_ROIPooling();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_ROIPooling return nullptr";
    return RET_ERROR;
  }

  auto val_offset = schema::CreateROIPooling(*fbb, attr->pooledH(), attr->pooledW(), attr->scale());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_ROIPooling, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *ROIPoolingCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<ROIPooling>(primitive);
}
Registry ROIPoolingRegistry(schema::PrimitiveType_ROIPooling, ROIPoolingCreator);
#endif

int ROIPooling::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  if (inputs_.size() != kDoubleNum) {
    MS_LOG(ERROR) << "inputs number is not equal to " << kDoubleNum;
    return RET_ERROR;
  }
  auto input = inputs_.front();
  if (input == nullptr) {
    return RET_NULL_PTR;
  }
  auto roi = inputs_.at(1);
  if (roi == nullptr) {
    return RET_NULL_PTR;
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

  auto new_h = GetPooledH();
  auto new_w = GetPooledW();
  auto shape_data = roi->shape();
  std::vector<int> output_shape;
  output_shape.push_back(shape_data[0]);
  output_shape.push_back(new_h);
  output_shape.push_back(new_w);
  output_shape.push_back(input->Channel());
  output->set_shape(output_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
