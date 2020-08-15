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

#include "c_ops/roi_pooling.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
int ROIPooling::GetPooledH() const { return this->primitive->value.AsROIPooling()->pooledH; }
int ROIPooling::GetPooledW() const { return this->primitive->value.AsROIPooling()->pooledW; }
float ROIPooling::GetScale() const { return this->primitive->value.AsROIPooling()->scale; }

void ROIPooling::SetPooledH(int pooled_h) { this->primitive->value.AsROIPooling()->pooledH = pooled_h; }
void ROIPooling::SetPooledW(int pooled_w) { this->primitive->value.AsROIPooling()->pooledW = pooled_w; }
void ROIPooling::SetScale(float scale) { this->primitive->value.AsROIPooling()->scale = scale; }

#else

int ROIPooling::GetPooledH() const { return this->primitive->value_as_ROIPooling()->pooledH(); }
int ROIPooling::GetPooledW() const { return this->primitive->value_as_ROIPooling()->pooledW(); }
float ROIPooling::GetScale() const { return this->primitive->value_as_ROIPooling()->scale(); }

void ROIPooling::SetPooledH(int pooled_h) {}
void ROIPooling::SetPooledW(int pooled_w) {}
void ROIPooling::SetScale(float scale) {}
#endif

int ROIPooling::InferShape(std::vector<lite::tensor::Tensor *> inputs_, std::vector<lite::tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive != nullptr);
  if (inputs_.size() != kDoubleNum) {
    MS_LOG(ERROR) << "inputs number is not equal to " << kDoubleNum;
    return 1;
  }
  auto input = inputs_.front();
  if (input == nullptr) {
    return 1;
  }
  auto roi = inputs_.at(1);
  if (roi == nullptr) {
    return 1;
  }
  auto output = outputs_.front();
  if (output == nullptr) {
    return 1;
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
  output->set_data_type(input->data_type());
  output->SetFormat(input->GetFormat());

  return 0;
}
}  // namespace mindspore
