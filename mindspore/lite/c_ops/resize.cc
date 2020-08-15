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

#include "c_ops/resize.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
int Resize::GetFormat() const { return this->primitive->value.AsResize()->format; }
int Resize::GetMethod() const { return this->primitive->value.AsResize()->method; }
long Resize::GetNewHeight() const { return this->primitive->value.AsResize()->newHeight; }
long Resize::GetNewWidth() const { return this->primitive->value.AsResize()->newWidth; }
bool Resize::GetAlignCorners() const { return this->primitive->value.AsResize()->alignCorners; }
bool Resize::GetPreserveAspectRatio() const { return this->primitive->value.AsResize()->preserveAspectRatio; }

void Resize::SetFormat(int format) { this->primitive->value.AsResize()->format = (schema::Format)format; }
void Resize::SetMethod(int method) { this->primitive->value.AsResize()->method = (schema::ResizeMethod)method; }
void Resize::SetNewHeight(long new_height) { this->primitive->value.AsResize()->newHeight = new_height; }
void Resize::SetNewWidth(long new_width) { this->primitive->value.AsResize()->newWidth = new_width; }
void Resize::SetAlignCorners(bool align_corners) { this->primitive->value.AsResize()->alignCorners = align_corners; }
void Resize::SetPreserveAspectRatio(bool preserve_aspect_ratio) {
  this->primitive->value.AsResize()->preserveAspectRatio = preserve_aspect_ratio;
}

#else

int Resize::GetFormat() const { return this->primitive->value_as_Resize()->format(); }
int Resize::GetMethod() const { return this->primitive->value_as_Resize()->method(); }
long Resize::GetNewHeight() const { return this->primitive->value_as_Resize()->newHeight(); }
long Resize::GetNewWidth() const { return this->primitive->value_as_Resize()->newWidth(); }
bool Resize::GetAlignCorners() const { return this->primitive->value_as_Resize()->alignCorners(); }
bool Resize::GetPreserveAspectRatio() const { return this->primitive->value_as_Resize()->preserveAspectRatio(); }

void Resize::SetFormat(int format) {}
void Resize::SetMethod(int method) {}
void Resize::SetNewHeight(long new_height) {}
void Resize::SetNewWidth(long new_width) {}
void Resize::SetAlignCorners(bool align_corners) {}
void Resize::SetPreserveAspectRatio(bool preserve_aspect_ratio) {}
#endif
namespace {
constexpr int kInputRank = 4;
}  // namespace
int Resize::InferShape(std::vector<lite::tensor::Tensor *> inputs_, std::vector<lite::tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive != nullptr);
  auto input = inputs_.front();
  if (input == nullptr) {
    return 1;
  }
  MS_ASSERT(input->shape().size() == kInputRank);

  auto output = outputs_.front();
  if (output == nullptr) {
    return 1;
  }
  auto new_height = GetNewHeight();
  auto new_width = GetNewWidth();

  std::vector<int> output_shape;
  output_shape.push_back(input->Batch());
  output_shape.push_back(new_height);
  output_shape.push_back(new_width);
  output_shape.push_back(input->Channel());
  output->set_shape(output_shape);
  output->set_data_type(input->data_type());
  output->SetFormat(input->GetFormat());

  return 0;
}
}  // namespace mindspore
