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

#include "c_ops/pad.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> Pad::GetPaddings() const { return this->primitive->value.AsPad()->paddings; }
int Pad::GetPaddingMode() const { return this->primitive->value.AsPad()->paddingMode; }
float Pad::GetConstantValue() const { return this->primitive->value.AsPad()->constantValue; }

void Pad::SetPaddings(const std::vector<int> &paddings) { this->primitive->value.AsPad()->paddings = paddings; }
void Pad::SetPaddingMode(int padding_mode) { this->primitive->value.AsPad()->paddingMode = padding_mode; }
void Pad::SetConstantValue(float constant_value) { this->primitive->value.AsPad()->constantValue = constant_value; }

#else

std::vector<int> Pad::GetPaddings() const {
  auto fb_vector = this->primitive->value_as_Pad()->paddings();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
int Pad::GetPaddingMode() const { return this->primitive->value_as_Pad()->paddingMode(); }
float Pad::GetConstantValue() const { return this->primitive->value_as_Pad()->constantValue(); }

void Pad::SetPaddings(const std::vector<int> &paddings) {}
void Pad::SetPaddingMode(int padding_mode) {}
void Pad::SetConstantValue(float constant_value) {}
#endif
namespace {
const size_t kPaddingsSize = 8;
const size_t kInputRank = 4;
}  // namespace
int Pad::InferShape(std::vector<lite::tensor::Tensor *> inputs, std::vector<lite::tensor::Tensor *> outputs) {
  MS_ASSERT(this->primitive != nullptr);
  if (this->primitive == nullptr) {
    return 1;
  }

  auto paddings = GetPaddings();

  auto input = inputs.front();
  if (input == nullptr) {
    return 1;
  }
  auto input_shape = input->shape();
  std::vector<int> output_shape;
  MS_ASSERT(input->shape().size() <= kInputRank);
  for (size_t i = 0; i < input_shape.size(); i++) {
    auto paddings_index = i + kInputRank - input_shape.size();
    auto shape = input_shape[i] + (paddings)[2 * paddings_index] + (paddings)[2 * paddings_index + 1];
    output_shape.push_back(shape);
  }

  auto output = outputs.front();
  if (output == nullptr) {
    return 1;
  }
  output->SetFormat(input->GetFormat());
  output->set_shape(output_shape);
  output->set_data_type(input->data_type());
  return 0;
}
}  // namespace mindspore
