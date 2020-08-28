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

#include "src/ops/pad.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> Pad::GetPaddings() const { return this->primitive_->value.AsPad()->paddings; }
int Pad::GetPaddingMode() const { return this->primitive_->value.AsPad()->paddingMode; }
float Pad::GetConstantValue() const { return this->primitive_->value.AsPad()->constantValue; }

void Pad::SetPaddings(const std::vector<int> &paddings) { this->primitive_->value.AsPad()->paddings = paddings; }
void Pad::SetPaddingMode(int padding_mode) {
  this->primitive_->value.AsPad()->paddingMode = (schema::PaddingMode)padding_mode;
}
void Pad::SetConstantValue(float constant_value) { this->primitive_->value.AsPad()->constantValue = constant_value; }

#else

std::vector<int> Pad::GetPaddings() const {
  auto fb_vector = this->primitive_->value_as_Pad()->paddings();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
int Pad::GetPaddingMode() const { return this->primitive_->value_as_Pad()->paddingMode(); }
float Pad::GetConstantValue() const { return this->primitive_->value_as_Pad()->constantValue(); }

int Pad::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Pad();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Pad return nullptr";
    return RET_ERROR;
  }
  std::vector<int32_t> paddings;
  if (attr->paddings() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->paddings()->size()); i++) {
      paddings.push_back(attr->paddings()->data()[i]);
    }
  }
  auto val_offset = schema::CreatePadDirect(*fbb, &paddings, attr->paddingMode(), attr->constantValue());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Pad, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
#endif
namespace {
const size_t kInputRank = 4;
}  // namespace
int Pad::InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) {
  MS_ASSERT(this->primitive_ != nullptr);
  if (this->primitive_ == nullptr) {
    return RET_NULL_PTR;
  }

  auto paddings = GetPaddings();

  auto input = inputs.front();
  if (input == nullptr) {
    return RET_NULL_PTR;
  }
  auto output = outputs.front();
  if (output == nullptr) {
    return RET_NULL_PTR;
  }
  output->SetFormat(input->GetFormat());
  output->set_data_type(input->data_type());
  if (!GetInferFlag()) {
    return RET_OK;
  }
  auto input_shape = input->shape();
  std::vector<int> output_shape;
  MS_ASSERT(input->shape().size() <= kInputRank);
  for (size_t i = 0; i < input_shape.size(); i++) {
    auto paddings_index = i + kInputRank - input_shape.size();
    auto shape = input_shape[i] + paddings[2 * paddings_index] + paddings[2 * paddings_index + 1];
    output_shape.push_back(shape);
  }

  output->set_shape(output_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
