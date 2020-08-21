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

#include "src/ops/crop.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
long Crop::GetAxis() const { return this->primitive_->value.AsCrop()->axis; }
std::vector<long> Crop::GetOffsets() const { return this->primitive_->value.AsCrop()->offsets; }

void Crop::SetAxis(long axis) { this->primitive_->value.AsCrop()->axis = axis; }
void Crop::SetOffsets(const std::vector<long> &offsets) { this->primitive_->value.AsCrop()->offsets = offsets; }

#else

long Crop::GetAxis() const { return this->primitive_->value_as_Crop()->axis(); }
std::vector<long> Crop::GetOffsets() const {
  auto fb_vector = this->primitive_->value_as_Crop()->offsets();
  return std::vector<long>(fb_vector->begin(), fb_vector->end());
}

void Crop::SetAxis(long axis) {}
void Crop::SetOffsets(const std::vector<long> &offsets) {}
#endif
namespace {
constexpr int kCropOutputNum = 1;
constexpr int kCropInputNum = 2;
}  // namespace
int Crop::InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) {
  if (outputs.size() != kCropOutputNum || inputs.size() != kCropInputNum) {
    MS_LOG(ERROR) << "Invalid output/input size! output size: " << outputs.size() << ",input size: " << inputs.size();
    return RET_PARAM_INVALID;
  }
  outputs[0]->SetFormat(inputs[0]->GetFormat());
  outputs[0]->set_data_type(inputs[0]->data_type());
  if (!GetInferFlag()) {
    return RET_OK;
  }
  outputs[0]->set_shape(inputs[1]->shape());
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
