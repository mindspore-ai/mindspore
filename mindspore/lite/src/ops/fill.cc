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

#include "src/ops/fill.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> Fill::GetDims() const { return this->primitive->value.AsFill()->dims; }

void Fill::SetDims(const std::vector<int> &dims) { this->primitive->value.AsFill()->dims = dims; }

#else

std::vector<int> Fill::GetDims() const {
  auto fb_vector = this->primitive->value_as_Fill()->dims();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}

void Fill::SetDims(const std::vector<int> &dims) {}
#endif

int Fill::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive != nullptr);
  auto input = inputs_.front();
  auto output = outputs_.front();
  if (input == nullptr || output == nullptr) {
    MS_LOG(ERROR) << "Fill input or output is null!";
    return RET_ERROR;
  }
  if (inputs_.size() != kSingleNum || outputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "input size: " << inputs_.size() << ", output size: " << outputs_.size();
    return RET_INPUT_TENSOR_ERROR;
  }
  auto fill_prim = this->primitive->value_as_Fill();
  if (fill_prim == nullptr) {
    MS_LOG(ERROR) << "Fill primitive is null!";
    return RET_ERROR;
  }
  std::vector<int> output_shape;
  (void)output_shape.insert(output_shape.begin(), fill_prim->dims()->begin(), fill_prim->dims()->end());
  output->set_shape(output_shape);
  output->set_data_type(input->data_type());
  output->SetFormat(input->GetFormat());
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
