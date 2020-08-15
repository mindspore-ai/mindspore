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

#include "c_ops/reduce.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> Reduce::GetAxes() const { return this->primitive->value.AsReduce()->axes; }
int Reduce::GetKeepDims() const { return this->primitive->value.AsReduce()->keepDims; }
int Reduce::GetMode() const { return this->primitive->value.AsReduce()->mode; }

void Reduce::SetAxes(const std::vector<int> &axes) { this->primitive->value.AsReduce()->axes = axes; }
void Reduce::SetKeepDims(int keep_dims) { this->primitive->value.AsReduce()->keepDims = keep_dims; }
void Reduce::SetMode(int mode) { this->primitive->value.AsReduce()->mode = (schema::ReduceMode)mode; }

#else

std::vector<int> Reduce::GetAxes() const {
  auto fb_vector = this->primitive->value_as_Reduce()->axes();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
int Reduce::GetKeepDims() const { return this->primitive->value_as_Reduce()->keepDims(); }
int Reduce::GetMode() const { return this->primitive->value_as_Reduce()->mode(); }

void Reduce::SetAxes(const std::vector<int> &axes) {}
void Reduce::SetKeepDims(int keep_dims) {}
void Reduce::SetMode(int mode) {}
#endif
namespace {
constexpr size_t kInputSize = 1;
constexpr size_t kOutputSize = 1;
}  // namespace
int Reduce::InferShape(std::vector<lite::tensor::Tensor *> inputs_, std::vector<lite::tensor::Tensor *> outputs_) {
  if (inputs_.size() != kInputSize || outputs_.size() != kOutputSize) {
    return 1;
  }
  auto input = inputs_.front();
  auto output = outputs_.front();
  if (input == nullptr || output == nullptr) {
    return 1;
  }
  if (this->primitive == nullptr) {
    return 1;
  }

  bool keep_dims = static_cast<bool>(GetKeepDims());
  std::vector<int> in_shape = input->shape();
  std::vector<int> out_shape;
  const auto &axes = GetAxes();
  auto num_axes = axes.size();
  // reduce on all axes
  if (num_axes == 0) {
    if (keep_dims) {
      for (auto i = 0; i < in_shape.size(); i++) {
        out_shape.push_back(1);
      }
    }
    output->set_shape(out_shape);
    output->set_data_type(input->data_type());
    return 0;
  }

  // reduce on selected axes
  for (size_t i = 0; i < in_shape.size(); i++) {
    bool reduce_axis = false;
    for (int idx = 0; idx < num_axes; ++idx) {
      if (static_cast<size_t>((axes)[idx]) == i) {
        reduce_axis = true;
        break;
      }
    }
    if (reduce_axis) {
      if (keep_dims) {
        out_shape.push_back(1);
      }
    } else {
      out_shape.push_back(in_shape[i]);
    }
  }
  output->set_shape(out_shape);
  output->set_data_type(input->data_type());
  output->SetFormat(input->GetFormat());

  return 0;
}
}  // namespace mindspore
