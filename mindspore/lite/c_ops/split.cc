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

#include "c_ops/split.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
int Split::GetNumberSplit() const { return this->primitive->value.AsSplit()->numberSplit; }
std::vector<int> Split::GetSizeSplits() const { return this->primitive->value.AsSplit()->sizeSplits; }
int Split::GetSplitDim() const { return this->primitive->value.AsSplit()->splitDim; }

void Split::SetNumberSplit(int number_split) { this->primitive->value.AsSplit()->numberSplit = number_split; }
void Split::SetSizeSplits(const std::vector<int> &size_splits) {
  this->primitive->value.AsSplit()->sizeSplits = size_splits;
}
void Split::SetSplitDim(int split_dim) { this->primitive->value.AsSplit()->splitDim = split_dim; }

#else

int Split::GetNumberSplit() const { return this->primitive->value_as_Split()->numberSplit(); }
std::vector<int> Split::GetSizeSplits() const {
  auto fb_vector = this->primitive->value_as_Split()->sizeSplits();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
int Split::GetSplitDim() const { return this->primitive->value_as_Split()->splitDim(); }

void Split::SetNumberSplit(int number_split) {}
void Split::SetSizeSplits(const std::vector<int> &size_splits) {}
void Split::SetSplitDim(int split_dim) {}
#endif
namespace {
constexpr int kSplitInputNum = 1;
}  // namespace
int Split::InferShape(std::vector<lite::tensor::Tensor *> inputs_, std::vector<lite::tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);

  MS_ASSERT(spilt_prim != nullptr);
  if (inputs_.size() != kSplitInputNum) {
    MS_LOG(ERROR) << "inputs number is not equal to " << kSplitInputNum;
    return 1;
  }
  auto output = outputs_.front();
  if (output == nullptr) {
    MS_LOG(ERROR) << "output null pointer dereferencing.";
    return 1;
  }
  int number_split = GetNumberSplit();
  if (outputs_.size() != number_split) {
    MS_LOG(ERROR) << "outputs number is not equal to " << number_split;
    return 1;
  }
  int split_dim = GetSplitDim();
  std::vector<int> input_shape = input->shape();
  std::vector<int> size_split;
  size_split.insert(size_split.begin(), GetSizeSplits().begin(), GetSizeSplits().end());

  for (int i = 0; i < number_split; ++i) {
    std::vector<int> output_shape;
    output_shape.insert(output_shape.begin(), input_shape.begin(), input_shape.end());
    auto split_dim_i = size_split.empty() ? input_shape[split_dim] / number_split : size_split[i];
    output_shape[split_dim] = split_dim_i;
    outputs_[i]->set_shape(output_shape);
    outputs_[i]->set_data_type(input->data_type());
    outputs_[i]->SetFormat(input->GetFormat());
  }
  return 0;
}
}  // namespace mindspore
