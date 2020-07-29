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

#include "src/ops/ops.h"
#include "include/errorcode.h"
#include "utils/log_adapter.h"
#include "src/ir/tensor.h"

namespace mindspore::lite {
namespace {
constexpr int kSplitInputNum = 1;
}  // namespace
int Split::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto spilt_prim = this->primitive->value_as_Split();
  MS_ASSERT(spilt_prim != nullptr);
  if (inputs_.size() != kSplitInputNum) {
    MS_LOG(ERROR) << "inputs number is not equal to " << kSplitInputNum;
    return RET_ERROR;
  }
  auto output = outputs_.front();
  if (output == nullptr) {
    MS_LOG(ERROR) << "output null pointer dereferencing.";
    return RET_ERROR;
  }
  int number_split = spilt_prim->numberSplit();
  if (outputs_.size() != number_split) {
    MS_LOG(ERROR) << "outputs number is not equal to " << number_split;
    return RET_ERROR;
  }
  int split_dim = spilt_prim->splitDim();
  std::vector<int> input_shape = input->shape();
  std::vector<int> size_split;
  size_split.insert(size_split.begin(), spilt_prim->sizeSplits()->begin(), spilt_prim->sizeSplits()->end());

  for (int i = 0; i < number_split; ++i) {
    std::vector<int> output_shape;
    output_shape.insert(output_shape.begin(), input_shape.begin(), input_shape.end());
    auto split_dim_i = size_split.empty() ? input_shape[split_dim] / number_split : size_split[i];
    output_shape[split_dim] = split_dim_i;
    outputs_[i]->set_shape(output_shape);
    outputs_[i]->set_data_type(input->data_type());
    outputs_[i]->SetFormat(input->GetFormat());
  }
  return RET_OK;
}
}  // namespace mindspore::lite
