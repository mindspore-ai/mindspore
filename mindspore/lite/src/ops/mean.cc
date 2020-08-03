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
constexpr size_t kInputSize = 1;
constexpr size_t kOutputSize = 1;
}  // namespace
int Mean::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  if (inputs_.size() != kInputSize || outputs_.size() != kOutputSize) {
    return RET_ERROR;
  }
  auto input = inputs_.front();
  auto output = outputs_.front();
  if (input == nullptr || output == nullptr) {
    return RET_NULL_PTR;
  }
  if (this->primitive == nullptr) {
    return RET_NULL_PTR;
  }
  auto mean_prim = this->primitive->value_as_Mean();
  bool keep_dims = static_cast<bool>(mean_prim->keepDims());
  std::vector<int> in_shape = input->shape();
  std::vector<int> out_shape;
  const auto &axes = mean_prim->axis();
  auto num_axes = axes->size();
  // reduce on all axes
  if (num_axes == 0) {
    if (keep_dims) {
      for (auto i = 0; i < in_shape.size(); i++) {
        out_shape.push_back(1);
      }
    }
    output->set_shape(out_shape);
    output->set_data_type(input->data_type());
    return RET_OK;
  }

  // reduce on selected axes
  for (size_t i = 0; i < in_shape.size(); i++) {
    bool reduce_axis = false;
    for (int idx = 0; idx < num_axes; ++idx) {
      if (static_cast<size_t>((*axes)[idx]) == i) {
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
  return RET_OK;
}
}  // namespace mindspore::lite
