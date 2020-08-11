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
const size_t kPaddingsSize = 8;
const size_t kInputRank = 4;
}  // namespace
int Pad::InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) {
  MS_ASSERT(this->primitive != nullptr);
  if (this->primitive == nullptr) {
    return RET_NULL_PTR;
  }
  auto pad_prim = this->primitive->value_as_Pad();
  if (pad_prim == nullptr) {
    return RET_NULL_PTR;
  }
  auto paddings = pad_prim->paddings();
  if (paddings == nullptr) {
    return RET_NULL_PTR;
  }

  auto input = inputs.front();
  if (input == nullptr) {
    return RET_NULL_PTR;
  }
  auto input_shape = input->shape();
  std::vector<int> output_shape;
  MS_ASSERT(input->shape().size() <= kInputRank);
  for (size_t i = 0; i < input_shape.size(); i++) {
    auto paddings_index = i + kInputRank - input_shape.size();
    auto shape = input_shape[i] + (*paddings)[2 * paddings_index] + (*paddings)[2 * paddings_index + 1];
    output_shape.push_back(shape);
  }

  auto output = outputs.front();
  if (output == nullptr) {
    return RET_NULL_PTR;
  }
  output->SetFormat(input->GetFormat());
  output->set_shape(output_shape);
  output->set_data_type(input->data_type());
  return RET_OK;
}
}  // namespace mindspore::lite
