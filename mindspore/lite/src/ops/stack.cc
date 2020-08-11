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
constexpr int kStackOutputNum = 1;
constexpr int kStackMinInputNum = 2;
}  // namespace

int Stack::InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) {
  MS_ASSERT(this->primitive != nullptr);
  if (outputs.size() != kStackOutputNum) {
    MS_LOG(ERROR) << "Invalid output size:" << outputs.size();
    return RET_PARAM_INVALID;
  }
  if (inputs.size() < kStackMinInputNum) {
    MS_LOG(ERROR) << "Invalid input size " << inputs.size();
    return RET_PARAM_INVALID;
  }
  auto input = inputs.at(0);
  auto input_shape = input->shape();
  auto stack_prim = this->primitive->value_as_Stack();
  std::vector<int32_t> output_shape = input_shape;
  int axis = stack_prim->axis() < 0 ? stack_prim->axis() + input_shape.size() : stack_prim->axis();
  if (axis < 0 || axis > input_shape.size()) {
    MS_LOG(ERROR) << "Invalid axis " << stack_prim->axis();
    return RET_PARAM_INVALID;
  }
  schema::Format input0_format = input->GetFormat();
  for (size_t i = 1; i < inputs.size(); ++i) {
    if (inputs[i]->GetFormat() != input0_format) {
      MS_LOG(ERROR) << "All inputs should have the same format!";
      return RET_PARAM_INVALID;
    }

    auto input_shape_tmp = inputs[i]->shape();
    if (input_shape_tmp.size() != input_shape.size()) {
      MS_LOG(ERROR) << "All input shape size should be the same!";
      return RET_PARAM_INVALID;
    }
    for (size_t j = 0; j < input_shape.size(); ++j) {
      if (input_shape_tmp[j] != input_shape[j]) {
        MS_LOG(ERROR) << "All input shape should be the same!";
        return RET_PARAM_INVALID;
      }
    }
  }

  output_shape.insert(output_shape.begin() + axis, inputs.size());
  outputs[0]->set_shape(output_shape);
  outputs[0]->set_data_type(input->data_type());
  outputs[0]->SetFormat(input->GetFormat());

  return RET_OK;
}
}  // namespace mindspore::lite
