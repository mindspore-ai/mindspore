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
int FullConnection::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive != nullptr);
  auto input0 = inputs_.front();
  MS_ASSERT(input0 != nullptr);
  auto input1 = inputs_[1];
  MS_ASSERT(input1 != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  auto fc_prim = this->primitive->value_as_FullConnection();
  if ((fc_prim->hasBias() && inputs_.size() != kMultiNum) || (!fc_prim->hasBias() && inputs_.size() != kDoubleNum)) {
    MS_LOG(ERROR) << "Input tensors num error";
    return RET_INPUT_TENSOR_ERROR;
  }
  auto axis = fc_prim->axis();
  auto use_axis = fc_prim->useAxis();
  if (use_axis && (axis < 1 || axis >= input0->shape().size())) {
    MS_LOG(ERROR) << "FullConnection axis invalid";
    return RET_INPUT_TENSOR_ERROR;
  }
  int new_k = 1;
  if (use_axis) {
    for (int i = axis; i < input0->shape().size(); ++i) {
      new_k *= input0->shape()[i];
    }
    if (new_k != input1->shape()[1]) {
      MS_LOG(ERROR) << "Input1 size invalid";
      return RET_PARAM_INVALID;
    }
  } else {
    new_k = input1->shape()[1];
  }

  if (fc_prim->hasBias()) {
    if (inputs_[2]->shape()[0] != input1->shape()[0]) {
      MS_LOG(ERROR) << "bias size invalid";
      return RET_PARAM_INVALID;
    }
  }
  std::vector<int> out_shape{inputs_[0]->shape()};
  if (use_axis) {
    out_shape.resize(fc_prim->axis() + 1);
    out_shape[fc_prim->axis()] = input1->shape()[0];
  } else {
    int total = 1;
    for (int i = 0; i < input0->shape().size(); ++i) {
      total *= input0->shape()[i];
    }
    out_shape.resize(2);
    auto batch_size = total / new_k;
    out_shape[0] = batch_size;
    out_shape[1] = input1->shape()[0];
  }
  output->set_shape(out_shape);
  output->set_data_type(input0->data_type());
  output->SetFormat(input0->GetFormat());

  return RET_OK;
}
}  // namespace mindspore::lite
