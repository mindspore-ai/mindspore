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
int MatMul::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive != nullptr);
  if (inputs_.size() != kDoubleNum) {
    MS_LOG(ERROR) << "OpMatMul inputs size: " << inputs_.size();
    return RET_INPUT_TENSOR_ERROR;
  }
  auto input0 = inputs_.front();
  MS_ASSERT(input0 != nullptr);
  auto input1 = inputs_.at(1);
  MS_ASSERT(input1 != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);

  std::vector<int> x_shape = input0->shape();
  std::vector<int> w_shape = input1->shape();
  if (x_shape.size() < 2 || w_shape.size() < 2) {
    MS_LOG(ERROR) << "inputs shape is invalid";
    return RET_INPUT_TENSOR_ERROR;
  }

  auto matmul_prim = this->primitive->value_as_MatMul();
  if (matmul_prim->transposeA()) {
    int tmp = x_shape.back();
    x_shape[x_shape.size() - 1] = x_shape[x_shape.size() - 2];
    x_shape[x_shape.size() - 2] = tmp;
  }
  if (matmul_prim->transposeB()) {
    int tmp = w_shape.back();
    w_shape[w_shape.size() - 1] = w_shape[w_shape.size() - 2];
    w_shape[w_shape.size() - 2] = tmp;
  }
  auto y_shape_size = std::max(x_shape.size(), w_shape.size());
  std::vector<int> y_shape(y_shape_size);
  y_shape = x_shape;
  y_shape[y_shape_size - 1] = w_shape[w_shape.size() - 1];
  output->set_shape(y_shape);
  output->set_data_type(input0->data_type());
  output->SetFormat(input0->GetFormat());

  return RET_OK;
}
}  // namespace mindspore::lite
