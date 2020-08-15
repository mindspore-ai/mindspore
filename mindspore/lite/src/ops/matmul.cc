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
  auto input0 = inputs_.front();
  MS_ASSERT(input0 != nullptr);
  auto input1 = inputs_.at(1);
  MS_ASSERT(input1 != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);

  std::vector<int> a_shape = input0->shape();
  std::vector<int> b_shape = input1->shape();
  if (a_shape.size() < 2 || b_shape.size() < 2) {
    MS_LOG(ERROR) << "inputs shape is invalid";
    return RET_INPUT_TENSOR_ERROR;
  }

  for (int i = 0; i < a_shape.size() - 2; ++i) {
    if (a_shape[i] != b_shape[i]) {
      MS_LOG(ERROR) << "Op MatMul's dimensions must be equal";
      return RET_INPUT_TENSOR_ERROR;
    }
  }

  auto matmul_prim = this->primitive->value_as_MatMul();
  if (matmul_prim->transposeA()) {
    std::swap(a_shape[a_shape.size() - 1], a_shape[a_shape.size() - 2]);
  }
  if (matmul_prim->transposeB()) {
    std::swap(b_shape[b_shape.size() - 1], b_shape[b_shape.size() - 2]);
  }
  std::vector<int> c_shape(a_shape);
  c_shape[c_shape.size() - 1] = b_shape[b_shape.size() - 1];
  output->set_shape(c_shape);
  output->set_data_type(input0->data_type());
  output->SetFormat(input0->GetFormat());

  return RET_OK;
}
}  // namespace mindspore::lite
