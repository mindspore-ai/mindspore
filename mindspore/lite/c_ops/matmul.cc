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

#include "c_ops/matmul.h"
#include <utility>

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
bool MatMul::GetTransposeA() const { return this->primitive->value.AsMatMul()->transposeA; }
bool MatMul::GetTransposeB() const { return this->primitive->value.AsMatMul()->transposeB; }

void MatMul::SetTransposeA(bool transpose_a) { this->primitive->value.AsMatMul()->transposeA = transpose_a; }
void MatMul::SetTransposeB(bool transpose_b) { this->primitive->value.AsMatMul()->transposeB = transpose_b; }

#else

bool MatMul::GetTransposeA() const { return this->primitive->value_as_MatMul()->transposeA(); }
bool MatMul::GetTransposeB() const { return this->primitive->value_as_MatMul()->transposeB(); }

void MatMul::SetTransposeA(bool transpose_a) {}
void MatMul::SetTransposeB(bool transpose_b) {}
#endif
int MatMul::InferShape(std::vector<lite::tensor::Tensor *> inputs_, std::vector<lite::tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive != nullptr);
  if (inputs_.size() != kDoubleNum) {
    MS_LOG(ERROR) << "OpMatMul inputs size: " << inputs_.size();
    return 1;
  }
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
    return 1;
  }

  for (int i = 0; i < a_shape.size() - 2; ++i) {
    if (a_shape[i] != b_shape[i]) {
      MS_LOG(ERROR) << "Op MatMul's dimensions must be equal";
      return 1;
    }
  }

  if (GetTransposeA()) {
    std::swap(a_shape[a_shape.size() - 1], a_shape[a_shape.size() - 2]);
  }
  if (GetTransposeB()) {
    std::swap(b_shape[b_shape.size() - 1], b_shape[b_shape.size() - 2]);
  }
  std::vector<int> c_shape(a_shape);
  c_shape[c_shape.size() - 1] = b_shape[b_shape.size() - 1];
  output->set_shape(c_shape);
  output->set_data_type(input0->data_type());
  output->SetFormat(input0->GetFormat());

  return 0;
}
}  // namespace mindspore
