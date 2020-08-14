/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_MATRIX_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_MATRIX_H_

#include <stdlib.h>
#include <vector>
#include "src/runtime/kernel/arm/nnacl/winograd_utils.h"

namespace mindspore::kernel {
class Matrix {
 public:
  Matrix() = default;
  ~Matrix() {
    if (data_ != nullptr) {
      free(data_);
    }
  }

  void SetData(void *data) { this->data_ = data; }

  void *GetData() { return this->data_; }

  void SetNDim(int dim) { this->n_dim_ = dim; }

  int GetNDim() { return this->n_dim_; }

  void SetShape(std::vector<int> shape) { this->shape_ = shape; }

  std::vector<int> GetShape() { return this->shape_; }

  void SetStride(std::vector<int> stride) { this->stride_ = stride; }

  std::vector<int> GetStride() { return this->stride_; }

  void SetNum(int m, int k) {
    this->m_ = m;
    this->k_ = k;
  }

  int GetM() { return this->m_; }

  int GetK() { return this->k_; }

 protected:
  void *data_;
  std::vector<int> shape_;
  std::vector<int> stride_;
  int m_;
  int k_;
  int n_dim_;
  bool row_major_;
};

Matrix *TransformMatrixGenerator(int m, int k);

// Chinese Remainder Theorem interp: 0.5
void ChooseMatrixG(Matrix *matrix_g, Matrix *matrix_gt);

void MatrixMultiply(const float *matrix_a, const float *matrix_b, float *matrix_c, int m, int k, int n, bool row);
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_MATRIX_H_
