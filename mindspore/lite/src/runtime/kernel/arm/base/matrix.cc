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

#include "src/runtime/kernel/arm/base/matrix.h"
#include "utils/log_adapter.h"

namespace mindspore::kernel {
Matrix *TransformMatrixGenerator(int m, int k) {
  auto matrix = new Matrix;
  auto aa = malloc(m * k * sizeof(float));
  matrix->SetData(aa);
  matrix->SetNum(m, k);
  return matrix;
}

void ChooseMatrixG(Matrix *matrix_g, Matrix *matrix_gt) {
  int m = matrix_g->GetM();
  int k = matrix_g->GetK();
  auto matrix_g_data = reinterpret_cast<float *>(matrix_g->GetData());
  auto matrix_gt_data = reinterpret_cast<float *>(matrix_gt->GetData());
  // m represents input unit, only 4 or 8 can be accepted for input unit.
  // k represents kernel unit, varies from 2 to 7.
  if (m == 4 && k == 2) {
    MatrixG4x2(matrix_g_data);
    MatrixGT2x4(matrix_gt_data);
  } else if (m == 8 && k == 2) {
    MatrixG8x2(matrix_g_data);
    MatrixGT2x8(matrix_gt_data);
  } else if (m == 8 && k == 3) {
    MatrixG8x3(matrix_g_data);
    MatrixGT3x8(matrix_gt_data);
  } else if (m == 8 && k == 4) {
    MatrixG8x4(matrix_g_data);
    MatrixGT4x8(matrix_gt_data);
  } else if (m == 8 && k == 5) {
    MatrixG8x5(matrix_g_data);
    MatrixGT5x8(matrix_gt_data);
  } else if (m == 8 && k == 6) {
    MatrixG8x6(matrix_g_data);
    MatrixGT6x8(matrix_gt_data);
  } else if (m == 8 && k == 7) {
    MatrixG8x7(matrix_g_data);
    MatrixGT7x8(matrix_gt_data);
  } else {
    MS_LOG(ERROR) << "Unsupported input unit or kernel unit.";
    return;
  }
}

void MatrixMultiply(const float *matrix_a, const float *matrix_b, float *matrix_c, int m, int k, int n, bool row) {
  // row-major implementation
  int count = 0;
  for (int h = 0; h < m; h++) {
    int h_offset = h * k;
    for (int w = 0; w < n; w++) {
      float res = 0;
      for (int i = 0; i < k; i++) {
        res += *(matrix_a + h_offset + i) * *(matrix_b + w + i * n);
      }
      *(matrix_c + count) = res;
      count++;
    }
  }
}
}  // namespace mindspore::kernel
