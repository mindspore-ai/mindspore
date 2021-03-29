/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "wrapper/int8/matmul_int8_wrapper.h"

void InitInt8MatrixA(int8_t *src_ptr, int32_t *input_sums, int8_t *dst_ptr, int batch, int row, int deep, int input_zp,
                     const int *weight_zp, bool a_transpose) {
  for (int i = 0; i < batch; ++i) {
    int8_t *cur_a_ptr = src_ptr + i * row * deep;
    if (a_transpose) {
      RowMajor2Col16x4MajorInt8(cur_a_ptr, deep, row, dst_ptr);
      CalcInputSums(cur_a_ptr, row, deep, *weight_zp, input_sums, ColMajor);
    } else {
      RowMajor2Row16x4MajorInt8(cur_a_ptr, dst_ptr, row, deep);
      CalcInputSums(cur_a_ptr, row, deep, *weight_zp, input_sums, RowMajor);
    }
  }
}

void InitInt8MatrixB(int8_t *weight_ptr, int32_t *weight_bias_sums_batch_, int8_t *dst_ptr, int batch, int deep,
                     int col, int col_align, int deep_16, int input_zp, int *weight_zp, const int *bias_ptr,
                     bool b_transpose, bool filter_per_channel) {
  for (int i = 0; i < batch; ++i) {
    int8_t *cur_b = weight_ptr + i * deep * col;
    int8_t *cur_b_pack = dst_ptr + i * col_align * deep_16;
    int32_t *cur_sums = weight_bias_sums_batch_ + i * col_align;
    if (b_transpose) {
#ifdef ENABLE_ARM32
      RowMajor2Row2x16MajorInt8(cur_b, cur_b_pack, col, deep);
#else
      RowMajor2Row16x4MajorInt8(cur_b, cur_b_pack, col, deep);
#endif
      CalcWeightBiasSums(cur_b, deep, col, input_zp, weight_zp, bias_ptr, cur_sums, ColMajor, filter_per_channel);
    } else {
#ifdef ENABLE_ARM32
      RowMajor2Col16x2MajorInt8(cur_b, cur_b_pack, deep, col);
#else
      RowMajor2Col16x4MajorInt8(cur_b, deep, col, cur_b_pack);
#endif
      CalcWeightBiasSums(cur_b, deep, col, input_zp, weight_zp, bias_ptr, cur_sums, RowMajor, false);
    }
  }
}
