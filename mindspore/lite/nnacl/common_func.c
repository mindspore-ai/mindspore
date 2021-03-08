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

#include "nnacl/common_func.h"

int offset(const int *shape, const int dim0, const int dim1, const int dim2, const int dim3) {
  return ((dim0 * shape[1] + dim1) * shape[2] + dim2) * shape[3] + dim3;
}

int offsetComm(const int *shape, const int dim0, const int dim1, const int dim2) {
  return ((dim0 * shape[1] + dim1) * shape[2] + dim2) * shape[3];
}

int offset4d(const int *shape, const int *dims) { return offset(shape, dims[0], dims[1], dims[2], dims[3]); }

int8_t MinInt8(int8_t a, int8_t b) { return b ^ ((a ^ b) & -(a < b)); }

int8_t MaxInt8(int8_t a, int8_t b) { return a ^ ((a ^ b) & -(a < b)); }

void ReluFp32(float *data, float *dst, int ele_num) {
  int index = 0;
#ifdef ENABLE_AVX
  int c8_block = DOWN_DIV(ele_num, C8NUM) * C8NUM;
  for (; index < c8_block; index += C8NUM) {
    MS_FLOAT32X8 relu_data = MS_LD256_F32(data + index);
    MS_FLOAT32X8 zero_data = MS_MOV256_F32(0.0f);
    relu_data = MS_MAX256_F32(relu_data, zero_data);
    MS_ST256_F32(dst + index, relu_data);
  }
#endif
#if defined(ENABLE_NEON) || defined(ENABLE_SSE)
  int c4_block = DOWN_DIV(ele_num, C4NUM) * C4NUM;
  for (; index < c4_block; index += C4NUM) {
    MS_FLOAT32X4 relu_data = MS_LDQ_F32(data + index);
    MS_FLOAT32X4 zero_data = MS_MOVQ_F32(0.0f);
    relu_data = MS_MAXQ_F32(relu_data, zero_data);
    MS_STQ_F32(dst + index, relu_data);
  }
#endif
  for (; index < ele_num; ++index) {
    data[index] = data[index] < 0.0f ? 0.0f : data[index];
  }
}

void Relu6Fp32(float *data, float *dst, int ele_num) {
  int index = 0;
#ifdef ENABLE_AVX
  int c8_block = DOWN_DIV(ele_num, C8NUM) * C8NUM;
  for (; index < c8_block; index += C8NUM) {
    MS_FLOAT32X8 relu6_data = MS_LD256_F32(data + index);
    MS_FLOAT32X8 zero_data = MS_MOV256_F32(0.0f);
    MS_FLOAT32X8 six_data = MS_MOV256_F32(6.0f);
    relu6_data = MS_MAX256_F32(relu6_data, zero_data);
    relu6_data = MS_MIN256_F32(relu6_data, six_data);
    MS_ST256_F32(dst + index, relu6_data);
  }
#endif

#if defined(ENABLE_NEON) || defined(ENABLE_SSE)
  int c4_block = DOWN_DIV(ele_num, C4NUM) * C4NUM;
  for (; index < c4_block; index += C4NUM) {
    MS_FLOAT32X4 relu6_data = MS_LDQ_F32(data + index);
    MS_FLOAT32X4 zero_data = MS_MOVQ_F32(0.0f);
    MS_FLOAT32X4 six_data = MS_MOVQ_F32(6.0f);
    relu6_data = MS_MAXQ_F32(relu6_data, zero_data);
    relu6_data = MS_MINQ_F32(relu6_data, six_data);
    MS_STQ_F32(dst + index, relu6_data);
  }
#endif
  for (; index < ele_num; ++index) {
    data[index] = data[index] < 0.0f ? 0.0f : data[index];
    data[index] = data[index] > 6.0f ? 6.0f : data[index];
  }
}
