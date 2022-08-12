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
#include "nnacl/fp32/matmul_avx512_fp32.h"
#include "nnacl/op_base.h"
#include "nnacl/intrinsics/ms_simd_instructions.h"

void GemmRowxColKernelFp32(float *dst, const float *src, const float *weight, const float *bias, const size_t act_flag,
                           const size_t row_block, const size_t col_block, const size_t depth, const size_t src_stride,
                           const size_t dst_stride, const size_t inc_flag) {
  __m512 dst_data[27];
  const float *src_sw[20];
  __m512 weight_data[6];
  for (int i = 0; i < C6NUM; ++i) {
    weight_data[i] = _mm512_set1_ps(0.0f);
  }
  for (int i = 0; i < row_block; ++i) {
    if (inc_flag & 0x01) {
      for (int j = 0; j < col_block; ++j) {
        dst_data[i * col_block + j] = _mm512_loadu_ps(dst + i * dst_stride + j * C16NUM);
      }
    } else if (bias != NULL) {
      for (int j = 0; j < col_block; ++j) {
        dst_data[i * col_block + j] = _mm512_loadu_ps(bias + j * C16NUM);
      }
    } else {
      for (int j = 0; j < col_block; ++j) {
        dst_data[i * col_block + j] = _mm512_set1_ps(0.0f);
      }
    }
    src_sw[i] = src + i * src_stride;
  }
  const float *weight_kernel = weight;
  for (int k = 0; k < depth; ++k) {
    for (int j = 0; j < col_block; ++j) {
      weight_data[j] = _mm512_loadu_ps(weight_kernel + j * C16NUM);
    }
    for (int i = 0; i < row_block; ++i) {
      for (int j = 0; j < col_block; ++j) {
        dst_data[i * col_block + j] =
          _mm512_fmadd_ps(_mm512_set1_ps(src_sw[i][k]), weight_data[j], dst_data[i * col_block + j]);
      }
    }
    weight_kernel += C16NUM * col_block;
  }  // k loop
  // add bias and relu
  for (int i = 0; i < row_block; ++i) {
    for (int j = 0; j < col_block; ++j) {
      if (inc_flag & 0x02) {
        if (0x1 & act_flag) {  // relu6
          dst_data[i * col_block + j] = _mm512_min_ps(dst_data[i * col_block + j], _mm512_set1_ps(6.0f));
        }
        if (0x2 & act_flag) {  // relu
          dst_data[i * col_block + j] = _mm512_max_ps(dst_data[i * col_block + j], _mm512_set1_ps(0.0f));
        }
      }
      _mm512_storeu_ps(dst + i * dst_stride + j * C16NUM, dst_data[i * col_block + j]);
    }
  }
}

void MatMulAvx512Fp32(const float *a, const float *b, float *c, const float *bias, const int act_type, const int depth,
                      const int cur_col, const int col_align, const int row) {
  int k_block = C1500NUM;
  int act_flag = 0;
  if (act_type == ActType_Relu6) {
    act_flag += 1;
  }
  if (act_type == ActType_Relu || act_type == ActType_Relu6) {
    act_flag += C2NUM;
  }
  GemmAvx512Kernel kernel[C4NUM][C13NUM];
  int max_shape[C4NUM] = {C12NUM, C12NUM, C8NUM, C6NUM};

#ifdef ENABLE_DEBUG
  for (int i = 0; i < C4NUM; i++) {
    for (int j = 0; j < C13NUM; j++) {
      kernel[i][j] = GemmRowxColKernelFp32;
    }
  }
#else
  kernel[0][1] = nnacl_gemm_avx512_1x16_kernel_nhwc_fp32;
  kernel[0][2] = nnacl_gemm_avx512_2x16_kernel_nhwc_fp32;
  kernel[0][3] = nnacl_gemm_avx512_3x16_kernel_nhwc_fp32;
  kernel[0][4] = nnacl_gemm_avx512_4x16_kernel_nhwc_fp32;
  kernel[0][5] = nnacl_gemm_avx512_5x16_kernel_nhwc_fp32;
  kernel[0][6] = nnacl_gemm_avx512_6x16_kernel_nhwc_fp32;
  kernel[0][7] = nnacl_gemm_avx512_7x16_kernel_nhwc_fp32;
  kernel[0][8] = nnacl_gemm_avx512_8x16_kernel_nhwc_fp32;
  kernel[0][9] = nnacl_gemm_avx512_9x16_kernel_nhwc_fp32;
  kernel[0][10] = nnacl_gemm_avx512_10x16_kernel_nhwc_fp32;
  kernel[0][11] = nnacl_gemm_avx512_11x16_kernel_nhwc_fp32;
  kernel[0][12] = nnacl_gemm_avx512_12x16_kernel_nhwc_fp32;

  kernel[1][1] = nnacl_gemm_avx512_1x32_kernel_nhwc_fp32;
  kernel[1][2] = nnacl_gemm_avx512_2x32_kernel_nhwc_fp32;
  kernel[1][3] = nnacl_gemm_avx512_3x32_kernel_nhwc_fp32;
  kernel[1][4] = nnacl_gemm_avx512_4x32_kernel_nhwc_fp32;
  kernel[1][5] = nnacl_gemm_avx512_5x32_kernel_nhwc_fp32;
  kernel[1][6] = nnacl_gemm_avx512_6x32_kernel_nhwc_fp32;
  kernel[1][7] = nnacl_gemm_avx512_7x32_kernel_nhwc_fp32;
  kernel[1][8] = nnacl_gemm_avx512_8x32_kernel_nhwc_fp32;
  kernel[1][9] = nnacl_gemm_avx512_9x32_kernel_nhwc_fp32;
  kernel[1][10] = nnacl_gemm_avx512_10x32_kernel_nhwc_fp32;
  kernel[1][11] = nnacl_gemm_avx512_11x32_kernel_nhwc_fp32;
  kernel[1][12] = nnacl_gemm_avx512_12x32_kernel_nhwc_fp32;

  kernel[2][1] = nnacl_gemm_avx512_1x48_kernel_nhwc_fp32;
  kernel[2][2] = nnacl_gemm_avx512_2x48_kernel_nhwc_fp32;
  kernel[2][3] = nnacl_gemm_avx512_3x48_kernel_nhwc_fp32;
  kernel[2][4] = nnacl_gemm_avx512_4x48_kernel_nhwc_fp32;
  kernel[2][5] = nnacl_gemm_avx512_5x48_kernel_nhwc_fp32;
  kernel[2][6] = nnacl_gemm_avx512_6x48_kernel_nhwc_fp32;
  kernel[2][7] = nnacl_gemm_avx512_7x48_kernel_nhwc_fp32;
  kernel[2][8] = nnacl_gemm_avx512_8x48_kernel_nhwc_fp32;

  kernel[3][1] = nnacl_gemm_avx512_1x64_kernel_nhwc_fp32;
  kernel[3][2] = nnacl_gemm_avx512_2x64_kernel_nhwc_fp32;
  kernel[3][3] = nnacl_gemm_avx512_3x64_kernel_nhwc_fp32;
  kernel[3][4] = nnacl_gemm_avx512_4x64_kernel_nhwc_fp32;
  kernel[3][5] = nnacl_gemm_avx512_5x64_kernel_nhwc_fp32;
  kernel[3][6] = nnacl_gemm_avx512_6x64_kernel_nhwc_fp32;
#endif

  int inc_flag;
  for (int k = 0; k < depth; k += k_block) {
    if (depth - k <= k_block) {
      k_block = depth - k;
      inc_flag = C3NUM - (k == 0);
    } else {
      inc_flag = 1 - (k == 0);
    }
    const float *bias_data = bias;
    // one time process 64 out_channel
    int col_block = C64NUM;
    for (int col_index = 0; col_index < cur_col; col_index += col_block) {
      col_block = MSMIN(col_block, cur_col - col_index);
      int row_block = max_shape[(col_block >> C4NUM) - 1];
      for (int m = 0; m < row; m += row_block) {
        row_block = MSMIN(row_block, row - m);
        kernel[(col_block >> C4NUM) - 1][row_block](c + col_index + m * col_align, a + m * depth + k,
                                                    b + col_index * depth + k * col_block, bias_data, act_flag,
                                                    row_block, col_block >> C4NUM, k_block, depth, col_align, inc_flag);
      }
      if (bias_data != NULL) {
        bias_data += col_block;
      }
    }
  }
}

void MatVecMulAvx512Fp32(const float *a, const float *b, float *c, const float *bias, const int act_type,
                         const int depth, const int cur_col, const int col_align) {
  // one time process 64 out_channel
  int k_block = C1500NUM;
  int act_flag = 0;
  if (act_type == ActType_Relu6) {
    act_flag += 1;
  }
  if (act_type == ActType_Relu || act_type == ActType_Relu6) {
    act_flag += C2NUM;
  }
#ifdef ENABLE_DEBUG
  GemmAvx512Kernel kernel[C4NUM] = {GemmRowxColKernelFp32, GemmRowxColKernelFp32, GemmRowxColKernelFp32,
                                    GemmRowxColKernelFp32};
#else
  GemmAvx512Kernel kernel[C4NUM] = {nnacl_gemm_avx512_1x16_kernel_nhwc_fp32, nnacl_gemm_avx512_1x32_kernel_nhwc_fp32,
                                    nnacl_gemm_avx512_1x48_kernel_nhwc_fp32, nnacl_gemm_avx512_1x64_kernel_nhwc_fp32};
#endif
  int inc_flag;
  for (int k = 0; k < depth; k += k_block) {
    if (depth - k <= k_block) {
      k_block = depth - k;
      inc_flag = C3NUM - (k == 0);
    } else {
      inc_flag = 1 - (k == 0);
    }
    const float *bias_data = bias;
    int col_block = C64NUM;
    for (int col_index = 0; col_index < cur_col; col_index += col_block) {
      col_block = MSMIN(col_block, cur_col - col_index);
      kernel[(col_block >> C4NUM) - 1](c + col_index, a + k, b + col_index * depth + k * col_block, bias_data, act_flag,
                                       1, col_block >> C4NUM, k_block, depth, col_align, inc_flag);
      if (bias_data != NULL) {
        bias_data += col_block;
      }
    }
  }
}

// act_type must be 0, 1, 2. 0: no_act, 1: relu, 3: relu6.
int64_t GemmIsNotPackOptimizeAVX512(int64_t m_index, const float *a, const float *b, float *c, const float *bias, int m,
                                    int k, int act_type) {
  // gemm dot is [m, k] * [k, 1] ==>> [m, 1]
  // block 8
  MS_FLOAT32X8 down_threshold256 = _mm256_setzero_ps();
  MS_FLOAT32X8 up_threshold256 = _mm256_set1_ps(C6NUM);
  for (; m_index <= m - C8NUM; m_index += C8NUM) {
    int k_index = 0;
    MS_FLOAT32X8 dst = MS_MOV256_F32(bias[0]);
    MS_SET_ZERO512X8_F32(dst16_)
    for (; k_index <= k - C16NUM; k_index += C16NUM) {
      __m512 weight = _mm512_loadu_ps(b + k_index);
      MS_LOAD512X8_F32(src, a + m_index * k + k_index, k)
      MS_FMADD512X8_F32(src, weight, dst16_)
    }
    MS_F32X8_GETI(dst, 0) += MS_REDUCE_ADD512_F32(dst16_1);
    MS_F32X8_GETI(dst, 1) += MS_REDUCE_ADD512_F32(dst16_2);
    MS_F32X8_GETI(dst, C2NUM) += MS_REDUCE_ADD512_F32(dst16_3);
    MS_F32X8_GETI(dst, C3NUM) += MS_REDUCE_ADD512_F32(dst16_4);
    MS_F32X8_GETI(dst, C4NUM) += MS_REDUCE_ADD512_F32(dst16_5);
    MS_F32X8_GETI(dst, C5NUM) += MS_REDUCE_ADD512_F32(dst16_6);
    MS_F32X8_GETI(dst, C6NUM) += MS_REDUCE_ADD512_F32(dst16_7);
    MS_F32X8_GETI(dst, C7NUM) += MS_REDUCE_ADD512_F32(dst16_8);
    for (; k_index < k; k_index++) {
      MS_F32X8_GETI(dst, 0) += b[k_index] * a[m_index * k + k_index];
      MS_F32X8_GETI(dst, 1) += b[k_index] * a[m_index * k + k_index + k];
      MS_F32X8_GETI(dst, C2NUM) += b[k_index] * a[m_index * k + k_index + C2NUM * k];
      MS_F32X8_GETI(dst, C3NUM) += b[k_index] * a[m_index * k + k_index + C3NUM * k];
      MS_F32X8_GETI(dst, C4NUM) += b[k_index] * a[m_index * k + k_index + C4NUM * k];
      MS_F32X8_GETI(dst, C5NUM) += b[k_index] * a[m_index * k + k_index + C5NUM * k];
      MS_F32X8_GETI(dst, C6NUM) += b[k_index] * a[m_index * k + k_index + C6NUM * k];
      MS_F32X8_GETI(dst, C7NUM) += b[k_index] * a[m_index * k + k_index + C7NUM * k];
    }

    if (act_type != 0) {
      dst = MS_MAX256_F32(dst, down_threshold256);
      if (act_type == 3) {  // 3: relu6
        dst = MS_MIN256_F32(dst, up_threshold256);
      }
    }

    MS_ST256_F32(c + m_index, dst);
  }
  return m_index;
}
