/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "nnacl/fp32/matmul_avx512_mask_fp32.h"
#include "nnacl/op_base.h"
#include "nnacl/intrinsics/ms_simd_instructions.h"

void GemmRowxColMaskKernelFp32(float *dst, const float *src, const float *weight, const float *bias,
                               const size_t act_flag, const size_t row_block, const size_t col_block,
                               const size_t depth, const size_t src_stride, const size_t dst_stride,
                               const size_t inc_flag, const u_int16_t *mask) {
  __m512 dst_data[27];
  const float *src_sw[20];
  __m512 weight_data[6];
  __mmask16 mask16 = (__mmask16)(*mask);
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
        if (j == col_block - 1) {
          dst_data[i * col_block + j] =
            _mm512_mask3_fmadd_ps(_mm512_set1_ps(src_sw[i][k]), weight_data[j], dst_data[i * col_block + j], mask16);
        } else {
          dst_data[i * col_block + j] =
            _mm512_fmadd_ps(_mm512_set1_ps(src_sw[i][k]), weight_data[j], dst_data[i * col_block + j]);
        }
      }
    }
    weight_kernel += C16NUM * col_block;
  }  // k loop
  // add bias and relu
  for (int i = 0; i < row_block; ++i) {
    for (int j = 0; j < col_block; ++j) {
      if (j == col_block - 1) {
        if (inc_flag & 0x02) {
          if (0x1 & act_flag) {  // relu6
            dst_data[i * col_block + j] =
              _mm512_maskz_min_ps(mask16, dst_data[i * col_block + j], _mm512_set1_ps(6.0f));
          }
          if (0x2 & act_flag) {  // relu
            dst_data[i * col_block + j] =
              _mm512_maskz_max_ps(mask16, dst_data[i * col_block + j], _mm512_set1_ps(0.0f));
          }
        }
        _mm512_mask_storeu_ps(dst + i * dst_stride + j * C16NUM, mask16, dst_data[i * col_block + j]);
      } else {
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
}

void MatMulMaskAvx512Fp32(const float *a, const float *b, float *c, const float *bias, const int act_type,
                          const int depth, const int cur_col, const int col_, const int row) {
  int k_block = C1500NUM;
  int act_flag = 0;
  if (act_type == ActType_Relu6) {
    act_flag += 1;
  }
  if (act_type == ActType_Relu || act_type == ActType_Relu6) {
    act_flag += C2NUM;
  }
  GemmAvx512MaskKernel kernel[C4NUM][C13NUM];
  int max_shape[C4NUM] = {C12NUM, C12NUM, C8NUM, C6NUM};

#ifdef ENABLE_DEBUG
  for (int i = 0; i < C4NUM; i++) {
    for (int j = 0; j < C13NUM; j++) {
      kernel[i][j] = GemmRowxColMaskKernelFp32;
    }
  }
#else
  kernel[0][1] = nnacl_gemm_avx512_1x16_mask_kernel_nhwc_fp32;
  kernel[0][2] = nnacl_gemm_avx512_2x16_mask_kernel_nhwc_fp32;
  kernel[0][3] = nnacl_gemm_avx512_3x16_mask_kernel_nhwc_fp32;
  kernel[0][4] = nnacl_gemm_avx512_4x16_mask_kernel_nhwc_fp32;
  kernel[0][5] = nnacl_gemm_avx512_5x16_mask_kernel_nhwc_fp32;
  kernel[0][6] = nnacl_gemm_avx512_6x16_mask_kernel_nhwc_fp32;
  kernel[0][7] = nnacl_gemm_avx512_7x16_mask_kernel_nhwc_fp32;
  kernel[0][8] = nnacl_gemm_avx512_8x16_mask_kernel_nhwc_fp32;
  kernel[0][9] = nnacl_gemm_avx512_9x16_mask_kernel_nhwc_fp32;
  kernel[0][10] = nnacl_gemm_avx512_10x16_mask_kernel_nhwc_fp32;
  kernel[0][11] = nnacl_gemm_avx512_11x16_mask_kernel_nhwc_fp32;
  kernel[0][12] = nnacl_gemm_avx512_12x16_mask_kernel_nhwc_fp32;

  kernel[1][1] = nnacl_gemm_avx512_1x32_mask_kernel_nhwc_fp32;
  kernel[1][2] = nnacl_gemm_avx512_2x32_mask_kernel_nhwc_fp32;
  kernel[1][3] = nnacl_gemm_avx512_3x32_mask_kernel_nhwc_fp32;
  kernel[1][4] = nnacl_gemm_avx512_4x32_mask_kernel_nhwc_fp32;
  kernel[1][5] = nnacl_gemm_avx512_5x32_mask_kernel_nhwc_fp32;
  kernel[1][6] = nnacl_gemm_avx512_6x32_mask_kernel_nhwc_fp32;
  kernel[1][7] = nnacl_gemm_avx512_7x32_mask_kernel_nhwc_fp32;
  kernel[1][8] = nnacl_gemm_avx512_8x32_mask_kernel_nhwc_fp32;
  kernel[1][9] = nnacl_gemm_avx512_9x32_mask_kernel_nhwc_fp32;
  kernel[1][10] = nnacl_gemm_avx512_10x32_mask_kernel_nhwc_fp32;
  kernel[1][11] = nnacl_gemm_avx512_11x32_mask_kernel_nhwc_fp32;
  kernel[1][12] = nnacl_gemm_avx512_12x32_mask_kernel_nhwc_fp32;

  kernel[2][1] = nnacl_gemm_avx512_1x48_mask_kernel_nhwc_fp32;
  kernel[2][2] = nnacl_gemm_avx512_2x48_mask_kernel_nhwc_fp32;
  kernel[2][3] = nnacl_gemm_avx512_3x48_mask_kernel_nhwc_fp32;
  kernel[2][4] = nnacl_gemm_avx512_4x48_mask_kernel_nhwc_fp32;
  kernel[2][5] = nnacl_gemm_avx512_5x48_mask_kernel_nhwc_fp32;
  kernel[2][6] = nnacl_gemm_avx512_6x48_mask_kernel_nhwc_fp32;
  kernel[2][7] = nnacl_gemm_avx512_7x48_mask_kernel_nhwc_fp32;
  kernel[2][8] = nnacl_gemm_avx512_8x48_mask_kernel_nhwc_fp32;

  kernel[3][1] = nnacl_gemm_avx512_1x64_mask_kernel_nhwc_fp32;
  kernel[3][2] = nnacl_gemm_avx512_2x64_mask_kernel_nhwc_fp32;
  kernel[3][3] = nnacl_gemm_avx512_3x64_mask_kernel_nhwc_fp32;
  kernel[3][4] = nnacl_gemm_avx512_4x64_mask_kernel_nhwc_fp32;
  kernel[3][5] = nnacl_gemm_avx512_5x64_mask_kernel_nhwc_fp32;
  kernel[3][6] = nnacl_gemm_avx512_6x64_mask_kernel_nhwc_fp32;
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
    u_int16_t avx512_mask = 0xFFFF;
    for (int col_index = 0; col_index < cur_col; col_index += col_block) {
      int less_tmp = cur_col - col_index;
      if (less_tmp < col_block) {
        col_block = UP_ROUND(less_tmp, C16NUM);
        avx512_mask = (0xFFFF >> (col_block - less_tmp));
      }
      int col_block_num = col_block >> C4NUM;
      int row_block = max_shape[col_block_num - 1];
      for (int m = 0; m < row; m += row_block) {
        row_block = MSMIN(row_block, row - m);
        kernel[col_block_num - 1][row_block](c + col_index + m * col_, a + m * depth + k,
                                             b + col_index * depth + k * col_block, bias_data, act_flag, row_block,
                                             col_block_num, k_block, depth, col_, inc_flag, &avx512_mask);
      }
      if (bias_data != NULL) {
        bias_data += col_block;
      }
    }
  }
}

void MatVecMulMaskAvx512Fp32(const float *a, const float *b, float *c, const float *bias, const int act_type,
                             const int depth, const int cur_col, const int col_) {
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
  GemmAvx512MaskKernel kernel[C4NUM] = {GemmRowxColMaskKernelFp32, GemmRowxColMaskKernelFp32, GemmRowxColMaskKernelFp32,
                                        GemmRowxColMaskKernelFp32};
#else
  GemmAvx512MaskKernel kernel[C4NUM] = {
    nnacl_gemm_avx512_1x16_mask_kernel_nhwc_fp32, nnacl_gemm_avx512_1x32_mask_kernel_nhwc_fp32,
    nnacl_gemm_avx512_1x48_mask_kernel_nhwc_fp32, nnacl_gemm_avx512_1x64_mask_kernel_nhwc_fp32};
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
    u_int16_t avx512_mask = 0xFFFF;
    for (int col_index = 0; col_index < cur_col; col_index += col_block) {
      int less_tmp = cur_col - col_index;
      if (less_tmp < col_block) {
        col_block = UP_ROUND(less_tmp, C16NUM);
        avx512_mask = (0xFFFF >> (col_block - less_tmp));
      }
      int col_block_num = col_block >> C4NUM;

      kernel[col_block_num - 1](c + col_index, a + k, b + col_index * depth + k * col_block, bias_data, act_flag, 1,
                                col_block_num, k_block, depth, col_, inc_flag, &avx512_mask);
      if (bias_data != NULL) {
        bias_data += col_block;
      }
    }
  }
}
