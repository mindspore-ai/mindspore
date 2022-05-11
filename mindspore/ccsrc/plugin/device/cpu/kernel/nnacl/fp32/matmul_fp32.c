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

#include "nnacl/fp32/matmul_fp32.h"
#include "nnacl/fp32/pack_fp32.h"
#include "nnacl/intrinsics/ms_simd_instructions.h"

#ifndef ENABLE_ARM
void MatVecMulFp32(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int col) {
  for (int ci = 0; ci < col; ci++) {
    float value = 0;
    for (int di = 0; di < depth; di++) {
      value += a[di] * b[ci * depth + di];
    }
    if (bias != NULL) value += bias[ci];
    if (act_type == ActType_Relu6) value = MSMIN(6.0f, value);
    if (act_type == ActType_Relu || act_type == ActType_Relu6) value = MSMAX(0.0f, value);
    c[ci] = value;
  }
}

void MatVecMulFp32Block8(const float *a, const float *b, float *c, const float *bias, int act_type, int depth,
                         int col) {
  int col8 = col / C8NUM * C8NUM;
  int ci = 0;
  for (; ci < col8; ci += C8NUM, c += C8NUM) {
#ifdef ENABLE_NEON
    float32x4_t value0 = vdupq_n_f32(0.0f);
    float32x4_t value1 = vdupq_n_f32(0.0f);
    for (int di = 0; di < depth; ++di, b += C8NUM) {
      value0 += vdupq_n_f32(a[di]) * vld1q_f32(b);
      value1 += vdupq_n_f32(a[di]) * vld1q_f32(b + 4);
    }
    if (bias != NULL) {
      value0 += vld1q_f32(bias[ci]);
      value1 += vld1q_f32(bias[ci + 4]);
    }
    if (act_type == ActType_Relu || act_type == ActType_Relu6) {
      value0 = vmaxq_f32(value0, 0.0f);
      value1 = vmaxq_f32(value1, 0.0f);
    }
    if (act_type == ActType_Relu6) {
      value0 = vminq_f32(value0, 6.0f);
      value1 = vminq_f32(value1, 6.0f);
    }
    vst1q_f32(c, value0);
    vst1q_f32(c + 4, value1);
#else
    float value[C8NUM] = {0};
    for (int di = 0; di < depth; ++di, b += C8NUM) {
      for (int j = 0; j < C8NUM; ++j) {
        value[j] += a[di] * b[j];
      }
    }
    for (int j = 0; j < C8NUM; ++j) {
      ADD_BIAS(value[j], bias, ci + j);
      DO_RELU(value[j], act_type);
      DO_RELU6(value[j], act_type);
    }
    memcpy(c, value, C8NUM * sizeof(float));
#endif
  }
  int res = col - col8;
  float value[C8NUM] = {0};
  for (int di = 0; di < depth; ++di, b += C8NUM) {
    for (int j = 0; j < res; ++j) {
      value[j] += a[di] * b[j];
    }
  }
  for (int j = 0; j < res; ++j) {
    ADD_BIAS(value[j], bias, ci + j);
    DO_RELU(value[j], act_type);
    DO_RELU6(value[j], act_type);
  }
  memcpy(c, value, res * sizeof(float));
}
#endif

#ifdef ENABLE_ARM32
void MatVecMulFp32Block4(const float *a, const float *b, float *c, const float *bias, int act_type, int depth,
                         int col) {
  int col4 = col / C4NUM * C4NUM;
  int ci = 0;
  for (; ci < col4; ci += C4NUM, c += C4NUM) {
#ifdef ENABLE_NEON
    float32x4_t value = vdupq_n_f32(0.0f);
    for (int di = 0; di < depth; ++di, b += C4NUM) {
      value += vdupq_n_f32(a[di]) * vld1q_f32(b);
    }
    if (bias != NULL) {
      value += vld1q_f32(&(bias[ci]));
    }
    if (act_type == ActType_Relu || act_type == ActType_Relu6) {
      value = vmaxq_f32(value, vdupq_n_f32(0.0f));
    }
    if (act_type == ActType_Relu6) {
      value = vminq_f32(value, vdupq_n_f32(6.0f));
    }
    vst1q_f32(c, value);
#else
    float value[C4NUM] = {0};
    for (int di = 0; di < depth; ++di, b += C4NUM) {
      for (int j = 0; j < C4NUM; ++j) {
        value[j] += a[di] * b[j];
      }
    }
    for (int j = 0; j < C4NUM; ++j) {
      ADD_BIAS(value[j], bias, ci + j);
      DO_RELU(value[j], act_type);
      DO_RELU6(value[j], act_type);
    }
    memcpy(c, value, C4NUM * sizeof(float));
#endif
  }
  int res = col - col4;
  float value[C4NUM] = {0};
  for (int di = 0; di < depth; ++di, b += C4NUM) {
    for (int j = 0; j < res; ++j) {
      value[j] += a[di] * b[j];
    }
  }
  for (int j = 0; j < res; ++j) {
    ADD_BIAS(value[j], bias, ci + j);
    DO_RELU(value[j], act_type);
    DO_RELU6(value[j], act_type);
  }
  memcpy(c, value, res * sizeof(float));
}
#endif

#ifdef ENABLE_ARM64
// 4x8
void MatVecMulFp32Neon64(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int col,
                         int align_col) {
  int ci = 0;
  for (; ci < align_col - C8NUM + 1; ci += C8NUM) {
    float32x4_t acc_0;
    float32x4_t acc_1;
    if (bias != NULL) {
      acc_0 = vld1q_f32(bias + ci);
      acc_1 = vld1q_f32(bias + ci + C4NUM);
    } else {
      acc_0 = vdupq_n_f32(0.0f);
      acc_1 = vdupq_n_f32(0.0f);
    }
    const float *bv_base = b + ci * depth;
    int di = 0;
    for (; di < depth - C4NUM + 1; di += C4NUM) {
      float32x4_t av = vld1q_f32(a + di);
      float32x4_t bv_00 = vld1q_f32(bv_base);
      float32x4_t bv_10 = vld1q_f32(bv_base + C4NUM);
      bv_base += C8NUM;
      float32x4_t bv_01 = vld1q_f32(bv_base);
      float32x4_t bv_11 = vld1q_f32(bv_base + C4NUM);
      bv_base += C8NUM;
      float32x4_t bv_02 = vld1q_f32(bv_base);
      float32x4_t bv_12 = vld1q_f32(bv_base + C4NUM);
      bv_base += C8NUM;
      float32x4_t bv_03 = vld1q_f32(bv_base);
      float32x4_t bv_13 = vld1q_f32(bv_base + C4NUM);
      bv_base += C8NUM;
      acc_0 = vmlaq_n_f32(acc_0, bv_00, av[0]);
      acc_1 = vmlaq_n_f32(acc_1, bv_10, av[0]);
      acc_0 = vmlaq_n_f32(acc_0, bv_01, av[1]);
      acc_1 = vmlaq_n_f32(acc_1, bv_11, av[1]);
      acc_0 = vmlaq_n_f32(acc_0, bv_02, av[2]);
      acc_1 = vmlaq_n_f32(acc_1, bv_12, av[2]);
      acc_0 = vmlaq_n_f32(acc_0, bv_03, av[3]);
      acc_1 = vmlaq_n_f32(acc_1, bv_13, av[3]);
    }
    if (di < depth) {
      for (; di < depth; ++di) {
        float ai = a[di];
        float32x4_t bv0 = vld1q_f32(bv_base);
        float32x4_t bv1 = vld1q_f32(bv_base + C4NUM);
        acc_0 = vmlaq_n_f32(acc_0, bv0, ai);
        acc_1 = vmlaq_n_f32(acc_1, bv1, ai);
        bv_base += C8NUM;
      }
    }  // only save actual col num data
    if (ci + C4NUM - 1 >= col) {
      int c_remain = col - ci;
      for (int i = 0; i < c_remain; ++i) {
        if (act_type == ActType_Relu) {
          c[i] = MSMAX(acc_0[i], 0.0f);
        } else if (act_type == ActType_Relu6) {
          c[i] = MSMIN(MSMAX(acc_0[i], 0.0f), 6.0f);
        } else {
          c[i] = acc_0[i];
        }
      }
      return;
    }
    if (act_type == ActType_Relu) {
      acc_0 = vmaxq_f32(acc_0, vdupq_n_f32(0.0f));
    } else if (act_type == ActType_Relu6) {
      acc_0 = vminq_f32(vmaxq_f32(acc_0, vdupq_n_f32(0.0f)), vdupq_n_f32(6.0f));
    }
    vst1q_f32(c, acc_0);
    if (ci + C8NUM - 1 >= col) {
      int c_remain = col - ci - C4NUM;
      for (int i = 0; i < c_remain; ++i) {
        if (act_type == ActType_Relu) {
          c[C4NUM + i] = MSMAX(acc_1[i], 0.0f);
        } else if (act_type == ActType_Relu6) {
          c[C4NUM + i] = MSMIN(MSMAX(acc_1[i], 0.0f), 6.0f);
        } else {
          c[C4NUM + i] = acc_1[i];
        }
      }
      return;
    }
    if (act_type == ActType_Relu) {
      acc_1 = vmaxq_f32(acc_1, vdupq_n_f32(0.0f));
    } else if (act_type == ActType_Relu6) {
      acc_1 = vminq_f32(vmaxq_f32(acc_1, vdupq_n_f32(0.0f)), vdupq_n_f32(6.0f));
    }
    vst1q_f32(c + C4NUM, acc_1);
    c += C8NUM;
  }
}
#endif

void MatMul12x8(const float *a, const float *b, float *dst, const float *bias, ActType act_type, int deep, int row,
                int col, int stride, int out_type) {
  if (out_type == OutType_Nhwc) {
    for (int r = 0; r < row; r++) {
      for (int c = 0; c < col; c++) {
        int r12div = r / 12, r12mod = r % 12;
        int c8div = c / 8, c8mod = c % 8;
        size_t ci = r * stride + c;
        float value = 0;
        for (int d = 0; d < deep; d++) {
          size_t ai = r12div * deep * 12 + d * 12 + r12mod;
          size_t bi = c8div * deep * 8 + d * 8 + c8mod;
          value = value + a[ai] * b[bi];
        }
        ADD_BIAS(value, bias, c)
        DO_RELU(value, act_type)
        DO_RELU6(value, act_type)
        dst[ci] = value;
      }
    }
  } else if (out_type == OutType_C8) {
    int col_8 = UP_ROUND(col, C8NUM);
    int row_12 = UP_ROUND(row, C12NUM);
    for (int r = 0; r < row_12; r++) {
      for (int c = 0; c < col_8; c++) {
        int r12div = r / C12NUM, r12mod = r % C12NUM;
        int c8div = c / C8NUM, c8mod = c % C8NUM;
        size_t ci = (c8div * C8NUM * row_12 + r * C8NUM + c8mod);
        float value = 0;
        for (int d = 0; d < deep; d++) {
          size_t ai = r12div * deep * C12NUM + d * C12NUM + r12mod;
          size_t bi = c8div * deep * C8NUM + d * C8NUM + c8mod;
          value = value + a[ai] * b[bi];
        }
        ADD_BIAS(value, bias, c)
        DO_RELU(value, act_type)
        DO_RELU6(value, act_type)
        dst[ci] = value;
      }
    }
  } else if (out_type == OutType_TileC8) {
    for (int i = 0; i < row; ++i) {
      int src_r_offset = i;
      int dst_r_offset = i * col * stride;
      for (int j = 0; j < col; ++j) {
        int c8div = j / 8, c8mod = j % 8;
        size_t ci = dst_r_offset + c8div * 8 * stride + c8mod;
        float value = 0;
        for (int d = 0; d < deep; ++d) {
          size_t ai = src_r_offset + d * C12NUM;
          size_t bi = c8div * deep * 8 + d * 8 + c8mod;
          value = value + a[ai] * b[bi];
        }
        ADD_BIAS(value, bias, j)
        DO_RELU(value, act_type)
        DO_RELU6(value, act_type)
        dst[ci] = value;
      }
    }
  }
}

void MatMulOpt(const float *a, const float *b, float *c, const float *bias, ActType act_type, int deep, int row,
               int col, size_t stride, int out_type) {
#ifdef ENABLE_ARM64
  if (out_type == OutType_C8) {
    MatmulFloatNeon64(a, b, c, bias, (int)act_type, deep, row, col, stride, 0, 0);
  } else {
    MatmulFloatNeon64Opt(a, b, c, bias, (int)act_type, deep, row, col, stride, (int)(out_type));
  }
#elif ENABLE_ARM32
  if (out_type == OutType_C8) {
    MatmulFloatNeon32(a, b, c, bias, (int)act_type, deep, row, col, stride, 0, 0);
  } else if (out_type == OutType_Nhwc) {
    MatmulFloatNeon32Opt12x4(a, b, c, bias, (int)act_type, deep, row, col, stride, 1);
  } else {
    MatmulFloatNeon32Opt(a, b, c, bias, (int)act_type, deep, row, col, stride, (int)(out_type));
  }
#elif ENABLE_AVX
  MatmulFloatAvxOpt(a, b, c, bias, (size_t)act_type, deep, row, col, stride, (size_t)(out_type));
#elif ENABLE_SSE
  MatmulFloatSse64Opt(a, b, c, bias, (int)act_type, deep, row, col, stride, (int)(out_type));
#else
  MatMul12x8(a, b, c, bias, act_type, deep, row, col, stride, out_type);
#endif
}

#ifdef ENABLE_AVX
void MatVecMulAvxFp32(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int cur_col,
                      int col_align) {
  // one time process 32 out_channel
  int col_block = C32NUM;
  int act_flag = 0;
  if (act_type == ActType_Relu6) {
    act_flag += 1;
  }
  if (act_type == ActType_Relu || act_type == ActType_Relu6) {
    act_flag += 2;
  }
  MatVecMulKernel kernel[4] = {MatVecMul1x8Kernel, MatVecMul1x16Kernel, MatVecMul1x24Kernel, MatVecMul1x32Kernel};
  const float *bias_data = bias;
  for (int col_index = 0; col_index < cur_col; col_index += col_block) {
    col_block = cur_col - col_index < col_block ? cur_col - col_index : col_block;
    kernel[(col_block >> 3) - 1](c + col_index, a, b + col_index * depth, bias_data, act_flag, 1, col_block >> 3,
                                 col_align, depth);
    if (bias_data != NULL) {
      bias_data += col_block;
    }
  }
}

void MatMulAvxFp32(const float *a, const float *b, float *c, const float *bias, const int act_type, const int depth,
                   const int cur_col, const int col_align, const int row) {
  // one time process 32 out_channel
  int col_block = C32NUM;
  int act_flag = 0;
  if (act_type == ActType_Relu6) {
    act_flag += 1;
  }
  if (act_type == ActType_Relu || act_type == ActType_Relu6) {
    act_flag += C2NUM;
  }
  int row_tile[4] = {C8NUM, C6NUM, C4NUM, C3NUM};
  MatVecMulKernel kernel[4][2] = {{MatVecMul1x8Kernel, MatMul8x8Kernel},
                                  {MatVecMul1x16Kernel, MatMul6x16Kernel},
                                  {MatVecMul1x24Kernel, MatMul4x24Kernel},
                                  {MatVecMul1x32Kernel, MatMul3x32Kernel}};
  const float *bias_data = bias;
  for (int col_index = 0; col_index < cur_col; col_index += col_block) {
    col_block = cur_col - col_index < col_block ? cur_col - col_index : col_block;
    int row_block = row_tile[(col_block >> C3NUM) - 1];
    for (int r = 0; r < row; r += row_block) {
      if (row_block > row - r) {
        row_block = 1;
      }
      kernel[(col_block >> C3NUM) - 1][row_block / row_tile[(col_block >> C3NUM) - 1]](
        c + col_index + r * col_align, a + r * depth, b + col_index * depth, bias_data, act_flag, row_block,
        col_block >> C3NUM, col_align, depth);
    }
    if (bias_data != NULL) {
      bias_data += col_block;
    }
  }
}

void MatMul3x32Kernel(float *dst, const float *src, const float *weight, const float *bias, const size_t act_flag,
                      const size_t row_block, const size_t col_block, size_t col_algin, const size_t deep) {
  col_algin *= sizeof(float);
  asm volatile(
    "cmpq $0, %2\n"
    "je 0f\n"
    "vmovups (%2), %%ymm0\n"
    "vmovups 0x20(%2), %%ymm1\n"
    "vmovups 0x40(%2), %%ymm2\n"
    "vmovups 0x60(%2), %%ymm3\n"
    "vmovups (%2), %%ymm4\n"
    "vmovups 0x20(%2), %%ymm5\n"
    "vmovups 0x40(%2), %%ymm6\n"
    "vmovups 0x60(%2), %%ymm7\n"
    "vmovups (%2), %%ymm8\n"
    "vmovups 0x20(%2), %%ymm9\n"
    "vmovups 0x40(%2), %%ymm10\n"
    "vmovups 0x60(%2), %%ymm11\n"
    "jmp 1f\n"
    "0:\n"
    "vxorps %%ymm0, %%ymm0, %%ymm0\n"
    "vxorps %%ymm1, %%ymm1, %%ymm1\n"
    "vxorps %%ymm2, %%ymm2, %%ymm2\n"
    "vxorps %%ymm3, %%ymm3, %%ymm3\n"
    "vxorps %%ymm4, %%ymm4, %%ymm4\n"
    "vxorps %%ymm5, %%ymm5, %%ymm5\n"
    "vxorps %%ymm6, %%ymm6, %%ymm6\n"
    "vxorps %%ymm7, %%ymm7, %%ymm7\n"
    "vxorps %%ymm8, %%ymm8, %%ymm8\n"
    "vxorps %%ymm9, %%ymm9, %%ymm9\n"
    "vxorps %%ymm10, %%ymm10, %%ymm10\n"
    "vxorps %%ymm11, %%ymm11, %%ymm11\n"

    "1:\n"                          // deep
    "vbroadcastss (%0), %%ymm12\n"  // src
    "vbroadcastss (%0, %7), %%ymm13\n"
    "vbroadcastss (%0, %7, 2), %%ymm14\n"
    "vmovups (%1), %%ymm15\n"  // weight
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm0\n"
    "vfmadd231ps %%ymm15, %%ymm13, %%ymm4\n"
    "vfmadd231ps %%ymm15, %%ymm14, %%ymm8\n"

    "vmovups 0x20(%1), %%ymm15\n"  // weight
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm1\n"
    "vfmadd231ps %%ymm15, %%ymm13, %%ymm5\n"
    "vfmadd231ps %%ymm15, %%ymm14, %%ymm9\n"

    "vmovups 0x40(%1), %%ymm15\n"  // weight
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm2\n"
    "vfmadd231ps %%ymm15, %%ymm13, %%ymm6\n"
    "vfmadd231ps %%ymm15, %%ymm14, %%ymm10\n"

    "vmovups 0x60(%1), %%ymm15\n"  // weight
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm3\n"
    "vfmadd231ps %%ymm15, %%ymm13, %%ymm7\n"
    "vfmadd231ps %%ymm15, %%ymm14, %%ymm11\n"
    "addq $128, %1\n"
    "addq $4, %0\n"
    "dec %3\n"
    "jg 1b\n"

    "and $0x3, %%eax\n"  // act_type
    "je 6f\n"
    // Relu
    "vxorps %%ymm12, %%ymm12, %%ymm12\n"
    "vmaxps %%ymm12, %%ymm0, %%ymm0\n"
    "vmaxps %%ymm12, %%ymm1, %%ymm1\n"
    "vmaxps %%ymm12, %%ymm2, %%ymm2\n"
    "vmaxps %%ymm12, %%ymm3, %%ymm3\n"
    "vmaxps %%ymm12, %%ymm4, %%ymm4\n"
    "vmaxps %%ymm12, %%ymm5, %%ymm5\n"
    "vmaxps %%ymm12, %%ymm6, %%ymm6\n"
    "vmaxps %%ymm12, %%ymm7, %%ymm7\n"
    "vmaxps %%ymm12, %%ymm8, %%ymm8\n"
    "vmaxps %%ymm12, %%ymm9, %%ymm9\n"
    "vmaxps %%ymm12, %%ymm10, %%ymm10\n"
    "vmaxps %%ymm12, %%ymm11, %%ymm11\n"
    "and $0x1, %%eax\n"
    "je 6f\n"
    // relu6
    "mov $0x40C00000, %%ecx\n"
    "vmovd %%ecx, %%xmm14\n"
    "vpermps %%ymm14, %%ymm12, %%ymm14\n"
    "vminps %%ymm14, %%ymm0, %%ymm0\n"
    "vminps %%ymm14, %%ymm1, %%ymm1\n"
    "vminps %%ymm14, %%ymm2, %%ymm2\n"
    "vminps %%ymm14, %%ymm3, %%ymm3\n"
    "vminps %%ymm14, %%ymm4, %%ymm4\n"
    "vminps %%ymm14, %%ymm5, %%ymm5\n"
    "vminps %%ymm14, %%ymm6, %%ymm6\n"
    "vminps %%ymm14, %%ymm7, %%ymm7\n"
    "vminps %%ymm14, %%ymm8, %%ymm8\n"
    "vminps %%ymm14, %%ymm9, %%ymm9\n"
    "vminps %%ymm14, %%ymm10, %%ymm10\n"
    "vminps %%ymm14, %%ymm11, %%ymm11\n"
    "6:\n"
    "vmovups %%ymm0, (%5)\n"  // dst_0
    "vmovups %%ymm1, 0x20(%5)\n"
    "vmovups %%ymm2, 0x40(%5)\n"
    "vmovups %%ymm3, 0x60(%5)\n"
    "vmovups %%ymm4, (%5, %6)\n"  // dst_1
    "vmovups %%ymm5, 0x20(%5, %6)\n"
    "vmovups %%ymm6, 0x40(%5, %6)\n"
    "vmovups %%ymm7, 0x60(%5, %6)\n"
    "vmovups %%ymm8, (%5, %6, 2)\n"  // dst_2
    "vmovups %%ymm9, 0x20(%5, %6, 2)\n"
    "vmovups %%ymm10, 0x40(%5, %6, 2)\n"
    "vmovups %%ymm11, 0x60(%5, %6, 2)\n"
    :
    : "r"(src), "r"(weight), "r"(bias), "r"(deep), "a"(act_flag), "r"(dst), "r"(col_algin),
      "r"(deep * sizeof(float))  // 7
    : "%rcx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10",
      "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15");
}

void MatVecMul1x32Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                         size_t row_block, size_t col_block, size_t col_algin, size_t deep) {
  asm volatile(
    "cmpq $0, %2\n"
    "je 0f\n"
    "vmovups (%2), %%ymm0\n"
    "vmovups 0x20(%2), %%ymm1\n"
    "vmovups 0x40(%2), %%ymm2\n"
    "vmovups 0x60(%2), %%ymm3\n"
    "jmp 1f\n"
    "0:\n"
    "vxorps %%ymm0, %%ymm0, %%ymm0\n"
    "vxorps %%ymm1, %%ymm1, %%ymm1\n"
    "vxorps %%ymm2, %%ymm2, %%ymm2\n"
    "vxorps %%ymm3, %%ymm3, %%ymm3\n"
    "1:\n"  // deep_c8
    "movq %3, %%rcx\n"
    "shr $3, %%ecx\n"
    "je 3f\n"
    "2:\n"
    "vbroadcastss (%0), %%ymm4\n"
    "vfmadd231ps (%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 0x20(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 0x40(%1), %%ymm4, %%ymm2\n"
    "vfmadd231ps 0x60(%1), %%ymm4, %%ymm3\n"

    "vbroadcastss 4(%0), %%ymm4\n"
    "vfmadd231ps 128(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 160(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 192(%1), %%ymm4, %%ymm2\n"
    "vfmadd231ps 224(%1), %%ymm4, %%ymm3\n"

    "vbroadcastss 8(%0), %%ymm4\n"
    "vfmadd231ps 256(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 288(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 320(%1), %%ymm4, %%ymm2\n"
    "vfmadd231ps 352(%1), %%ymm4, %%ymm3\n"

    "vbroadcastss 12(%0), %%ymm4\n"
    "vfmadd231ps 384(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 416(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 448(%1), %%ymm4, %%ymm2\n"
    "vfmadd231ps 480(%1), %%ymm4, %%ymm3\n"

    "vbroadcastss 16(%0), %%ymm4\n"
    "vfmadd231ps 512(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 544(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 576(%1), %%ymm4, %%ymm2\n"
    "vfmadd231ps 608(%1), %%ymm4, %%ymm3\n"

    "vbroadcastss 20(%0), %%ymm4\n"
    "vfmadd231ps 640(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 672(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 704(%1), %%ymm4, %%ymm2\n"
    "vfmadd231ps 736(%1), %%ymm4, %%ymm3\n"

    "vbroadcastss 24(%0), %%ymm4\n"
    "vfmadd231ps 768(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 800(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 832(%1), %%ymm4, %%ymm2\n"
    "vfmadd231ps 864(%1), %%ymm4, %%ymm3\n"

    "vbroadcastss 28(%0), %%ymm4\n"
    "vfmadd231ps 896(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 928(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 960(%1), %%ymm4, %%ymm2\n"
    "vfmadd231ps 992(%1), %%ymm4, %%ymm3\n"
    "addq $1024, %1\n"
    "addq $32, %0\n"
    "dec %%ecx\n"
    "jg 2b\n"

    "3:\n"
    "and $7, %3\n"  // deep_remainder
    "je 5f\n"
    "4:\n"
    "vbroadcastss (%0), %%ymm4\n"
    "vfmadd231ps (%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 0x20(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 0x40(%1), %%ymm4, %%ymm2\n"
    "vfmadd231ps 0x60(%1), %%ymm4, %%ymm3\n"
    "addq $128, %1\n"
    "addq $4, %0\n"
    "dec %3\n"
    "jg 4b\n"

    "5:\n"
    "and $0x3, %%eax\n"  // act_type
    "je 6f\n"
    // Relu
    "vxorps %%ymm12, %%ymm12, %%ymm12\n"
    "vmaxps %%ymm12, %%ymm0, %%ymm0\n"
    "vmaxps %%ymm12, %%ymm1, %%ymm1\n"
    "vmaxps %%ymm12, %%ymm2, %%ymm2\n"
    "vmaxps %%ymm12, %%ymm3, %%ymm3\n"
    "and $0x1, %%eax\n"
    "je 6f\n"
    // relu6
    "mov $0x40C00000, %%ecx\n"
    "vmovd %%ecx, %%xmm14\n"
    "vpermps %%ymm14, %%ymm12, %%ymm14\n"
    "vminps %%ymm14, %%ymm0, %%ymm0\n"
    "vminps %%ymm14, %%ymm1, %%ymm1\n"
    "vminps %%ymm14, %%ymm2, %%ymm2\n"
    "vminps %%ymm14, %%ymm3, %%ymm3\n"
    "6:\n"
    "vmovups %%ymm0, (%5)\n"  // dst_0
    "vmovups %%ymm1, 0x20(%5)\n"
    "vmovups %%ymm2, 0x40(%5)\n"
    "vmovups %%ymm3, 0x60(%5)\n"
    :
    : "r"(src), "r"(weight), "r"(bias), "r"(deep), "a"(act_flag), "r"(dst)  // 5
    : "%rcx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm12", "%ymm4", "%ymm14");
}

void MatMul4x24Kernel(float *dst, const float *src, const float *weight, const float *bias, const size_t act_flag,
                      const size_t row_block, const size_t col_block, size_t col_algin, const size_t deep) {
  float *dst_3 = dst + C3NUM * col_algin;
  col_algin *= sizeof(float);
  size_t src_3_step = C3NUM * deep * sizeof(float);
  asm volatile(
    "cmpq $0, %2\n"
    "je 0f\n"
    "vmovups (%2), %%ymm0\n"
    "vmovups 0x20(%2), %%ymm1\n"
    "vmovups 0x40(%2), %%ymm2\n"
    "vmovups (%2), %%ymm3\n"
    "vmovups 0x20(%2), %%ymm4\n"
    "vmovups 0x40(%2), %%ymm5\n"
    "vmovups (%2), %%ymm6\n"
    "vmovups 0x20(%2), %%ymm7\n"
    "vmovups 0x40(%2), %%ymm8\n"
    "vmovups (%2), %%ymm9\n"
    "vmovups 0x20(%2), %%ymm10\n"
    "vmovups 0x40(%2), %%ymm11\n"
    "jmp 1f\n"
    "0:\n"
    "vxorps %%ymm0, %%ymm0, %%ymm0\n"
    "vxorps %%ymm1, %%ymm1, %%ymm1\n"
    "vxorps %%ymm2, %%ymm2, %%ymm2\n"
    "vxorps %%ymm3, %%ymm3, %%ymm3\n"
    "vxorps %%ymm4, %%ymm4, %%ymm4\n"
    "vxorps %%ymm5, %%ymm5, %%ymm5\n"
    "vxorps %%ymm6, %%ymm6, %%ymm6\n"
    "vxorps %%ymm7, %%ymm7, %%ymm7\n"
    "vxorps %%ymm8, %%ymm8, %%ymm8\n"
    "vxorps %%ymm9, %%ymm9, %%ymm9\n"
    "vxorps %%ymm10, %%ymm10, %%ymm10\n"
    "vxorps %%ymm11, %%ymm11, %%ymm11\n"

    "1:\n"                     // deep
    "vmovups (%1), %%ymm12\n"  // weight
    "vmovups 0x20(%1), %%ymm13\n"
    "vmovups 0x40(%1), %%ymm14\n"

    "vbroadcastss (%0), %%ymm15\n"  // src
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm0\n"
    "vfmadd231ps %%ymm15, %%ymm13, %%ymm1\n"
    "vfmadd231ps %%ymm15, %%ymm14, %%ymm2\n"

    "vbroadcastss (%0, %9), %%ymm15\n"
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm3\n"
    "vfmadd231ps %%ymm15, %%ymm13, %%ymm4\n"
    "vfmadd231ps %%ymm15, %%ymm14, %%ymm5\n"

    "vbroadcastss (%0, %9, 2), %%ymm15\n"
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm6\n"
    "vfmadd231ps %%ymm15, %%ymm13, %%ymm7\n"
    "vfmadd231ps %%ymm15, %%ymm14, %%ymm8\n"

    "vbroadcastss (%0, %7), %%ymm15\n"
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm9\n"
    "vfmadd231ps %%ymm15, %%ymm13, %%ymm10\n"
    "vfmadd231ps %%ymm15, %%ymm14, %%ymm11\n"
    "addq $96, %1\n"
    "addq $4, %0\n"
    "dec %3\n"
    "jg 1b\n"

    "and $0x3, %%eax\n"  // act_type
    "je 6f\n"
    // Relu
    "vxorps %%ymm12, %%ymm12, %%ymm12\n"
    "vmaxps %%ymm12, %%ymm0, %%ymm0\n"
    "vmaxps %%ymm12, %%ymm1, %%ymm1\n"
    "vmaxps %%ymm12, %%ymm2, %%ymm2\n"
    "vmaxps %%ymm12, %%ymm3, %%ymm3\n"
    "vmaxps %%ymm12, %%ymm4, %%ymm4\n"
    "vmaxps %%ymm12, %%ymm5, %%ymm5\n"
    "vmaxps %%ymm12, %%ymm6, %%ymm6\n"
    "vmaxps %%ymm12, %%ymm7, %%ymm7\n"
    "vmaxps %%ymm12, %%ymm8, %%ymm8\n"
    "vmaxps %%ymm12, %%ymm9, %%ymm9\n"
    "vmaxps %%ymm12, %%ymm10, %%ymm10\n"
    "vmaxps %%ymm12, %%ymm11, %%ymm11\n"
    "and $0x1, %%eax\n"
    "je 6f\n"
    // relu6
    "mov $0x40C00000, %%ecx\n"
    "vmovd %%ecx, %%xmm14\n"
    "vpermps %%ymm14, %%ymm12, %%ymm14\n"
    "vminps %%ymm14, %%ymm0, %%ymm0\n"
    "vminps %%ymm14, %%ymm1, %%ymm1\n"
    "vminps %%ymm14, %%ymm2, %%ymm2\n"
    "vminps %%ymm14, %%ymm3, %%ymm3\n"
    "vminps %%ymm14, %%ymm4, %%ymm4\n"
    "vminps %%ymm14, %%ymm5, %%ymm5\n"
    "vminps %%ymm14, %%ymm6, %%ymm6\n"
    "vminps %%ymm14, %%ymm7, %%ymm7\n"
    "vminps %%ymm14, %%ymm8, %%ymm8\n"
    "vminps %%ymm14, %%ymm9, %%ymm9\n"
    "vminps %%ymm14, %%ymm10, %%ymm10\n"
    "vminps %%ymm14, %%ymm11, %%ymm11\n"
    "6:\n"
    "vmovups %%ymm0, (%5)\n"  // dst_0
    "vmovups %%ymm1, 0x20(%5)\n"
    "vmovups %%ymm2, 0x40(%5)\n"
    "vmovups %%ymm3, (%5, %6)\n"
    "vmovups %%ymm4, 0x20(%5, %6)\n"  // dst_1
    "vmovups %%ymm5, 0x40(%5, %6)\n"
    "vmovups %%ymm6, (%5, %6, 2)\n"
    "vmovups %%ymm7, 0x20(%5, %6, 2)\n"
    "vmovups %%ymm8, 0x40(%5, %6, 2)\n"  // dst_2
    "vmovups %%ymm9, (%8)\n"
    "vmovups %%ymm10, 0x20(%8)\n"
    "vmovups %%ymm11, 0x40(%8)\n"
    :
    : "r"(src), "r"(weight), "r"(bias), "r"(deep), "a"(act_flag), "r"(dst), "r"(col_algin), "r"(src_3_step), "r"(dst_3),
      "r"(deep * sizeof(float))  // 9
    : "%rcx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10",
      "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15");
}

void MatVecMul1x24Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                         size_t row_block, size_t col_block, size_t col_algin, size_t deep) {
  asm volatile(
    "cmpq $0, %2\n"
    "je 0f\n"
    "vmovups (%2), %%ymm0\n"
    "vmovups 0x20(%2), %%ymm1\n"
    "vmovups 0x40(%2), %%ymm2\n"
    "jmp 1f\n"
    "0:\n"
    "vxorps %%ymm0, %%ymm0, %%ymm0\n"
    "vxorps %%ymm1, %%ymm1, %%ymm1\n"
    "vxorps %%ymm2, %%ymm2, %%ymm2\n"

    "1:\n"  // deep
    "movq %3, %%rcx\n"
    "shr $3, %%ecx\n"
    "je 3f\n"
    "2:\n"
    "vbroadcastss (%0), %%ymm4\n"
    "vfmadd231ps (%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 0x20(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 0x40(%1), %%ymm4, %%ymm2\n"

    "vbroadcastss 4(%0), %%ymm4\n"
    "vfmadd231ps 96(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 128(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 160(%1), %%ymm4, %%ymm2\n"

    "vbroadcastss 8(%0), %%ymm4\n"
    "vfmadd231ps 192(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 224(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 256(%1), %%ymm4, %%ymm2\n"

    "vbroadcastss 12(%0), %%ymm4\n"
    "vfmadd231ps 288(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 320(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 352(%1), %%ymm4, %%ymm2\n"

    "vbroadcastss 16(%0), %%ymm4\n"
    "vfmadd231ps 384(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 416(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 448(%1), %%ymm4, %%ymm2\n"

    "vbroadcastss 20(%0), %%ymm4\n"
    "vfmadd231ps 480(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 512(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 544(%1), %%ymm4, %%ymm2\n"

    "vbroadcastss 24(%0), %%ymm4\n"
    "vfmadd231ps 576(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 608(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 640(%1), %%ymm4, %%ymm2\n"

    "vbroadcastss 28(%0), %%ymm4\n"
    "vfmadd231ps 672(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 704(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 736(%1), %%ymm4, %%ymm2\n"
    "addq $768, %1\n"
    "addq $32, %0\n"
    "dec %%ecx\n"
    "jg 2b\n"

    "3:\n"
    "and $7, %3\n"  // deep_remainder
    "je 5f\n"
    "4:\n"
    "vbroadcastss (%0), %%ymm4\n"
    "vfmadd231ps (%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 0x20(%1), %%ymm4, %%ymm1\n"
    "vfmadd231ps 0x40(%1), %%ymm4, %%ymm2\n"
    "addq $96, %1\n"
    "addq $4, %0\n"
    "dec %3\n"
    "jg 4b\n"

    "5:\n"
    "and $0x3, %%eax\n"  // act_type
    "je 6f\n"
    // Relu
    "vxorps %%ymm12, %%ymm12, %%ymm12\n"
    "vmaxps %%ymm12, %%ymm0, %%ymm0\n"
    "vmaxps %%ymm12, %%ymm1, %%ymm1\n"
    "vmaxps %%ymm12, %%ymm2, %%ymm2\n"

    "and $0x1, %%eax\n"
    "je 6f\n"
    // relu6
    "mov $0x40C00000, %%ecx\n"
    "vmovd %%ecx, %%xmm14\n"
    "vpermps %%ymm14, %%ymm12, %%ymm14\n"
    "vminps %%ymm14, %%ymm0, %%ymm0\n"
    "vminps %%ymm14, %%ymm1, %%ymm1\n"
    "vminps %%ymm14, %%ymm2, %%ymm2\n"

    "6:\n"
    "vmovups %%ymm0, (%5)\n"  // dst_0
    "vmovups %%ymm1, 0x20(%5)\n"
    "vmovups %%ymm2, 0x40(%5)\n"

    :
    : "r"(src), "r"(weight), "r"(bias), "r"(deep), "a"(act_flag), "r"(dst)  // 5
    : "%rcx", "%ymm0", "%ymm1", "%ymm2", "%ymm12", "%ymm4", "%ymm14");
}

void MatMul6x16Kernel(float *dst, const float *src, const float *weight, const float *bias, const size_t act_flag,
                      const size_t row_block, const size_t col_block, size_t col_algin, const size_t deep) {
  float *dst_3 = dst + 3 * col_algin;
  float *dst_5 = dst + 5 * col_algin;
  col_algin *= sizeof(float);
  size_t src_3_step = 3 * deep * sizeof(float);
  size_t src_5_step = 5 * deep * sizeof(float);
  asm volatile(
    "cmpq $0, %2\n"
    "je 0f\n"
    "vmovups (%2), %%ymm0\n"
    "vmovups 0x20(%2), %%ymm1\n"
    "vmovups (%2), %%ymm2\n"
    "vmovups 0x20(%2), %%ymm3\n"
    "vmovups (%2), %%ymm4\n"
    "vmovups 0x20(%2), %%ymm5\n"
    "vmovups (%2), %%ymm6\n"
    "vmovups 0x20(%2), %%ymm7\n"
    "vmovups (%2), %%ymm8\n"
    "vmovups 0x20(%2), %%ymm9\n"
    "vmovups (%2), %%ymm10\n"
    "vmovups 0x20(%2), %%ymm11\n"
    "jmp 1f\n"
    "0:\n"
    "vxorps %%ymm0, %%ymm0, %%ymm0\n"
    "vxorps %%ymm1, %%ymm1, %%ymm1\n"
    "vxorps %%ymm2, %%ymm2, %%ymm2\n"
    "vxorps %%ymm3, %%ymm3, %%ymm3\n"
    "vxorps %%ymm4, %%ymm4, %%ymm4\n"
    "vxorps %%ymm5, %%ymm5, %%ymm5\n"
    "vxorps %%ymm6, %%ymm6, %%ymm6\n"
    "vxorps %%ymm7, %%ymm7, %%ymm7\n"
    "vxorps %%ymm8, %%ymm8, %%ymm8\n"
    "vxorps %%ymm9, %%ymm9, %%ymm9\n"
    "vxorps %%ymm10, %%ymm10, %%ymm10\n"
    "vxorps %%ymm11, %%ymm11, %%ymm11\n"

    "1:\n"                     // deep
    "vmovups (%1), %%ymm12\n"  // weight
    "vmovups 0x20(%1), %%ymm13\n"

    "vbroadcastss (%0), %%ymm14\n"       // src_0
    "vbroadcastss (%0, %11), %%ymm15\n"  // src_1
    "vfmadd231ps %%ymm14, %%ymm12, %%ymm0\n"
    "vfmadd231ps %%ymm14, %%ymm13, %%ymm1\n"
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm2\n"
    "vfmadd231ps %%ymm15, %%ymm13, %%ymm3\n"

    "vbroadcastss (%0, %11, 2), %%ymm14\n"  // src_2
    "vbroadcastss (%0, %8), %%ymm15\n"      // src_3
    "vfmadd231ps %%ymm14, %%ymm12, %%ymm4\n"
    "vfmadd231ps %%ymm14, %%ymm13, %%ymm5\n"
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm6\n"
    "vfmadd231ps %%ymm15, %%ymm13, %%ymm7\n"

    "vbroadcastss (%0, %11, 4), %%ymm14\n"  // src_4
    "vbroadcastss (%0, %9), %%ymm15\n"      // src_5
    "vfmadd231ps %%ymm14, %%ymm12, %%ymm8\n"
    "vfmadd231ps %%ymm14, %%ymm13, %%ymm9\n"
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm10\n"
    "vfmadd231ps %%ymm15, %%ymm13, %%ymm11\n"

    "addq $64, %1\n"
    "addq $4, %0\n"
    "dec %3\n"
    "jg 1b\n"

    "and $0x3, %%eax\n"  // act_type
    "je 6f\n"
    // Relu
    "vxorps %%ymm12, %%ymm12, %%ymm12\n"
    "vmaxps %%ymm12, %%ymm0, %%ymm0\n"
    "vmaxps %%ymm12, %%ymm1, %%ymm1\n"
    "vmaxps %%ymm12, %%ymm2, %%ymm2\n"
    "vmaxps %%ymm12, %%ymm3, %%ymm3\n"
    "vmaxps %%ymm12, %%ymm4, %%ymm4\n"
    "vmaxps %%ymm12, %%ymm5, %%ymm5\n"
    "vmaxps %%ymm12, %%ymm6, %%ymm6\n"
    "vmaxps %%ymm12, %%ymm7, %%ymm7\n"
    "vmaxps %%ymm12, %%ymm8, %%ymm8\n"
    "vmaxps %%ymm12, %%ymm9, %%ymm9\n"
    "vmaxps %%ymm12, %%ymm10, %%ymm10\n"
    "vmaxps %%ymm12, %%ymm11, %%ymm11\n"
    "and $0x1, %%eax\n"
    "je 6f\n"
    // relu6
    "mov $0x40C00000, %%ecx\n"
    "vmovd %%ecx, %%xmm14\n"
    "vpermps %%ymm14, %%ymm12, %%ymm14\n"
    "vminps %%ymm14, %%ymm0, %%ymm0\n"
    "vminps %%ymm14, %%ymm1, %%ymm1\n"
    "vminps %%ymm14, %%ymm2, %%ymm2\n"
    "vminps %%ymm14, %%ymm3, %%ymm3\n"
    "vminps %%ymm14, %%ymm4, %%ymm4\n"
    "vminps %%ymm14, %%ymm5, %%ymm5\n"
    "vminps %%ymm14, %%ymm6, %%ymm6\n"
    "vminps %%ymm14, %%ymm7, %%ymm7\n"
    "vminps %%ymm14, %%ymm8, %%ymm8\n"
    "vminps %%ymm14, %%ymm9, %%ymm9\n"
    "vminps %%ymm14, %%ymm10, %%ymm10\n"
    "vminps %%ymm14, %%ymm11, %%ymm11\n"
    "6:\n"
    "vmovups %%ymm0, (%5)\n"  // dst_0
    "vmovups %%ymm1, 0x20(%5)\n"
    "vmovups %%ymm2, (%5, %6)\n"  // dst_1
    "vmovups %%ymm3, 0x20(%5, %6)\n"
    "vmovups %%ymm4, (%5, %6, 2)\n"  // dst_2
    "vmovups %%ymm5, 0x20(%5, %6, 2)\n"
    "vmovups %%ymm6, (%7)\n"  // dst_3
    "vmovups %%ymm7, 0x20(%7)\n"
    "vmovups %%ymm8, (%5, %6, 4)\n"  // dst_4
    "vmovups %%ymm9, 0x20(%5, %6, 4)\n"
    "vmovups %%ymm10, (%10)\n"  // dst_5
    "vmovups %%ymm11, 0x20(%10)\n"
    :
    : "r"(src), "r"(weight), "r"(bias), "r"(deep), "a"(act_flag), "r"(dst), "r"(col_algin), "r"(dst_3), "r"(src_3_step),
      "r"(src_5_step), "r"(dst_5), "r"(deep * sizeof(float))  // 11
    : "%rcx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10",
      "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15");
}

void MatVecMul1x16Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                         size_t row_block, size_t col_block, size_t col_algin, size_t deep) {
  asm volatile(
    "cmpq $0, %2\n"
    "je 0f\n"
    "vmovups (%2), %%ymm0\n"
    "vmovups 0x20(%2), %%ymm1\n"
    "jmp 1f\n"
    "0:\n"
    "vxorps %%ymm0, %%ymm0, %%ymm0\n"
    "vxorps %%ymm1, %%ymm1, %%ymm1\n"
    "1:\n"
    "movq %3, %%rcx\n"
    "shr $3, %%ecx\n"
    "je 3f\n"
    "2:\n"  // deep_c8
    "vbroadcastss (%0), %%ymm4\n"
    "vfmadd231ps (%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 0x20(%1), %%ymm4, %%ymm1\n"

    "vbroadcastss 4(%0), %%ymm4\n"
    "vfmadd231ps 64(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 96(%1), %%ymm4, %%ymm1\n"

    "vbroadcastss 8(%0), %%ymm4\n"
    "vfmadd231ps 128(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 160(%1), %%ymm4, %%ymm1\n"

    "vbroadcastss 12(%0), %%ymm4\n"
    "vfmadd231ps 192(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 224(%1), %%ymm4, %%ymm1\n"

    "vbroadcastss 16(%0), %%ymm4\n"
    "vfmadd231ps 256(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 288(%1), %%ymm4, %%ymm1\n"

    "vbroadcastss 20(%0), %%ymm4\n"
    "vfmadd231ps 320(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 352(%1), %%ymm4, %%ymm1\n"

    "vbroadcastss 24(%0), %%ymm4\n"
    "vfmadd231ps 384(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 416(%1), %%ymm4, %%ymm1\n"

    "vbroadcastss 28(%0), %%ymm4\n"
    "vfmadd231ps 448(%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 480(%1), %%ymm4, %%ymm1\n"
    "addq $512, %1\n"
    "addq $32, %0\n"
    "dec %%ecx\n"
    "jg 2b\n"

    "3:\n"
    "and $7, %3\n"
    "je 5f\n"
    "4:\n"
    "vbroadcastss (%0), %%ymm4\n"
    "vfmadd231ps (%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 0x20(%1), %%ymm4, %%ymm1\n"
    "addq $64, %1\n"
    "addq $4, %0\n"
    "dec %3\n"
    "jg 4b\n"

    "5:\n"
    "and $0x3, %%eax\n"  // act_type
    "je 6f\n"
    // Relu
    "vxorps %%ymm12, %%ymm12, %%ymm12\n"
    "vmaxps %%ymm12, %%ymm0, %%ymm0\n"
    "vmaxps %%ymm12, %%ymm1, %%ymm1\n"

    "and $0x1, %%eax\n"
    "je 6f\n"
    // relu6
    "mov $0x40C00000, %%ecx\n"
    "vmovd %%ecx, %%xmm14\n"
    "vpermps %%ymm14, %%ymm12, %%ymm14\n"
    "vminps %%ymm14, %%ymm0, %%ymm0\n"
    "vminps %%ymm14, %%ymm1, %%ymm1\n"

    "6:\n"
    "vmovups %%ymm0, (%5)\n"  // dst_0
    "vmovups %%ymm1, 0x20(%5)\n"

    :
    : "r"(src), "r"(weight), "r"(bias), "r"(deep), "a"(act_flag), "r"(dst)  // 5
    : "%rcx", "%ymm0", "%ymm1", "%ymm12", "%ymm4", "%ymm14");
}

void MatVecMul1x8Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                        size_t row_block, size_t col_block, size_t col_algin, size_t deep) {
  asm volatile(
    "cmpq $0, %2\n"
    "je 0f\n"
    "vmovups (%2), %%ymm0\n"
    "jmp 1f\n"
    "0:\n"
    "vxorps %%ymm0, %%ymm0, %%ymm0\n"
    "1:\n"
    "movq %3, %%rcx\n"
    "shr $3, %%ecx\n"
    "je 3f\n"
    "2:\n"  // deep_c8
    "vbroadcastss (%0), %%ymm4\n"
    "vfmadd231ps (%1), %%ymm4, %%ymm0\n"
    "vbroadcastss 4(%0), %%ymm4\n"
    "vfmadd231ps 32(%1), %%ymm4, %%ymm0\n"
    "vbroadcastss 8(%0), %%ymm4\n"
    "vfmadd231ps 64(%1), %%ymm4, %%ymm0\n"
    "vbroadcastss 12(%0), %%ymm4\n"
    "vfmadd231ps 96(%1), %%ymm4, %%ymm0\n"
    "vbroadcastss 16(%0), %%ymm4\n"
    "vfmadd231ps 128(%1), %%ymm4, %%ymm0\n"
    "vbroadcastss 20(%0), %%ymm4\n"
    "vfmadd231ps 160(%1), %%ymm4, %%ymm0\n"
    "vbroadcastss 24(%0), %%ymm4\n"
    "vfmadd231ps 192(%1), %%ymm4, %%ymm0\n"
    "vbroadcastss 28(%0), %%ymm4\n"
    "vfmadd231ps 224(%1), %%ymm4, %%ymm0\n"
    "addq $256, %1\n"
    "addq $32, %0\n"
    "dec %%ecx\n"
    "jg 2b\n"

    "3:\n"
    "and $7, %3\n"
    "je 5f\n"
    "4:\n"
    "vbroadcastss (%0), %%ymm4\n"
    "vfmadd231ps (%1), %%ymm4, %%ymm0\n"
    "addq $32, %1\n"
    "addq $4, %0\n"
    "dec %3\n"
    "jg 4b\n"

    "5:\n"
    "and $0x3, %%eax\n"  // act_type
    "je 6f\n"
    // Relu
    "vxorps %%ymm12, %%ymm12, %%ymm12\n"
    "vmaxps %%ymm12, %%ymm0, %%ymm0\n"

    "and $0x1, %%eax\n"
    "je 6f\n"
    // relu6
    "mov $0x40C00000, %%ecx\n"
    "vmovd %%ecx, %%xmm14\n"
    "vpermps %%ymm14, %%ymm12, %%ymm14\n"
    "vminps %%ymm14, %%ymm0, %%ymm0\n"

    "6:\n"
    "vmovups %%ymm0, (%5)\n"  // dst_0

    :
    : "r"(src), "r"(weight), "r"(bias), "r"(deep), "a"(act_flag), "r"(dst)  // 5
    : "%rcx", "%ymm0", "%ymm1", "%ymm12", "%ymm4", "%ymm14");
}

void MatMul8x8Kernel(float *dst, const float *src, const float *weight, const float *bias, const size_t act_flag,
                     const size_t row_block, const size_t col_block, size_t col_algin, const size_t deep) {
  float *dst_5 = dst + C5NUM * col_algin;
  col_algin *= sizeof(float);
  size_t dst_3_step = C3NUM * col_algin;
  size_t src_3_step = C3NUM * deep * sizeof(float);
  const float *src_5 = C5NUM * deep + src;
  asm volatile(
    "cmpq $0, %2\n"
    "je 0f\n"
    "vmovups (%2), %%ymm0\n"
    "vmovups (%2), %%ymm1\n"
    "vmovups (%2), %%ymm2\n"
    "vmovups (%2), %%ymm3\n"
    "vmovups (%2), %%ymm4\n"
    "vmovups (%2), %%ymm5\n"
    "vmovups (%2), %%ymm6\n"
    "vmovups (%2), %%ymm7\n"
    "jmp 1f\n"
    "0:\n"
    "vxorps %%ymm0, %%ymm0, %%ymm0\n"
    "vxorps %%ymm1, %%ymm1, %%ymm1\n"
    "vxorps %%ymm2, %%ymm2, %%ymm2\n"
    "vxorps %%ymm3, %%ymm3, %%ymm3\n"
    "vxorps %%ymm4, %%ymm4, %%ymm4\n"
    "vxorps %%ymm5, %%ymm5, %%ymm5\n"
    "vxorps %%ymm6, %%ymm6, %%ymm6\n"
    "vxorps %%ymm7, %%ymm7, %%ymm7\n"

    "1:\n"                     // deep
    "vmovups (%1), %%ymm15\n"  // weight

    "vbroadcastss (%0), %%ymm8\n"           // src_0
    "vbroadcastss (%0, %11), %%ymm9\n"      // src_1
    "vbroadcastss (%0, %11, 2), %%ymm10\n"  // src_2
    "vbroadcastss (%0, %8), %%ymm11\n"      // src_3
    "vfmadd231ps %%ymm8, %%ymm15, %%ymm0\n"
    "vfmadd231ps %%ymm9, %%ymm15, %%ymm1\n"
    "vfmadd231ps %%ymm10, %%ymm15, %%ymm2\n"
    "vfmadd231ps %%ymm11, %%ymm15, %%ymm3\n"

    "vbroadcastss (%0, %11, 4), %%ymm8\n"   // src_4
    "vbroadcastss (%9), %%ymm9\n"           // src_5
    "vbroadcastss (%9, %11, 1), %%ymm10\n"  // src_6
    "vbroadcastss (%9, %11, 2), %%ymm11\n"  // src_7
    "vfmadd231ps %%ymm8, %%ymm15, %%ymm4\n"
    "vfmadd231ps %%ymm9, %%ymm15, %%ymm5\n"
    "vfmadd231ps %%ymm10, %%ymm15, %%ymm6\n"
    "vfmadd231ps %%ymm11, %%ymm15, %%ymm7\n"

    "addq $32, %1\n"
    "addq $4, %0\n"
    "addq $4, %9\n"
    "dec %3\n"
    "jg 1b\n"

    "and $0x3, %%eax\n"  // act_type
    "je 6f\n"
    // Relu
    "vxorps %%ymm12, %%ymm12, %%ymm12\n"
    "vmaxps %%ymm12, %%ymm0, %%ymm0\n"
    "vmaxps %%ymm12, %%ymm1, %%ymm1\n"
    "vmaxps %%ymm12, %%ymm2, %%ymm2\n"
    "vmaxps %%ymm12, %%ymm3, %%ymm3\n"
    "vmaxps %%ymm12, %%ymm4, %%ymm4\n"
    "vmaxps %%ymm12, %%ymm5, %%ymm5\n"
    "vmaxps %%ymm12, %%ymm6, %%ymm6\n"
    "vmaxps %%ymm12, %%ymm7, %%ymm7\n"
    "and $0x1, %%eax\n"
    "je 6f\n"
    // relu6
    "mov $0x40C00000, %%ecx\n"
    "vmovd %%ecx, %%xmm14\n"
    "vpermps %%ymm14, %%ymm12, %%ymm14\n"
    "vminps %%ymm14, %%ymm0, %%ymm0\n"
    "vminps %%ymm14, %%ymm1, %%ymm1\n"
    "vminps %%ymm14, %%ymm2, %%ymm2\n"
    "vminps %%ymm14, %%ymm3, %%ymm3\n"
    "vminps %%ymm14, %%ymm4, %%ymm4\n"
    "vminps %%ymm14, %%ymm5, %%ymm5\n"
    "vminps %%ymm14, %%ymm6, %%ymm6\n"
    "vminps %%ymm14, %%ymm7, %%ymm7\n"
    "6:\n"
    "vmovups %%ymm0, (%5)\n"  // dst_0
    "vmovups %%ymm1, (%5, %6)\n"
    "vmovups %%ymm2, (%5, %6, 2)\n"
    "vmovups %%ymm3, (%5, %7)\n"
    "vmovups %%ymm4, (%5, %6, 4)\n"
    "vmovups %%ymm5, (%10)\n"
    "vmovups %%ymm6, (%10, %6)\n"
    "vmovups %%ymm7, (%10, %6, 2)\n"
    :
    : "r"(src), "r"(weight), "r"(bias), "r"(deep), "a"(act_flag), "r"(dst), "r"(col_algin), "r"(dst_3_step),  // 7
      "r"(src_3_step), "r"(src_5), "r"(dst_5), "r"(deep * sizeof(float))                                      // 11
    : "%rcx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10",
      "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15");
}

#ifdef ENABLE_DEBUG
void MatVecMulRowxColKernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                            size_t row_block, size_t col_block, size_t col_algin, size_t deep) {
  __m256 dst_data[12];
  const float *src_sw[12];
  __m256 weight_data[4];
  for (int i = 0; i < 4; ++i) {
    weight_data[i] = _mm256_set1_ps(0.0f);
  }
  for (int i = 0; i < row_block; ++i) {
    if (bias != NULL) {
      for (int j = 0; j < col_block; ++j) {
        dst_data[i * col_block + j] = _mm256_loadu_ps(bias + j * 8);
      }
    } else {
      for (int j = 0; j < col_block; ++j) {
        dst_data[i * col_block + j] = _mm256_set1_ps(0.0f);
      }
    }
    src_sw[i] = src + i * deep;
  }
  const float *weight_kernel = weight;
  for (int ic = 0; ic < deep; ++ic) {
    for (int j = 0; j < col_block; ++j) {
      weight_data[j] = _mm256_loadu_ps(weight_kernel + j * C8NUM);
    }
    for (int i = 0; i < row_block; ++i) {
      for (int j = 0; j < col_block; ++j) {
        dst_data[i * col_block + j] =
          _mm256_fmadd_ps(_mm256_set1_ps(src_sw[i][ic]), weight_data[j], dst_data[i * col_block + j]);
      }
    }
    weight_kernel += C8NUM * col_block;
  }  // ic loop
  // add bias and relu
  for (int i = 0; i < row_block; ++i) {
    for (int j = 0; j < col_block; ++j) {
      if (0x1 & act_flag) {  // relu6
        dst_data[i * col_block + j] = _mm256_min_ps(dst_data[i * col_block + j], _mm256_set1_ps(6.0f));
      }
      if (0x2 & act_flag) {  // relu
        dst_data[i * col_block + j] = _mm256_max_ps(dst_data[i * col_block + j], _mm256_set1_ps(0.0f));
      }
      _mm256_storeu_ps(dst + i * col_algin + j * C8NUM, dst_data[i * col_block + j]);
    }
  }
}
#endif
#endif

#define ActCompute(bit_num, down_threshold, up_threshold) \
  if (act_type != 0) {                                    \
    dst = MS_MAX##bit_num##_F32(dst, down_threshold);     \
    if (act_type == 3) {                                  \
      dst = MS_MIN##bit_num##_F32(dst, up_threshold);     \
    }                                                     \
  }

// act_type must be 0, 1, 2. 0: no_act, 1: relu, 3: relu6.
void GemmIsNotPack(const float *a, const float *b, float *c, const float *bias, int row, int deep, int act_type) {
  int index = 0;
#ifdef ENABLE_AVX512
  __m512 down_threshold512 = _mm512_setzero_ps();
  __m512 up_threshold512 = _mm512_set1_ps(C6NUM);
  __m512 b_data16 = _mm512_set1_ps(b[0]);
  __m512 bias_data16 = _mm512_set1_ps(bias[0]);
  for (; index < row - C16NUM; index += C16NUM) {
    __m512 a_data = _mm512_loadu_ps(a + index);
    __m512 dst = b_data16 * a_data + bias_data16;
    ActCompute(512, down_threshold512, up_threshold512);
    _mm512_storeu_ps(c + index, dst);
  }
#endif

#ifdef ENABLE_AVX
  __m256 down_threshold256 = _mm256_setzero_ps();
  __m256 up_threshold256 = _mm256_set1_ps(C6NUM);
  __m256 b_data8 = _mm256_set1_ps(b[0]);
  __m256 bias_data8 = _mm256_set1_ps(bias[0]);
  for (; index < row - C8NUM; index += C8NUM) {
    __m256 a_data = _mm256_loadu_ps(a + index);
    __m256 dst = b_data8 * a_data + bias_data8;
    ActCompute(256, down_threshold256, up_threshold256);
    _mm256_storeu_ps(c + index, dst);
  }
#endif

#if defined(ENABLE_SSE) || defined(ENABLE_ARM)
  MS_FLOAT32X4 down_threshold128 = MS_MOVQ_F32(0);
  MS_FLOAT32X4 up_threshold128 = MS_MOVQ_F32(C6NUM);
  MS_FLOAT32X4 b_data4 = MS_MOVQ_F32(b[0]);
  MS_FLOAT32X4 bias_data4 = MS_MOVQ_F32(bias[0]);
  for (; index < row - C4NUM; index += C4NUM) {
    MS_FLOAT32X4 a_data = MS_LDQ_F32(a + index);
    MS_FLOAT32X4 dst = MS_ADD128_F32(MS_MUL128_F32(b_data4, a_data), bias_data4);
    ActCompute(128, down_threshold128, up_threshold128);
    MS_STQ_F32(c + index, dst);
  }
#endif

  for (; index < row; ++index) {
    float dst = a[index] * b[0] + bias[0];
    ActCompute(32, 0, C6NUM);
    c[index] = dst;
  }
}

// act_type must be 0, 1, 2. 0: no_act, 1: relu, 3: relu6.
void GemmIsNotPackOptimize(const float *a, const float *b, float *c, const float *bias, int m, int k, int act_type) {
  // gemm dot is [m, k] * [k, 1] ==>> [m, 1]
  int m_index = 0;
#ifdef ENABLE_AVX512
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
    ActCompute(256, down_threshold256, up_threshold256);
    MS_ST256_F32(c + m_index, dst);
  }
#endif
#ifdef ENABLE_AVX
  // block 4
  MS_FLOAT32X4 down_threshold128 = MS_MOVQ_F32(0);
  MS_FLOAT32X4 up_threshold128 = MS_MOVQ_F32(C6NUM);
  for (; m_index <= m - C4NUM; m_index += C4NUM) {
    int k_index = 0;
    MS_FLOAT32X4 dst = MS_MOV128_F32(bias[0]);
    MS_SET_ZERO256X4_F32(dst_)
    for (; k_index <= k - C8NUM; k_index += C8NUM) {
      MS_FLOAT32X8 weight = MS_LD256_F32(b + k_index);
      MS_LOAD256X4_F32(src, a + m_index * k + k_index, k);
      MS_FMADD256X4_F32(src, weight, dst_);
    }
    MS_F32X8_GETI(dst, 0) += MS_REDUCE_ADD256_F32(dst_1);
    MS_F32X8_GETI(dst, 1) += MS_REDUCE_ADD256_F32(dst_2);
    MS_F32X8_GETI(dst, C2NUM) += MS_REDUCE_ADD256_F32(dst_3);
    MS_F32X8_GETI(dst, C3NUM) += MS_REDUCE_ADD256_F32(dst_4);
    for (; k_index < k; ++k_index) {
      MS_F32X8_GETI(dst, 0) += b[k_index] * a[m_index * k + k_index];
      MS_F32X8_GETI(dst, 1) += b[k_index] * a[m_index * k + k_index + k];
      MS_F32X8_GETI(dst, C2NUM) += b[k_index] * a[m_index * k + k_index + C2NUM * k];
      MS_F32X8_GETI(dst, C3NUM) += b[k_index] * a[m_index * k + k_index + C3NUM * k];
    }
    ActCompute(128, down_threshold128, up_threshold128);
    MS_ST128_F32(c + m_index, dst);
  }
#endif

  // block 1
  for (; m_index < m; m_index++) {
    float dst = bias[0];
    int k_index = 0;
#ifdef ENABLE_AVX512
    __m512 dst1 = _mm512_setzero_ps();
    for (; k_index <= k - C16NUM; k_index += C16NUM) {
      __m512 weight = _mm512_loadu_ps(b + k_index);
      __m512 a1 = _mm512_loadu_ps(a + m_index * k + k_index);
      dst1 = _mm512_fmadd_ps(weight, a1, dst1);
    }
    dst += _mm512_reduce_add_ps(dst1);
#endif
#ifdef ENABLE_AVX
    __m256 dst2 = _mm256_setzero_ps();
    for (; k_index <= k - C8NUM; k_index += C8NUM) {
      __m256 weight = _mm256_loadu_ps(b + k_index);
      __m256 src = _mm256_loadu_ps(a + m_index * k + k_index);
      dst2 = _mm256_fmadd_ps(weight, src, dst2);
    }
    dst += MS_REDUCE_ADD256_F32(dst2);
#endif
    for (; k_index < k; k_index++) {
      dst += b[k_index] * a[m_index * k + k_index];
    }
    ActCompute(32, 0, C6NUM);
    c[m_index] = dst;
  }
}

#ifdef ENABLE_ARM64
// act_type must be 0, 1, 2. 0: no_act, 1: relu, 3: relu6.
void MatMul4x1Kernel(const float *input, const float *weight, float *output, const float *bias, size_t deep,
                     size_t act_type) {
  // 1: LoopD16, 2: LoopD12, 3: LoopD8, 4: LoopD4, 5: LoopD1, 6: LoopDEnd, 7: LoopDTail, 8: LoopDTailCompute
  // 9: WriteBack
  asm volatile(
    "mov x8, %[input]\n"
    "mov x9, %[weight]\n"
    "mov x10, %[deep]\n"
    "add x5, %[input], %[deep], LSL #2\n"
    "add x6, %[input], %[deep], LSL #3\n"
    "add x7, x5, %[deep], LSL #3\n"
    "dup v0.2d, xzr\n"
    "dup v1.2d, xzr\n"
    "dup v2.2d, xzr\n"
    "dup v3.2d, xzr\n"
    "subs x10, x10, #16\n"
    "blt 2f\n"
    "1:\n"  // LoopD16
    "ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x8], #64\n"
    "ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x5], #64\n"
    "ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x6], #64\n"
    "ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x7], #64\n"
    "ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x9], #64\n"
    "fmla v0.4s, v4.4s, v28.4s\n"
    "fmla v1.4s, v16.4s, v28.4s\n"
    "fmla v2.4s, v20.4s, v28.4s\n"
    "fmla v3.4s, v24.4s, v28.4s\n"
    "fmla v0.4s, v5.4s, v29.4s\n"
    "fmla v1.4s, v17.4s, v29.4s\n"
    "fmla v2.4s, v21.4s, v29.4s\n"
    "fmla v3.4s, v25.4s, v29.4s\n"
    "fmla v0.4s, v6.4s, v30.4s\n"
    "fmla v1.4s, v18.4s, v30.4s\n"
    "fmla v2.4s, v22.4s, v30.4s\n"
    "fmla v3.4s, v26.4s, v30.4s\n"
    "fmla v0.4s, v7.4s, v31.4s\n"
    "fmla v1.4s, v19.4s, v31.4s\n"
    "fmla v2.4s, v23.4s, v31.4s\n"
    "fmla v3.4s, v27.4s, v31.4s\n"
    "subs x10, x10, #16\n"
    "bge 1b\n"
    "2:\n"  // LoopD12
    "adds x10, x10, #16\n"
    "cbz x10, 6f\n"
    "cmp x10, #12\n"
    "blt 3f\n"
    "ld1 {v4.4s, v5.4s, v6.4s}, [x8], #48\n"
    "ld1 {v16.4s, v17.4s, v18.4s}, [x5], #48\n"
    "ld1 {v20.4s, v21.4s, v22.4s}, [x6], #48\n"
    "ld1 {v24.4s, v25.4s, v26.4s}, [x7], #48\n"
    "ld1 {v28.4s, v29.4s, v30.4s}, [x9], #48\n"
    "fmla v0.4s, v4.4s, v28.4s\n"
    "fmla v1.4s, v16.4s, v28.4s\n"
    "fmla v2.4s, v20.4s, v28.4s\n"
    "fmla v3.4s, v24.4s, v28.4s\n"
    "fmla v0.4s, v5.4s, v29.4s\n"
    "fmla v1.4s, v17.4s, v29.4s\n"
    "fmla v2.4s, v21.4s, v29.4s\n"
    "fmla v3.4s, v25.4s, v29.4s\n"
    "fmla v0.4s, v6.4s, v30.4s\n"
    "fmla v1.4s, v18.4s, v30.4s\n"
    "fmla v2.4s, v22.4s, v30.4s\n"
    "fmla v3.4s, v26.4s, v30.4s\n"
    "sub x10, x10, #12\n"
    "b 7f\n"
    "3:\n"  // LoopD8
    "cmp x10, #8\n"
    "blt 4f\n"
    "ld1 {v4.4s, v5.4s}, [x8], #32\n"
    "ld1 {v16.4s, v17.4s}, [x5], #32\n"
    "ld1 {v20.4s, v21.4s}, [x6], #32\n"
    "ld1 {v24.4s, v25.4s}, [x7], #32\n"
    "ld1 {v28.4s, v29.4s}, [x9], #32\n"
    "fmla v0.4s, v4.4s, v28.4s\n"
    "fmla v1.4s, v16.4s, v28.4s\n"
    "fmla v2.4s, v20.4s, v28.4s\n"
    "fmla v3.4s, v24.4s, v28.4s\n"
    "fmla v0.4s, v5.4s, v29.4s\n"
    "fmla v1.4s, v17.4s, v29.4s\n"
    "fmla v2.4s, v21.4s, v29.4s\n"
    "fmla v3.4s, v25.4s, v29.4s\n"
    "sub x10, x10, #8\n"
    "b 7f\n"
    "4:\n"  // LoopD4
    "cmp x10, #4\n"
    "blt 7f\n"
    "ld1 {v4.4s}, [x8], #16\n"
    "ld1 {v16.4s}, [x5], #16\n"
    "ld1 {v20.4s}, [x6], #16\n"
    "ld1 {v24.4s}, [x7], #16\n"
    "ld1 {v28.4s}, [x9], #16\n"
    "fmla v0.4s, v4.4s, v28.4s\n"
    "fmla v1.4s, v16.4s, v28.4s\n"
    "fmla v2.4s, v20.4s, v28.4s\n"
    "fmla v3.4s, v24.4s, v28.4s\n"
    "sub x10, x10, #4\n"
    "7:\n"
    "cbz x10, 6f\n"
    "dup v4.2d, xzr\n"
    "dup v16.2d, xzr\n"
    "dup v20.2d, xzr\n"
    "dup v24.2d, xzr\n"
    "dup v28.2d, xzr\n"
    "subs x10, x10, #2\n"
    "blt 5f\n"
    "ld1 {v4.d}[0], [x8], #8\n"  // LoopD2
    "ld1 {v16.d}[0], [x5], #8\n"
    "ld1 {v20.d}[0], [x6], #8\n"
    "ld1 {v24.d}[0], [x7], #8\n"
    "ld1 {v28.d}[0], [x9], #8\n"
    "cbz x10, 8f\n"
    "5:\n"  // LoopD1
    "ld1 {v4.s}[2], [x8]\n"
    "ld1 {v16.s}[2], [x5]\n"
    "ld1 {v20.s}[2], [x6]\n"
    "ld1 {v24.s}[2], [x7]\n"
    "ld1 {v28.s}[2], [x9]\n"
    "8:\n"
    "fmla v0.4s, v4.4s, v28.4s\n"
    "fmla v1.4s, v16.4s, v28.4s\n"
    "fmla v2.4s, v20.4s, v28.4s\n"
    "fmla v3.4s, v24.4s, v28.4s\n"
    "6:\n"
    "faddp v4.4s, v0.4s, v1.4s\n"
    "faddp v5.4s, v2.4s, v3.4s\n"
    "faddp v0.4s, v4.4s, v5.4s\n"
    "cbz %[bias], 9f\n"
    "ld1r {v1.4s}, [%[bias]]\n"
    "fadd v0.4s, v0.4s, v1.4s\n"
    "9:\n"
    "cbz %[act], 10f\n"
    "dup v1.2d, xzr\n"
    "fmax v0.4s, v0.4s, v1.4s\n"
    "cmp %[act], #3\n"
    "bne 10f\n"
    "movi v1.4s, #6\n"
    "scvtf v1.4s, v1.4s\n"
    "fmin v0.4s, v0.4s, v1.4s\n"
    "10:\n"
    "st1 {v0.4s}, [%[output]]\n"

    :
    : [ input ] "r"(input), [ weight ] "r"(weight), [ output ] "r"(output), [ bias ] "r"(bias), [ deep ] "r"(deep),
      [ act ] "r"(act_type)
    : "cc", "x5", "x6", "x7", "x8", "x9", "x10", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18",
      "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}

void MatMul2x1Kernel(const float *input, const float *weight, float *output, const float *bias, size_t deep,
                     size_t act_type) {
  // 1: LoopD16, 2: LoopD12, 3: LoopD8, 4: LoopD4, 5: LoopD1, 6: LoopDEnd, 7: LoopDTail, 8: LoopDTailCompute
  // 9: WriteBack
  asm volatile(
    "mov x8, %[input]\n"
    "mov x9, %[weight]\n"
    "mov x10, %[deep]\n"
    "add x5, %[input], %[deep], LSL #2\n"
    "dup v0.2d, xzr\n"
    "dup v1.2d, xzr\n"
    "subs x10, x10, #16\n"
    "blt 2f\n"
    "1:\n"  // LoopD16
    "ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x8], #64\n"
    "ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x5], #64\n"
    "ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x9], #64\n"
    "fmla v0.4s, v4.4s, v28.4s\n"
    "fmla v1.4s, v16.4s, v28.4s\n"
    "fmla v0.4s, v5.4s, v29.4s\n"
    "fmla v1.4s, v17.4s, v29.4s\n"
    "fmla v0.4s, v6.4s, v30.4s\n"
    "fmla v1.4s, v18.4s, v30.4s\n"
    "fmla v0.4s, v7.4s, v31.4s\n"
    "fmla v1.4s, v19.4s, v31.4s\n"
    "subs x10, x10, #16\n"
    "bge 1b\n"
    "2:\n"  // LoopD12
    "adds x10, x10, #16\n"
    "cbz x10, 6f\n"
    "cmp x10, #12\n"
    "blt 3f\n"
    "ld1 {v4.4s, v5.4s, v6.4s}, [x8], #48\n"
    "ld1 {v16.4s, v17.4s, v18.4s}, [x5], #48\n"
    "ld1 {v28.4s, v29.4s, v30.4s}, [x9], #48\n"
    "fmla v0.4s, v4.4s, v28.4s\n"
    "fmla v1.4s, v16.4s, v28.4s\n"
    "fmla v0.4s, v5.4s, v29.4s\n"
    "fmla v1.4s, v17.4s, v29.4s\n"
    "fmla v0.4s, v6.4s, v30.4s\n"
    "fmla v1.4s, v18.4s, v30.4s\n"
    "sub x10, x10, #12\n"
    "b 7f\n"
    "3:\n"  // LoopD8
    "cmp x10, #8\n"
    "blt 4f\n"
    "ld1 {v4.4s, v5.4s}, [x8], #32\n"
    "ld1 {v16.4s, v17.4s}, [x5], #32\n"
    "ld1 {v28.4s, v29.4s}, [x9], #32\n"
    "fmla v0.4s, v4.4s, v28.4s\n"
    "fmla v1.4s, v16.4s, v28.4s\n"
    "fmla v0.4s, v5.4s, v29.4s\n"
    "fmla v1.4s, v17.4s, v29.4s\n"
    "sub x10, x10, #8\n"
    "b 7f\n"
    "4:\n"  // LoopD4
    "cmp x10, #4\n"
    "blt 7f\n"
    "ld1 {v4.4s}, [x8], #16\n"
    "ld1 {v16.4s}, [x5], #16\n"
    "ld1 {v28.4s}, [x9], #16\n"
    "fmla v0.4s, v4.4s, v28.4s\n"
    "fmla v1.4s, v16.4s, v28.4s\n"
    "sub x10, x10, #4\n"
    "7:\n"
    "cbz x10, 6f\n"
    "dup v4.2d, xzr\n"
    "dup v16.2d, xzr\n"
    "subs x10, x10, #2\n"
    "blt 5f\n"
    "ld1 {v4.d}[0], [x8], #8\n"  // LoopD2
    "ld1 {v16.d}[0], [x5], #8\n"
    "ld1 {v28.d}[0], [x9], #8\n"
    "cbz x10, 8f\n"
    "5:\n"  // LoopD1
    "ld1 {v4.s}[2], [x8]\n"
    "ld1 {v16.s}[2], [x5]\n"
    "ld1 {v28.s}[2], [x9]\n"
    "8:\n"
    "fmla v0.4s, v4.4s, v28.4s\n"
    "fmla v1.4s, v16.4s, v28.4s\n"
    "6:\n"
    "faddp v4.4s, v0.4s, v1.4s\n"
    "faddp v0.4s, v4.4s, v4.4s\n"
    "cbz %[bias], 9f\n"
    "ld1r {v1.4s}, [%[bias]]\n"
    "fadd v0.2s, v0.2s, v1.2s\n"
    "9:\n"
    "cbz %[act], 10f\n"
    "fmov d1, xzr\n"
    "fmax v0.2s, v0.2s, v1.2s\n"
    "cmp %[act], #3\n"
    "bne 10f\n"
    "movi v1.2s, #6\n"
    "scvtf v1.2s, v1.2s\n"
    "fmin v0.2s, v0.2s, v1.2s\n"
    "10:\n"
    "st1 {v0.2s}, [%[output]]\n"

    :
    : [ input ] "r"(input), [ weight ] "r"(weight), [ output ] "r"(output), [ bias ] "r"(bias), [ deep ] "r"(deep),
      [ act ] "r"(act_type)
    : "cc", "x5", "x8", "x9", "x10", "v0", "v1", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v28", "v29",
      "v30", "v31", "memory");
}

void MatMul1x1Kernel(const float *input, const float *weight, float *output, const float *bias, size_t deep,
                     size_t act_type) {
  // 1: LoopD16, 2: LoopD12, 3: LoopD8, 4: LoopD4, 5: LoopD1, 6: LoopDEnd, 7: LoopDTail, 8: LoopDTailCompute
  // 9: WriteBack
  asm volatile(
    "mov x8, %[input]\n"
    "mov x9, %[weight]\n"
    "mov x10, %[deep]\n"
    "dup v0.2d, xzr\n"
    "subs x10, x10, #16\n"
    "blt 2f\n"
    "1:\n"  // LoopD16
    "ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x8], #64\n"
    "ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x9], #64\n"
    "fmla v0.4s, v4.4s, v28.4s\n"
    "fmla v0.4s, v5.4s, v29.4s\n"
    "fmla v0.4s, v6.4s, v30.4s\n"
    "fmla v0.4s, v7.4s, v31.4s\n"
    "subs x10, x10, #16\n"
    "bge 1b\n"
    "2:\n"  // LoopD12
    "adds x10, x10, #16\n"
    "cbz x10, 6f\n"
    "cmp x10, #12\n"
    "blt 3f\n"
    "ld1 {v4.4s, v5.4s, v6.4s}, [x8], #48\n"
    "ld1 {v28.4s, v29.4s, v30.4s}, [x9], #48\n"
    "fmla v0.4s, v4.4s, v28.4s\n"
    "fmla v0.4s, v5.4s, v29.4s\n"
    "fmla v0.4s, v6.4s, v30.4s\n"
    "sub x10, x10, #12\n"
    "b 7f\n"
    "3:\n"  // LoopD8
    "cmp x10, #8\n"
    "blt 4f\n"
    "ld1 {v4.4s, v5.4s}, [x8], #32\n"
    "ld1 {v28.4s, v29.4s}, [x9], #32\n"
    "fmla v0.4s, v4.4s, v28.4s\n"
    "fmla v0.4s, v5.4s, v29.4s\n"
    "sub x10, x10, #8\n"
    "b 7f\n"
    "4:\n"  // LoopD4
    "cmp x10, #4\n"
    "blt 7f\n"
    "ld1 {v4.4s}, [x8], #16\n"
    "ld1 {v28.4s}, [x9], #16\n"
    "fmla v0.4s, v4.4s, v28.4s\n"
    "sub x10, x10, #4\n"
    "7:\n"
    "cbz x10, 6f\n"
    "dup v4.2d, xzr\n"
    "subs x10, x10, #2\n"
    "blt 5f\n"
    "ld1 {v4.d}[0], [x8], #8\n"  // LoopD2
    "ld1 {v28.d}[0], [x9], #8\n"
    "cbz x10, 8f\n"
    "5:\n"  // LoopD1
    "ld1 {v4.s}[3], [x8]\n"
    "ld1 {v28.s}[3], [x9]\n"
    "8:\n"
    "fmla v0.4s, v4.4s, v28.4s\n"
    "6:\n"
    "faddp v4.4s, v0.4s, v0.4s\n"
    "faddp v0.4s, v4.4s, v4.4s\n"
    "cbz %[bias], 9f\n"
    "ld1 {v1.s}[0], [%[bias]]\n"
    "fadd s0, s0, s1\n"
    "9:\n"
    "cbz %[act], 10f\n"
    "fmov s1, wzr\n"
    "fmax s0, s0, s1\n"
    "cmp %[act], #3\n"
    "bne 10f\n"
    "mov x10, #6\n"
    "scvtf s1, x10\n"
    "fmin s0, s0, s1\n"
    "10:\n"
    "str s0, [%[output]]\n"

    :
    : [ input ] "r"(input), [ weight ] "r"(weight), [ output ] "r"(output), [ bias ] "r"(bias), [ deep ] "r"(deep),
      [ act ] "r"(act_type)
    : "cc", "x8", "x9", "x10", "v0", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v28", "v29", "v30", "v31");
}

void GemmIsNotPackByRow(const float *a, const float *b, float *c, const float *bias, int start_row, int end_row,
                        int deep, int act_type) {
  const float *input = a + start_row * deep;
  float *output = c + start_row;
  const int step = C4NUM * deep;
  for (; start_row <= end_row - C4NUM; start_row += C4NUM) {
    MatMul4x1Kernel(input, b, output, bias, deep, act_type);
    input += step;
    output += C4NUM;
  }
  for (; start_row <= end_row - C2NUM; start_row += C2NUM) {
    MatMul2x1Kernel(input, b, output, bias, deep, act_type);
    input += C2NUM * deep;
    output += C2NUM;
  }
  if (start_row == end_row - 1) {
    MatMul1x1Kernel(input, b, output, bias, deep, act_type);
  }
}
#endif
