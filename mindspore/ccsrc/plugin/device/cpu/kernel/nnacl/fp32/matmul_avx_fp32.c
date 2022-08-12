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
#include "nnacl/fp32/matmul_avx_fp32.h"
#include "nnacl/intrinsics/ms_simd_avx_instructions.h"

void MatVecMulAvxFp32(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int cur_col,
                      int col_align) {
  // one time process 32 out_channel
  int col_block = C32NUM;
  int act_flag = C0NUM;
  if (act_type == ActType_Relu6) {
    act_flag += C1NUM;
  }
  if (act_type == ActType_Relu || act_type == ActType_Relu6) {
    act_flag += C2NUM;
  }
  MatVecMulKernel kernel[4] = {MatVecMul1x8Kernel, MatVecMul1x16Kernel, MatVecMul1x24Kernel, MatVecMul1x32Kernel};
  const float *bias_data = bias;
  for (int col_index = 0; col_index < cur_col; col_index += col_block) {
    col_block = cur_col - col_index < col_block ? cur_col - col_index : col_block;
    kernel[(col_block >> C3NUM) - 1](c + col_index, a, b + col_index * depth, bias_data, act_flag, 1,
                                     col_block >> C3NUM, col_align, depth);
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
  for (int i = 0; i < C4NUM; ++i) {
    weight_data[i] = _mm256_set1_ps(0.0f);
  }
  for (int i = 0; i < row_block; ++i) {
    if (bias != NULL) {
      for (int j = 0; j < col_block; ++j) {
        dst_data[i * col_block + j] = _mm256_loadu_ps(bias + j * C8NUM);
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
