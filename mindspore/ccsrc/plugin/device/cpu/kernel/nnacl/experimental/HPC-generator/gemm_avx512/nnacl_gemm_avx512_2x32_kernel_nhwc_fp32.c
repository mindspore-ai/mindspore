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
#include <x86intrin.h>
#include "nnacl/fp32/matmul_avx512_fp32.h"

// nnacl gemm in x86 avx512 asm code
void nnacl_gemm_avx512_2x32_kernel_nhwc_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                             const size_t act_flag, const size_t row_block, const size_t col_block,
                                             const size_t depth, const size_t src_stride, const size_t dst_stride,
                                             const size_t inc_flag) {
  size_t dst_stride_t = dst_stride << 2;
  asm volatile(
    // inc in depth
    "movq %[inc_flag], %%rax\n"
    "and $0x1, %%rax\n"
    "je 0f\n"
    "vmovups 0(%[dst_0]), %%zmm0\n"
    "vmovups 64(%[dst_0]), %%zmm1\n"
    "vmovups 0(%[dst_0], %[dst_stride], 1), %%zmm2\n"
    "vmovups 64(%[dst_0], %[dst_stride], 1), %%zmm3\n"
    "jmp 2f\n"
    ".align 16\n"
    "0:\n"
    "cmpq $0, %[bias]\n"
    "je 1f\n"
    "vmovups 0(%[bias]), %%zmm0\n"
    "vmovups 64(%[bias]), %%zmm1\n"
    "vmovups 0(%[bias]), %%zmm2\n"
    "vmovups 64(%[bias]), %%zmm3\n"
    "jmp 2f\n"
    ".align 16\n"
    "1:\n"
    "vxorps %%zmm0, %%zmm0, %%zmm0\n"
    "vxorps %%zmm1, %%zmm1, %%zmm1\n"
    "vxorps %%zmm2, %%zmm2, %%zmm2\n"
    "vxorps %%zmm3, %%zmm3, %%zmm3\n"
    ".align 16\n"
    "2:\n"
    :
    : [ dst_0 ] "r"(dst), [ bias ] "r"(bias), [ dst_stride ] "r"(dst_stride_t), [ inc_flag ] "r"(inc_flag)
    : "%zmm0", "%zmm1", "%zmm2", "%zmm3");
  size_t src_stride_t = src_stride << 2;
  asm volatile(
    "cmp $16, %[depth]\n"
    "jb 1f\n"
    ".align 16\n"
    "0:\n"
    // block 0
    "vmovups 0(%[weight]), %%zmm31\n"
    "vmovups 64(%[weight]), %%zmm30\n"
    "vbroadcastss 0(%[src_0]), %%zmm29\n"
    "vbroadcastss 0(%[src_0], %[src_stride], 1), %%zmm28\n"
    "vfmadd231ps %%zmm31, %%zmm29, %%zmm0\n"
    "vfmadd231ps %%zmm30, %%zmm29, %%zmm1\n"
    "vfmadd231ps %%zmm31, %%zmm28, %%zmm2\n"
    "vfmadd231ps %%zmm30, %%zmm28, %%zmm3\n"
    // block 1
    "vmovups 128(%[weight]), %%zmm31\n"
    "vmovups 192(%[weight]), %%zmm30\n"
    "vbroadcastss 4(%[src_0]), %%zmm29\n"
    "vbroadcastss 4(%[src_0], %[src_stride], 1), %%zmm28\n"
    "vfmadd231ps %%zmm31, %%zmm29, %%zmm0\n"
    "vfmadd231ps %%zmm30, %%zmm29, %%zmm1\n"
    "vfmadd231ps %%zmm31, %%zmm28, %%zmm2\n"
    "vfmadd231ps %%zmm30, %%zmm28, %%zmm3\n"
    // block 2
    "vmovups 256(%[weight]), %%zmm31\n"
    "vmovups 320(%[weight]), %%zmm30\n"
    "vbroadcastss 8(%[src_0]), %%zmm29\n"
    "vbroadcastss 8(%[src_0], %[src_stride], 1), %%zmm28\n"
    "vfmadd231ps %%zmm31, %%zmm29, %%zmm0\n"
    "vfmadd231ps %%zmm30, %%zmm29, %%zmm1\n"
    "vfmadd231ps %%zmm31, %%zmm28, %%zmm2\n"
    "vfmadd231ps %%zmm30, %%zmm28, %%zmm3\n"
    // block 3
    "vmovups 384(%[weight]), %%zmm31\n"
    "vmovups 448(%[weight]), %%zmm30\n"
    "vbroadcastss 12(%[src_0]), %%zmm29\n"
    "vbroadcastss 12(%[src_0], %[src_stride], 1), %%zmm28\n"
    "vfmadd231ps %%zmm31, %%zmm29, %%zmm0\n"
    "vfmadd231ps %%zmm30, %%zmm29, %%zmm1\n"
    "vfmadd231ps %%zmm31, %%zmm28, %%zmm2\n"
    "vfmadd231ps %%zmm30, %%zmm28, %%zmm3\n"
    // block 4
    "vmovups 512(%[weight]), %%zmm31\n"
    "vmovups 576(%[weight]), %%zmm30\n"
    "vbroadcastss 16(%[src_0]), %%zmm29\n"
    "vbroadcastss 16(%[src_0], %[src_stride], 1), %%zmm28\n"
    "vfmadd231ps %%zmm31, %%zmm29, %%zmm0\n"
    "vfmadd231ps %%zmm30, %%zmm29, %%zmm1\n"
    "vfmadd231ps %%zmm31, %%zmm28, %%zmm2\n"
    "vfmadd231ps %%zmm30, %%zmm28, %%zmm3\n"
    // block 5
    "vmovups 640(%[weight]), %%zmm31\n"
    "vmovups 704(%[weight]), %%zmm30\n"
    "vbroadcastss 20(%[src_0]), %%zmm29\n"
    "vbroadcastss 20(%[src_0], %[src_stride], 1), %%zmm28\n"
    "vfmadd231ps %%zmm31, %%zmm29, %%zmm0\n"
    "vfmadd231ps %%zmm30, %%zmm29, %%zmm1\n"
    "vfmadd231ps %%zmm31, %%zmm28, %%zmm2\n"
    "vfmadd231ps %%zmm30, %%zmm28, %%zmm3\n"
    // block 6
    "vmovups 768(%[weight]), %%zmm31\n"
    "vmovups 832(%[weight]), %%zmm30\n"
    "vbroadcastss 24(%[src_0]), %%zmm29\n"
    "vbroadcastss 24(%[src_0], %[src_stride], 1), %%zmm28\n"
    "vfmadd231ps %%zmm31, %%zmm29, %%zmm0\n"
    "vfmadd231ps %%zmm30, %%zmm29, %%zmm1\n"
    "vfmadd231ps %%zmm31, %%zmm28, %%zmm2\n"
    "vfmadd231ps %%zmm30, %%zmm28, %%zmm3\n"
    // block 7
    "vmovups 896(%[weight]), %%zmm31\n"
    "vmovups 960(%[weight]), %%zmm30\n"
    "vbroadcastss 28(%[src_0]), %%zmm29\n"
    "vbroadcastss 28(%[src_0], %[src_stride], 1), %%zmm28\n"
    "vfmadd231ps %%zmm31, %%zmm29, %%zmm0\n"
    "vfmadd231ps %%zmm30, %%zmm29, %%zmm1\n"
    "vfmadd231ps %%zmm31, %%zmm28, %%zmm2\n"
    "vfmadd231ps %%zmm30, %%zmm28, %%zmm3\n"
    // block 8
    "vmovups 1024(%[weight]), %%zmm31\n"
    "vmovups 1088(%[weight]), %%zmm30\n"
    "vbroadcastss 32(%[src_0]), %%zmm29\n"
    "vbroadcastss 32(%[src_0], %[src_stride], 1), %%zmm28\n"
    "vfmadd231ps %%zmm31, %%zmm29, %%zmm0\n"
    "vfmadd231ps %%zmm30, %%zmm29, %%zmm1\n"
    "vfmadd231ps %%zmm31, %%zmm28, %%zmm2\n"
    "vfmadd231ps %%zmm30, %%zmm28, %%zmm3\n"
    // block 9
    "vmovups 1152(%[weight]), %%zmm31\n"
    "vmovups 1216(%[weight]), %%zmm30\n"
    "vbroadcastss 36(%[src_0]), %%zmm29\n"
    "vbroadcastss 36(%[src_0], %[src_stride], 1), %%zmm28\n"
    "vfmadd231ps %%zmm31, %%zmm29, %%zmm0\n"
    "vfmadd231ps %%zmm30, %%zmm29, %%zmm1\n"
    "vfmadd231ps %%zmm31, %%zmm28, %%zmm2\n"
    "vfmadd231ps %%zmm30, %%zmm28, %%zmm3\n"
    // block 10
    "vmovups 1280(%[weight]), %%zmm31\n"
    "vmovups 1344(%[weight]), %%zmm30\n"
    "vbroadcastss 40(%[src_0]), %%zmm29\n"
    "vbroadcastss 40(%[src_0], %[src_stride], 1), %%zmm28\n"
    "vfmadd231ps %%zmm31, %%zmm29, %%zmm0\n"
    "vfmadd231ps %%zmm30, %%zmm29, %%zmm1\n"
    "vfmadd231ps %%zmm31, %%zmm28, %%zmm2\n"
    "vfmadd231ps %%zmm30, %%zmm28, %%zmm3\n"
    // block 11
    "vmovups 1408(%[weight]), %%zmm31\n"
    "vmovups 1472(%[weight]), %%zmm30\n"
    "vbroadcastss 44(%[src_0]), %%zmm29\n"
    "vbroadcastss 44(%[src_0], %[src_stride], 1), %%zmm28\n"
    "vfmadd231ps %%zmm31, %%zmm29, %%zmm0\n"
    "vfmadd231ps %%zmm30, %%zmm29, %%zmm1\n"
    "vfmadd231ps %%zmm31, %%zmm28, %%zmm2\n"
    "vfmadd231ps %%zmm30, %%zmm28, %%zmm3\n"
    // block 12
    "vmovups 1536(%[weight]), %%zmm31\n"
    "vmovups 1600(%[weight]), %%zmm30\n"
    "vbroadcastss 48(%[src_0]), %%zmm29\n"
    "vbroadcastss 48(%[src_0], %[src_stride], 1), %%zmm28\n"
    "vfmadd231ps %%zmm31, %%zmm29, %%zmm0\n"
    "vfmadd231ps %%zmm30, %%zmm29, %%zmm1\n"
    "vfmadd231ps %%zmm31, %%zmm28, %%zmm2\n"
    "vfmadd231ps %%zmm30, %%zmm28, %%zmm3\n"
    // block 13
    "vmovups 1664(%[weight]), %%zmm31\n"
    "vmovups 1728(%[weight]), %%zmm30\n"
    "vbroadcastss 52(%[src_0]), %%zmm29\n"
    "vbroadcastss 52(%[src_0], %[src_stride], 1), %%zmm28\n"
    "vfmadd231ps %%zmm31, %%zmm29, %%zmm0\n"
    "vfmadd231ps %%zmm30, %%zmm29, %%zmm1\n"
    "vfmadd231ps %%zmm31, %%zmm28, %%zmm2\n"
    "vfmadd231ps %%zmm30, %%zmm28, %%zmm3\n"
    // block 14
    "vmovups 1792(%[weight]), %%zmm31\n"
    "vmovups 1856(%[weight]), %%zmm30\n"
    "vbroadcastss 56(%[src_0]), %%zmm29\n"
    "vbroadcastss 56(%[src_0], %[src_stride], 1), %%zmm28\n"
    "vfmadd231ps %%zmm31, %%zmm29, %%zmm0\n"
    "vfmadd231ps %%zmm30, %%zmm29, %%zmm1\n"
    "vfmadd231ps %%zmm31, %%zmm28, %%zmm2\n"
    "vfmadd231ps %%zmm30, %%zmm28, %%zmm3\n"
    // block 15
    "vmovups 1920(%[weight]), %%zmm31\n"
    "vmovups 1984(%[weight]), %%zmm30\n"
    "vbroadcastss 60(%[src_0]), %%zmm29\n"
    "vbroadcastss 60(%[src_0], %[src_stride], 1), %%zmm28\n"
    "vfmadd231ps %%zmm31, %%zmm29, %%zmm0\n"
    "vfmadd231ps %%zmm30, %%zmm29, %%zmm1\n"
    "vfmadd231ps %%zmm31, %%zmm28, %%zmm2\n"
    "vfmadd231ps %%zmm30, %%zmm28, %%zmm3\n"
    "add $2048, %[weight]\n"
    "add $64, %[src_0]\n"
    "sub $16, %[depth]\n"
    "cmp $16, %[depth]\n"
    "jge 0b\n"
    "cmp $0, %[depth]\n"
    "je 2f\n"
    ".align 16\n"
    "1:\n"
    // block 0
    "vmovups 0(%[weight]), %%zmm31\n"
    "vmovups 64(%[weight]), %%zmm30\n"
    "vbroadcastss 0(%[src_0]), %%zmm29\n"
    "vbroadcastss 0(%[src_0], %[src_stride], 1), %%zmm28\n"
    "vfmadd231ps %%zmm31, %%zmm29, %%zmm0\n"
    "vfmadd231ps %%zmm30, %%zmm29, %%zmm1\n"
    "vfmadd231ps %%zmm31, %%zmm28, %%zmm2\n"
    "vfmadd231ps %%zmm30, %%zmm28, %%zmm3\n"
    "add $128, %[weight]\n"
    "add $4, %[src_0]\n"
    "dec %[depth]\n"
    "jg 1b\n"
    ".align 16\n"
    "2:\n"
    "and $0x2, %[inc_flag]\n"
    "je 3f\n"
    "and $0x3, %[act_flag]\n"
    "je 3f\n"
    // relu
    "vxorps %%zmm31, %%zmm31, %%zmm31\n"
    "vmaxps %%zmm0, %%zmm31, %%zmm0\n"
    "vmaxps %%zmm1, %%zmm31, %%zmm1\n"
    "vmaxps %%zmm2, %%zmm31, %%zmm2\n"
    "vmaxps %%zmm3, %%zmm31, %%zmm3\n"
    "and $0x1, %[act_flag]\n"
    "je 3f\n"
    // relu6
    "mov $0x40C00000, %%eax\n"
    "vmovd %%eax, %%xmm30\n"
    "vbroadcastss %%xmm30, %%zmm30\n"
    "vminps %%zmm0, %%zmm30, %%zmm0\n"
    "vminps %%zmm1, %%zmm30, %%zmm1\n"
    "vminps %%zmm2, %%zmm30, %%zmm2\n"
    "vminps %%zmm3, %%zmm30, %%zmm3\n"
    ".align 16\n"
    "3:\n"
    "vmovups %%zmm0, 0(%[dst_0])\n"
    "vmovups %%zmm1, 64(%[dst_0])\n"
    "vmovups %%zmm2, 0(%[dst_0], %[dst_stride], 1)\n"
    "vmovups %%zmm3, 64(%[dst_0], %[dst_stride], 1)\n"
    :
    : [ src_0 ] "r"(src), [ src_stride ] "r"(src_stride_t), [ weight ] "r"(weight), [ depth ] "r"(depth),
      [ inc_flag ] "r"(inc_flag), [ act_flag ] "r"(act_flag), [ dst_0 ] "r"(dst), [ dst_stride ] "r"(dst_stride_t)
    : "%rax", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10",
      "%zmm11", "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16", "%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21",
      "%zmm22", "%zmm23", "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30", "%zmm31");
}
