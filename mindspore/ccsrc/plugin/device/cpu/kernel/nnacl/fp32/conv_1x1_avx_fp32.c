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
#include "nnacl/fp32/conv_1x1_avx_fp32.h"
#include "nnacl/intrinsics/ms_simd_avx_instructions.h"

void Conv1x1SW3x32AVXKernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                            size_t ow_block, size_t oc_block, size_t oc_align, size_t ic_align, size_t in_sw_step,
                            size_t dst_flag) {
  asm volatile(
    "movq %8, %%rax\n"
    "and $0x1, %%eax\n"
    "je 0f\n"
    "vmovups (%7), %%ymm0\n"
    "vmovups 0x20(%7), %%ymm1\n"
    "vmovups 0x40(%7), %%ymm2\n"
    "vmovups 0x60(%7), %%ymm3\n"
    "vmovups (%7, %6, 1), %%ymm4\n"
    "vmovups 0x20(%7, %6, 1), %%ymm5\n"
    "vmovups 0x40(%7, %6, 1), %%ymm6\n"
    "vmovups 0x60(%7, %6, 1), %%ymm7\n"
    "vmovups (%7, %6, 2), %%ymm8\n"
    "vmovups 0x20(%7, %6, 2), %%ymm9\n"
    "vmovups 0x40(%7, %6, 2), %%ymm10\n"
    "vmovups 0x60(%7, %6, 2), %%ymm11\n"
    "jmp 2f\n"
    "0:\n"
    "cmpq $0, %2\n"
    "je 1f\n"
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
    "jmp 2f\n"
    "1:\n"
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

    "2:\n"  // LoopIC
    "vbroadcastss (%0), %%ymm13\n"
    "vbroadcastss (%0, %4), %%ymm14\n"
    "vbroadcastss (%0, %4, 2), %%ymm15\n"
    "vmovups (%1), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm8\n"
    "vmovups 0x20(%1), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm1\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm5\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm9\n"
    "vmovups 0x40(%1), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm2\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm6\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm10\n"
    "vmovups 0x60(%1), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm3\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm7\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm11\n"

    "vbroadcastss 4(%0), %%ymm13\n"
    "vbroadcastss 4(%0, %4), %%ymm14\n"
    "vbroadcastss 4(%0, %4, 2), %%ymm15\n"
    "vmovups 128(%1), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm8\n"
    "vmovups 160(%1), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm1\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm5\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm9\n"
    "vmovups 192(%1), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm2\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm6\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm10\n"
    "vmovups 224(%1), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm3\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm7\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm11\n"

    "vbroadcastss 8(%0), %%ymm13\n"
    "vbroadcastss 8(%0, %4), %%ymm14\n"
    "vbroadcastss 8(%0, %4, 2), %%ymm15\n"
    "vmovups 256(%1), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm8\n"
    "vmovups 288(%1), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm1\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm5\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm9\n"
    "vmovups 320(%1), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm2\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm6\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm10\n"
    "vmovups 352(%1), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm3\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm7\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm11\n"

    "vbroadcastss 12(%0), %%ymm13\n"
    "vbroadcastss 12(%0, %4), %%ymm14\n"
    "vbroadcastss 12(%0, %4, 2), %%ymm15\n"
    "vmovups 384(%1), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm8\n"
    "vmovups 416(%1), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm1\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm5\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm9\n"
    "vmovups 448(%1), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm2\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm6\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm10\n"
    "vmovups 480(%1), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm3\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm7\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm11\n"

    "vbroadcastss 16(%0), %%ymm13\n"
    "vbroadcastss 16(%0, %4), %%ymm14\n"
    "vbroadcastss 16(%0, %4, 2), %%ymm15\n"
    "vmovups 512(%1), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm8\n"
    "vmovups 544(%1), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm1\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm5\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm9\n"
    "vmovups 576(%1), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm2\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm6\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm10\n"
    "vmovups 608(%1), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm3\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm7\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm11\n"

    "vbroadcastss 20(%0), %%ymm13\n"
    "vbroadcastss 20(%0, %4), %%ymm14\n"
    "vbroadcastss 20(%0, %4, 2), %%ymm15\n"
    "vmovups 640(%1), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm8\n"
    "vmovups 672(%1), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm1\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm5\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm9\n"
    "vmovups 704(%1), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm2\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm6\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm10\n"
    "vmovups 736(%1), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm3\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm7\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm11\n"

    "vbroadcastss 24(%0), %%ymm13\n"
    "vbroadcastss 24(%0, %4), %%ymm14\n"
    "vbroadcastss 24(%0, %4, 2), %%ymm15\n"
    "vmovups 768(%1), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm8\n"
    "vmovups 800(%1), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm1\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm5\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm9\n"
    "vmovups 832(%1), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm2\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm6\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm10\n"
    "vmovups 864(%1), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm3\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm7\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm11\n"

    "vbroadcastss 28(%0), %%ymm13\n"
    "vbroadcastss 28(%0, %4), %%ymm14\n"
    "vbroadcastss 28(%0, %4, 2), %%ymm15\n"
    "vmovups 896(%1), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm8\n"
    "vmovups 928(%1), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm1\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm5\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm9\n"
    "vmovups 960(%1), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm2\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm6\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm10\n"
    "vmovups 992(%1), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm3\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm7\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm11\n"

    "addq $1024, %1\n"
    "addq $32, %0\n"
    "dec %3\n"
    "jg 2b\n"

    "movq %8, %%rax\n"
    "and $0x2, %%eax\n"
    "je 3f\n"
    "movq %5, %%rax\n"
    "and $0x3, %%eax\n"
    "je 3f\n"
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
    "je 3f\n"
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

    "3:\n"
    "vmovups %%ymm0, (%7)\n"  // dst_0
    "vmovups %%ymm1, 0x20(%7)\n"
    "vmovups %%ymm2, 0x40(%7)\n"
    "vmovups %%ymm3, 0x60(%7)\n"
    "vmovups %%ymm4, (%7, %6, 1)\n"
    "vmovups %%ymm5, 0x20(%7, %6, 1)\n"
    "vmovups %%ymm6, 0x40(%7, %6, 1)\n"
    "vmovups %%ymm7, 0x60(%7, %6, 1)\n"
    "vmovups %%ymm8, (%7, %6, 2)\n"
    "vmovups %%ymm9, 0x20(%7, %6, 2)\n"
    "vmovups %%ymm10, 0x40(%7, %6, 2)\n"
    "vmovups %%ymm11, 0x60(%7, %6, 2)\n"
    :
    : "r"(src), "r"(weight), "r"(bias), "r"(ic_align), "r"(in_sw_step), "r"(act_flag), "r"(oc_align), "r"(dst),
      "r"(dst_flag)  // 8
    : "%rax", "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9",
      "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15");
}

void Conv1x1SW1x32AVXKernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                            size_t ow_block, size_t oc_block, size_t oc_align, size_t ic_align, size_t in_sw_step,
                            size_t dst_flag) {
  asm volatile(
    "movq %8, %%rax\n"
    "and $0x1, %%eax\n"
    "je 0f\n"
    "vmovups (%7), %%ymm0\n"
    "vmovups 0x20(%7), %%ymm1\n"
    "vmovups 0x40(%7), %%ymm2\n"
    "vmovups 0x60(%7), %%ymm3\n"
    "jmp 2f\n"
    "0:\n"
    "cmpq $0, %2\n"
    "je 1f\n"
    "vmovups (%2), %%ymm0\n"
    "vmovups 0x20(%2), %%ymm1\n"
    "vmovups 0x40(%2), %%ymm2\n"
    "vmovups 0x60(%2), %%ymm3\n"
    "jmp 2f\n"
    "1:\n"
    "vxorps %%ymm0, %%ymm0, %%ymm0\n"
    "vxorps %%ymm1, %%ymm1, %%ymm1\n"
    "vxorps %%ymm2, %%ymm2, %%ymm2\n"
    "vxorps %%ymm3, %%ymm3, %%ymm3\n"

    "2:\n"  // LoopIC
    "vbroadcastss (%0), %%ymm13\n"
    "vmovups (%1), %%ymm4\n"
    "vmovups 0x20(%1), %%ymm5\n"
    "vmovups 0x40(%1), %%ymm6\n"
    "vmovups 0x60(%1), %%ymm7\n"
    "vfmadd231ps %%ymm4, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm5, %%ymm13, %%ymm1\n"
    "vfmadd231ps %%ymm6, %%ymm13, %%ymm2\n"
    "vfmadd231ps %%ymm7, %%ymm13, %%ymm3\n"

    "vbroadcastss 4(%0), %%ymm13\n"
    "vmovups 128(%1), %%ymm4\n"
    "vmovups 160(%1), %%ymm5\n"
    "vmovups 192(%1), %%ymm6\n"
    "vmovups 224(%1), %%ymm7\n"
    "vfmadd231ps %%ymm4, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm5, %%ymm13, %%ymm1\n"
    "vfmadd231ps %%ymm6, %%ymm13, %%ymm2\n"
    "vfmadd231ps %%ymm7, %%ymm13, %%ymm3\n"

    "vbroadcastss 8(%0), %%ymm13\n"
    "vmovups 256(%1), %%ymm4\n"
    "vmovups 288(%1), %%ymm5\n"
    "vmovups 320(%1), %%ymm6\n"
    "vmovups 352(%1), %%ymm7\n"
    "vfmadd231ps %%ymm4, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm5, %%ymm13, %%ymm1\n"
    "vfmadd231ps %%ymm6, %%ymm13, %%ymm2\n"
    "vfmadd231ps %%ymm7, %%ymm13, %%ymm3\n"

    "vbroadcastss 12(%0), %%ymm13\n"
    "vmovups 384(%1), %%ymm4\n"
    "vmovups 416(%1), %%ymm5\n"
    "vmovups 448(%1), %%ymm6\n"
    "vmovups 480(%1), %%ymm7\n"
    "vfmadd231ps %%ymm4, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm5, %%ymm13, %%ymm1\n"
    "vfmadd231ps %%ymm6, %%ymm13, %%ymm2\n"
    "vfmadd231ps %%ymm7, %%ymm13, %%ymm3\n"

    "vbroadcastss 16(%0), %%ymm13\n"
    "vmovups 512(%1), %%ymm4\n"
    "vmovups 544(%1), %%ymm5\n"
    "vmovups 576(%1), %%ymm6\n"
    "vmovups 608(%1), %%ymm7\n"
    "vfmadd231ps %%ymm4, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm5, %%ymm13, %%ymm1\n"
    "vfmadd231ps %%ymm6, %%ymm13, %%ymm2\n"
    "vfmadd231ps %%ymm7, %%ymm13, %%ymm3\n"

    "vbroadcastss 20(%0), %%ymm13\n"
    "vmovups 640(%1), %%ymm4\n"
    "vmovups 672(%1), %%ymm5\n"
    "vmovups 704(%1), %%ymm6\n"
    "vmovups 736(%1), %%ymm7\n"
    "vfmadd231ps %%ymm4, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm5, %%ymm13, %%ymm1\n"
    "vfmadd231ps %%ymm6, %%ymm13, %%ymm2\n"
    "vfmadd231ps %%ymm7, %%ymm13, %%ymm3\n"

    "vbroadcastss 24(%0), %%ymm13\n"
    "vmovups 768(%1), %%ymm4\n"
    "vmovups 800(%1), %%ymm5\n"
    "vmovups 832(%1), %%ymm6\n"
    "vmovups 864(%1), %%ymm7\n"
    "vfmadd231ps %%ymm4, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm5, %%ymm13, %%ymm1\n"
    "vfmadd231ps %%ymm6, %%ymm13, %%ymm2\n"
    "vfmadd231ps %%ymm7, %%ymm13, %%ymm3\n"

    "vbroadcastss 28(%0), %%ymm13\n"
    "vmovups 896(%1), %%ymm4\n"
    "vmovups 928(%1), %%ymm5\n"
    "vmovups 960(%1), %%ymm6\n"
    "vmovups 992(%1), %%ymm7\n"
    "vfmadd231ps %%ymm4, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm5, %%ymm13, %%ymm1\n"
    "vfmadd231ps %%ymm6, %%ymm13, %%ymm2\n"
    "vfmadd231ps %%ymm7, %%ymm13, %%ymm3\n"
    "addq $1024, %1\n"
    "addq $32, %0\n"
    "dec %3\n"
    "jg 2b\n"

    "movq %8, %%rax\n"
    "and $0x2, %%eax\n"
    "je 3f\n"
    "movq %5, %%rax\n"
    "and $0x3, %%eax\n"
    "je 3f\n"
    // Relu
    "vxorps %%ymm12, %%ymm12, %%ymm12\n"
    "vmaxps %%ymm12, %%ymm0, %%ymm0\n"
    "vmaxps %%ymm12, %%ymm1, %%ymm1\n"
    "vmaxps %%ymm12, %%ymm2, %%ymm2\n"
    "vmaxps %%ymm12, %%ymm3, %%ymm3\n"

    "and $0x1, %%eax\n"
    "je 3f\n"
    // relu6
    "mov $0x40C00000, %%ecx\n"
    "vmovd %%ecx, %%xmm14\n"
    "vpermps %%ymm14, %%ymm12, %%ymm14\n"
    "vminps %%ymm14, %%ymm0, %%ymm0\n"
    "vminps %%ymm14, %%ymm1, %%ymm1\n"
    "vminps %%ymm14, %%ymm2, %%ymm2\n"
    "vminps %%ymm14, %%ymm3, %%ymm3\n"

    "3:\n"
    "vmovups %%ymm0, (%7)\n"  // dst_0
    "vmovups %%ymm1, 0x20(%7)\n"
    "vmovups %%ymm2, 0x40(%7)\n"
    "vmovups %%ymm3, 0x60(%7)\n"
    :
    : "r"(src), "r"(weight), "r"(bias), "r"(ic_align), "r"(in_sw_step), "r"(act_flag), "r"(oc_align), "r"(dst),
      "r"(dst_flag)  // 8
    : "%rax", "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm12", "%ymm13",
      "%ymm14");
}

void Conv1x1SW4x24AVXKernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                            size_t ow_block, size_t oc_block, size_t oc_align, size_t ic_align, size_t in_sw_step,
                            size_t dst_flag) {
  size_t src_3_step = 3 * in_sw_step;
  float *dst_3 = dst + 3 * oc_align / sizeof(float);
  asm volatile(
    "movq %10, %%rax\n"  // dst_flag
    "and $0x1, %%eax\n"
    "je 0f\n"
    "vmovups (%8), %%ymm0\n"  // dst_0
    "vmovups 0x20(%8), %%ymm1\n"
    "vmovups 0x40(%8), %%ymm2\n"
    "vmovups (%8, %7, 1), %%ymm3\n"
    "vmovups 0x20(%8, %7, 1), %%ymm4\n"
    "vmovups 0x40(%8, %7, 1), %%ymm5\n"
    "vmovups (%8, %7, 2), %%ymm6\n"
    "vmovups 0x20(%8, %7, 2), %%ymm7\n"
    "vmovups 0x40(%8, %7, 2), %%ymm8\n"
    "vmovups (%9), %%ymm9\n"
    "vmovups 0x20(%9), %%ymm10\n"
    "vmovups 0x40(%9), %%ymm11\n"
    "jmp 2f\n"
    "0:\n"
    "cmpq $0, %2\n"
    "je 1f\n"
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
    "jmp 2f\n"
    "1:\n"
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

    "2:\n"  // LoopIC
    "vmovups (%1), %%ymm13\n"
    "vmovups 0x20(%1), %%ymm14\n"
    "vmovups 0x40(%1), %%ymm15\n"
    "vbroadcastss (%0), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm1\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm2\n"
    "vbroadcastss (%0, %4), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm3\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm5\n"
    "vbroadcastss (%0, %4, 2), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm6\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm7\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm8\n"
    "vbroadcastss (%0, %5), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm9\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm10\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm11\n"

    "vmovups 96(%1), %%ymm13\n"
    "vmovups 128(%1), %%ymm14\n"
    "vmovups 160(%1), %%ymm15\n"
    "vbroadcastss 4(%0), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm1\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm2\n"
    "vbroadcastss 4(%0, %4), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm3\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm5\n"
    "vbroadcastss 4(%0, %4, 2), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm6\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm7\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm8\n"
    "vbroadcastss 4(%0, %5), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm9\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm10\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm11\n"

    "vmovups 192(%1), %%ymm13\n"
    "vmovups 224(%1), %%ymm14\n"
    "vmovups 256(%1), %%ymm15\n"
    "vbroadcastss 8(%0), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm1\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm2\n"
    "vbroadcastss 8(%0, %4), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm3\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm5\n"
    "vbroadcastss 8(%0, %4, 2), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm6\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm7\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm8\n"
    "vbroadcastss 8(%0, %5), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm9\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm10\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm11\n"

    "vmovups 288(%1), %%ymm13\n"
    "vmovups 320(%1), %%ymm14\n"
    "vmovups 352(%1), %%ymm15\n"
    "vbroadcastss 12(%0), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm1\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm2\n"
    "vbroadcastss 12(%0, %4), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm3\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm5\n"
    "vbroadcastss 12(%0, %4, 2), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm6\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm7\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm8\n"
    "vbroadcastss 12(%0, %5), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm9\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm10\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm11\n"

    "vmovups 384(%1), %%ymm13\n"
    "vmovups 416(%1), %%ymm14\n"
    "vmovups 448(%1), %%ymm15\n"
    "vbroadcastss 16(%0), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm1\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm2\n"
    "vbroadcastss 16(%0, %4), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm3\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm5\n"
    "vbroadcastss 16(%0, %4, 2), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm6\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm7\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm8\n"
    "vbroadcastss 16(%0, %5), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm9\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm10\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm11\n"

    "vmovups 480(%1), %%ymm13\n"
    "vmovups 512(%1), %%ymm14\n"
    "vmovups 544(%1), %%ymm15\n"
    "vbroadcastss 20(%0), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm1\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm2\n"
    "vbroadcastss 20(%0, %4), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm3\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm5\n"
    "vbroadcastss 20(%0, %4, 2), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm6\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm7\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm8\n"
    "vbroadcastss 20(%0, %5), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm9\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm10\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm11\n"

    "vmovups 576(%1), %%ymm13\n"
    "vmovups 608(%1), %%ymm14\n"
    "vmovups 640(%1), %%ymm15\n"
    "vbroadcastss 24(%0), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm1\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm2\n"
    "vbroadcastss 24(%0, %4), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm3\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm5\n"
    "vbroadcastss 24(%0, %4, 2), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm6\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm7\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm8\n"
    "vbroadcastss 24(%0, %5), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm9\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm10\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm11\n"

    "vmovups 672(%1), %%ymm13\n"
    "vmovups 704(%1), %%ymm14\n"
    "vmovups 736(%1), %%ymm15\n"
    "vbroadcastss 28(%0), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm1\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm2\n"
    "vbroadcastss 28(%0, %4), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm3\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm5\n"
    "vbroadcastss 28(%0, %4, 2), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm6\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm7\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm8\n"
    "vbroadcastss 28(%0, %5), %%ymm12\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm9\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm10\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm11\n"

    "addq $768, %1\n"
    "addq $32, %0\n"
    "dec %3\n"
    "jg 2b\n"

    "movq %10, %%rax\n"
    "and $0x2, %%eax\n"
    "je 3f\n"
    "movq %6, %%rax\n"
    "and $0x3, %%eax\n"
    "je 3f\n"
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
    "je 3f\n"
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

    "3:\n"
    "vmovups %%ymm0, (%8)\n"  // dst_0
    "vmovups %%ymm1, 0x20(%8)\n"
    "vmovups %%ymm2, 0x40(%8)\n"
    "vmovups %%ymm3, (%8, %7, 1)\n"
    "vmovups %%ymm4, 0x20(%8, %7, 1)\n"
    "vmovups %%ymm5, 0x40(%8, %7, 1)\n"
    "vmovups %%ymm6, (%8, %7, 2)\n"
    "vmovups %%ymm7, 0x20(%8, %7, 2)\n"
    "vmovups %%ymm8, 0x40(%8, %7, 2)\n"
    "vmovups %%ymm9, (%9)\n"
    "vmovups %%ymm10, 0x20(%9)\n"
    "vmovups %%ymm11, 0x40(%9)\n"
    :
    : "r"(src), "r"(weight), "r"(bias), "r"(ic_align), "r"(in_sw_step), "r"(src_3_step), "r"(act_flag),  // 6
      "r"(oc_align), "r"(dst), "r"(dst_3), "r"(dst_flag)                                                 // 10
    : "%rax", "%rcx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9",
      "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15");
}

void Conv1x1SW1x24AVXKernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                            size_t ow_block, size_t oc_block, size_t oc_align, size_t ic_align, size_t in_sw_step,
                            size_t dst_flag) {
  asm volatile(
    "movq %8, %%rax\n"
    "and $0x1, %%eax\n"
    "je 0f\n"
    "vmovups (%7), %%ymm0\n"
    "vmovups 0x20(%7), %%ymm1\n"
    "vmovups 0x40(%7), %%ymm2\n"
    "jmp 2f\n"
    "0:\n"
    "cmpq $0, %2\n"
    "je 1f\n"
    "vmovups (%2), %%ymm0\n"
    "vmovups 0x20(%2), %%ymm1\n"
    "vmovups 0x40(%2), %%ymm2\n"
    "jmp 2f\n"
    "1:\n"
    "vxorps %%ymm0, %%ymm0, %%ymm0\n"
    "vxorps %%ymm1, %%ymm1, %%ymm1\n"
    "vxorps %%ymm2, %%ymm2, %%ymm2\n"

    "2:\n"  // LoopIC
    "vbroadcastss (%0), %%ymm13\n"
    "vmovups (%1), %%ymm4\n"
    "vmovups 0x20(%1), %%ymm5\n"
    "vmovups 0x40(%1), %%ymm6\n"
    "vfmadd231ps %%ymm4, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm5, %%ymm13, %%ymm1\n"
    "vfmadd231ps %%ymm6, %%ymm13, %%ymm2\n"

    "vbroadcastss 4(%0), %%ymm13\n"
    "vmovups 96(%1), %%ymm4\n"
    "vmovups 128(%1), %%ymm5\n"
    "vmovups 160(%1), %%ymm6\n"
    "vfmadd231ps %%ymm4, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm5, %%ymm13, %%ymm1\n"
    "vfmadd231ps %%ymm6, %%ymm13, %%ymm2\n"

    "vbroadcastss 8(%0), %%ymm13\n"
    "vmovups 192(%1), %%ymm4\n"
    "vmovups 224(%1), %%ymm5\n"
    "vmovups 256(%1), %%ymm6\n"
    "vfmadd231ps %%ymm4, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm5, %%ymm13, %%ymm1\n"
    "vfmadd231ps %%ymm6, %%ymm13, %%ymm2\n"

    "vbroadcastss 12(%0), %%ymm13\n"
    "vmovups 288(%1), %%ymm4\n"
    "vmovups 320(%1), %%ymm5\n"
    "vmovups 352(%1), %%ymm6\n"
    "vfmadd231ps %%ymm4, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm5, %%ymm13, %%ymm1\n"
    "vfmadd231ps %%ymm6, %%ymm13, %%ymm2\n"

    "vbroadcastss 16(%0), %%ymm13\n"
    "vmovups 384(%1), %%ymm4\n"
    "vmovups 416(%1), %%ymm5\n"
    "vmovups 448(%1), %%ymm6\n"
    "vfmadd231ps %%ymm4, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm5, %%ymm13, %%ymm1\n"
    "vfmadd231ps %%ymm6, %%ymm13, %%ymm2\n"

    "vbroadcastss 20(%0), %%ymm13\n"
    "vmovups 480(%1), %%ymm4\n"
    "vmovups 512(%1), %%ymm5\n"
    "vmovups 544(%1), %%ymm6\n"
    "vfmadd231ps %%ymm4, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm5, %%ymm13, %%ymm1\n"
    "vfmadd231ps %%ymm6, %%ymm13, %%ymm2\n"

    "vbroadcastss 24(%0), %%ymm13\n"
    "vmovups 576(%1), %%ymm4\n"
    "vmovups 608(%1), %%ymm5\n"
    "vmovups 640(%1), %%ymm6\n"
    "vfmadd231ps %%ymm4, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm5, %%ymm13, %%ymm1\n"
    "vfmadd231ps %%ymm6, %%ymm13, %%ymm2\n"

    "vbroadcastss 28(%0), %%ymm13\n"
    "vmovups 672(%1), %%ymm4\n"
    "vmovups 704(%1), %%ymm5\n"
    "vmovups 736(%1), %%ymm6\n"
    "vfmadd231ps %%ymm4, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm5, %%ymm13, %%ymm1\n"
    "vfmadd231ps %%ymm6, %%ymm13, %%ymm2\n"

    "addq $768, %1\n"
    "addq $32, %0\n"
    "dec %3\n"
    "jg 2b\n"

    "movq %8, %%rax\n"
    "and $0x2, %%eax\n"
    "je 3f\n"
    "movq %5, %%rax\n"
    "and $0x3, %%eax\n"
    "je 3f\n"
    // Relu
    "vxorps %%ymm12, %%ymm12, %%ymm12\n"
    "vmaxps %%ymm12, %%ymm0, %%ymm0\n"
    "vmaxps %%ymm12, %%ymm1, %%ymm1\n"
    "vmaxps %%ymm12, %%ymm2, %%ymm2\n"

    "and $0x1, %%eax\n"
    "je 3f\n"
    // relu6
    "mov $0x40C00000, %%ecx\n"
    "vmovd %%ecx, %%xmm14\n"
    "vpermps %%ymm14, %%ymm12, %%ymm14\n"
    "vminps %%ymm14, %%ymm0, %%ymm0\n"
    "vminps %%ymm14, %%ymm1, %%ymm1\n"
    "vminps %%ymm14, %%ymm2, %%ymm2\n"

    "3:\n"
    "vmovups %%ymm0, (%7)\n"  // dst_0
    "vmovups %%ymm1, 0x20(%7)\n"
    "vmovups %%ymm2, 0x40(%7)\n"
    :
    : "r"(src), "r"(weight), "r"(bias), "r"(ic_align), "r"(in_sw_step), "r"(act_flag), "r"(oc_align), "r"(dst),
      "r"(dst_flag)  // 8
    : "%rax", "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm4", "%ymm5", "%ymm6", "%ymm12", "%ymm13", "%ymm14");
}

void Conv1x1SW6x16AVXKernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                            size_t ow_block, size_t oc_block, size_t oc_align, size_t ic_align, size_t in_sw_step,
                            size_t dst_flag) {
  size_t src_3_step = 3 * in_sw_step;
  float *dst_3 = dst + 3 * oc_align / sizeof(float);
  asm volatile(
    "movq %10, %%rax\n"  // dst_flag
    "and $0x1, %%eax\n"
    "je 0f\n"
    "vmovups (%8), %%ymm0\n"  // dst_0
    "vmovups 0x20(%8), %%ymm1\n"
    "vmovups (%8, %7, 1), %%ymm2\n"
    "vmovups 0x20(%8, %7, 1), %%ymm3\n"
    "vmovups (%8, %7, 2), %%ymm4\n"
    "vmovups 0x20(%8, %7, 2), %%ymm5\n"
    "vmovups (%9), %%ymm6\n"
    "vmovups 0x20(%9), %%ymm7\n"
    "vmovups (%9, %7, 1), %%ymm8\n"
    "vmovups 0x20(%9, %7, 1), %%ymm9\n"
    "vmovups (%9, %7, 2), %%ymm10\n"
    "vmovups 0x20(%9, %7, 2), %%ymm11\n"
    "jmp 2f\n"
    "0:\n"
    "cmpq $0, %2\n"
    "je 1f\n"
    "vmovups (%2), %%ymm0\n"
    "vmovups 0x20(%2), %%ymm1\n"
    // We need to copy ymm0 to ymm3 to reduce IO time, but unfortunately I didn't find the corresponding instruction.
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
    "jmp 2f\n"
    "1:\n"
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

    "2:\n"  // LoopIC
    "movq %0, %%rax\n"
    "addq %5, %%rax\n"

    "vmovups (%1), %%ymm12\n"
    "vmovups 0x20(%1), %%ymm13\n"
    "vbroadcastss (%0), %%ymm14\n"
    "vbroadcastss (%0, %4), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm0\n"
    "vfmadd231ps %%ymm13, %%ymm14, %%ymm1\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm2\n"
    "vfmadd231ps %%ymm13, %%ymm15, %%ymm3\n"
    "vbroadcastss (%0, %4, 2), %%ymm14\n"
    "vbroadcastss (%%rax), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n"
    "vfmadd231ps %%ymm13, %%ymm14, %%ymm5\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm6\n"
    "vfmadd231ps %%ymm13, %%ymm15, %%ymm7\n"
    "vbroadcastss (%%rax, %4), %%ymm14\n"
    "vbroadcastss (%%rax, %4, 2), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm8\n"
    "vfmadd231ps %%ymm13, %%ymm14, %%ymm9\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm10\n"
    "vfmadd231ps %%ymm13, %%ymm15, %%ymm11\n"

    "vmovups 64(%1), %%ymm12\n"
    "vmovups 96(%1), %%ymm13\n"
    "vbroadcastss 4(%0), %%ymm14\n"
    "vbroadcastss 4(%0, %4), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm0\n"
    "vfmadd231ps %%ymm13, %%ymm14, %%ymm1\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm2\n"
    "vfmadd231ps %%ymm13, %%ymm15, %%ymm3\n"
    "vbroadcastss 4(%0, %4, 2), %%ymm14\n"
    "vbroadcastss 4(%%rax), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n"
    "vfmadd231ps %%ymm13, %%ymm14, %%ymm5\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm6\n"
    "vfmadd231ps %%ymm13, %%ymm15, %%ymm7\n"
    "vbroadcastss 4(%%rax, %4), %%ymm14\n"
    "vbroadcastss 4(%%rax, %4, 2), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm8\n"
    "vfmadd231ps %%ymm13, %%ymm14, %%ymm9\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm10\n"
    "vfmadd231ps %%ymm13, %%ymm15, %%ymm11\n"

    "vmovups 128(%1), %%ymm12\n"
    "vmovups 160(%1), %%ymm13\n"
    "vbroadcastss 8(%0), %%ymm14\n"
    "vbroadcastss 8(%0, %4), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm0\n"
    "vfmadd231ps %%ymm13, %%ymm14, %%ymm1\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm2\n"
    "vfmadd231ps %%ymm13, %%ymm15, %%ymm3\n"
    "vbroadcastss 8(%0, %4, 2), %%ymm14\n"
    "vbroadcastss 8(%%rax), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n"
    "vfmadd231ps %%ymm13, %%ymm14, %%ymm5\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm6\n"
    "vfmadd231ps %%ymm13, %%ymm15, %%ymm7\n"
    "vbroadcastss 8(%%rax, %4), %%ymm14\n"
    "vbroadcastss 8(%%rax, %4, 2), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm8\n"
    "vfmadd231ps %%ymm13, %%ymm14, %%ymm9\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm10\n"
    "vfmadd231ps %%ymm13, %%ymm15, %%ymm11\n"

    "vmovups 192(%1), %%ymm12\n"
    "vmovups 224(%1), %%ymm13\n"
    "vbroadcastss 12(%0), %%ymm14\n"
    "vbroadcastss 12(%0, %4), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm0\n"
    "vfmadd231ps %%ymm13, %%ymm14, %%ymm1\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm2\n"
    "vfmadd231ps %%ymm13, %%ymm15, %%ymm3\n"
    "vbroadcastss 12(%0, %4, 2), %%ymm14\n"
    "vbroadcastss 12(%%rax), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n"
    "vfmadd231ps %%ymm13, %%ymm14, %%ymm5\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm6\n"
    "vfmadd231ps %%ymm13, %%ymm15, %%ymm7\n"
    "vbroadcastss 12(%%rax, %4), %%ymm14\n"
    "vbroadcastss 12(%%rax, %4, 2), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm8\n"
    "vfmadd231ps %%ymm13, %%ymm14, %%ymm9\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm10\n"
    "vfmadd231ps %%ymm13, %%ymm15, %%ymm11\n"

    "vmovups 256(%1), %%ymm12\n"
    "vmovups 288(%1), %%ymm13\n"
    "vbroadcastss 16(%0), %%ymm14\n"
    "vbroadcastss 16(%0, %4), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm0\n"
    "vfmadd231ps %%ymm13, %%ymm14, %%ymm1\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm2\n"
    "vfmadd231ps %%ymm13, %%ymm15, %%ymm3\n"
    "vbroadcastss 16(%0, %4, 2), %%ymm14\n"
    "vbroadcastss 16(%%rax), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n"
    "vfmadd231ps %%ymm13, %%ymm14, %%ymm5\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm6\n"
    "vfmadd231ps %%ymm13, %%ymm15, %%ymm7\n"
    "vbroadcastss 16(%%rax, %4), %%ymm14\n"
    "vbroadcastss 16(%%rax, %4, 2), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm8\n"
    "vfmadd231ps %%ymm13, %%ymm14, %%ymm9\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm10\n"
    "vfmadd231ps %%ymm13, %%ymm15, %%ymm11\n"

    "vmovups 320(%1), %%ymm12\n"
    "vmovups 352(%1), %%ymm13\n"
    "vbroadcastss 20(%0), %%ymm14\n"
    "vbroadcastss 20(%0, %4), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm0\n"
    "vfmadd231ps %%ymm13, %%ymm14, %%ymm1\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm2\n"
    "vfmadd231ps %%ymm13, %%ymm15, %%ymm3\n"
    "vbroadcastss 20(%0, %4, 2), %%ymm14\n"
    "vbroadcastss 20(%%rax), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n"
    "vfmadd231ps %%ymm13, %%ymm14, %%ymm5\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm6\n"
    "vfmadd231ps %%ymm13, %%ymm15, %%ymm7\n"
    "vbroadcastss 20(%%rax, %4), %%ymm14\n"
    "vbroadcastss 20(%%rax, %4, 2), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm8\n"
    "vfmadd231ps %%ymm13, %%ymm14, %%ymm9\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm10\n"
    "vfmadd231ps %%ymm13, %%ymm15, %%ymm11\n"

    "vmovups 384(%1), %%ymm12\n"
    "vmovups 416(%1), %%ymm13\n"
    "vbroadcastss 24(%0), %%ymm14\n"
    "vbroadcastss 24(%0, %4), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm0\n"
    "vfmadd231ps %%ymm13, %%ymm14, %%ymm1\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm2\n"
    "vfmadd231ps %%ymm13, %%ymm15, %%ymm3\n"
    "vbroadcastss 24(%0, %4, 2), %%ymm14\n"
    "vbroadcastss 24(%%rax), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n"
    "vfmadd231ps %%ymm13, %%ymm14, %%ymm5\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm6\n"
    "vfmadd231ps %%ymm13, %%ymm15, %%ymm7\n"
    "vbroadcastss 24(%%rax, %4), %%ymm14\n"
    "vbroadcastss 24(%%rax, %4, 2), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm8\n"
    "vfmadd231ps %%ymm13, %%ymm14, %%ymm9\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm10\n"
    "vfmadd231ps %%ymm13, %%ymm15, %%ymm11\n"

    "vmovups 448(%1), %%ymm12\n"
    "vmovups 480(%1), %%ymm13\n"
    "vbroadcastss 28(%0), %%ymm14\n"
    "vbroadcastss 28(%0, %4), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm0\n"
    "vfmadd231ps %%ymm13, %%ymm14, %%ymm1\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm2\n"
    "vfmadd231ps %%ymm13, %%ymm15, %%ymm3\n"
    "vbroadcastss 28(%0, %4, 2), %%ymm14\n"
    "vbroadcastss 28(%%rax), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n"
    "vfmadd231ps %%ymm13, %%ymm14, %%ymm5\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm6\n"
    "vfmadd231ps %%ymm13, %%ymm15, %%ymm7\n"
    "vbroadcastss 28(%%rax, %4), %%ymm14\n"
    "vbroadcastss 28(%%rax, %4, 2), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm8\n"
    "vfmadd231ps %%ymm13, %%ymm14, %%ymm9\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm10\n"
    "vfmadd231ps %%ymm13, %%ymm15, %%ymm11\n"

    "addq $512, %1\n"
    "addq $32, %0\n"
    "dec %3\n"
    "jg 2b\n"

    "movq %10, %%rax\n"
    "and $0x2, %%eax\n"
    "je 3f\n"
    "movq %6, %%rax\n"
    "and $0x3, %%eax\n"
    "je 3f\n"
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
    "je 3f\n"
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

    "3:\n"
    "vmovups %%ymm0, (%8)\n"  // dst_0
    "vmovups %%ymm1, 0x20(%8)\n"
    "vmovups %%ymm2, (%8, %7, 1)\n"
    "vmovups %%ymm3, 0x20(%8, %7, 1)\n"
    "vmovups %%ymm4, (%8, %7, 2)\n"
    "vmovups %%ymm5, 0x20(%8, %7, 2)\n"
    "vmovups %%ymm6, (%9)\n"  // dst+3
    "vmovups %%ymm7, 0x20(%9)\n"
    "vmovups %%ymm8, (%9, %7, 1)\n"
    "vmovups %%ymm9, 0x20(%9, %7, 1)\n"
    "vmovups %%ymm10, (%9, %7, 2)\n"
    "vmovups %%ymm11, 0x20(%9, %7, 2)\n"
    :
    : "r"(src), "r"(weight), "r"(bias), "r"(ic_align), "r"(in_sw_step), "r"(src_3_step), "r"(act_flag),  // 6
      "r"(oc_align), "r"(dst), "r"(dst_3), "r"(dst_flag)                                                 // 10
    : "%rax", "%rcx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9",
      "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15");
}

void Conv1x1SW1x16AVXKernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                            size_t ow_block, size_t oc_block, size_t oc_align, size_t ic_align, size_t in_sw_step,
                            size_t dst_flag) {
  asm volatile(
    "movq %8, %%rax\n"
    "and $0x1, %%eax\n"
    "je 0f\n"
    "vmovups (%7), %%ymm0\n"
    "vmovups 0x20(%7), %%ymm1\n"
    "jmp 2f\n"
    "0:\n"
    "cmpq $0, %2\n"
    "je 1f\n"
    "vmovups (%2), %%ymm0\n"
    "vmovups 0x20(%2), %%ymm1\n"
    "jmp 2f\n"
    "1:\n"
    "vxorps %%ymm0, %%ymm0, %%ymm0\n"
    "vxorps %%ymm1, %%ymm1, %%ymm1\n"

    "2:\n"  // LoopIC
    "vbroadcastss (%0), %%ymm12\n"
    "vmovups (%1), %%ymm13\n"
    "vmovups 0x20(%1), %%ymm14\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm1\n"

    "vbroadcastss 4(%0), %%ymm12\n"
    "vmovups 64(%1), %%ymm13\n"
    "vmovups 96(%1), %%ymm14\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm1\n"

    "vbroadcastss 8(%0), %%ymm12\n"
    "vmovups 128(%1), %%ymm13\n"
    "vmovups 160(%1), %%ymm14\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm1\n"

    "vbroadcastss 12(%0), %%ymm12\n"
    "vmovups 192(%1), %%ymm13\n"
    "vmovups 224(%1), %%ymm14\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm1\n"

    "vbroadcastss 16(%0), %%ymm12\n"
    "vmovups 256(%1), %%ymm13\n"
    "vmovups 288(%1), %%ymm14\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm1\n"

    "vbroadcastss 20(%0), %%ymm12\n"
    "vmovups 320(%1), %%ymm13\n"
    "vmovups 352(%1), %%ymm14\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm1\n"

    "vbroadcastss 24(%0), %%ymm12\n"
    "vmovups 384(%1), %%ymm13\n"
    "vmovups 416(%1), %%ymm14\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm1\n"

    "vbroadcastss 28(%0), %%ymm12\n"
    "vmovups 448(%1), %%ymm13\n"
    "vmovups 480(%1), %%ymm14\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm1\n"

    "addq $512, %1\n"
    "addq $32, %0\n"
    "dec %3\n"
    "jg 2b\n"

    "movq %8, %%rax\n"
    "and $0x2, %%eax\n"
    "je 3f\n"
    "movq %5, %%rax\n"
    "and $0x3, %%eax\n"
    "je 3f\n"
    // Relu
    "vxorps %%ymm12, %%ymm12, %%ymm12\n"
    "vmaxps %%ymm12, %%ymm0, %%ymm0\n"
    "vmaxps %%ymm12, %%ymm1, %%ymm1\n"

    "and $0x1, %%eax\n"
    "je 3f\n"
    // relu6
    "mov $0x40C00000, %%ecx\n"
    "vmovd %%ecx, %%xmm14\n"
    "vpermps %%ymm14, %%ymm12, %%ymm14\n"
    "vminps %%ymm14, %%ymm0, %%ymm0\n"
    "vminps %%ymm14, %%ymm1, %%ymm1\n"

    "3:\n"
    "vmovups %%ymm0, (%7)\n"  // dst_0
    "vmovups %%ymm1, 0x20(%7)\n"
    :
    : "r"(src), "r"(weight), "r"(bias), "r"(ic_align), "r"(in_sw_step), "r"(act_flag), "r"(oc_align), "r"(dst),
      "r"(dst_flag)  // 8
    : "%rax", "%ecx", "%ymm0", "%ymm1", "%ymm12", "%ymm13", "%ymm14");
}

void Conv1x1SW12x8AVXKernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                            size_t ow_block, size_t oc_block, size_t oc_align, size_t ic_align, size_t in_sw_step,
                            size_t dst_flag) {
  ic_align <<= 3;
  size_t src_3_step = 3 * in_sw_step;
  float *dst_3 = dst + 3 * oc_align / sizeof(float);
  float *dst_5 = dst + 5 * oc_align / sizeof(float);
  float *dst_9 = dst + 9 * oc_align / sizeof(float);
  asm volatile(
    "movq %12, %%rax\n"
    "and $0x1, %%eax\n"
    "je 0f\n"
    "vmovups (%8), %%ymm0\n"  // dst_0
    "vmovups (%8, %7), %%ymm1\n"
    "vmovups (%8, %7, 2), %%ymm2\n"
    "vmovups (%9), %%ymm3\n"  // dst_3
    "vmovups (%8, %7, 4), %%ymm4\n"
    "vmovups (%10), %%ymm5\n"  // dst_5
    "vmovups (%10, %7, 1), %%ymm6\n"
    "vmovups (%10, %7, 2), %%ymm7\n"
    "vmovups (%8, %7, 8), %%ymm8\n"
    "vmovups (%11), %%ymm9\n"  // dst_9
    "vmovups (%11, %7, 1), %%ymm10\n"
    "vmovups (%11, %7, 2), %%ymm11\n"
    "jmp 2f\n"
    "0:\n"
    "cmpq $0, %2\n"
    "je 1f\n"
    "vmovups (%2), %%ymm0\n"
    "vmovups (%2), %%ymm1\n"
    "vmovups (%2), %%ymm2\n"
    "vmovups (%2), %%ymm3\n"
    "vmovups (%2), %%ymm4\n"
    "vmovups (%2), %%ymm5\n"
    "vmovups (%2), %%ymm6\n"
    "vmovups (%2), %%ymm7\n"
    "vmovups (%2), %%ymm8\n"
    "vmovups (%2), %%ymm9\n"
    "vmovups (%2), %%ymm10\n"
    "vmovups (%2), %%ymm11\n"
    "jmp 2f\n"
    "1:\n"
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

    "2:\n"  // LoopIC
    "vmovups (%1), %%ymm12\n"
    "movq %0, %%rax\n"
    "vbroadcastss (%%rax), %%ymm13\n"
    "vbroadcastss (%%rax, %4), %%ymm14\n"
    "vbroadcastss (%%rax, %4, 2), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm1\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm2\n"
    "addq %5, %%rax\n"
    "vbroadcastss (%%rax), %%ymm13\n"
    "vbroadcastss (%%rax, %4), %%ymm14\n"
    "vbroadcastss (%%rax, %4, 2), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm3\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm5\n"
    "addq %5, %%rax\n"
    "vbroadcastss (%%rax), %%ymm13\n"
    "vbroadcastss (%%rax, %4), %%ymm14\n"
    "vbroadcastss (%%rax, %4, 2), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm6\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm7\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm8\n"
    "addq %5, %%rax\n"
    "vbroadcastss (%%rax), %%ymm13\n"
    "vbroadcastss (%%rax, %4), %%ymm14\n"
    "vbroadcastss (%%rax, %4, 2), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm9\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm10\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm11\n"
    "addq $32, %1\n"
    "addq $4, %0\n"
    "dec %3\n"
    "jg 2b\n"
    :
    : "r"(src), "r"(weight), "r"(bias), "r"(ic_align), "r"(in_sw_step), "r"(src_3_step), "r"(act_flag),  // 6
      "r"(oc_align), "r"(dst), "r"(dst_3), "r"(dst_5), "r"(dst_9), "r"(dst_flag)                         // 12
    : "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10",
      "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15");

  asm volatile(
    "and $0x2, %%eax\n"
    "je 0f\n"
    "movq %0, %%rax\n"
    "and $0x3, %%eax\n"
    "je 0f\n"
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
    "je 0f\n"
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

    "0:\n"
    "vmovups %%ymm0, (%2)\n"  // dst_0
    "vmovups %%ymm1, (%2, %1)\n"
    "vmovups %%ymm2, (%2, %1, 2)\n"
    "vmovups %%ymm3, (%3)\n"  // dst_3
    "vmovups %%ymm4, (%2, %1, 4)\n"
    "vmovups %%ymm5, (%4)\n"  // dst_5
    "vmovups %%ymm6, (%4, %1, 1)\n"
    "vmovups %%ymm7, (%4, %1, 2)\n"
    "vmovups %%ymm8, (%2, %1, 8)\n"
    "vmovups %%ymm9, (%5)\n"  // dst_9
    "vmovups %%ymm10, (%5, %1, 1)\n"
    "vmovups %%ymm11, (%5, %1, 2)\n"
    :
    : "r"(act_flag), "r"(oc_align), "r"(dst), "r"(dst_3), "r"(dst_5), "r"(dst_9), "a"(dst_flag)  // 6
    : "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10",
      "%ymm11", "%ymm12", "%ymm14");
}

void Conv1x1SW1x8AVXKernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                           size_t ow_block, size_t oc_block, size_t oc_align, size_t ic_align, size_t in_sw_step,
                           size_t dst_flag) {
  asm volatile(
    "movq %8, %%rax\n"
    "and $0x1, %%eax\n"
    "je 0f\n"
    "vmovups (%7), %%ymm0\n"
    "jmp 2f\n"
    "0:\n"
    "cmpq $0, %2\n"
    "je 1f\n"
    "vmovups (%2), %%ymm0\n"
    "jmp 2f\n"
    "1:\n"
    "vxorps %%ymm0, %%ymm0, %%ymm0\n"

    "2:\n"  // LoopIC
    "vbroadcastss (%0), %%ymm12\n"
    "vmovups (%1), %%ymm13\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"

    "vbroadcastss 4(%0), %%ymm12\n"
    "vmovups 32(%1), %%ymm13\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"

    "vbroadcastss 8(%0), %%ymm12\n"
    "vmovups 64(%1), %%ymm13\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"

    "vbroadcastss 12(%0), %%ymm12\n"
    "vmovups 96(%1), %%ymm13\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"

    "vbroadcastss 16(%0), %%ymm12\n"
    "vmovups 128(%1), %%ymm13\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"

    "vbroadcastss 20(%0), %%ymm12\n"
    "vmovups 160(%1), %%ymm13\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"

    "vbroadcastss 24(%0), %%ymm12\n"
    "vmovups 192(%1), %%ymm13\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"

    "vbroadcastss 28(%0), %%ymm12\n"
    "vmovups 224(%1), %%ymm13\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"
    "addq $256, %1\n"
    "addq $32, %0\n"
    "dec %3\n"
    "jg 2b\n"

    "movq %8, %%rax\n"
    "and $0x2, %%eax\n"
    "je 3f\n"
    "movq %5, %%rax\n"
    "and $0x3, %%eax\n"
    "je 3f\n"
    // Relu
    "vxorps %%ymm12, %%ymm12, %%ymm12\n"
    "vmaxps %%ymm12, %%ymm0, %%ymm0\n"

    "and $0x1, %%eax\n"
    "je 3f\n"
    // relu6
    "mov $0x40C00000, %%ecx\n"
    "vmovd %%ecx, %%xmm14\n"
    "vpermps %%ymm14, %%ymm12, %%ymm14\n"
    "vminps %%ymm14, %%ymm0, %%ymm0\n"

    "3:\n"
    "vmovups %%ymm0, (%7)\n"  // dst_0
    :
    : "r"(src), "r"(weight), "r"(bias), "r"(ic_align), "r"(in_sw_step), "r"(act_flag), "r"(oc_align), "r"(dst),
      "r"(dst_flag)  // 8
    : "%rax", "%ecx", "%ymm0", "%ymm12", "%ymm13");
}

// sliding window to compate 1x1 conv in x86
void Conv1x1SWAVXFp32(const float *input_data, const float *packed_weight, const float *bias_data, float *output_data,
                      int task_id, ConvParameter *conv_param, SlidingWindowParam *sw_param) {
  int output_w = conv_param->output_w_;
  int output_h = conv_param->output_h_;
  int ohw = output_h * output_w;
  int ohw_step = UP_DIV(ohw, conv_param->thread_num_);
  int ohw_start = ohw_step * task_id;
  int ohw_end = MSMIN(ohw_start + ohw_step, ohw);
  if (ohw_start >= ohw_end) {
    return;
  }
  int act_type = C0NUM;
  int oc_tile_ = C8NUM;  // oc in algin to C8NUM in x86_64_avx
  if (conv_param->act_type_ == ActType_Relu6) {
    act_type += C1NUM;
  }
  if (conv_param->act_type_ == ActType_Relu6 || conv_param->act_type_ == ActType_Relu) {
    act_type += C2NUM;
  }
  int pad_d = conv_param->pad_d_;
  int pad_l = conv_param->pad_l_;
  int pad_r = conv_param->pad_r_;
  int pad_u = conv_param->pad_u_;
  int oc_align = sw_param->block_channel_;
  int oc_align_float = oc_align * sizeof(float);
  int ic_align = sw_param->ic_align_;
  int in_sw_step = sw_param->in_sw_step_;
  int in_sw_step_float = sw_param->in_sw_step_ * sizeof(float);
  int kernel_step = sw_param->kernel_step_;
  int oc_num = sw_param->c_block_;
  int in_step = sw_param->in_step_;
  int out_step = sw_param->out_step_;
  const int ow_block_num[4] = {12, 6, 4, 3};
  const Conv1x1SWAVXKernel kernel[4][2] = {{Conv1x1SW1x8AVXKernel, Conv1x1SW12x8AVXKernel},
                                           {Conv1x1SW1x16AVXKernel, Conv1x1SW6x16AVXKernel},
                                           {Conv1x1SW1x24AVXKernel, Conv1x1SW4x24AVXKernel},
                                           {Conv1x1SW1x32AVXKernel, Conv1x1SW3x32AVXKernel}};
  for (int b = 0; b < conv_param->output_batch_; b++) {
    int ic_block = 128;
    int dst_flag = 0;
    for (int ic = 0; ic < ic_align; ic += ic_block) {
      if (ic_align - ic <= ic_block) {
        ic_block = ic_align - ic;
        dst_flag = C3NUM - (ic == 0);
      } else {
        dst_flag = 1 - (ic == 0);
      }
      if (pad_d == 0 && pad_l == 0 && pad_r == 0 && pad_u == 0) {
        const float *bias = bias_data;
        int oc_block = 0;
        for (int oc = 0; oc < oc_num; oc += oc_block) {
          oc_block = MSMIN(C4NUM, oc_num - oc);  // 4 3 2 1
          const float *weight = packed_weight + oc * kernel_step + ic * C8NUM * oc_block;
          if (bias != NULL) {
            bias = bias_data + oc * oc_tile_;
          }
          const float *src_w = input_data + ic + ohw_start * in_sw_step;
          float *dst_oc = output_data + oc * oc_tile_;
          int hw_block = ow_block_num[oc_block - 1];
          for (int hw = ohw_start; hw < ohw_end; hw += hw_block) {
            if (hw_block > ohw_end - hw) {  // ow is not enough and process one ow
              hw_block = 1;
            }
            float *dst_w = dst_oc + hw * oc_align;
            kernel[oc_block - 1][hw_block / ow_block_num[oc_block - 1]](dst_w, src_w, weight, bias, act_type, hw_block,
                                                                        oc_block, oc_align_float, ic_block >> C3NUM,
                                                                        in_sw_step_float, dst_flag);
            src_w += hw_block * in_sw_step;
          }
        }
      }
    }
    input_data += in_step;
    output_data += out_step;
  }  // batch loop
}

#ifdef ENABLE_DEBUG
void Conv1x1SWOWxOCAVXKernel(float *dst, const float *src, const float *weight, const float *bias, size_t act_flag,
                             size_t ow_block, size_t oc_block, size_t oc_align, size_t ic_align, size_t in_sw_step,
                             size_t dst_flag) {
  oc_align /= sizeof(float);
  in_sw_step /= sizeof(float);
  ic_align <<= C3NUM;
  __m256 dst_data[12];
  const float *src_sw[12];
  __m256 weight_data[4];
  for (int i = 0; i < C4NUM; ++i) {
    weight_data[i] = _mm256_set1_ps(0.0f);
  }
  for (int i = 0; i < ow_block; ++i) {
    if (dst_flag & 0x01) {
      for (int j = 0; j < oc_block; ++j) {
        dst_data[i * oc_block + j] = _mm256_loadu_ps(dst + i * oc_align + j * C8NUM);
      }
    } else {
      if (bias != NULL) {
        for (int j = 0; j < oc_block; ++j) {
          dst_data[i * oc_block + j] = _mm256_loadu_ps(bias + j * C8NUM);
        }
      } else {
        for (int j = 0; j < oc_block; ++j) {
          dst_data[i * oc_block + j] = _mm256_set1_ps(0.0f);
        }
      }
    }
    src_sw[i] = src + i * in_sw_step;
  }
  const float *weight_kernel = weight;
  for (int ic = 0; ic < ic_align; ++ic) {
    for (int j = 0; j < oc_block; ++j) {
      weight_data[j] = _mm256_loadu_ps(weight_kernel + j * C8NUM);
    }
    for (int i = 0; i < ow_block; ++i) {
      for (int j = 0; j < oc_block; ++j) {
        dst_data[i * oc_block + j] += src_sw[i][ic] * weight_data[j];
      }
    }
    weight_kernel += C8NUM * oc_block;
  }  // ic loop
  // add bias and relu
  for (int i = 0; i < ow_block; ++i) {
    for (int j = 0; j < oc_block; ++j) {
      if (dst_flag & 0x02) {
        if (0x1 & act_flag) {  // relu6
          dst_data[i * oc_block + j] = _mm256_min_ps(dst_data[i * oc_block + j], _mm256_set1_ps(6.0f));
        }
        if (0x2 & act_flag) {  // relu
          dst_data[i * oc_block + j] = _mm256_max_ps(dst_data[i * oc_block + j], _mm256_set1_ps(0.0f));
        }
      }
      _mm256_storeu_ps(dst + i * oc_align + j * C8NUM, dst_data[i * oc_block + j]);
    }
  }
}
#endif
