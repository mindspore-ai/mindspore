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
#ifdef ENABLE_AVX
#include "nnacl/assembly_global.h"

.text
.align 4

// void ConvDwAVXFp32Row(float *output_ptr, const float *input_tmp, const float *weight_ptr, size_t num_pixels,
//                          size_t output_channel, size_t input_step);
// in linux x64 platform:
// rdi: output_ptr
// rsi: input_ptr
// rdx: weight_ptr
// rcx: num_pixels
// r8: output_channel
// r9: input_step

// in win x64 platform: "shadow space" needs to be opened up for first four parameters ==> 32 bites
// rcx: output_ptr
// rdx: input_ptr
// r8: weight_ptr
// r9: num_pixels
// 40: output_channel
// 48: input_step

asm_function ConvDwAVXFp32Row
    pushq %r15
    pushq %r14
    pushq %r13
    pushq %r12
    pushq %rsi
    pushq %rdi
    addq $48, %rsp

#ifdef _WIN32
    movq %rcx, %rdi // output_ptr
    movq %rdx, %rsi // input_ptr
    movq %r8, %rdx // weight_ptr
    movq %r9, %rcx // num_pixels
    movq 40(%rsp), %r8 // output_channel
    movq 48(%rsp), %r9 // input_step
    movq 56(%rsp), %r11 // first_calc_flag
    movq 64(%rsp), %r10 // bias
#else
    movq 8(%rsp), %r11 // first_calc_flag
    movq 16(%rsp), %r10 // bias
#endif


    movq $4, %r13
    imul %r13, %r9
    movq %r8, %r12
    imul %r13, %r12
    movq %rsi, %r13 // input_ptr
    movq %rdx, %r14 // weight_ptr
    movq %r8, %r15 // output_channel

    cmpq $1, %r11
    je OutputInitByBias
    jmp OutputInitBySelf

OutputInitByBias:
    cmpq $3, %rcx
    jae BiasLoopPixelNum4
    cmpq $0, %rcx
    ja BiasLoopPixel
    je End

    BiasLoopPixelNum4:
        movq %r13, %rsi // input_tmp
        movq %r14, %rdx // weight_tmp
        movq %r15, %r8 // channel_tmp
        movq 16(%rsp), %r10 // bias_tmp

        cmpq $8, %r8
        jae BiasLoopC8Num4
        cmpq $0, %r8
        ja BiasLoopCNum4
        jmp BiasLoopCEndNum4

        BiasLoopC8Num4:
            vmovups (%rsi), %ymm0 // input_tmp
            vmovups (%rsi, %r9), %ymm1
            vmovups (%rsi, %r9, 2), %ymm2
            // vmovups (%rsi, %r9, 3), %ymm3

            vmovups (%r10), %ymm8 // output_tmp
            vmovups (%r10), %ymm9 // output_tmp
            vmovups (%r10), %ymm10 // output_tmp
            // vmovups (%r10), %ymm11 // output_tmp
            addq $32, %rsi
            addq $32, %r10

            vmovups (%rdx), %ymm15 // weight_tmp
            vfmadd231ps %ymm15, %ymm0, %ymm8
            vfmadd231ps %ymm15, %ymm1, %ymm9
            vfmadd231ps %ymm15, %ymm2, %ymm10
            // vfmadd231ps %ymm15, %ymm3, %ymm11

            addq $32, %rdx
            vmovups %ymm8, (%rdi)
            vmovups %ymm9, (%rdi, %r12)
            vmovups %ymm10, (%rdi, %r12, 2)
            // vmovups %ymm11, (%rdi, %r12, 3)
            addq $32, %rdi

            subq $8, %r8
            cmpq $8, %r8
            jae BiasLoopC8Num4
            cmpq $0, %r8
            ja BiasLoopCNum4
            jmp BiasLoopCEndNum4

        BiasLoopCNum4:
            vmovss (%rsi), %xmm0  // input_tmp
            vmovss (%rsi, %r9), %xmm1
            vmovss (%rsi, %r9, 2), %xmm2
            // vmovss (%rsi, %r9, 3), %xmm3

            vmovss (%r10), %xmm8  // output_ptr
            vmovss (%r10), %xmm9 // output_tmp
            vmovss (%r10), %xmm10 // output_tmp
            // vmovss (%r10), %xmm11 // output_tmp
            addq $4, %r10

            vmovss (%rdx), %xmm15 // weight_tmp
            vfmadd231ss %xmm15, %xmm0, %xmm8
            vfmadd231ss %xmm15, %xmm1, %xmm9
            vfmadd231ss %xmm15, %xmm2, %xmm10
            // vfmadd231ss %xmm15, %xmm3, %xmm11

            addq $4, %rsi
            addq $4, %rdx
            vmovss %xmm8, (%rdi)
            vmovss %xmm9, (%rdi, %r12)
            vmovss %xmm10, (%rdi, %r12, 2)
            // vmovss %xmm11, (%rdi, %r12, 3)
            addq $4, %rdi

            subq $1, %r8
            cmpq $0, %r8
            ja BiasLoopCNum4
            jmp BiasLoopCEndNum4

        BiasLoopCEndNum4:
            subq $3, %rcx // num_pixel -= 3
            addq %r12, %rdi
            addq %r12, %rdi

            addq %r9, %r13
            addq %r9, %r13
            addq %r9, %r13
            cmpq $3, %rcx
            jae BiasLoopPixelNum4
            cmpq $0, %rcx
            ja BiasLoopPixel
            je End

    BiasLoopPixel:
        movq %r13, %rsi // input_tmp
        movq %r14, %rdx // weight_tmp
        movq %r15, %r8 // channel_tmp
        movq 16(%rsp), %r10 // bias_tmp

        cmpq $8, %r8
        jae BiasLoopC8
        cmpq $0, %r8
        ja BiasLoopC
        jmp BiasLoopCEnd

        BiasLoopC8:
            vmovups (%rsi), %ymm0 // input_tmp
            vmovups (%r10), %ymm8 // output_tmp
            addq $32, %rsi
            addq $32, %r10

            vfmadd231ps (%rdx), %ymm0, %ymm8

            addq $32, %rdx
            vmovups %ymm8, (%rdi)
            addq $32, %rdi

            subq $8, %r8
            cmpq $8, %r8
            jae BiasLoopC8
            cmpq $0, %r8
            ja BiasLoopC
            jmp BiasLoopCEnd

        BiasLoopC:
            vmovss (%rsi), %xmm0  // input_tmp
            vmovss (%r10), %xmm8  // output_ptr
            addq $4, %r10

            vfmadd231ss (%rdx), %xmm0, %xmm8

            addq $4, %rsi
            addq $4, %rdx
            vmovss %xmm8, (%rdi)
            addq $4, %rdi

            subq $1, %r8
            cmpq $0, %r8
            ja BiasLoopC
            jmp BiasLoopCEnd

        BiasLoopCEnd:
            subq $1, %rcx // num_pixel -= 1
            cmpq $0, %rcx
            je End
            addq %r9, %r13
            jmp BiasLoopPixel

OutputInitBySelf:
    cmpq $3, %rcx
    jae LoopPixelNum4
    cmpq $0, %rcx
    ja LoopPixel
    je End

    LoopPixelNum4:
        movq %r13, %rsi // input_tmp
        movq %r14, %rdx // weight_tmp
        movq %r15, %r8 // channel_tmp

        cmpq $8, %r8
        jae LoopC8Num4
        cmpq $0, %r8
        ja LoopCNum4
        jmp LoopCEndNum4

        LoopC8Num4:
            vmovups (%rsi), %ymm0 // input_tmp
            vmovups (%rsi, %r9), %ymm1
            vmovups (%rsi, %r9, 2), %ymm2
            // vmovups (%rsi, %r9, 3), %ymm3

            vmovups (%rdi), %ymm8 // output_tmp
            vmovups (%rdi, %r12), %ymm9 // output_tmp
            vmovups (%rdi, %r12, 2), %ymm10 // output_tmp
            // vmovups (%rdi, %r12, 3), %ymm11 // output_tmp
            addq $32, %rsi

            vmovups (%rdx), %ymm15 // weight_tmp
            vfmadd231ps %ymm15, %ymm0, %ymm8
            vfmadd231ps %ymm15, %ymm1, %ymm9
            vfmadd231ps %ymm15, %ymm2, %ymm10
            // vfmadd231ps %ymm15, %ymm3, %ymm11

            addq $32, %rdx
            vmovups %ymm8, (%rdi)
            vmovups %ymm9, (%rdi, %r12)
            vmovups %ymm10, (%rdi, %r12, 2)
            // vmovups %ymm11, (%rdi, %r12, 3)
            addq $32, %rdi

            subq $8, %r8
            cmpq $8, %r8
            jae LoopC8Num4
            cmpq $0, %r8
            ja LoopCNum4
            jmp LoopCEndNum4

        LoopCNum4:
            vmovss (%rsi), %xmm0  // input_tmp
            vmovss (%rsi, %r9), %xmm1
            vmovss (%rsi, %r9, 2), %xmm2
            // vmovss (%rsi, %r9, 3), %xmm3

            vmovss (%rdi), %xmm8  // output_ptr
            vmovss (%rdi, %r12), %xmm9 // output_tmp
            vmovss (%rdi, %r12, 2), %xmm10 // output_tmp
            // vmovss (%rdi, %r12, 3), %xmm11 // output_tmp

            vmovss (%rdx), %xmm15 // weight_tmp
            vfmadd231ss %xmm15, %xmm0, %xmm8
            vfmadd231ss %xmm15, %xmm1, %xmm9
            vfmadd231ss %xmm15, %xmm2, %xmm10
            // vfmadd231ss %xmm15, %xmm3, %xmm11

            addq $4, %rsi
            addq $4, %rdx
            vmovss %xmm8, (%rdi)
            vmovss %xmm9, (%rdi, %r12)
            vmovss %xmm10, (%rdi, %r12, 2)
            // vmovss %xmm11, (%rdi, %r12, 3)
            addq $4, %rdi

            subq $1, %r8
            cmpq $0, %r8
            ja LoopCNum4
            jmp LoopCEndNum4

        LoopCEndNum4:
            subq $3, %rcx // num_pixel -= 3
            addq %r12, %rdi
            addq %r12, %rdi

            addq %r9, %r13
            addq %r9, %r13
            addq %r9, %r13
            cmpq $3, %rcx
            jae LoopPixelNum4
            cmpq $0, %rcx
            ja LoopPixel
            je End

    LoopPixel:
        movq %r13, %rsi // input_tmp
        movq %r14, %rdx // weight_tmp
        movq %r15, %r8 // channel_tmp

        cmpq $8, %r8
        jae LoopC8
        cmpq $0, %r8
        ja LoopC
        jmp LoopCEnd

        LoopC8:
            vmovups (%rsi), %ymm0 // input_tmp
            vmovups (%rdi), %ymm8 // output_tmp
            addq $32, %rsi

            vfmadd231ps (%rdx), %ymm0, %ymm8

            addq $32, %rdx
            vmovups %ymm8, (%rdi)
            addq $32, %rdi

            subq $8, %r8
            cmpq $8, %r8
            jae LoopC8
            cmpq $0, %r8
            ja LoopC
            jmp LoopCEnd

        LoopC:
            vmovss (%rsi), %xmm0  // input_tmp
            vmovss (%rdi), %xmm8  // output_ptr

            vfmadd231ss (%rdx), %xmm0, %xmm8

            addq $4, %rsi
            addq $4, %rdx
            vmovss %xmm8, (%rdi)
            addq $4, %rdi

            subq $1, %r8
            cmpq $0, %r8
            ja LoopC
            jmp LoopCEnd

        LoopCEnd:
            subq $1, %rcx // num_pixel -= 1
            cmpq $0, %rcx
            je End
            addq %r9, %r13
            jmp LoopPixel
End:
    subq $48, %rsp
    popq %rdi
    popq %rsi
    popq %r12
    popq %r13
    popq %r14
    popq %r15
    retq
#endif
