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
#ifdef ENABLE_ARM64
#include "nnacl/assembly_global.h"

.text
.align 5

// void ConvDwFp32Border(float *dst, const float *src, const float *weight, const float *bias, size_t height, size_t width,
//                       size_t in_kh_step, size_t in_kw_step, size_t kernel_w, size_t relu, size_t relu6)

// x0: dst, x1: src, x2: weight, x3: bias, x4: height, x5: width, x6: in_kh_step, x7: in_kw_step,
// x8: kernel_w, x9: relu, x10: relu6
asm_function ConvDwFp32Border
    // registers v8 ~ v15 must be preserved by a callee across subroutine calls, according to
    // https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst#simd-and-floating-point-registers
    // x19 ~ x29 should be also preserved
    // whereas our coding style do not permit such amount of parameters
    ldr x8, [sp]
    ldr x9, [sp, #8]
    ldr x10, [sp, #16]

    ld1 {v0.4s}, [x3] // bias
    movi v1.4s, #6    // relu 6
    scvtf v1.4s, v1.4s
    dup v2.4s, wzr    // relu

    mov x13, x1
    mov x14, x2
    LoopH:
        mov x15, x13
        mov x16, x14
        mov x17, x5
        LoopW:
            ld1 {v3.4s}, [x15], x7
            ld1 {v4.4s}, [x16], #16
            fmla v0.4s, v3.4s, v4.4s
            subs x17, x17, #1
            bne LoopW
        subs x4, x4, #1
        add x13, x13, x6
        add x14, x14, x8
        bne LoopH
    cbnz x10, Relu6
    cbnz x9, Relu
    b Write
    Relu6:
        fmin v0.4s, v0.4s, v1.4s
    Relu:
        fmax v0.4s, v0.4s, v2.4s
    Write:
        st1 {v0.4s}, [x0]

    ret
#endif
