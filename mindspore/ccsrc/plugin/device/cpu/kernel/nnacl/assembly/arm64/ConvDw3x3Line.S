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

// void ConvDw3x3Line(float *dst, float **lines, const float *weight, const float *bias_data, int width, int ori_channel,
//                    bool relu, bool relu6)

// x0: dst, x1: lines, x2: weight, x3: bias, x4: width, x5: ori_channel, x6: relu, x7: relu6
asm_function ConvDw3x3Line
    // registers v8 ~ v15 must be preserved by a callee across subroutine calls, according to
    // https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst#simd-and-floating-point-registers
    // x19 ~ x29 should be also preserved
    // whereas our coding style do not permit such amount of parameters
    sub sp, sp, #128
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64

    ldr x8, [x1]
    ldr x9, [x1, #8]
    ldr x10, [x1, #16]
    mov x11, x5
    mov x16, #4
    mul x16, x5, x16

    mov w14, #6
    dup v30.4s, w14
    scvtf v30.4s, v30.4s

    LoopC4:
        cbz x3, NoBias
        ld1 {v31.4s}, [x3], #16
    NoBias:
        ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x2], #64
        ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x2], #64
        ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x2], #64
        mov x12, x0
        mov x13, x4

        cmp x13, #2
        blt LoopOwRemain
        LoopOw2:
            ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x8], #64
            ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x9], #64
            ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x10], #64
            fmul v24.4s, v12.4s, v0.4s
            fmul v25.4s, v13.4s, v1.4s
            fmul v26.4s, v14.4s, v2.4s
            fmul v27.4s, v15.4s, v3.4s
            fmla v24.4s, v16.4s, v4.4s
            fmla v25.4s, v17.4s, v5.4s
            fmla v26.4s, v18.4s, v6.4s
            fmla v27.4s, v19.4s, v7.4s
            fmla v24.4s, v20.4s, v8.4s
            fmla v25.4s, v21.4s, v9.4s
            fmla v26.4s, v22.4s, v10.4s
            fmla v27.4s, v23.4s, v11.4s

            fadd v28.4s, v25.4s, v26.4s
            fadd v28.4s, v28.4s, v24.4s
            fsub v29.4s, v27.4s, v26.4s
            fadd v29.4s, v29.4s, v25.4s

            cbz x3, Activation
        Bias:
            fadd v28.4s, v28.4s, v31.4s
            fadd v29.4s, v29.4s, v31.4s

        Activation:
            cbnz x7, Relu6
            cbnz x6, Relu
            b Write
        Relu6:
            fmin v28.4s, v28.4s, v30.4s
            fmin v29.4s, v29.4s, v30.4s
        Relu:
            movi v27.16b, #0
            fmax v28.4s, v28.4s, v27.4s
            fmax v29.4s, v29.4s, v27.4s
        Write:
            add x15, x12, x16
            cmp x11, #4
            bge Write4
            cmp x11, #3
            beq Write3
            cmp x11, #2
            beq Write2
            cmp x11, #1
            beq Write1

        Write1:
            str s28, [x12]
            str s29, [x15]
            b WriteEnd
        Write2:
            st1 {v28.2s}, [x12]
            st1 {v29.2s}, [x15]
            b WriteEnd
        Write3:
            st1 {v28.2s}, [x12]
            add x17, x12, #8
            st1 {v28.s}[2], [x17]
            st1 {v29.2s}, [x15]
            add x18, x15, #8
            st1 {v29.s}[2], [x18]
            b WriteEnd
        Write4:
            st1 {v28.4s}, [x12]
            st1 {v29.4s}, [x15]

    WriteEnd:
        add x12, x15, x16
        sub x13, x13, #2
        cmp x13, #2
        bge LoopOw2
        cmp x13, #0
        beq LoopOwEnd

        LoopOwRemain:
            ld1 {v12.4s, v13.4s, v14.4s}, [x8]
            add x8, x8, #64
            ld1 {v16.4s, v17.4s, v18.4s}, [x9]
            add x9, x9, #64
            ld1 {v20.4s, v21.4s, v22.4s}, [x10]
            add x10, x10, #64
            fmul v24.4s, v12.4s, v0.4s
            fmul v25.4s, v13.4s, v1.4s
            fmul v26.4s, v14.4s, v2.4s

            fmla v24.4s, v16.4s, v4.4s
            fmla v25.4s, v17.4s, v5.4s
            fmla v26.4s, v18.4s, v6.4s

            fmla v24.4s, v20.4s, v8.4s
            fmla v25.4s, v21.4s, v9.4s
            fmla v26.4s, v22.4s, v10.4s

            fadd v28.4s, v25.4s, v26.4s
            fadd v28.4s, v28.4s, v24.4s

            cbz x3, ActivationRemain
        BiasRemain:
            fadd v28.4s, v28.4s, v31.4s

        ActivationRemain:
            cbnz x7, Relu6Remain
            cbnz x6, ReluRemain
            b WriteRemain
        Relu6Remain:
            fmin v28.4s, v28.4s, v30.4s
        ReluRemain:
            movi v27.16b, #0
            fmax v28.4s, v28.4s, v27.4s
        WriteRemain:
            cmp x11, #4
            bge Write4Remain
            cmp x11, #3
            beq Write3Remain
            cmp x11, #2
            beq Write2Remain
            cmp x11, #1
            beq Write1Remain

        Write1Remain:
            str s28, [x12]
            b LoopOwEnd
        Write2Remain:
            st1 {v28.2s}, [x12]
            b LoopOwEnd
        Write3Remain:
            st1 {v28.2s}, [x12]
            add x17, x12, #8
            st1 {v28.s}[2], [x17]
            b LoopOwEnd
        Write4Remain:
            st1 {v28.4s}, [x12]

    LoopOwEnd:  
        subs x11, x11, #4
        add x0, x0, #16
        bgt LoopC4

    sub sp, sp, #128
    ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
    ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
    ret
#endif
