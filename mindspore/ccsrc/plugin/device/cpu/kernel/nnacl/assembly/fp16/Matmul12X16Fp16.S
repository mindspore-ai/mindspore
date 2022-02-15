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

// void MatMul12x16Fp16Opt(const float16_t *a, const float16_t *b, float16_t *c, const float16_t *bias, int act_type,
//                      int depth, int row, int col, size_t stride, size_t writeMode)
// x0: a
// x1: b
// x2: c
// x3: bias
// x4: act_type     : ActType_No, ActType_Relu, ActType_Sigmod, ActType_Relu6, ActType_Prelu
// x5: depth        : Ic
// x6: row          : remain_row
// x7: col
// x8: stride       : output_stride  x8 = x8 * 2
// x9: writeMode    : OutType_C8 = 0, OutType_Nhwc = 1, OutType_TileC8 = 2

// x17              : input_stride

asm_function MatMul12x16Fp16Opt
    sub sp, sp, #160
    st1 {v8.8h, v9.8h, v10.8h, v11.8h}, [sp], #64
    st1 {v12.8h, v13.8h, v14.8h, v15.8h}, [sp], #64
    stp x19, x20, [sp], #16
    stp x21, x22, [sp], #16

    ldr x8, [sp]
    ldr x9, [sp, #8]

.macro CLEAR_OUTPUT_V8_V9
    dup v8.4s, wzr
    dup v9.4s, wzr
.endm

.macro CLEAR_OUTPUT_V8_V11
    dup v8.4s, wzr
    dup v9.4s, wzr
    dup v10.4s, wzr
    dup v11.4s, wzr
.endm

.macro CLEAR_OUTPUT_V8_V15
    CLEAR_OUTPUT_V8_V11
    dup v12.4s, wzr
    dup v13.4s, wzr
    dup v14.4s, wzr
    dup v15.4s, wzr
.endm

.macro CLEAR_OUTPUT_V8_V23
    CLEAR_OUTPUT_V8_V15
    dup v16.4s, wzr
    dup v17.4s, wzr
    dup v18.4s, wzr
    dup v19.4s, wzr
    dup v20.4s, wzr
    dup v21.4s, wzr
    dup v22.4s, wzr
    dup v23.4s, wzr
.endm

.macro CLEAR_OUTPUT_V8_V31
    CLEAR_OUTPUT_V8_V23
    dup v24.4s, wzr
    dup v25.4s, wzr
    dup v26.4s, wzr
    dup v27.4s, wzr
    dup v28.4s, wzr
    dup v29.4s, wzr
    dup v30.4s, wzr
    dup v31.4s, wzr
.endm

    mov x21, #24
    mul x17, x5, x21    // input_stride : 12 * Ic * sizeof(float16_t)
    mov x21, #2
    mul x8, x8, x21     // output_stride

LoopRowStart:
    cmp x6, #1
    ble LoopRow1
    cmp x6, #2
    ble LoopRow2
    cmp x6, #4
    ble LoopRow4
    cmp x6, #8
    ble LoopRow8

LoopRow12:
    mov x14, x1 // cur_weight
    mov x13, x7 // reload_col
    mov x12, x3 // reload_bias

    LoopCol12:
        mov x11, x2 // cur_output
        mov x10, x0 // cur_input
        mov x19, x5 // reload_depth
        CLEAR_OUTPUT_V8_V31
        cmp x19, #2
        blt LoopDepth12One

    LoopDepth12:
        ld1 {v0.8h}, [x10], #16                // cur_input
        ld1 {v1.4h}, [x10], #8
        ld1 {v4.8h, v5.8h}, [x14], #32         // cur_weight
        fmla v8.8h, v4.8h, v0.h[0]
        fmla v9.8h, v5.8h, v0.h[0]
        fmla v10.8h, v4.8h, v0.h[1]
        fmla v11.8h, v5.8h, v0.h[1]
        fmla v12.8h, v4.8h, v0.h[2]
        fmla v13.8h, v5.8h, v0.h[2]
        fmla v14.8h, v4.8h, v0.h[3]
        fmla v15.8h, v5.8h, v0.h[3]
        fmla v16.8h, v4.8h, v0.h[4]
        fmla v17.8h, v5.8h, v0.h[4]
        fmla v18.8h, v4.8h, v0.h[5]
        fmla v19.8h, v5.8h, v0.h[5]
        ld1 {v2.8h}, [x10], #16  // cur_input
        ld1 {v3.4h}, [x10], #8
        ld1 {v6.8h, v7.8h}, [x14], #32         // cur_weight
        fmla v20.8h, v4.8h, v0.h[6]
        fmla v21.8h, v5.8h, v0.h[6]
        fmla v22.8h, v4.8h, v0.h[7]
        fmla v23.8h, v5.8h, v0.h[7]
        fmla v24.8h, v4.8h, v1.h[0]
        fmla v25.8h, v5.8h, v1.h[0]
        fmla v26.8h, v4.8h, v1.h[1]
        fmla v27.8h, v5.8h, v1.h[1]
        fmla v28.8h, v4.8h, v1.h[2]
        fmla v29.8h, v5.8h, v1.h[2]
        fmla v30.8h, v4.8h, v1.h[3]
        fmla v31.8h, v5.8h, v1.h[3]

        fmla v8.8h, v6.8h, v2.h[0]
        fmla v9.8h, v7.8h, v2.h[0]
        fmla v10.8h, v6.8h, v2.h[1]
        fmla v11.8h, v7.8h, v2.h[1]
        fmla v12.8h, v6.8h, v2.h[2]
        fmla v13.8h, v7.8h, v2.h[2]
        fmla v14.8h, v6.8h, v2.h[3]
        fmla v15.8h, v7.8h, v2.h[3]
        fmla v16.8h, v6.8h, v2.h[4]
        fmla v17.8h, v7.8h, v2.h[4]
        fmla v18.8h, v6.8h, v2.h[5]
        fmla v19.8h, v7.8h, v2.h[5]
        fmla v20.8h, v6.8h, v2.h[6]
        fmla v21.8h, v7.8h, v2.h[6]
        fmla v22.8h, v6.8h, v2.h[7]
        fmla v23.8h, v7.8h, v2.h[7]
        fmla v24.8h, v6.8h, v3.h[0]
        fmla v25.8h, v7.8h, v3.h[0]
        fmla v26.8h, v6.8h, v3.h[1]
        fmla v27.8h, v7.8h, v3.h[1]
        fmla v28.8h, v6.8h, v3.h[2]
        fmla v29.8h, v7.8h, v3.h[2]
        fmla v30.8h, v6.8h, v3.h[3]
        fmla v31.8h, v7.8h, v3.h[3]
        subs x19, x19, #2
        beq Bias12
        cmp x19, #2
        bge LoopDepth12

        LoopDepth12One:
            ld1 {v0.4h, v1.4h, v2.4h}, [x10], #24  // cur_input
            ld1 {v3.8h, v4.8h}, [x14], #32         // cur_weight
            fmla v8.8h, v3.8h, v0.h[0]
            fmla v9.8h, v4.8h, v0.h[0]
            fmla v10.8h, v3.8h, v0.h[1]
            fmla v11.8h, v4.8h, v0.h[1]
            fmla v12.8h, v3.8h, v0.h[2]
            fmla v13.8h, v4.8h, v0.h[2]
            fmla v14.8h, v3.8h, v0.h[3]
            fmla v15.8h, v4.8h, v0.h[3]
            fmla v16.8h, v3.8h, v1.h[0]
            fmla v17.8h, v4.8h, v1.h[0]
            fmla v18.8h, v3.8h, v1.h[1]
            fmla v19.8h, v4.8h, v1.h[1]
            fmla v20.8h, v3.8h, v1.h[2]
            fmla v21.8h, v4.8h, v1.h[2]
            fmla v22.8h, v3.8h, v1.h[3]
            fmla v23.8h, v4.8h, v1.h[3]
            fmla v24.8h, v3.8h, v2.h[0]
            fmla v25.8h, v4.8h, v2.h[0]
            fmla v26.8h, v3.8h, v2.h[1]
            fmla v27.8h, v4.8h, v2.h[1]
            fmla v28.8h, v3.8h, v2.h[2]
            fmla v29.8h, v4.8h, v2.h[2]
            fmla v30.8h, v3.8h, v2.h[3]
            fmla v31.8h, v4.8h, v2.h[3]
            subs x19, x19, #1
            bgt LoopDepth12One

        Bias12:
            cbz x3, Activation12
            ld1 {v0.8h, v1.8h}, [x12], #32
            fadd v8.8h, v8.8h, v0.8h
            fadd v9.8h, v9.8h, v1.8h
            fadd v10.8h, v10.8h, v0.8h
            fadd v11.8h, v11.8h, v1.8h
            fadd v12.8h, v12.8h, v0.8h
            fadd v13.8h, v13.8h, v1.8h
            fadd v14.8h, v14.8h, v0.8h
            fadd v15.8h, v15.8h, v1.8h
            fadd v16.8h, v16.8h, v0.8h
            fadd v17.8h, v17.8h, v1.8h
            fadd v18.8h, v18.8h, v0.8h
            fadd v19.8h, v19.8h, v1.8h
            fadd v20.8h, v20.8h, v0.8h
            fadd v21.8h, v21.8h, v1.8h
            fadd v22.8h, v22.8h, v0.8h
            fadd v23.8h, v23.8h, v1.8h
            fadd v24.8h, v24.8h, v0.8h
            fadd v25.8h, v25.8h, v1.8h
            fadd v26.8h, v26.8h, v0.8h
            fadd v27.8h, v27.8h, v1.8h
            fadd v28.8h, v28.8h, v0.8h
            fadd v29.8h, v29.8h, v1.8h
            fadd v30.8h, v30.8h, v0.8h
            fadd v31.8h, v31.8h, v1.8h

        Activation12:
            cmp x4, #3
            beq Relu612
            cmp x4, #1
            beq Relu12
            b Write

        Relu612:
            movi v2.8h, #0x46, lsl #8
            fmin v8.8h, v8.8h, v2.8h
            fmin v9.8h, v9.8h, v2.8h
            fmin v10.8h, v10.8h, v2.8h
            fmin v11.8h, v11.8h, v2.8h
            fmin v12.8h, v12.8h, v2.8h
            fmin v13.8h, v13.8h, v2.8h
            fmin v14.8h, v14.8h, v2.8h
            fmin v15.8h, v15.8h, v2.8h
            fmin v16.8h, v16.8h, v2.8h
            fmin v17.8h, v17.8h, v2.8h
            fmin v18.8h, v18.8h, v2.8h
            fmin v19.8h, v19.8h, v2.8h
            fmin v20.8h, v20.8h, v2.8h
            fmin v21.8h, v21.8h, v2.8h
            fmin v22.8h, v22.8h, v2.8h
            fmin v23.8h, v23.8h, v2.8h
            fmin v24.8h, v24.8h, v2.8h
            fmin v25.8h, v25.8h, v2.8h
            fmin v26.8h, v26.8h, v2.8h
            fmin v27.8h, v27.8h, v2.8h
            fmin v28.8h, v28.8h, v2.8h
            fmin v29.8h, v29.8h, v2.8h
            fmin v30.8h, v30.8h, v2.8h
            fmin v31.8h, v31.8h, v2.8h
        
        Relu12:
            dup v2.8h, wzr
            fmax v8.8h, v8.8h, v2.8h
            fmax v9.8h, v9.8h, v2.8h
            fmax v10.8h, v10.8h, v2.8h
            fmax v11.8h, v11.8h, v2.8h
            fmax v12.8h, v12.8h, v2.8h
            fmax v13.8h, v13.8h, v2.8h
            fmax v14.8h, v14.8h, v2.8h
            fmax v15.8h, v15.8h, v2.8h
            fmax v16.8h, v16.8h, v2.8h
            fmax v17.8h, v17.8h, v2.8h
            fmax v18.8h, v18.8h, v2.8h
            fmax v19.8h, v19.8h, v2.8h
            fmax v20.8h, v20.8h, v2.8h
            fmax v21.8h, v21.8h, v2.8h
            fmax v22.8h, v22.8h, v2.8h
            fmax v23.8h, v23.8h, v2.8h
            fmax v24.8h, v24.8h, v2.8h
            fmax v25.8h, v25.8h, v2.8h
            fmax v26.8h, v26.8h, v2.8h
            fmax v27.8h, v27.8h, v2.8h
            fmax v28.8h, v28.8h, v2.8h
            fmax v29.8h, v29.8h, v2.8h
            fmax v30.8h, v30.8h, v2.8h
            fmax v31.8h, v31.8h, v2.8h
            b Write

LoopRow8:
    mov x14, x1 // cur_weight
    mov x13, x7 // reload_col
    mov x12, x3 // reload_bias

    LoopCol8:
        mov x11, x2 // cur_output
        mov x10, x0 // cur_input
        mov x19, x5 // reload_depth
        CLEAR_OUTPUT_V8_V23
        cmp x19, #2
        blt LoopDepth8One

    LoopDepth8:
        ld1 {v0.8h}, [x10], #16                // cur_input
        ld1 {v1.4h}, [x10], #8
        ld1 {v4.8h, v5.8h}, [x14], #32         // cur_weight
        fmla v8.8h, v4.8h, v0.h[0]
        fmla v9.8h, v5.8h, v0.h[0]
        fmla v10.8h, v4.8h, v0.h[1]
        fmla v11.8h, v5.8h, v0.h[1]
        fmla v12.8h, v4.8h, v0.h[2]
        fmla v13.8h, v5.8h, v0.h[2]
        fmla v14.8h, v4.8h, v0.h[3]
        fmla v15.8h, v5.8h, v0.h[3]
        fmla v16.8h, v4.8h, v0.h[4]
        fmla v17.8h, v5.8h, v0.h[4]
        fmla v18.8h, v4.8h, v0.h[5]
        fmla v19.8h, v5.8h, v0.h[5]
        ld1 {v2.8h}, [x10], #16  // cur_input
        ld1 {v3.4h}, [x10], #8
        ld1 {v6.8h, v7.8h}, [x14], #32         // cur_weight
        fmla v20.8h, v4.8h, v0.h[6]
        fmla v21.8h, v5.8h, v0.h[6]
        fmla v22.8h, v4.8h, v0.h[7]
        fmla v23.8h, v5.8h, v0.h[7]

        fmla v8.8h, v6.8h, v2.h[0]
        fmla v9.8h, v7.8h, v2.h[0]
        fmla v10.8h, v6.8h, v2.h[1]
        fmla v11.8h, v7.8h, v2.h[1]
        fmla v12.8h, v6.8h, v2.h[2]
        fmla v13.8h, v7.8h, v2.h[2]
        fmla v14.8h, v6.8h, v2.h[3]
        fmla v15.8h, v7.8h, v2.h[3]
        fmla v16.8h, v6.8h, v2.h[4]
        fmla v17.8h, v7.8h, v2.h[4]
        fmla v18.8h, v6.8h, v2.h[5]
        fmla v19.8h, v7.8h, v2.h[5]
        fmla v20.8h, v6.8h, v2.h[6]
        fmla v21.8h, v7.8h, v2.h[6]
        fmla v22.8h, v6.8h, v2.h[7]
        fmla v23.8h, v7.8h, v2.h[7]
        subs x19, x19, #2
        beq Bias8
        cmp x19, #2
        bge LoopDepth8

        LoopDepth8One:
            ld1 {v0.4h, v1.4h, v2.4h}, [x10], #24  // cur_input
            ld1 {v3.8h, v4.8h}, [x14], #32         // cur_weight
            fmla v8.8h, v3.8h, v0.h[0]
            fmla v9.8h, v4.8h, v0.h[0]
            fmla v10.8h, v3.8h, v0.h[1]
            fmla v11.8h, v4.8h, v0.h[1]
            fmla v12.8h, v3.8h, v0.h[2]
            fmla v13.8h, v4.8h, v0.h[2]
            fmla v14.8h, v3.8h, v0.h[3]
            fmla v15.8h, v4.8h, v0.h[3]
            fmla v16.8h, v3.8h, v1.h[0]
            fmla v17.8h, v4.8h, v1.h[0]
            fmla v18.8h, v3.8h, v1.h[1]
            fmla v19.8h, v4.8h, v1.h[1]
            fmla v20.8h, v3.8h, v1.h[2]
            fmla v21.8h, v4.8h, v1.h[2]
            fmla v22.8h, v3.8h, v1.h[3]
            fmla v23.8h, v4.8h, v1.h[3]
            subs x19, x19, #1
            bgt LoopDepth8One

        Bias8:
            cbz x3, Activation8
            ld1 {v0.8h, v1.8h}, [x12], #32
            fadd v8.8h, v8.8h, v0.8h
            fadd v9.8h, v9.8h, v1.8h
            fadd v10.8h, v10.8h, v0.8h
            fadd v11.8h, v11.8h, v1.8h
            fadd v12.8h, v12.8h, v0.8h
            fadd v13.8h, v13.8h, v1.8h
            fadd v14.8h, v14.8h, v0.8h
            fadd v15.8h, v15.8h, v1.8h
            fadd v16.8h, v16.8h, v0.8h
            fadd v17.8h, v17.8h, v1.8h
            fadd v18.8h, v18.8h, v0.8h
            fadd v19.8h, v19.8h, v1.8h
            fadd v20.8h, v20.8h, v0.8h
            fadd v21.8h, v21.8h, v1.8h
            fadd v22.8h, v22.8h, v0.8h
            fadd v23.8h, v23.8h, v1.8h

        Activation8:
            cmp x4, #3
            beq Relu68
            cmp x4, #1
            beq Relu8
            b Write

        Relu68:
            movi v2.8h, #0x46, lsl #8
            fmin v8.8h, v8.8h, v2.8h
            fmin v9.8h, v9.8h, v2.8h
            fmin v10.8h, v10.8h, v2.8h
            fmin v11.8h, v11.8h, v2.8h
            fmin v12.8h, v12.8h, v2.8h
            fmin v13.8h, v13.8h, v2.8h
            fmin v14.8h, v14.8h, v2.8h
            fmin v15.8h, v15.8h, v2.8h
            fmin v16.8h, v16.8h, v2.8h
            fmin v17.8h, v17.8h, v2.8h
            fmin v18.8h, v18.8h, v2.8h
            fmin v19.8h, v19.8h, v2.8h
            fmin v20.8h, v20.8h, v2.8h
            fmin v21.8h, v21.8h, v2.8h
            fmin v22.8h, v22.8h, v2.8h
            fmin v23.8h, v23.8h, v2.8h
        
        Relu8:
            dup v2.8h, wzr
            fmax v8.8h, v8.8h, v2.8h
            fmax v9.8h, v9.8h, v2.8h
            fmax v10.8h, v10.8h, v2.8h
            fmax v11.8h, v11.8h, v2.8h
            fmax v12.8h, v12.8h, v2.8h
            fmax v13.8h, v13.8h, v2.8h
            fmax v14.8h, v14.8h, v2.8h
            fmax v15.8h, v15.8h, v2.8h
            fmax v16.8h, v16.8h, v2.8h
            fmax v17.8h, v17.8h, v2.8h
            fmax v18.8h, v18.8h, v2.8h
            fmax v19.8h, v19.8h, v2.8h
            fmax v20.8h, v20.8h, v2.8h
            fmax v21.8h, v21.8h, v2.8h
            fmax v22.8h, v22.8h, v2.8h
            fmax v23.8h, v23.8h, v2.8h
            b Write

LoopRow4:
    mov x14, x1 // cur_weight
    mov x13, x7 // reload_col
    mov x12, x3 // reload_bias

    LoopCol4:
        mov x11, x2 // cur_output
        mov x10, x0 // cur_input
        mov x19, x5 // reload_depth
        CLEAR_OUTPUT_V8_V15
        cmp x19, #2
        blt LoopDepth4One

    LoopDepth4:
        ld1 {v0.8h}, [x10], #16                // cur_input
        ld1 {v1.4h}, [x10], #8
        ld1 {v4.8h, v5.8h}, [x14], #32         // cur_weight
        fmla v8.8h, v4.8h, v0.h[0]
        fmla v9.8h, v5.8h, v0.h[0]
        fmla v10.8h, v4.8h, v0.h[1]
        fmla v11.8h, v5.8h, v0.h[1]
        fmla v12.8h, v4.8h, v0.h[2]
        fmla v13.8h, v5.8h, v0.h[2]
        fmla v14.8h, v4.8h, v0.h[3]
        fmla v15.8h, v5.8h, v0.h[3]
        ld1 {v2.8h}, [x10], #16  // cur_input
        ld1 {v3.4h}, [x10], #8
        ld1 {v6.8h, v7.8h}, [x14], #32         // cur_weight

        fmla v8.8h, v6.8h, v2.h[0]
        fmla v9.8h, v7.8h, v2.h[0]
        fmla v10.8h, v6.8h, v2.h[1]
        fmla v11.8h, v7.8h, v2.h[1]
        fmla v12.8h, v6.8h, v2.h[2]
        fmla v13.8h, v7.8h, v2.h[2]
        fmla v14.8h, v6.8h, v2.h[3]
        fmla v15.8h, v7.8h, v2.h[3]
        subs x19, x19, #2
        beq Bias4
        cmp x19, #2
        bge LoopDepth4

        LoopDepth4One:
            ld1 {v0.4h, v1.4h, v2.4h}, [x10], #24  // cur_input
            ld1 {v3.8h, v4.8h}, [x14], #32         // cur_weight
            fmla v8.8h, v3.8h, v0.h[0]
            fmla v9.8h, v4.8h, v0.h[0]
            fmla v10.8h, v3.8h, v0.h[1]
            fmla v11.8h, v4.8h, v0.h[1]
            fmla v12.8h, v3.8h, v0.h[2]
            fmla v13.8h, v4.8h, v0.h[2]
            fmla v14.8h, v3.8h, v0.h[3]
            fmla v15.8h, v4.8h, v0.h[3]
            subs x19, x19, #1
            bgt LoopDepth4One

        Bias4:
            cbz x3, Activation4
            ld1 {v0.8h, v1.8h}, [x12], #32
            fadd v8.8h, v8.8h, v0.8h
            fadd v9.8h, v9.8h, v1.8h
            fadd v10.8h, v10.8h, v0.8h
            fadd v11.8h, v11.8h, v1.8h
            fadd v12.8h, v12.8h, v0.8h
            fadd v13.8h, v13.8h, v1.8h
            fadd v14.8h, v14.8h, v0.8h
            fadd v15.8h, v15.8h, v1.8h

        Activation4:
            cmp x4, #3
            beq Relu64
            cmp x4, #1
            beq Relu4
            b Write

        Relu64:
            movi v2.8h, #0x46, lsl #8
            fmin v8.8h, v8.8h, v2.8h
            fmin v9.8h, v9.8h, v2.8h
            fmin v10.8h, v10.8h, v2.8h
            fmin v11.8h, v11.8h, v2.8h
            fmin v12.8h, v12.8h, v2.8h
            fmin v13.8h, v13.8h, v2.8h
            fmin v14.8h, v14.8h, v2.8h
            fmin v15.8h, v15.8h, v2.8h
        
        Relu4:
            dup v2.8h, wzr
            fmax v8.8h, v8.8h, v2.8h
            fmax v9.8h, v9.8h, v2.8h
            fmax v10.8h, v10.8h, v2.8h
            fmax v11.8h, v11.8h, v2.8h
            fmax v12.8h, v12.8h, v2.8h
            fmax v13.8h, v13.8h, v2.8h
            fmax v14.8h, v14.8h, v2.8h
            fmax v15.8h, v15.8h, v2.8h
            b Write

LoopRow2:
    mov x14, x1 // cur_weight
    mov x13, x7 // reload_col
    mov x12, x3 // reload_bias

    LoopCol2:
        mov x11, x2 // cur_output
        mov x10, x0 // cur_input
        mov x19, x5 // reload_depth
        CLEAR_OUTPUT_V8_V11
        cmp x19, #2
        blt LoopDepth2One

    LoopDepth2:
        ld1 {v0.8h}, [x10], #16                // cur_input
        ld1 {v1.4h}, [x10], #8
        ld1 {v4.8h, v5.8h}, [x14], #32         // cur_weight
        fmla v8.8h, v4.8h, v0.h[0]
        fmla v9.8h, v5.8h, v0.h[0]
        fmla v10.8h, v4.8h, v0.h[1]
        fmla v11.8h, v5.8h, v0.h[1]
        ld1 {v2.8h}, [x10], #16  // cur_input
        ld1 {v3.4h}, [x10], #8
        ld1 {v6.8h, v7.8h}, [x14], #32         // cur_weight

        fmla v8.8h, v6.8h, v2.h[0]
        fmla v9.8h, v7.8h, v2.h[0]
        fmla v10.8h, v6.8h, v2.h[1]
        fmla v11.8h, v7.8h, v2.h[1]
        subs x19, x19, #2
        beq Bias2
        cmp x19, #2
        bge LoopDepth2

        LoopDepth2One:
            ld1 {v0.4h, v1.4h, v2.4h}, [x10], #24  // cur_input
            ld1 {v3.8h, v4.8h}, [x14], #32         // cur_weight
            fmla v8.8h, v3.8h, v0.h[0]
            fmla v9.8h, v4.8h, v0.h[0]
            fmla v10.8h, v3.8h, v0.h[1]
            fmla v11.8h, v4.8h, v0.h[1]
            subs x19, x19, #1
            bgt LoopDepth2One

        Bias2:
            cbz x3, Activation2
            ld1 {v0.8h, v1.8h}, [x12], #32
            fadd v8.8h, v8.8h, v0.8h
            fadd v9.8h, v9.8h, v1.8h
            fadd v10.8h, v10.8h, v0.8h
            fadd v11.8h, v11.8h, v1.8h

        Activation2:
            cmp x4, #3
            beq Relu62
            cmp x4, #1
            beq Relu2
            b Write

        Relu62:
            movi v2.8h, #0x46, lsl #8
            fmin v8.8h, v8.8h, v2.8h
            fmin v9.8h, v9.8h, v2.8h
            fmin v10.8h, v10.8h, v2.8h
            fmin v11.8h, v11.8h, v2.8h
        
        Relu2:
            dup v2.8h, wzr
            fmax v8.8h, v8.8h, v2.8h
            fmax v9.8h, v9.8h, v2.8h
            fmax v10.8h, v10.8h, v2.8h
            fmax v11.8h, v11.8h, v2.8h
            b Write

LoopRow1:
    mov x14, x1 // cur_weight
    mov x13, x7 // reload_col
    mov x12, x3 // reload_bias

    LoopCol1:
        mov x11, x2 // cur_output
        mov x10, x0 // cur_input
        mov x19, x5 // reload_depth
        CLEAR_OUTPUT_V8_V9
        cmp x19, #2
        blt LoopDepth1One

    LoopDepth1:
        ld1 {v0.8h}, [x10], #16                // cur_input
        ld1 {v1.4h}, [x10], #8
        ld1 {v4.8h, v5.8h}, [x14], #32         // cur_weight
        fmla v8.8h, v4.8h, v0.h[0]
        fmla v9.8h, v5.8h, v0.h[0]
        ld1 {v2.8h}, [x10], #16  // cur_input
        ld1 {v3.4h}, [x10], #8
        ld1 {v6.8h, v7.8h}, [x14], #32         // cur_weight

        fmla v8.8h, v6.8h, v2.h[0]
        fmla v9.8h, v7.8h, v2.h[0]
        subs x19, x19, #2
        beq Bias1
        cmp x19, #2
        bge LoopDepth1

        LoopDepth1One:
            ld1 {v0.4h, v1.4h, v2.4h}, [x10], #24  // cur_input
            ld1 {v3.8h, v4.8h}, [x14], #32         // cur_weight
            fmla v8.8h, v3.8h, v0.h[0]
            fmla v9.8h, v4.8h, v0.h[0]
            subs x19, x19, #1
            bgt LoopDepth1One

        Bias1:
            cbz x3, Activation1
            ld1 {v0.8h, v1.8h}, [x12], #32
            fadd v8.8h, v8.8h, v0.8h
            fadd v9.8h, v9.8h, v1.8h

        Activation1:
            cmp x4, #3
            beq Relu61
            cmp x4, #1
            beq Relu1
            b Write

        Relu61:
            movi v2.8h, #0x46, lsl #8
            fmin v8.8h, v8.8h, v2.8h
            fmin v9.8h, v9.8h, v2.8h
        
        Relu1:
            dup v2.8h, wzr
            fmax v8.8h, v8.8h, v2.8h
            fmax v9.8h, v9.8h, v2.8h
            b Write

        Write:
            cmp x13, #1
            beq Write1
            cmp x13, #2
            beq Write2
            cmp x13, #3
            beq Write3
            cmp x13, #4
            beq Write4
            cmp x13, #5
            beq Write5
            cmp x13, #6
            beq Write6
            cmp x13, #7
            beq Write7
            cmp x13, #8
            beq Write8
            cmp x13, #9
            beq Write9
            cmp x13, #10
            beq Write10
            cmp x13, #11
            beq Write11
            cmp x13, #12
            beq Write12
            cmp x13, #13
            beq Write13
            cmp x13, #14
            beq Write14
            cmp x13, #15
            beq Write15
            b Write16

        Write1:
            add x2, x2, #2
            str h8, [x11]
            cmp x6, #1
            beq WriteEnd
            add x11, x11, x8
            str h10, [x11]
            cmp x6, #2
            beq WriteEnd
            add x11, x11, x8
            str h12, [x11]
            cmp x6, #3
            beq WriteEnd
            add x11, x11, x8
            str h14, [x11]
            cmp x6, #4
            beq WriteEnd
            add x11, x11, x8
            str h16, [x11]
            cmp x6, #5
            beq WriteEnd
            add x11, x11, x8
            str h18, [x11]
            cmp x6, #6
            beq WriteEnd
            add x11, x11, x8
            str h20, [x11]
            cmp x6, #7
            beq WriteEnd
            add x11, x11, x8
            str h22, [x11]
            cmp x6, #8
            beq WriteEnd
            add x11, x11, x8
            str h24, [x11]
            cmp x6, #9
            beq WriteEnd
            add x11, x11, x8
            str h26, [x11]
            cmp x6, #10
            beq WriteEnd
            add x11, x11, x8
            str h28, [x11]
            cmp x6, #11
            beq WriteEnd
            add x11, x11, x8
            str h30, [x11]
            add x11, x11, x8
            add x11, x11, #2
            b WriteEnd

        Write2:
            add x2, x2, #4
            add x19, x11, #2
            st1 {v8.h}[0], [x11], x8
            st1 {v8.h}[1], [x19], x8
            cmp x6, #1
            beq WriteEnd
            st1 {v10.h}[0], [x11], x8
            st1 {v10.h}[1], [x19], x8
            cmp x6, #2
            beq WriteEnd
            st1 {v12.h}[0], [x11], x8
            st1 {v12.h}[1], [x19], x8
            cmp x6, #3
            beq WriteEnd
            st1 {v14.h}[0], [x11], x8
            st1 {v14.h}[1], [x19], x8
            cmp x6, #4
            beq WriteEnd
            st1 {v16.h}[0], [x11], x8
            st1 {v16.h}[1], [x19], x8
            cmp x6, #5
            beq WriteEnd
            st1 {v18.h}[0], [x11], x8
            st1 {v18.h}[1], [x19], x8
            cmp x6, #6
            beq WriteEnd
            st1 {v20.h}[0], [x11], x8
            st1 {v20.h}[1], [x19], x8
            cmp x6, #7
            beq WriteEnd
            st1 {v22.h}[0], [x11], x8
            st1 {v22.h}[1], [x19], x8
            cmp x6, #8
            beq WriteEnd
            st1 {v24.h}[0], [x11], x8
            st1 {v24.h}[1], [x19], x8
            cmp x6, #9
            beq WriteEnd
            st1 {v26.h}[0], [x11], x8
            st1 {v26.h}[1], [x19], x8
            cmp x6, #10
            beq WriteEnd
            st1 {v28.h}[0], [x11], x8
            st1 {v28.h}[1], [x19], x8
            cmp x6, #11
            beq WriteEnd
            st1 {v30.h}[0], [x11], x8
            st1 {v30.h}[1], [x19], x8
            add x11, x11, #4
            b WriteEnd

        Write3:
            add x2, x2, #6
            add x19, x11, #4
            add x20, x11, #2
            st1 {v8.h}[0], [x11], x8
            st1 {v8.h}[1], [x20], x8
            st1 {v8.h}[2], [x19], x8
            cmp x6, #1
            beq WriteEnd
            st1 {v10.h}[0], [x11], x8
            st1 {v10.h}[1], [x20], x8
            st1 {v10.h}[2], [x19], x8
            cmp x6, #2
            beq WriteEnd
            st1 {v12.h}[0], [x11], x8
            st1 {v12.h}[1], [x20], x8
            st1 {v12.h}[2], [x19], x8
            cmp x6, #3
            beq WriteEnd
            st1 {v14.h}[0], [x11], x8
            st1 {v14.h}[1], [x20], x8
            st1 {v14.h}[2], [x19], x8
            cmp x6, #4
            beq WriteEnd
            st1 {v16.h}[0], [x11], x8
            st1 {v16.h}[1], [x20], x8
            st1 {v16.h}[2], [x19], x8
            cmp x6, #5
            beq WriteEnd
            st1 {v18.h}[0], [x11], x8
            st1 {v18.h}[1], [x20], x8
            st1 {v18.h}[2], [x19], x8
            cmp x6, #6
            beq WriteEnd
            st1 {v20.h}[0], [x11], x8
            st1 {v20.h}[1], [x20], x8
            st1 {v20.h}[2], [x19], x8
            cmp x6, #7
            beq WriteEnd
            st1 {v22.h}[0], [x11], x8
            st1 {v22.h}[1], [x20], x8
            st1 {v22.h}[2], [x19], x8
            cmp x6, #8
            beq WriteEnd
            st1 {v24.h}[0], [x11], x8
            st1 {v24.h}[1], [x20], x8
            st1 {v24.h}[2], [x19], x8
            cmp x6, #9
            beq WriteEnd
            st1 {v26.h}[0], [x11], x8
            st1 {v26.h}[1], [x20], x8
            st1 {v26.h}[2], [x19], x8
            cmp x6, #10
            beq WriteEnd
            st1 {v28.h}[0], [x11], x8
            st1 {v28.h}[1], [x20], x8
            st1 {v28.h}[2], [x19], x8
            cmp x6, #11
            beq WriteEnd
            st1 {v30.h}[0], [x11], x8
            st1 {v30.h}[1], [x20], x8
            st1 {v30.h}[2], [x19], x8
            add x11, x11, #6
            b WriteEnd

        Write4:
            add x2, x2, #8
            st1 {v8.4h}, [x11], x8
            cmp x6, #1
            beq WriteEnd
            st1 {v10.4h}, [x11], x8
            cmp x6, #2
            beq WriteEnd
            st1 {v12.4h}, [x11], x8
            cmp x6, #3
            beq WriteEnd
            st1 {v14.4h}, [x11], x8
            cmp x6, #4
            beq WriteEnd
            st1 {v16.4h}, [x11], x8
            cmp x6, #5
            beq WriteEnd
            st1 {v18.4h}, [x11], x8
            cmp x6, #6
            beq WriteEnd
            st1 {v20.4h}, [x11], x8
            cmp x6, #7
            beq WriteEnd
            st1 {v22.4h}, [x11], x8
            cmp x6, #8
            beq WriteEnd
            st1 {v24.4h}, [x11], x8
            cmp x6, #9
            beq WriteEnd
            st1 {v26.4h}, [x11], x8
            cmp x6, #10
            beq WriteEnd
            st1 {v28.4h}, [x11], x8
            cmp x6, #11
            beq WriteEnd
            st1 {v30.4h}, [x11], x8
            add x11, x11, #8
            b WriteEnd
        Write5:
            add x2, x2, #10
            add x19, x11, #8
            st1 {v8.4h}, [x11], x8
            st1 {v8.h}[4], [x19], x8
            cmp x6, #1
            beq WriteEnd
            st1 {v10.4h}, [x11], x8
            st1 {v10.h}[4], [x19], x8
            cmp x6, #2
            beq WriteEnd
            st1 {v12.4h}, [x11], x8
            st1 {v12.h}[4], [x19], x8
            cmp x6, #3
            beq WriteEnd
            st1 {v14.4h}, [x11], x8
            st1 {v14.h}[4], [x19], x8
            cmp x6, #4
            beq WriteEnd
            st1 {v16.4h}, [x11], x8
            st1 {v16.h}[4], [x19], x8
            cmp x6, #5
            beq WriteEnd
            st1 {v18.4h}, [x11], x8
            st1 {v18.h}[4], [x19], x8
            cmp x6, #6
            beq WriteEnd
            st1 {v20.4h}, [x11], x8
            st1 {v20.h}[4], [x19], x8
            cmp x6, #7
            beq WriteEnd
            st1 {v22.4h}, [x11], x8
            st1 {v22.h}[4], [x19], x8
            cmp x6, #8
            beq WriteEnd
            st1 {v24.4h}, [x11], x8
            st1 {v24.h}[4], [x19], x8
            cmp x6, #9
            beq WriteEnd
            st1 {v26.4h}, [x11], x8
            st1 {v26.h}[4], [x19], x8
            cmp x6, #10
            beq WriteEnd
            st1 {v28.4h}, [x11], x8
            st1 {v28.h}[4], [x19], x8
            cmp x6, #11
            beq WriteEnd
            st1 {v30.4h}, [x11], x8
            st1 {v30.h}[4], [x19], x8
            add x11, x11, #10
            b WriteEnd
        Write6:
            add x2, x2, #12
            add x19, x11, #8
            add x20, x11, #10
            st1 {v8.4h}, [x11], x8
            st1 {v8.h}[4], [x19], x8
            st1 {v8.h}[5], [x20], x8
            cmp x6, #1
            beq WriteEnd
            st1 {v10.4h}, [x11], x8
            st1 {v10.h}[4], [x19], x8
            st1 {v10.h}[5], [x20], x8
            cmp x6, #2
            beq WriteEnd
            st1 {v12.4h}, [x11], x8
            st1 {v12.h}[4], [x19], x8
            st1 {v12.h}[5], [x20], x8
            cmp x6, #3
            beq WriteEnd
            st1 {v14.4h}, [x11], x8
            st1 {v14.h}[4], [x19], x8
            st1 {v14.h}[5], [x20], x8
            cmp x6, #4
            beq WriteEnd
            st1 {v16.4h}, [x11], x8
            st1 {v16.h}[4], [x19], x8
            st1 {v16.h}[5], [x20], x8
            cmp x6, #5
            beq WriteEnd
            st1 {v18.4h}, [x11], x8
            st1 {v18.h}[4], [x19], x8
            st1 {v18.h}[5], [x20], x8
            cmp x6, #6
            beq WriteEnd
            st1 {v20.4h}, [x11], x8
            st1 {v20.h}[4], [x19], x8
            st1 {v20.h}[5], [x20], x8
            cmp x6, #7
            beq WriteEnd
            st1 {v22.4h}, [x11], x8
            st1 {v22.h}[4], [x19], x8
            st1 {v22.h}[5], [x20], x8
            cmp x6, #8
            beq WriteEnd
            st1 {v24.4h}, [x11], x8
            st1 {v24.h}[4], [x19], x8
            st1 {v24.h}[5], [x20], x8
            cmp x6, #9
            beq WriteEnd
            st1 {v26.4h}, [x11], x8
            st1 {v26.h}[4], [x19], x8
            st1 {v26.h}[5], [x20], x8
            cmp x6, #10
            beq WriteEnd
            st1 {v28.4h}, [x11], x8
            st1 {v28.h}[4], [x19], x8
            st1 {v28.h}[5], [x20], x8
            cmp x6, #11
            beq WriteEnd
            st1 {v30.4h}, [x11], x8
            st1 {v30.h}[4], [x19], x8
            st1 {v30.h}[5], [x20], x8
            add x11, x11, #12
            b WriteEnd
        Write7:
            add x2, x2, #14
            add x19, x11, #8
            add x20, x11, #10
            add x10, x11, #12
            st1 {v8.4h}, [x11], x8
            st1 {v8.h}[4], [x19], x8
            st1 {v8.h}[5], [x20], x8
            st1 {v8.h}[6], [x10], x8
            cmp x6, #1
            beq WriteEnd
            st1 {v10.4h}, [x11], x8
            st1 {v10.h}[4], [x19], x8
            st1 {v10.h}[5], [x20], x8
            st1 {v10.h}[6], [x10], x8
            cmp x6, #2
            beq WriteEnd
            st1 {v12.4h}, [x11], x8
            st1 {v12.h}[4], [x19], x8
            st1 {v12.h}[5], [x20], x8
            st1 {v12.h}[6], [x10], x8
            cmp x6, #3
            beq WriteEnd
            st1 {v14.4h}, [x11], x8
            st1 {v14.h}[4], [x19], x8
            st1 {v14.h}[5], [x20], x8
            st1 {v14.h}[6], [x10], x8
            cmp x6, #4
            beq WriteEnd
            st1 {v16.4h}, [x11], x8
            st1 {v16.h}[4], [x19], x8
            st1 {v16.h}[5], [x20], x8
            st1 {v16.h}[6], [x10], x8
            cmp x6, #5
            beq WriteEnd
            st1 {v18.4h}, [x11], x8
            st1 {v18.h}[4], [x19], x8
            st1 {v18.h}[5], [x20], x8
            st1 {v18.h}[6], [x10], x8
            cmp x6, #6
            beq WriteEnd
            st1 {v20.4h}, [x11], x8
            st1 {v20.h}[4], [x19], x8
            st1 {v20.h}[5], [x20], x8
            st1 {v20.h}[6], [x10], x8
            cmp x6, #7
            beq WriteEnd
            st1 {v22.4h}, [x11], x8
            st1 {v22.h}[4], [x19], x8
            st1 {v22.h}[5], [x20], x8
            st1 {v22.h}[6], [x10], x8
            cmp x6, #8
            beq WriteEnd
            st1 {v24.4h}, [x11], x8
            st1 {v24.h}[4], [x19], x8
            st1 {v24.h}[5], [x20], x8
            st1 {v24.h}[6], [x10], x8
            cmp x6, #9
            beq WriteEnd
            st1 {v26.4h}, [x11], x8
            st1 {v26.h}[4], [x19], x8
            st1 {v26.h}[5], [x20], x8
            st1 {v26.h}[6], [x10], x8
            cmp x6, #10
            beq WriteEnd
            st1 {v28.4h}, [x11], x8
            st1 {v28.h}[4], [x19], x8
            st1 {v28.h}[5], [x20], x8
            st1 {v28.h}[6], [x10], x8
            cmp x6, #11
            beq WriteEnd
            st1 {v30.4h}, [x11], x8
            st1 {v30.h}[4], [x19], x8
            st1 {v30.h}[5], [x20], x8
            st1 {v30.h}[6], [x10], x8
            add x11, x11, #14
            b WriteEnd
        Write8:
            add x2, x2, #16
            st1 {v8.8h}, [x11], x8
            cmp x6, #1
            beq WriteEnd
            st1 {v10.8h}, [x11], x8
            cmp x6, #2
            beq WriteEnd
            st1 {v12.8h}, [x11], x8
            cmp x6, #3
            beq WriteEnd
            st1 {v14.8h}, [x11], x8
            cmp x6, #4
            beq WriteEnd
            st1 {v16.8h}, [x11], x8
            cmp x6, #5
            beq WriteEnd
            st1 {v18.8h}, [x11], x8
            cmp x6, #6
            beq WriteEnd
            st1 {v20.8h}, [x11], x8
            cmp x6, #7
            beq WriteEnd
            st1 {v22.8h}, [x11], x8
            cmp x6, #8
            beq WriteEnd
            st1 {v24.8h}, [x11], x8
            cmp x6, #9
            beq WriteEnd
            st1 {v26.8h}, [x11], x8
            cmp x6, #10
            beq WriteEnd
            st1 {v28.8h}, [x11], x8
            cmp x6, #11
            beq WriteEnd
            st1 {v30.8h}, [x11], x8
            add x11, x11, #16
            b WriteEnd
        Write9:
            add x2, x2, #18
            add x19, x11, #16
            st1 {v8.8h}, [x11], x8
            st1 {v9.h}[0], [x19], x8
            cmp x6, #1
            beq WriteEnd
            st1 {v10.8h}, [x11], x8
            st1 {v11.h}[0], [x19], x8
            cmp x6, #2
            beq WriteEnd
            st1 {v12.8h}, [x11], x8
            st1 {v13.h}[0], [x19], x8
            cmp x6, #3
            beq WriteEnd
            st1 {v14.8h}, [x11], x8
            st1 {v15.h}[0], [x19], x8
            cmp x6, #4
            beq WriteEnd
            st1 {v16.8h}, [x11], x8
            st1 {v17.h}[0], [x19], x8
            cmp x6, #5
            beq WriteEnd
            st1 {v18.8h}, [x11], x8
            st1 {v19.h}[0], [x19], x8
            cmp x6, #6
            beq WriteEnd
            st1 {v20.8h}, [x11], x8
            st1 {v21.h}[0], [x19], x8
            cmp x6, #7
            beq WriteEnd
            st1 {v22.8h}, [x11], x8
            st1 {v23.h}[0], [x19], x8
            cmp x6, #8
            beq WriteEnd
            st1 {v24.8h}, [x11], x8
            st1 {v25.h}[0], [x19], x8
            cmp x6, #9
            beq WriteEnd
            st1 {v26.8h}, [x11], x8
            st1 {v27.h}[0], [x19], x8
            cmp x6, #10
            beq WriteEnd
            st1 {v28.8h}, [x11], x8
            st1 {v29.h}[0], [x19], x8
            cmp x6, #11
            beq WriteEnd
            st1 {v30.8h}, [x11], x8
            st1 {v31.h}[0], [x19], x8
            add x11, x11, #18
            b WriteEnd
        Write10:
            add x2, x2, #20
            add x19, x11, #16
            add x20, x11, #18
            st1 {v8.8h}, [x11], x8
            st1 {v9.h}[0], [x19], x8
            st1 {v9.h}[1], [x20], x8
            cmp x6, #1
            beq WriteEnd
            st1 {v10.8h}, [x11], x8
            st1 {v11.h}[0], [x19], x8
            st1 {v11.h}[1], [x20], x8
            cmp x6, #2
            beq WriteEnd
            st1 {v12.8h}, [x11], x8
            st1 {v13.h}[0], [x19], x8
            st1 {v13.h}[1], [x20], x8
            cmp x6, #3
            beq WriteEnd
            st1 {v14.8h}, [x11], x8
            st1 {v15.h}[0], [x19], x8
            st1 {v15.h}[1], [x20], x8
            cmp x6, #4
            beq WriteEnd
            st1 {v16.8h}, [x11], x8
            st1 {v17.h}[0], [x19], x8
            st1 {v17.h}[1], [x20], x8
            cmp x6, #5
            beq WriteEnd
            st1 {v18.8h}, [x11], x8
            st1 {v19.h}[0], [x19], x8
            st1 {v19.h}[1], [x20], x8
            cmp x6, #6
            beq WriteEnd
            st1 {v20.8h}, [x11], x8
            st1 {v21.h}[0], [x19], x8
            st1 {v21.h}[1], [x20], x8
            cmp x6, #7
            beq WriteEnd
            st1 {v22.8h}, [x11], x8
            st1 {v23.h}[0], [x19], x8
            st1 {v23.h}[1], [x20], x8
            cmp x6, #8
            beq WriteEnd
            st1 {v24.8h}, [x11], x8
            st1 {v25.h}[0], [x19], x8
            st1 {v25.h}[1], [x20], x8
            cmp x6, #9
            beq WriteEnd
            st1 {v26.8h}, [x11], x8
            st1 {v27.h}[0], [x19], x8
            st1 {v27.h}[1], [x20], x8
            cmp x6, #10
            beq WriteEnd
            st1 {v28.8h}, [x11], x8
            st1 {v29.h}[0], [x19], x8
            st1 {v29.h}[1], [x20], x8
            cmp x6, #11
            beq WriteEnd
            st1 {v30.8h}, [x11], x8
            st1 {v31.h}[0], [x19], x8
            st1 {v31.h}[1], [x20], x8
            add x11, x11, #20
            b WriteEnd
        Write11:
            add x2, x2, #22
            add x19, x11, #16
            add x20, x11, #18
            add x10, x11, #20
            st1 {v8.8h}, [x11], x8
            st1 {v9.h}[0], [x19], x8
            st1 {v9.h}[1], [x20], x8
            st1 {v9.h}[2], [x10], x8
            cmp x6, #1
            beq WriteEnd
            st1 {v10.8h}, [x11], x8
            st1 {v11.h}[0], [x19], x8
            st1 {v11.h}[1], [x20], x8
            st1 {v11.h}[2], [x10], x8
            cmp x6, #2
            beq WriteEnd
            st1 {v12.8h}, [x11], x8
            st1 {v13.h}[0], [x19], x8
            st1 {v13.h}[1], [x20], x8
            st1 {v13.h}[2], [x10], x8
            cmp x6, #3
            beq WriteEnd
            st1 {v14.8h}, [x11], x8
            st1 {v15.h}[0], [x19], x8
            st1 {v15.h}[1], [x20], x8
            st1 {v15.h}[2], [x10], x8
            cmp x6, #4
            beq WriteEnd
            st1 {v16.8h}, [x11], x8
            st1 {v17.h}[0], [x19], x8
            st1 {v17.h}[1], [x20], x8
            st1 {v17.h}[2], [x10], x8
            cmp x6, #5
            beq WriteEnd
            st1 {v18.8h}, [x11], x8
            st1 {v19.h}[0], [x19], x8
            st1 {v19.h}[1], [x20], x8
            st1 {v19.h}[2], [x10], x8
            cmp x6, #6
            beq WriteEnd
            st1 {v20.8h}, [x11], x8
            st1 {v21.h}[0], [x19], x8
            st1 {v21.h}[1], [x20], x8
            st1 {v21.h}[2], [x10], x8
            cmp x6, #7
            beq WriteEnd
            st1 {v22.8h}, [x11], x8
            st1 {v23.h}[0], [x19], x8
            st1 {v23.h}[1], [x20], x8
            st1 {v23.h}[2], [x10], x8
            cmp x6, #8
            beq WriteEnd
            st1 {v24.8h}, [x11], x8
            st1 {v25.h}[0], [x19], x8
            st1 {v25.h}[1], [x20], x8
            st1 {v25.h}[2], [x10], x8
            cmp x6, #9
            beq WriteEnd
            st1 {v26.8h}, [x11], x8
            st1 {v27.h}[0], [x19], x8
            st1 {v27.h}[1], [x20], x8
            st1 {v27.h}[2], [x10], x8
            cmp x6, #10
            beq WriteEnd
            st1 {v28.8h}, [x11], x8
            st1 {v29.h}[0], [x19], x8
            st1 {v29.h}[1], [x20], x8
            st1 {v29.h}[2], [x10], x8
            cmp x6, #11
            beq WriteEnd
            st1 {v30.8h}, [x11], x8
            st1 {v31.h}[0], [x19], x8
            st1 {v31.h}[1], [x20], x8
            st1 {v31.h}[2], [x10], x8
            add x11, x11, #22
            b WriteEnd
        Write12:
            add x2, x2, #24
            add x19, x11, #16
            st1 {v8.8h}, [x11], x8
            st1 {v9.4h}, [x19], x8
            cmp x6, #1
            beq WriteEnd
            st1 {v10.8h}, [x11], x8
            st1 {v11.4h}, [x19], x8
            cmp x6, #2
            beq WriteEnd
            st1 {v12.8h}, [x11], x8
            st1 {v13.4h}, [x19], x8
            cmp x6, #3
            beq WriteEnd
            st1 {v14.8h}, [x11], x8
            st1 {v15.4h}, [x19], x8
            cmp x6, #4
            beq WriteEnd
            st1 {v16.8h}, [x11], x8
            st1 {v17.4h}, [x19], x8
            cmp x6, #5
            beq WriteEnd
            st1 {v18.8h}, [x11], x8
            st1 {v19.4h}, [x19], x8
            cmp x6, #6
            beq WriteEnd
            st1 {v20.8h}, [x11], x8
            st1 {v21.4h}, [x19], x8
            cmp x6, #7
            beq WriteEnd
            st1 {v22.8h}, [x11], x8
            st1 {v23.4h}, [x19], x8
            cmp x6, #8
            beq WriteEnd
            st1 {v24.8h}, [x11], x8
            st1 {v25.4h}, [x19], x8
            cmp x6, #9
            beq WriteEnd
            st1 {v26.8h}, [x11], x8
            st1 {v27.4h}, [x19], x8
            cmp x6, #10
            beq WriteEnd
            st1 {v28.8h}, [x11], x8
            st1 {v29.4h}, [x19], x8
            cmp x6, #11
            beq WriteEnd
            st1 {v30.8h}, [x11], x8
            st1 {v31.4h}, [x19], x8
            add x11, x11, #24
            b WriteEnd
        Write13:
            add x2, x2, #26
            add x19, x11, #16
            add x20, x11, #24
            st1 {v8.8h}, [x11], x8
            st1 {v9.4h}, [x19], x8
            st1 {v9.h}[4], [x20], x8
            cmp x6, #1
            beq WriteEnd
            st1 {v10.8h}, [x11], x8
            st1 {v11.4h}, [x19], x8
            st1 {v11.h}[4], [x20], x8
            cmp x6, #2
            beq WriteEnd
            st1 {v12.8h}, [x11], x8
            st1 {v13.4h}, [x19], x8
            st1 {v13.h}[4], [x20], x8
            cmp x6, #3
            beq WriteEnd
            st1 {v14.8h}, [x11], x8
            st1 {v15.4h}, [x19], x8
            st1 {v15.h}[4], [x20], x8
            cmp x6, #4
            beq WriteEnd
            st1 {v16.8h}, [x11], x8
            st1 {v17.4h}, [x19], x8
            st1 {v17.h}[4], [x20], x8
            cmp x6, #5
            beq WriteEnd
            st1 {v18.8h}, [x11], x8
            st1 {v19.4h}, [x19], x8
            st1 {v19.h}[4], [x20], x8
            cmp x6, #6
            beq WriteEnd
            st1 {v20.8h}, [x11], x8
            st1 {v21.4h}, [x19], x8
            st1 {v21.h}[4], [x20], x8
            cmp x6, #7
            beq WriteEnd
            st1 {v22.8h}, [x11], x8
            st1 {v23.4h}, [x19], x8
            st1 {v23.h}[4], [x20], x8
            cmp x6, #8
            beq WriteEnd
            st1 {v24.8h}, [x11], x8
            st1 {v25.4h}, [x19], x8
            st1 {v25.h}[4], [x20], x8
            cmp x6, #9
            beq WriteEnd
            st1 {v26.8h}, [x11], x8
            st1 {v27.4h}, [x19], x8
            st1 {v27.h}[4], [x20], x8
            cmp x6, #10
            beq WriteEnd
            st1 {v28.8h}, [x11], x8
            st1 {v29.4h}, [x19], x8
            st1 {v29.h}[4], [x20], x8
            cmp x6, #11
            beq WriteEnd
            st1 {v30.8h}, [x11], x8
            st1 {v31.4h}, [x19], x8
            st1 {v31.h}[4], [x20], x8
            add x11, x11, #26
            b WriteEnd
        Write14:
            add x2, x2, #28
            add x19, x11, #16
            add x20, x11, #24
            add x10, x11, #26
            st1 {v8.8h}, [x11], x8
            st1 {v9.4h}, [x19], x8
            st1 {v9.h}[4], [x20], x8
            st1 {v9.h}[5], [x10], x8
            cmp x6, #1
            beq WriteEnd
            st1 {v10.8h}, [x11], x8
            st1 {v11.4h}, [x19], x8
            st1 {v11.h}[4], [x20], x8
            st1 {v11.h}[5], [x10], x8
            cmp x6, #2
            beq WriteEnd
            st1 {v12.8h}, [x11], x8
            st1 {v13.4h}, [x19], x8
            st1 {v13.h}[4], [x20], x8
            st1 {v13.h}[5], [x10], x8
            cmp x6, #3
            beq WriteEnd
            st1 {v14.8h}, [x11], x8
            st1 {v15.4h}, [x19], x8
            st1 {v15.h}[4], [x20], x8
            st1 {v15.h}[5], [x10], x8
            cmp x6, #4
            beq WriteEnd
            st1 {v16.8h}, [x11], x8
            st1 {v17.4h}, [x19], x8
            st1 {v17.h}[4], [x20], x8
            st1 {v17.h}[5], [x10], x8
            cmp x6, #5
            beq WriteEnd
            st1 {v18.8h}, [x11], x8
            st1 {v19.4h}, [x19], x8
            st1 {v19.h}[4], [x20], x8
            st1 {v19.h}[5], [x10], x8
            cmp x6, #6
            beq WriteEnd
            st1 {v20.8h}, [x11], x8
            st1 {v21.4h}, [x19], x8
            st1 {v21.h}[4], [x20], x8
            st1 {v21.h}[5], [x10], x8
            cmp x6, #7
            beq WriteEnd
            st1 {v22.8h}, [x11], x8
            st1 {v23.4h}, [x19], x8
            st1 {v23.h}[4], [x20], x8
            st1 {v23.h}[5], [x10], x8
            cmp x6, #8
            beq WriteEnd
            st1 {v24.8h}, [x11], x8
            st1 {v25.4h}, [x19], x8
            st1 {v25.h}[4], [x20], x8
            st1 {v25.h}[5], [x10], x8
            cmp x6, #9
            beq WriteEnd
            st1 {v26.8h}, [x11], x8
            st1 {v27.4h}, [x19], x8
            st1 {v27.h}[4], [x20], x8
            st1 {v27.h}[5], [x10], x8
            cmp x6, #10
            beq WriteEnd
            st1 {v28.8h}, [x11], x8
            st1 {v29.4h}, [x19], x8
            st1 {v29.h}[4], [x20], x8
            st1 {v29.h}[5], [x10], x8
            cmp x6, #11
            beq WriteEnd
            st1 {v30.8h}, [x11], x8
            st1 {v31.4h}, [x19], x8
            st1 {v31.h}[4], [x20], x8
            st1 {v31.h}[5], [x10], x8
            add x11, x11, #28
            b WriteEnd
        Write15:
            add x2, x2, #30
            add x19, x11, #16
            add x20, x11, #24
            add x10, x11, #26
            add x15, x11, #28
            st1 {v8.8h}, [x11], x8
            st1 {v9.4h}, [x19], x8
            st1 {v9.h}[4], [x20], x8
            st1 {v9.h}[5], [x10], x8
            st1 {v9.h}[6], [x15], x8
            cmp x6, #1
            beq WriteEnd
            st1 {v10.8h}, [x11], x8
            st1 {v11.4h}, [x19], x8
            st1 {v11.h}[4], [x20], x8
            st1 {v11.h}[5], [x10], x8
            st1 {v11.h}[6], [x15], x8
            cmp x6, #2
            beq WriteEnd
            st1 {v12.8h}, [x11], x8
            st1 {v13.4h}, [x19], x8
            st1 {v13.h}[4], [x20], x8
            st1 {v13.h}[5], [x10], x8
            st1 {v13.h}[6], [x15], x8
            cmp x6, #3
            beq WriteEnd
            st1 {v14.8h}, [x11], x8
            st1 {v15.4h}, [x19], x8
            st1 {v15.h}[4], [x20], x8
            st1 {v15.h}[5], [x10], x8
            st1 {v15.h}[6], [x15], x8
            cmp x6, #4
            beq WriteEnd
            st1 {v16.8h}, [x11], x8
            st1 {v17.4h}, [x19], x8
            st1 {v17.h}[4], [x20], x8
            st1 {v17.h}[5], [x10], x8
            st1 {v17.h}[6], [x15], x8
            cmp x6, #5
            beq WriteEnd
            st1 {v18.8h}, [x11], x8
            st1 {v19.4h}, [x19], x8
            st1 {v19.h}[4], [x20], x8
            st1 {v19.h}[5], [x10], x8
            st1 {v19.h}[6], [x15], x8
            cmp x6, #6
            beq WriteEnd
            st1 {v20.8h}, [x11], x8
            st1 {v21.4h}, [x19], x8
            st1 {v21.h}[4], [x20], x8
            st1 {v21.h}[5], [x10], x8
            st1 {v21.h}[6], [x15], x8
            cmp x6, #7
            beq WriteEnd
            st1 {v22.8h}, [x11], x8
            st1 {v23.4h}, [x19], x8
            st1 {v23.h}[4], [x20], x8
            st1 {v23.h}[5], [x10], x8
            st1 {v23.h}[6], [x15], x8
            cmp x6, #8
            beq WriteEnd
            st1 {v24.8h}, [x11], x8
            st1 {v25.4h}, [x19], x8
            st1 {v25.h}[4], [x20], x8
            st1 {v25.h}[5], [x10], x8
            st1 {v25.h}[6], [x15], x8
            cmp x6, #9
            beq WriteEnd
            st1 {v26.8h}, [x11], x8
            st1 {v27.4h}, [x19], x8
            st1 {v27.h}[4], [x20], x8
            st1 {v27.h}[5], [x10], x8
            st1 {v27.h}[6], [x15], x8
            cmp x6, #10
            beq WriteEnd
            st1 {v28.8h}, [x11], x8
            st1 {v29.4h}, [x19], x8
            st1 {v29.h}[4], [x20], x8
            st1 {v29.h}[5], [x10], x8
            st1 {v29.h}[6], [x15], x8
            cmp x6, #11
            beq WriteEnd
            st1 {v30.8h}, [x11], x8
            st1 {v31.4h}, [x19], x8
            st1 {v31.h}[4], [x20], x8
            st1 {v31.h}[5], [x10], x8
            st1 {v31.h}[6], [x15], x8
            add x11, x11, #30
            b WriteEnd
        Write16:
            add x2, x2, #32
            add x19, x11, #16
            st1 {v8.8h}, [x11], x8
            st1 {v9.8h}, [x19], x8
            cmp x6, #1
            beq WriteEnd
            st1 {v10.8h}, [x11], x8
            st1 {v11.8h}, [x19], x8
            cmp x6, #2
            beq WriteEnd
            st1 {v12.8h}, [x11], x8
            st1 {v13.8h}, [x19], x8
            cmp x6, #3
            beq WriteEnd
            st1 {v14.8h}, [x11], x8
            st1 {v15.8h}, [x19], x8
            cmp x6, #4
            beq WriteEnd
            st1 {v16.8h}, [x11], x8
            st1 {v17.8h}, [x19], x8
            cmp x6, #5
            beq WriteEnd
            st1 {v18.8h}, [x11], x8
            st1 {v19.8h}, [x19], x8
            cmp x6, #6
            beq WriteEnd
            st1 {v20.8h}, [x11], x8
            st1 {v21.8h}, [x19], x8
            cmp x6, #7
            beq WriteEnd
            st1 {v22.8h}, [x11], x8
            st1 {v23.8h}, [x19], x8
            cmp x6, #8
            beq WriteEnd
            st1 {v24.8h}, [x11], x8
            st1 {v25.8h}, [x19], x8
            cmp x6, #9
            beq WriteEnd
            st1 {v26.8h}, [x11], x8
            st1 {v27.8h}, [x19], x8
            cmp x6, #10
            beq WriteEnd
            st1 {v28.8h}, [x11], x8
            st1 {v29.8h}, [x19], x8
            cmp x6, #11
            beq WriteEnd
            st1 {v30.8h}, [x11], x8
            st1 {v31.8h}, [x19], x8
            add x11, x11, #32
            b WriteEnd

        WriteEnd:
            subs x13, x13, #16   // col - 16
            ble LoopColEnd
            cmp x6, #1
            ble LoopCol1
            cmp x6, #2
            ble LoopCol2
            cmp x6, #4
            ble LoopCol4
            cmp x6, #8
            ble LoopCol8
            b LoopCol12

LoopColEnd:
        add x0, x0, x17
        mov x21, #2
        mul x21, x21, x7
        sub x11, x11, x21
        mov x2, x11
        subs x6, x6, #12
        bgt LoopRowStart

    sub sp, sp, #160
    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [sp], #64
    ld1 {v12.8h, v13.8h, v14.8h, v15.8h}, [sp], #64
    ldp x19, x20, [sp], #16
    ldp x21, x22, [sp], #16
    ret
#endif
