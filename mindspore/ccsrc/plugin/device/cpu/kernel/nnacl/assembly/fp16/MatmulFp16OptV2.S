/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

// void MatmulFp16OptV2(const float16_t *a, const float16_t *b, float16_t *c, const float16_t *bias, int act_type,
//                      size_t depth, size_t row, size_t col, size_t stride, size_t writeMode)
// x0: a
// x1: b
// x2: c
// x3: bias
// x4: act_type
// x5: depth
// x6: row
// x7: col
// x8: stride
// x9: writeMode

asm_function MatmulFp16OptV2
    sub sp, sp, #192
    st1 {v8.8h, v9.8h, v10.8h, v11.8h}, [sp], #64
    st1 {v12.8h, v13.8h, v14.8h, v15.8h}, [sp], #64
    stp x19, x20, [sp], #16
    stp x21, x22, [sp], #16
    stp x23, x24, [sp], #16
    stp x29, x30, [sp], #16

    ldr x8, [sp]
    ldr x9, [sp, #8]  // writeMode
    lsl x8, x8, #1  // stride * sizeof(float16_t)

    lsl x15, x7, #1 // col * sizeof(float16_t)
    lsl x16, x5, #1  // depth * sizeof(float16_t)
    mov x11, x2
    movi v7.8h, #0x46, lsl #8
    subs x6, x6, #12
    blt LoopRow8
LoopRow12:
    mov x11, x1  // reload matrixB
    mov x12, x3  // reload bias
    mov x13, x7  // reload col
    mov x21, x2  // relocate output
    subs x13, x13, #16
    blt LoopCol12x8
    LoopCol12x16:
        mov x10, x0  // update matrixA
        ld1 {v0.8h}, [x10], #16
        mov x14, x5  // reload depth
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h}, [x11], #16
        cbnz x12, InitFromBias12x16
        dup v8.2d, xzr
        dup v9.2d, xzr
        dup v10.2d, xzr
        dup v11.2d, xzr
        dup v12.2d, xzr
        dup v13.2d, xzr
        dup v14.2d, xzr
        dup v15.2d, xzr
        dup v16.2d, xzr
        dup v17.2d, xzr
        dup v18.2d, xzr
        dup v19.2d, xzr
        dup v20.2d, xzr
        dup v21.2d, xzr
        dup v22.2d, xzr
        dup v23.2d, xzr
        dup v24.2d, xzr
        dup v25.2d, xzr
        dup v26.2d, xzr
        dup v27.2d, xzr
        dup v28.2d, xzr
        dup v29.2d, xzr
        dup v30.2d, xzr
        dup v31.2d, xzr
        b Compute12x16Enter
        InitFromBias12x16:
            ld1 {v8.8h, v9.8h}, [x12]
            ld1 {v10.8h, v11.8h}, [x12]
            ld1 {v12.8h, v13.8h}, [x12]
            ld1 {v14.8h, v15.8h}, [x12]
            ld1 {v16.8h, v17.8h}, [x12]
            ld1 {v18.8h, v19.8h}, [x12]
            ld1 {v20.8h, v21.8h}, [x12]
            ld1 {v22.8h, v23.8h}, [x12]
            ld1 {v24.8h, v25.8h}, [x12]
            ld1 {v26.8h, v27.8h}, [x12]
            ld1 {v28.8h, v29.8h}, [x12]
            ld1 {v30.8h, v31.8h}, [x12]
            add x12, x12, #32
    Compute12x16Enter:
        bl Compute12x16Unit
        Activation12x16:
            cmp x4, #3
            beq Relu612x16
            cmp x4, #1
            beq Relu12x16
            b Write12x16

            Relu612x16:
                fmin v8.8h, v8.8h, v7.8h
                fmin v9.8h, v9.8h, v7.8h
                fmin v10.8h, v10.8h, v7.8h
                fmin v11.8h, v11.8h, v7.8h
                fmin v12.8h, v12.8h, v7.8h
                fmin v13.8h, v13.8h, v7.8h
                fmin v14.8h, v14.8h, v7.8h
                fmin v15.8h, v15.8h, v7.8h
                fmin v16.8h, v16.8h, v7.8h
                fmin v17.8h, v17.8h, v7.8h
                fmin v18.8h, v18.8h, v7.8h
                fmin v19.8h, v19.8h, v7.8h
                fmin v20.8h, v20.8h, v7.8h
                fmin v21.8h, v21.8h, v7.8h
                fmin v22.8h, v22.8h, v7.8h
                fmin v23.8h, v23.8h, v7.8h
                fmin v24.8h, v24.8h, v7.8h
                fmin v25.8h, v25.8h, v7.8h
                fmin v26.8h, v26.8h, v7.8h
                fmin v27.8h, v27.8h, v7.8h
                fmin v28.8h, v28.8h, v7.8h
                fmin v29.8h, v29.8h, v7.8h
                fmin v30.8h, v30.8h, v7.8h
                fmin v31.8h, v31.8h, v7.8h

            Relu12x16:
                dup v6.8h, wzr
                fmax v8.8h, v8.8h, v6.8h
                fmax v9.8h, v9.8h, v6.8h
                fmax v10.8h, v10.8h, v6.8h
                fmax v11.8h, v11.8h, v6.8h
                fmax v12.8h, v12.8h, v6.8h
                fmax v13.8h, v13.8h, v6.8h
                fmax v14.8h, v14.8h, v6.8h
                fmax v15.8h, v15.8h, v6.8h
                fmax v16.8h, v16.8h, v6.8h
                fmax v17.8h, v17.8h, v6.8h
                fmax v18.8h, v18.8h, v6.8h
                fmax v19.8h, v19.8h, v6.8h
                fmax v20.8h, v20.8h, v6.8h
                fmax v21.8h, v21.8h, v6.8h
                fmax v22.8h, v22.8h, v6.8h
                fmax v23.8h, v23.8h, v6.8h
                fmax v24.8h, v24.8h, v6.8h
                fmax v25.8h, v25.8h, v6.8h
                fmax v26.8h, v26.8h, v6.8h
                fmax v27.8h, v27.8h, v6.8h
                fmax v28.8h, v28.8h, v6.8h
                fmax v29.8h, v29.8h, v6.8h
                fmax v30.8h, v30.8h, v6.8h
                fmax v31.8h, v31.8h, v6.8h
            Write12x16:
                mov x22, x21
                add x23, x21, x8, lsl #2
                add x24, x21, x8, lsl #3
                st1 {v8.8h, v9.8h}, [x22], x8
                st1 {v10.8h, v11.8h}, [x22], x8
                st1 {v12.8h, v13.8h}, [x22], x8
                st1 {v14.8h, v15.8h}, [x22]
                st1 {v16.8h, v17.8h}, [x23], x8
                st1 {v18.8h, v19.8h}, [x23], x8
                st1 {v20.8h, v21.8h}, [x23], x8
                st1 {v22.8h, v23.8h}, [x23]
                st1 {v24.8h, v25.8h}, [x24], x8
                st1 {v26.8h, v27.8h}, [x24], x8
                st1 {v28.8h, v29.8h}, [x24], x8
                st1 {v30.8h, v31.8h}, [x24]
                add x21, x21, #32
                subs x13, x13, #16
                bge LoopCol12x16

    LoopCol12x8:
        adds x13, x13, #16
        cbz x13, LoopRow12End
        subs x13, x13, #8
        blt LoopCol12x4 
        mov x10, x0  // update matrixA
        ld1 {v0.8h}, [x10], #16
        mov x14, x5  // reload depth
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h}, [x11], #16
        cbnz x12, InitFromBias12x8
        dup v8.2d, xzr
        dup v10.2d, xzr
        dup v12.2d, xzr
        dup v14.2d, xzr
        dup v16.2d, xzr
        dup v18.2d, xzr
        dup v20.2d, xzr
        dup v22.2d, xzr
        dup v24.2d, xzr
        dup v26.2d, xzr
        dup v28.2d, xzr
        dup v30.2d, xzr
        b Compute12x8Enter
        InitFromBias12x8:
            ld1 {v8.8h}, [x12]
            ld1 {v10.8h}, [x12]
            ld1 {v12.8h}, [x12]
            ld1 {v14.8h}, [x12]
            ld1 {v16.8h}, [x12]
            ld1 {v18.8h}, [x12]
            ld1 {v20.8h}, [x12]
            ld1 {v22.8h}, [x12]
            ld1 {v24.8h}, [x12]
            ld1 {v26.8h}, [x12]
            ld1 {v28.8h}, [x12]
            ld1 {v30.8h}, [x12]
            add x12, x12, #16
    Compute12x8Enter:
        bl Compute12x8Unit
        Activation12x8:
            cmp x4, #3
            beq Relu612x8
            cmp x4, #1
            beq Relu12x8
            b Write12x8

            Relu612x8:
                fmin v8.8h, v8.8h, v7.8h
                fmin v10.8h, v10.8h, v7.8h
                fmin v12.8h, v12.8h, v7.8h
                fmin v14.8h, v14.8h, v7.8h
                fmin v16.8h, v16.8h, v7.8h
                fmin v18.8h, v18.8h, v7.8h
                fmin v20.8h, v20.8h, v7.8h
                fmin v22.8h, v22.8h, v7.8h
                fmin v24.8h, v24.8h, v7.8h
                fmin v26.8h, v26.8h, v7.8h
                fmin v28.8h, v28.8h, v7.8h
                fmin v30.8h, v30.8h, v7.8h

            Relu12x8:
                dup v6.8h, wzr
                fmax v8.8h, v8.8h, v6.8h
                fmax v10.8h, v10.8h, v6.8h
                fmax v12.8h, v12.8h, v6.8h
                fmax v14.8h, v14.8h, v6.8h
                fmax v16.8h, v16.8h, v6.8h
                fmax v18.8h, v18.8h, v6.8h
                fmax v20.8h, v20.8h, v6.8h
                fmax v22.8h, v22.8h, v6.8h
                fmax v24.8h, v24.8h, v6.8h
                fmax v26.8h, v26.8h, v6.8h
                fmax v28.8h, v28.8h, v6.8h
                fmax v30.8h, v30.8h, v6.8h
            Write12x8:
                mov x22, x21
                add x23, x21, x8, lsl #2
                add x24, x21, x8, lsl #3
                st1 {v8.8h}, [x22], x8
                st1 {v10.8h}, [x22], x8
                st1 {v12.8h}, [x22], x8
                st1 {v14.8h}, [x22]
                st1 {v16.8h}, [x23], x8
                st1 {v18.8h}, [x23], x8
                st1 {v20.8h}, [x23], x8
                st1 {v22.8h}, [x23]
                st1 {v24.8h}, [x24], x8
                st1 {v26.8h}, [x24], x8
                st1 {v28.8h}, [x24], x8
                st1 {v30.8h}, [x24]
                add x21, x21, #16
                subs x13, x13, #8

    LoopCol12x4:
        adds x13, x13, #8
        cbz x13, LoopRow12End
    LoopCol12x4Core:
        mov x10, x0  // update matrixA
        ld1 {v0.8h}, [x10], #16
        mov x14, x5  // reload depth
        prfm pldl1strm, [x11, #632]
        ld1 {v3.4h}, [x11], #8
        cbnz x12, InitFromBias12x4
        dup v8.2s, wzr
        dup v10.2s, wzr
        dup v12.2s, wzr
        dup v14.2s, wzr
        dup v16.2s, wzr
        dup v18.2s, wzr
        dup v20.2s, wzr
        dup v22.2s, wzr
        dup v24.2s, wzr
        dup v26.2s, wzr
        dup v28.2s, wzr
        dup v30.2s, wzr
        b Compute12x4Enter
        InitFromBias12x4:
            ld1 {v8.4h}, [x12]
            ld1 {v10.4h}, [x12]
            ld1 {v12.4h}, [x12]
            ld1 {v14.4h}, [x12]
            ld1 {v16.4h}, [x12]
            ld1 {v18.4h}, [x12]
            ld1 {v20.4h}, [x12]
            ld1 {v22.4h}, [x12]
            ld1 {v24.4h}, [x12]
            ld1 {v26.4h}, [x12]
            ld1 {v28.4h}, [x12]
            ld1 {v30.4h}, [x12]
            add x12, x12, #8
    Compute12x4Enter:
        bl Compute12x4Unit
        Activation12x4:
            cmp x4, #3
            beq Relu612x4
            cmp x4, #1
            beq Relu12x4
            b Write12x4

            Relu612x4:
                fmin v8.4h, v8.4h, v7.4h
                fmin v10.4h, v10.4h, v7.4h
                fmin v12.4h, v12.4h, v7.4h
                fmin v14.4h, v14.4h, v7.4h
                fmin v16.4h, v16.4h, v7.4h
                fmin v18.4h, v18.4h, v7.4h
                fmin v20.4h, v20.4h, v7.4h
                fmin v22.4h, v22.4h, v7.4h
                fmin v24.4h, v24.4h, v7.4h
                fmin v26.4h, v26.4h, v7.4h
                fmin v28.4h, v28.4h, v7.4h
                fmin v30.4h, v30.4h, v7.4h

            Relu12x4:
                dup v6.4h, wzr
                fmax v8.4h, v8.4h, v6.4h
                fmax v10.4h, v10.4h, v6.4h
                fmax v12.4h, v12.4h, v6.4h
                fmax v14.4h, v14.4h, v6.4h
                fmax v16.4h, v16.4h, v6.4h
                fmax v18.4h, v18.4h, v6.4h
                fmax v20.4h, v20.4h, v6.4h
                fmax v22.4h, v22.4h, v6.4h
                fmax v24.4h, v24.4h, v6.4h
                fmax v26.4h, v26.4h, v6.4h
                fmax v28.4h, v28.4h, v6.4h
                fmax v30.4h, v30.4h, v6.4h
            Write12x4:
                mov x22, x21
                add x23, x21, x8, lsl #2
                add x24, x21, x8, lsl #3
                cmp x13, #1
                beq Write12x1
                cmp x13, #2
                beq Write12x2
                cmp x13, #3
                beq Write12x3
                st1 {v8.4h}, [x22], x8
                st1 {v10.4h}, [x22], x8
                st1 {v12.4h}, [x22], x8
                st1 {v14.4h}, [x22]
                st1 {v16.4h}, [x23], x8
                st1 {v18.4h}, [x23], x8
                st1 {v20.4h}, [x23], x8
                st1 {v22.4h}, [x23]
                st1 {v24.4h}, [x24], x8
                st1 {v26.4h}, [x24], x8
                st1 {v28.4h}, [x24], x8
                st1 {v30.4h}, [x24]
                add x21, x21, #8
                subs x13, x13, #4
                bgt LoopCol12x4Core
                b LoopRow12End
            Write12x1:
                st1 {v8.h}[0], [x22], x8
                st1 {v10.h}[0], [x22], x8
                st1 {v12.h}[0], [x22], x8
                st1 {v14.h}[0], [x22]
                st1 {v16.h}[0], [x23], x8
                st1 {v18.h}[0], [x23], x8
                st1 {v20.h}[0], [x23], x8
                st1 {v22.h}[0], [x23]
                st1 {v24.h}[0], [x24], x8
                st1 {v26.h}[0], [x24], x8
                st1 {v28.h}[0], [x24], x8
                st1 {v30.h}[0], [x24]
                b LoopRow12End
            Write12x2:                
                st1 {v8.s}[0], [x22], x8
                st1 {v10.s}[0], [x22], x8
                st1 {v12.s}[0], [x22], x8
                st1 {v14.s}[0], [x22]
                st1 {v16.s}[0], [x23], x8
                st1 {v18.s}[0], [x23], x8
                st1 {v20.s}[0], [x23], x8
                st1 {v22.s}[0], [x23]
                st1 {v24.s}[0], [x24], x8
                st1 {v26.s}[0], [x24], x8
                st1 {v28.s}[0], [x24], x8
                st1 {v30.s}[0], [x24]
                b LoopRow12End
            Write12x3:
                add x23, x22, #4
                st1 {v8.s}[0], [x22], x8
                st1 {v8.h}[2], [x23], x8
                st1 {v10.s}[0], [x22], x8
                st1 {v10.h}[2], [x23], x8
                st1 {v12.s}[0], [x22], x8
                st1 {v12.h}[2], [x23], x8
                st1 {v14.s}[0], [x22], x8
                st1 {v14.h}[2], [x23], x8
                st1 {v16.s}[0], [x22], x8
                st1 {v16.h}[2], [x23], x8
                st1 {v18.s}[0], [x22], x8
                st1 {v18.h}[2], [x23], x8
                st1 {v20.s}[0], [x22], x8
                st1 {v20.h}[2], [x23], x8
                st1 {v22.s}[0], [x22], x8
                st1 {v22.h}[2], [x23], x8
                st1 {v24.s}[0], [x22], x8
                st1 {v24.h}[2], [x23], x8
                st1 {v26.s}[0], [x22], x8
                st1 {v26.h}[2], [x23], x8
                st1 {v28.s}[0], [x22], x8
                st1 {v28.h}[2], [x23], x8
                st1 {v30.s}[0], [x22]
                st1 {v30.h}[2], [x23]
            LoopRow12End:
                add x0, x0, x16, lsl #3
                add x0, x0, x16, lsl #2
                add x2, x2, x8, lsl #3
                add x2, x2, x8, lsl #2
                subs x6, x6, #12
                bge LoopRow12

LoopRow8:
    adds x6, x6,#12
    cbz x6, End
    subs x6, x6, #8
    blt LoopRow4
    mov x11, x1  // reload matrixB
    mov x12, x3  // reload bias
    mov x13, x7  // reload col
    mov x21, x2  // relocate output
    subs x13, x13, #16
    blt LoopCol8x8
    LoopCol8x16:
        mov x10, x0  // update matrixA
        ld1 {v0.8h}, [x10], #16
        mov x14, x5  // reload depth
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h}, [x11], #16
        cbnz x12, InitFromBias8x16
        dup v8.2d, xzr
        dup v9.2d, xzr
        dup v10.2d, xzr
        dup v11.2d, xzr
        dup v12.2d, xzr
        dup v13.2d, xzr
        dup v14.2d, xzr
        dup v15.2d, xzr
        dup v16.2d, xzr
        dup v17.2d, xzr
        dup v18.2d, xzr
        dup v19.2d, xzr
        dup v20.2d, xzr
        dup v21.2d, xzr
        dup v22.2d, xzr
        dup v23.2d, xzr
        b Compute8x16Enter
        InitFromBias8x16:
            ld1 {v8.8h, v9.8h}, [x12]
            ld1 {v10.8h, v11.8h}, [x12]
            ld1 {v12.8h, v13.8h}, [x12]
            ld1 {v14.8h, v15.8h}, [x12]
            ld1 {v16.8h, v17.8h}, [x12]
            ld1 {v18.8h, v19.8h}, [x12]
            ld1 {v20.8h, v21.8h}, [x12]
            ld1 {v22.8h, v23.8h}, [x12]
            add x12, x12, #32
    Compute8x16Enter:
        bl Compute8x16Unit
        Activation8x16:
            cmp x4, #3
            beq Relu68x16
            cmp x4, #1
            beq Relu8x16
            b Write8x16

            Relu68x16:
                fmin v8.8h, v8.8h, v7.8h
                fmin v9.8h, v9.8h, v7.8h
                fmin v10.8h, v10.8h, v7.8h
                fmin v11.8h, v11.8h, v7.8h
                fmin v12.8h, v12.8h, v7.8h
                fmin v13.8h, v13.8h, v7.8h
                fmin v14.8h, v14.8h, v7.8h
                fmin v15.8h, v15.8h, v7.8h
                fmin v16.8h, v16.8h, v7.8h
                fmin v17.8h, v17.8h, v7.8h
                fmin v18.8h, v18.8h, v7.8h
                fmin v19.8h, v19.8h, v7.8h
                fmin v20.8h, v20.8h, v7.8h
                fmin v21.8h, v21.8h, v7.8h
                fmin v22.8h, v22.8h, v7.8h
                fmin v23.8h, v23.8h, v7.8h

            Relu8x16:
                dup v6.8h, wzr
                fmax v8.8h, v8.8h, v6.8h
                fmax v9.8h, v9.8h, v6.8h
                fmax v10.8h, v10.8h, v6.8h
                fmax v11.8h, v11.8h, v6.8h
                fmax v12.8h, v12.8h, v6.8h
                fmax v13.8h, v13.8h, v6.8h
                fmax v14.8h, v14.8h, v6.8h
                fmax v15.8h, v15.8h, v6.8h
                fmax v16.8h, v16.8h, v6.8h
                fmax v17.8h, v17.8h, v6.8h
                fmax v18.8h, v18.8h, v6.8h
                fmax v19.8h, v19.8h, v6.8h
                fmax v20.8h, v20.8h, v6.8h
                fmax v21.8h, v21.8h, v6.8h
                fmax v22.8h, v22.8h, v6.8h
                fmax v23.8h, v23.8h, v6.8h
            Write8x16:
                mov x22, x21
                add x23, x21, x8, lsl #2
                st1 {v8.8h, v9.8h}, [x22], x8
                st1 {v10.8h, v11.8h}, [x22], x8
                st1 {v12.8h, v13.8h}, [x22], x8
                st1 {v14.8h, v15.8h}, [x22]
                st1 {v16.8h, v17.8h}, [x23], x8
                st1 {v18.8h, v19.8h}, [x23], x8
                st1 {v20.8h, v21.8h}, [x23], x8
                st1 {v22.8h, v23.8h}, [x23]
                add x21, x21, #32
                subs x13, x13, #16
                bge LoopCol8x16

    LoopCol8x8:
        adds x13, x13, #16
        cbz x13, LoopRow8End
        subs x13, x13, #8
        blt LoopCol8x4
        mov x10, x0  // update matrixA
        ld1 {v0.8h}, [x10], #16
        mov x14, x5  // reload depth
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h}, [x11], #16
        cbnz x12, InitFromBias8x8
        dup v8.2d, xzr
        dup v10.2d, xzr
        dup v12.2d, xzr
        dup v14.2d, xzr
        dup v16.2d, xzr
        dup v18.2d, xzr
        dup v20.2d, xzr
        dup v22.2d, xzr
        b Compute8x8Enter
        InitFromBias8x8:
            ld1 {v8.8h}, [x12]
            ld1 {v10.8h}, [x12]
            ld1 {v12.8h}, [x12]
            ld1 {v14.8h}, [x12]
            ld1 {v16.8h}, [x12]
            ld1 {v18.8h}, [x12]
            ld1 {v20.8h}, [x12]
            ld1 {v22.8h}, [x12]
            add x12, x12, #16
    Compute8x8Enter:
        bl Compute8x8Unit
        Activation8x8:
            cmp x4, #3
            beq Relu68x8
            cmp x4, #1
            beq Relu8x8
            b Write8x8

            Relu68x8:
                fmin v8.8h, v8.8h, v7.8h
                fmin v10.8h, v10.8h, v7.8h
                fmin v12.8h, v12.8h, v7.8h
                fmin v14.8h, v14.8h, v7.8h
                fmin v16.8h, v16.8h, v7.8h
                fmin v18.8h, v18.8h, v7.8h
                fmin v20.8h, v20.8h, v7.8h
                fmin v22.8h, v22.8h, v7.8h

            Relu8x8:
                dup v6.8h, wzr
                fmax v8.8h, v8.8h, v6.8h
                fmax v10.8h, v10.8h, v6.8h
                fmax v12.8h, v12.8h, v6.8h
                fmax v14.8h, v14.8h, v6.8h
                fmax v16.8h, v16.8h, v6.8h
                fmax v18.8h, v18.8h, v6.8h
                fmax v20.8h, v20.8h, v6.8h
                fmax v22.8h, v22.8h, v6.8h
            Write8x8:
                mov x22, x21
                add x23, x21, x8, lsl #2
                st1 {v8.8h}, [x22], x8
                st1 {v10.8h}, [x22], x8
                st1 {v12.8h}, [x22], x8
                st1 {v14.8h}, [x22]
                st1 {v16.8h}, [x23], x8
                st1 {v18.8h}, [x23], x8
                st1 {v20.8h}, [x23], x8
                st1 {v22.8h}, [x23]
                add x21, x21, #16
                subs x13, x13, #8

    LoopCol8x4:
        adds x13, x13, #8
        cbz x13, LoopRow8End
    LoopCol8x4Core:
        mov x10, x0  // update matrixA
        ld1 {v0.8h}, [x10], #16
        mov x14, x5  // reload depth
        prfm pldl1strm, [x11, #632]
        ld1 {v3.4h}, [x11], #8
        cbnz x12, InitFromBias8x4
        dup v8.2s, wzr
        dup v10.2s, wzr
        dup v12.2s, wzr
        dup v14.2s, wzr
        dup v16.2s, wzr
        dup v18.2s, wzr
        dup v20.2s, wzr
        dup v22.2s, wzr
        b Compute8x4Enter
        InitFromBias8x4:
            ld1 {v8.4h}, [x12]
            ld1 {v10.4h}, [x12]
            ld1 {v12.4h}, [x12]
            ld1 {v14.4h}, [x12]
            ld1 {v16.4h}, [x12]
            ld1 {v18.4h}, [x12]
            ld1 {v20.4h}, [x12]
            ld1 {v22.4h}, [x12]
            add x12, x12, #8
    Compute8x4Enter:
        bl Compute8x4Unit
        Activation8x4:
            cmp x4, #3
            beq Relu68x4
            cmp x4, #1
            beq Relu8x4
            b Write8x4

            Relu68x4:
                fmin v8.4h, v8.4h, v7.4h
                fmin v10.4h, v10.4h, v7.4h
                fmin v12.4h, v12.4h, v7.4h
                fmin v14.4h, v14.4h, v7.4h
                fmin v16.4h, v16.4h, v7.4h
                fmin v18.4h, v18.4h, v7.4h
                fmin v20.4h, v20.4h, v7.4h
                fmin v22.4h, v22.4h, v7.4h

            Relu8x4:
                dup v6.4h, wzr
                fmax v8.4h, v8.4h, v6.4h
                fmax v10.4h, v10.4h, v6.4h
                fmax v12.4h, v12.4h, v6.4h
                fmax v14.4h, v14.4h, v6.4h
                fmax v16.4h, v16.4h, v6.4h
                fmax v18.4h, v18.4h, v6.4h
                fmax v20.4h, v20.4h, v6.4h
                fmax v22.4h, v22.4h, v6.4h
            Write8x4:
                mov x22, x21
                add x23, x21, x8, lsl #2
                cmp x13, #1
                beq Write8x1
                cmp x13, #2
                beq Write8x2
                cmp x13, #3
                beq Write8x3
                st1 {v8.4h}, [x22], x8
                st1 {v10.4h}, [x22], x8
                st1 {v12.4h}, [x22], x8
                st1 {v14.4h}, [x22]
                st1 {v16.4h}, [x23], x8
                st1 {v18.4h}, [x23], x8
                st1 {v20.4h}, [x23], x8
                st1 {v22.4h}, [x23]
                add x21, x21, #8
                subs x13, x13, #4
                bgt LoopCol8x4Core
                b LoopRow8End
            Write8x1:
                st1 {v8.h}[0], [x22], x8
                st1 {v10.h}[0], [x22], x8
                st1 {v12.h}[0], [x22], x8
                st1 {v14.h}[0], [x22]
                st1 {v16.h}[0], [x23], x8
                st1 {v18.h}[0], [x23], x8
                st1 {v20.h}[0], [x23], x8
                st1 {v22.h}[0], [x23]
                b LoopRow8End
            Write8x2:
                st1 {v8.s}[0], [x22], x8
                st1 {v10.s}[0], [x22], x8
                st1 {v12.s}[0], [x22], x8
                st1 {v14.s}[0], [x22]
                st1 {v16.s}[0], [x23], x8
                st1 {v18.s}[0], [x23], x8
                st1 {v20.s}[0], [x23], x8
                st1 {v22.s}[0], [x23]
                b LoopRow8End
            Write8x3:
                add x23, x22, #4
                st1 {v8.s}[0], [x22], x8
                st1 {v8.h}[2], [x23], x8
                st1 {v10.s}[0], [x22], x8
                st1 {v10.h}[2], [x23], x8
                st1 {v12.s}[0], [x22], x8
                st1 {v12.h}[2], [x23], x8
                st1 {v14.s}[0], [x22], x8
                st1 {v14.h}[2], [x23], x8
                st1 {v16.s}[0], [x22], x8
                st1 {v16.h}[2], [x23], x8
                st1 {v18.s}[0], [x22], x8
                st1 {v18.h}[2], [x23], x8
                st1 {v20.s}[0], [x22], x8
                st1 {v20.h}[2], [x23], x8
                st1 {v22.s}[0], [x22], x8
                st1 {v22.h}[2], [x23], x8
            LoopRow8End:
                add x0, x0, x16, lsl #3
                add x2, x2, x8, lsl #3
                subs x6, x6, #8

LoopRow4:
    adds x6, x6, #8
    cbz x6, End
    subs x6, x6, #4
    blt LoopRowTail
    mov x11, x1  // reload matrixB
    mov x12, x3  // reload bias
    mov x13, x7  // reload col
    mov x21, x2  // relocate output
    subs x13, x13, #16
    blt LoopCol4x8
    LoopCol4x16:
        mov x10, x0  // update matrixA
        ld1 {v0.4h}, [x10], #8
        mov x14, x5  // reload depth
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h}, [x11], #16
        cbnz x12, InitFromBias4x16
        dup v8.2d, xzr
        dup v9.2d, xzr
        dup v10.2d, xzr
        dup v11.2d, xzr
        dup v12.2d, xzr
        dup v13.2d, xzr
        dup v14.2d, xzr
        dup v15.2d, xzr
        b Compute4x16Enter
        InitFromBias4x16:
            ld1 {v8.8h, v9.8h}, [x12]
            ld1 {v10.8h, v11.8h}, [x12]
            ld1 {v12.8h, v13.8h}, [x12]
            ld1 {v14.8h, v15.8h}, [x12]
            add x12, x12, #32
    Compute4x16Enter:
        bl Compute4x16Unit
        Activation4x16:
            cmp x4, #3
            beq Relu64x16
            cmp x4, #1
            beq Relu4x16
            b Write4x16

            Relu64x16:
                fmin v8.8h, v8.8h, v7.8h
                fmin v9.8h, v9.8h, v7.8h
                fmin v10.8h, v10.8h, v7.8h
                fmin v11.8h, v11.8h, v7.8h
                fmin v12.8h, v12.8h, v7.8h
                fmin v13.8h, v13.8h, v7.8h
                fmin v14.8h, v14.8h, v7.8h
                fmin v15.8h, v15.8h, v7.8h

            Relu4x16:
                dup v6.8h, wzr
                fmax v8.8h, v8.8h, v6.8h
                fmax v9.8h, v9.8h, v6.8h
                fmax v10.8h, v10.8h, v6.8h
                fmax v11.8h, v11.8h, v6.8h
                fmax v12.8h, v12.8h, v6.8h
                fmax v13.8h, v13.8h, v6.8h
                fmax v14.8h, v14.8h, v6.8h
                fmax v15.8h, v15.8h, v6.8h
            Write4x16:
                mov x22, x21
                st1 {v8.8h, v9.8h}, [x22], x8
                st1 {v10.8h, v11.8h}, [x22], x8
                st1 {v12.8h, v13.8h}, [x22], x8
                st1 {v14.8h, v15.8h}, [x22]
                add x21, x21, #32
                subs x13, x13, #16
                bge LoopCol4x16

    LoopCol4x8:
        adds x13, x13, #16
        cbz x13, LoopRow4End
        subs x13, x13, #8
        blt LoopCol4x4
        mov x10, x0  // update matrixA
        ld1 {v0.4h}, [x10], #8
        mov x14, x5  // reload depth
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h}, [x11], #16
        cbnz x12, InitFromBias4x8
        dup v8.2d, xzr
        dup v10.2d, xzr
        dup v12.2d, xzr
        dup v14.2d, xzr
        b Compute4x8Enter
        InitFromBias4x8:
            ld1 {v8.8h}, [x12]
            ld1 {v10.8h}, [x12]
            ld1 {v12.8h}, [x12]
            ld1 {v14.8h}, [x12]
            add x12, x12, #16
    Compute4x8Enter:
        bl Compute4x8Unit
        Activation4x8:
            cmp x4, #3
            beq Relu64x8
            cmp x4, #1
            beq Relu4x8
            b Write4x8

            Relu64x8:
                fmin v8.8h, v8.8h, v7.8h
                fmin v10.8h, v10.8h, v7.8h
                fmin v12.8h, v12.8h, v7.8h
                fmin v14.8h, v14.8h, v7.8h

            Relu4x8:
                dup v6.8h, wzr
                fmax v8.8h, v8.8h, v6.8h
                fmax v10.8h, v10.8h, v6.8h
                fmax v12.8h, v12.8h, v6.8h
                fmax v14.8h, v14.8h, v6.8h
            Write4x8:
                mov x22, x21
                st1 {v8.8h}, [x22], x8
                st1 {v10.8h}, [x22], x8
                st1 {v12.8h}, [x22], x8
                st1 {v14.8h}, [x22]
                add x21, x21, #16
                subs x13, x13, #8

    LoopCol4x4:
        adds x13, x13, #8
        cbz x13, LoopRow4End
    LoopCol4x4Core:
        mov x10, x0  // update matrixA
        ld1 {v0.4h}, [x10], #8
        mov x14, x5  // reload depth
        prfm pldl1strm, [x11, #632]
        ld1 {v3.4h}, [x11], #8
        cbnz x12, InitFromBias4x4
        dup v8.2s, wzr
        dup v10.2s, wzr
        dup v12.2s, wzr
        dup v14.2s, wzr
        b Compute4x4Enter
        InitFromBias4x4:
            ld1 {v8.4h}, [x12]
            ld1 {v10.4h}, [x12]
            ld1 {v12.4h}, [x12]
            ld1 {v14.4h}, [x12]
            add x12, x12, #8
    Compute4x4Enter:
        bl Compute4x4Unit
        Activation4x4:
            cmp x4, #3
            beq Relu64x4
            cmp x4, #1
            beq Relu4x4
            b Write4x4

            Relu64x4:
                fmin v8.4h, v8.4h, v7.4h
                fmin v10.4h, v10.4h, v7.4h
                fmin v12.4h, v12.4h, v7.4h
                fmin v14.4h, v14.4h, v7.4h

            Relu4x4:
                dup v6.4h, wzr
                fmax v8.4h, v8.4h, v6.4h
                fmax v10.4h, v10.4h, v6.4h
                fmax v12.4h, v12.4h, v6.4h
                fmax v14.4h, v14.4h, v6.4h
            Write4x4:
                mov x22, x21
                cmp x13, #1
                beq Write4x1
                cmp x13, #2
                beq Write4x2
                cmp x13, #3
                beq Write4x3
                st1 {v8.4h}, [x22], x8
                st1 {v10.4h}, [x22], x8
                st1 {v12.4h}, [x22], x8
                st1 {v14.4h}, [x22]
                add x21, x21, #8
                subs x13, x13, #4
                bgt LoopCol4x4Core
                b LoopRow4End
            Write4x1:
                st1 {v8.h}[0], [x22], x8
                st1 {v10.h}[0], [x22], x8
                st1 {v12.h}[0], [x22], x8
                st1 {v14.h}[0], [x22]
                b LoopRow4End
            Write4x2:
                st1 {v8.s}[0], [x22], x8
                st1 {v10.s}[0], [x22], x8
                st1 {v12.s}[0], [x22], x8
                st1 {v14.s}[0], [x22]
                b LoopRow4End
            Write4x3:
                add x23, x22, #4
                st1 {v8.s}[0], [x22], x8
                st1 {v8.h}[2], [x23], x8
                st1 {v10.s}[0], [x22], x8
                st1 {v10.h}[2], [x23], x8
                st1 {v12.s}[0], [x22], x8
                st1 {v12.h}[2], [x23], x8
                st1 {v14.s}[0], [x22], x8
                st1 {v14.h}[2], [x23], x8
            LoopRow4End:
                add x0, x0, x16, lsl #2
                add x2, x2, x8, lsl #2
                subs x6, x6, #4

LoopRowTail:
    adds x6, x6, #4
    cbz x6, End
    cmp x6, #1
    beq LoopRow1
    cmp x6, #2
    beq LoopRow2
    // LoopRow3
    mov x11, x1  // reload matrixB
    mov x12, x3  // reload bias
    mov x13, x7  // reload col
    mov x21, x2  // relocate output
    subs x13, x13, #16
    blt LoopCol3x8
    LoopCol3x16:
        mov x10, x0  // update matrixA
        mov x14, x5  // reload depth
        cbnz x12, InitFromBias3x16
        dup v8.2d, xzr
        dup v9.2d, xzr
        dup v10.2d, xzr
        dup v11.2d, xzr
        dup v12.2d, xzr
        dup v13.2d, xzr
        b Compute3x16Enter
        InitFromBias3x16:
            ld1 {v8.8h, v9.8h}, [x12]
            ld1 {v10.8h, v11.8h}, [x12]
            ld1 {v12.8h, v13.8h}, [x12]
            add x12, x12, #32
    Compute3x16Enter:
        bl Compute3x16Unit
        Activation3x16:
            cmp x4, #3
            beq Relu63x16
            cmp x4, #1
            beq Relu3x16
            b Write3x16

            Relu63x16:
                fmin v8.8h, v8.8h, v7.8h
                fmin v9.8h, v9.8h, v7.8h
                fmin v10.8h, v10.8h, v7.8h
                fmin v11.8h, v11.8h, v7.8h
                fmin v12.8h, v12.8h, v7.8h
                fmin v13.8h, v13.8h, v7.8h

            Relu3x16:
                dup v6.8h, wzr
                fmax v8.8h, v8.8h, v6.8h
                fmax v9.8h, v9.8h, v6.8h
                fmax v10.8h, v10.8h, v6.8h
                fmax v11.8h, v11.8h, v6.8h
                fmax v12.8h, v12.8h, v6.8h
                fmax v13.8h, v13.8h, v6.8h
            Write3x16:
                mov x22, x21
                st1 {v8.8h, v9.8h}, [x22], x8
                st1 {v10.8h, v11.8h}, [x22], x8
                st1 {v12.8h, v13.8h}, [x22]
                add x21, x21, #32
                subs x13, x13, #16
                bge LoopCol3x16

    LoopCol3x8:
        adds x13, x13, #16
        cbz x13, End
        subs x13, x13, #8
        blt LoopCol3x4
        mov x10, x0  // update matrixA
        mov x14, x5  // reload depth
        cbnz x12, InitFromBias3x8
        dup v8.2d, xzr
        dup v10.2d, xzr
        dup v12.2d, xzr
        b Compute3x8Enter
        InitFromBias3x8:
            ld1 {v8.8h}, [x12]
            ld1 {v10.8h}, [x12]
            ld1 {v12.8h}, [x12]
            add x12, x12, #16
    Compute3x8Enter:
        bl Compute3x8Unit
        Activation3x8:
            cmp x4, #3
            beq Relu63x8
            cmp x4, #1
            beq Relu3x8
            b Write3x8

            Relu63x8:
                fmin v8.8h, v8.8h, v7.8h
                fmin v10.8h, v10.8h, v7.8h
                fmin v12.8h, v12.8h, v7.8h

            Relu3x8:
                dup v6.8h, wzr
                fmax v8.8h, v8.8h, v6.8h
                fmax v10.8h, v10.8h, v6.8h
                fmax v12.8h, v12.8h, v6.8h
            Write3x8:
                mov x22, x21
                st1 {v8.8h}, [x22], x8
                st1 {v10.8h}, [x22], x8
                st1 {v12.8h}, [x22]
                add x21, x21, #16
                subs x13, x13, #8

    LoopCol3x4:
        adds x13, x13, #8
        cbz x13, End
    LoopCol3x4Core:
        mov x10, x0  // update matrixA
        mov x14, x5  // reload depth
        cbnz x12, InitFromBias3x4
        dup v8.2s, wzr
        dup v10.2s, wzr
        dup v12.2s, wzr
        b Compute3x4Enter
        InitFromBias3x4:
            ld1 {v8.4h}, [x12]
            ld1 {v10.4h}, [x12]
            ld1 {v12.4h}, [x12]
            add x12, x12, #8
    Compute3x4Enter:
        bl Compute3x4Unit
        Activation3x4:
            cmp x4, #3
            beq Relu63x4
            cmp x4, #1
            beq Relu3x4
            b Write3x4

            Relu63x4:
                fmin v8.4h, v8.4h, v7.4h
                fmin v10.4h, v10.4h, v7.4h
                fmin v12.4h, v12.4h, v7.4h

            Relu3x4:
                dup v6.4h, wzr
                fmax v8.4h, v8.4h, v6.4h
                fmax v10.4h, v10.4h, v6.4h
                fmax v12.4h, v12.4h, v6.4h
            Write3x4:
                mov x22, x21
                cmp x13, #1
                beq Write3x1
                cmp x13, #2
                beq Write3x2
                cmp x13, #3
                beq Write3x3
                st1 {v8.4h}, [x22], x8
                st1 {v10.4h}, [x22], x8
                st1 {v12.4h}, [x22]
                add x21, x21, #8
                subs x13, x13, #4
                bgt LoopCol3x4Core
                b End
            Write3x1:
                st1 {v8.h}[0], [x22], x8
                st1 {v10.h}[0], [x22], x8
                st1 {v12.h}[0], [x22]
                b End
            Write3x2:
                st1 {v8.s}[0], [x22], x8
                st1 {v10.s}[0], [x22], x8
                st1 {v12.s}[0], [x22]
                b End
            Write3x3:
                add x23, x22, #4
                st1 {v8.s}[0], [x22], x8
                st1 {v8.h}[2], [x23], x8
                st1 {v10.s}[0], [x22], x8
                st1 {v10.h}[2], [x23], x8
                st1 {v12.s}[0], [x22], x8
                st1 {v12.h}[2], [x23], x8
                b End

LoopRow2:
    mov x11, x1  // reload matrixB
    mov x12, x3  // reload bias
    mov x13, x7  // reload col
    mov x21, x2  // relocate output
    subs x13, x13, #16
    blt LoopCol2x8
    LoopCol2x16:
        mov x10, x0  // update matrixA
        mov x14, x5  // reload depth
        cbnz x12, InitFromBias2x16
        dup v8.2d, xzr
        dup v9.2d, xzr
        dup v10.2d, xzr
        dup v11.2d, xzr
        b Compute2x16Enter
        InitFromBias2x16:
            ld1 {v8.8h, v9.8h}, [x12]
            ld1 {v10.8h, v11.8h}, [x12]
            add x12, x12, #32
    Compute2x16Enter:
        bl Compute2x16Unit
        Activation2x16:
            cmp x4, #3
            beq Relu62x16
            cmp x4, #1
            beq Relu2x16
            b Write2x16

            Relu62x16:
                fmin v8.8h, v8.8h, v7.8h
                fmin v9.8h, v9.8h, v7.8h
                fmin v10.8h, v10.8h, v7.8h
                fmin v11.8h, v11.8h, v7.8h

            Relu2x16:
                dup v6.8h, wzr
                fmax v8.8h, v8.8h, v6.8h
                fmax v9.8h, v9.8h, v6.8h
                fmax v10.8h, v10.8h, v6.8h
                fmax v11.8h, v11.8h, v6.8h
            Write2x16:
                mov x22, x21
                st1 {v8.8h, v9.8h}, [x22], x8
                st1 {v10.8h, v11.8h}, [x22]
                add x21, x21, #32
                subs x13, x13, #16
                bge LoopCol2x16

    LoopCol2x8:
        adds x13, x13, #16
        cbz x13, End
        subs x13, x13, #8
        blt LoopCol2x4
        mov x10, x0  // update matrixA
        mov x14, x5  // reload depth
        cbnz x12, InitFromBias2x8
        dup v8.2d, xzr
        dup v10.2d, xzr
        b Compute2x8Enter
        InitFromBias2x8:
            ld1 {v8.8h}, [x12]
            ld1 {v10.8h}, [x12]
            add x12, x12, #16
    Compute2x8Enter:
        bl Compute2x8Unit
        Activation2x8:
            cmp x4, #3
            beq Relu62x8
            cmp x4, #1
            beq Relu2x8
            b Write2x8

            Relu62x8:
                fmin v8.8h, v8.8h, v7.8h
                fmin v10.8h, v10.8h, v7.8h

            Relu2x8:
                dup v6.8h, wzr
                fmax v8.8h, v8.8h, v6.8h
                fmax v10.8h, v10.8h, v6.8h
            Write2x8:
                mov x22, x21
                st1 {v8.8h}, [x22], x8
                st1 {v10.8h}, [x22]
                add x21, x21, #16
                subs x13, x13, #8

    LoopCol2x4:
        adds x13, x13, #8
        cbz x13, End
    LoopCol2x4Core:
        mov x10, x0  // update matrixA
        mov x14, x5  // reload depth
        cbnz x12, InitFromBias2x4
        dup v8.2s, wzr
        dup v10.2s, wzr
        b Compute2x4Enter
        InitFromBias2x4:
            ld1 {v8.4h}, [x12]
            ld1 {v10.4h}, [x12]
            add x12, x12, #8
    Compute2x4Enter:
        bl Compute2x4Unit
        Activation2x4:
            cmp x4, #3
            beq Relu62x4
            cmp x4, #1
            beq Relu2x4
            b Write2x4

            Relu62x4:
                fmin v8.4h, v8.4h, v7.4h
                fmin v10.4h, v10.4h, v7.4h
            Relu2x4:
                dup v6.4h, wzr
                fmax v8.4h, v8.4h, v6.4h
                fmax v10.4h, v10.4h, v6.4h
            Write2x4:
                mov x22, x21
                cmp x13, #1
                beq Write2x1
                cmp x13, #2
                beq Write2x2
                cmp x13, #3
                beq Write2x3
                st1 {v8.4h}, [x22], x8
                st1 {v10.4h}, [x22]
                add x21, x21, #8
                subs x13, x13, #4
                bgt LoopCol2x4Core
                b End
            Write2x1:
                st1 {v8.h}[0], [x22], x8
                st1 {v10.h}[0], [x22]
                b End
            Write2x2:
                st1 {v8.s}[0], [x22], x8
                st1 {v10.s}[0], [x22]
                b End
            Write2x3:
                add x23, x22, #4
                st1 {v8.s}[0], [x22], x8
                st1 {v8.h}[2], [x23], x8
                st1 {v10.s}[0], [x22], x8
                st1 {v10.h}[2], [x23], x8
                b End

LoopRow1:
    mov x11, x1  // reload matrixB
    mov x12, x3  // reload bias
    mov x13, x7  // reload col
    mov x21, x2  // relocate output
    subs x13, x13, #16
    blt LoopCol1x8
    LoopCol1x16:
        mov x10, x0  // update matrixA
        mov x14, x5  // reload depth
        cbnz x12, InitFromBias1x16
        dup v8.2d, xzr
        dup v9.2d, xzr
        b Compute1x16Enter
        InitFromBias1x16:
            ld1 {v8.8h, v9.8h}, [x12], #32
    Compute1x16Enter:
        bl Compute1x16Unit
        Activation1x16:
            cmp x4, #3
            beq Relu61x16
            cmp x4, #1
            beq Relu1x16
            b Write1x16

            Relu61x16:
                fmin v8.8h, v8.8h, v7.8h
                fmin v9.8h, v9.8h, v7.8h

            Relu1x16:
                dup v6.8h, wzr
                fmax v8.8h, v8.8h, v6.8h
                fmax v9.8h, v9.8h, v6.8h
            Write1x16:
                st1 {v8.8h, v9.8h}, [x21], #32
                subs x13, x13, #16
                bge LoopCol1x16

    LoopCol1x8:
        adds x13, x13, #16
        cbz x13, End
        subs x13, x13, #8
        blt LoopCol1x4
        mov x10, x0  // update matrixA
        mov x14, x5  // reload depth
        cbnz x12, InitFromBias1x8
        dup v8.2d, xzr
        b Compute1x8Enter
        InitFromBias1x8:
            ld1 {v8.8h}, [x12], #16
    Compute1x8Enter:
        bl Compute1x8Unit
        Activation1x8:
            cmp x4, #3
            beq Relu61x8
            cmp x4, #1
            beq Relu1x8
            b Write1x8

            Relu61x8:
                fmin v8.8h, v8.8h, v7.8h

            Relu1x8:
                dup v6.8h, wzr
                fmax v8.8h, v8.8h, v6.8h
            Write1x8:
                st1 {v8.8h}, [x21], #16
                subs x13, x13, #8

    LoopCol1x4:
        adds x13, x13, #8
        cbz x13, End
    LoopCol1x4Core:
        mov x10, x0  // update matrixA
        mov x14, x5  // reload depth
        cbnz x12, InitFromBias1x4
        dup v8.2s, wzr
        b Compute1x4Enter
        InitFromBias1x4:
            ld1 {v8.4h}, [x12], #8
    Compute1x4Enter:
        bl Compute1x4Unit
        Activation1x4:
            cmp x4, #3
            beq Relu61x4
            cmp x4, #1
            beq Relu1x4
            b Write1x4

            Relu61x4:
                fmin v8.4h, v8.4h, v7.4h
            Relu1x4:
                dup v6.4h, wzr
                fmax v8.4h, v8.4h, v6.4h
            Write1x4:
                cmp x13, #1
                beq Write1x1
                cmp x13, #2
                beq Write1x2
                cmp x13, #3
                beq Write1x3
                st1 {v8.4h}, [x21], #8
                subs x13, x13, #4
                bgt LoopCol1x4Core
                b End
            Write1x1:
                st1 {v8.h}[0], [x21]
                b End
            Write1x2:
                st1 {v8.s}[0], [x21]
                b End
            Write1x3:
                add x22, x21, #4
                st1 {v8.s}[0], [x21]
                st1 {v8.h}[2], [x22]
                b End

Compute12x16Unit:
    subs x14, x14, #2
    ble Compute12x16End
    Compute12x16:
        prfm pldl1keep, [x10, #632]
        ld1 {v1.8h, v2.8h}, [x10], #32
        ld1 {v4.8h, v5.8h}, [x11], #32
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v0.h[1]
        fmla v12.8h, v3.8h, v0.h[2]
        fmla v14.8h, v3.8h, v0.h[3]
        fmla v16.8h, v3.8h, v0.h[4]
        fmla v18.8h, v3.8h, v0.h[5]
        fmla v20.8h, v3.8h, v0.h[6]
        fmla v22.8h, v3.8h, v0.h[7]
        fmla v24.8h, v3.8h, v1.h[0]
        fmla v26.8h, v3.8h, v1.h[1]
        fmla v28.8h, v3.8h, v1.h[2]
        fmla v30.8h, v3.8h, v1.h[3]
        prfm pldl1strm, [x11, #632]
        ld1 {v6.8h}, [x11], #16
        fmla v9.8h, v4.8h, v0.h[0]
        fmla v11.8h, v4.8h, v0.h[1]
        fmla v13.8h, v4.8h, v0.h[2]
        fmla v15.8h, v4.8h, v0.h[3]
        fmla v17.8h, v4.8h, v0.h[4]
        fmla v19.8h, v4.8h, v0.h[5]
        fmla v21.8h, v4.8h, v0.h[6]
        fmla v23.8h, v4.8h, v0.h[7]
        fmla v25.8h, v4.8h, v1.h[0]
        fmla v27.8h, v4.8h, v1.h[1]
        fmla v29.8h, v4.8h, v1.h[2]
        fmla v31.8h, v4.8h, v1.h[3]

        fmla v8.8h, v5.8h, v1.h[4]
        fmla v10.8h, v5.8h, v1.h[5]
        fmla v12.8h, v5.8h, v1.h[6]
        fmla v14.8h, v5.8h, v1.h[7]
        fmla v16.8h, v5.8h, v2.h[0]
        fmla v18.8h, v5.8h, v2.h[1]
        fmla v20.8h, v5.8h, v2.h[2]
        fmla v22.8h, v5.8h, v2.h[3]
        fmla v24.8h, v5.8h, v2.h[4]
        fmla v26.8h, v5.8h, v2.h[5]
        fmla v28.8h, v5.8h, v2.h[6]
        fmla v30.8h, v5.8h, v2.h[7]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h}, [x11], #16
        fmla v9.8h, v6.8h, v1.h[4]
        fmla v11.8h, v6.8h, v1.h[5]
        fmla v13.8h, v6.8h, v1.h[6]
        fmla v15.8h, v6.8h, v1.h[7]
        prfm pldl1keep, [x10, #632]        
        ld1 {v0.8h}, [x10], #16
        fmla v17.8h, v6.8h, v2.h[0]
        fmla v19.8h, v6.8h, v2.h[1]
        fmla v21.8h, v6.8h, v2.h[2]
        fmla v23.8h, v6.8h, v2.h[3]
        fmla v25.8h, v6.8h, v2.h[4]
        fmla v27.8h, v6.8h, v2.h[5]
        fmla v29.8h, v6.8h, v2.h[6]
        fmla v31.8h, v6.8h, v2.h[7]

        subs x14, x14, #2
        bgt Compute12x16
    Compute12x16End:
        cbnz x14, Compute12x16End1
        prfm pldl1keep, [x10, #632]
        ld1 {v1.4h}, [x10], #8
        ld1 {v4.8h}, [x11], #16
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v0.h[1]
        fmla v12.8h, v3.8h, v0.h[2]
        fmla v14.8h, v3.8h, v0.h[3]
        fmla v16.8h, v3.8h, v0.h[4]
        fmla v18.8h, v3.8h, v0.h[5]
        fmla v20.8h, v3.8h, v0.h[6]
        fmla v22.8h, v3.8h, v0.h[7]
        fmla v24.8h, v3.8h, v1.h[0]
        fmla v26.8h, v3.8h, v1.h[1]
        fmla v28.8h, v3.8h, v1.h[2]
        fmla v30.8h, v3.8h, v1.h[3]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h}, [x11], #16
        fmla v9.8h, v4.8h, v0.h[0]
        fmla v11.8h, v4.8h, v0.h[1]
        fmla v13.8h, v4.8h, v0.h[2]
        fmla v15.8h, v4.8h, v0.h[3]
        ld1 {v2.8h}, [x10], #16
        fmla v17.8h, v4.8h, v0.h[4]
        fmla v19.8h, v4.8h, v0.h[5]
        fmla v21.8h, v4.8h, v0.h[6]
        fmla v23.8h, v4.8h, v0.h[7]
        fmla v25.8h, v4.8h, v1.h[0]
        fmla v27.8h, v4.8h, v1.h[1]
        fmla v29.8h, v4.8h, v1.h[2]
        fmla v31.8h, v4.8h, v1.h[3]
        mov v0.16b, v2.16b
    Compute12x16End1:
        ld1 {v1.4h}, [x10]
        ld1 {v4.8h}, [x11], #16
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v0.h[1]
        fmla v12.8h, v3.8h, v0.h[2]
        fmla v14.8h, v3.8h, v0.h[3]
        fmla v16.8h, v3.8h, v0.h[4]
        fmla v18.8h, v3.8h, v0.h[5]
        fmla v20.8h, v3.8h, v0.h[6]
        fmla v22.8h, v3.8h, v0.h[7]
        fmla v24.8h, v3.8h, v1.h[0]
        fmla v26.8h, v3.8h, v1.h[1]
        fmla v28.8h, v3.8h, v1.h[2]
        fmla v30.8h, v3.8h, v1.h[3]
        fmla v9.8h, v4.8h, v0.h[0]
        fmla v11.8h, v4.8h, v0.h[1]
        fmla v13.8h, v4.8h, v0.h[2]
        fmla v15.8h, v4.8h, v0.h[3]
        fmla v17.8h, v4.8h, v0.h[4]
        fmla v19.8h, v4.8h, v0.h[5]
        fmla v21.8h, v4.8h, v0.h[6]
        fmla v23.8h, v4.8h, v0.h[7]
        fmla v25.8h, v4.8h, v1.h[0]
        fmla v27.8h, v4.8h, v1.h[1]
        fmla v29.8h, v4.8h, v1.h[2]
        fmla v31.8h, v4.8h, v1.h[3]
        ret
            
Compute12x8Unit:
    subs x14, x14, #2
    ble Compute12x8End
    Compute12x8:
        prfm pldl1keep, [x10, #632]
        ld1 {v1.8h, v2.8h}, [x10], #32
        ld1 {v4.8h}, [x11], #16
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v0.h[1]
        fmla v12.8h, v3.8h, v0.h[2]
        fmla v14.8h, v3.8h, v0.h[3]
        fmla v16.8h, v3.8h, v0.h[4]
        fmla v18.8h, v3.8h, v0.h[5]
        fmla v20.8h, v3.8h, v0.h[6]
        fmla v22.8h, v3.8h, v0.h[7]
        fmla v24.8h, v3.8h, v1.h[0]
        fmla v26.8h, v3.8h, v1.h[1]
        fmla v28.8h, v3.8h, v1.h[2]
        fmla v30.8h, v3.8h, v1.h[3]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h}, [x11], #16
        fmla v8.8h, v4.8h, v1.h[4]
        fmla v10.8h, v4.8h, v1.h[5]
        fmla v12.8h, v4.8h, v1.h[6]
        fmla v14.8h, v4.8h, v1.h[7]
        ld1 {v0.8h}, [x10], #16
        fmla v16.8h, v4.8h, v2.h[0]
        fmla v18.8h, v4.8h, v2.h[1]
        fmla v20.8h, v4.8h, v2.h[2]
        fmla v22.8h, v4.8h, v2.h[3]
        fmla v24.8h, v4.8h, v2.h[4]
        fmla v26.8h, v4.8h, v2.h[5]
        fmla v28.8h, v4.8h, v2.h[6]
        fmla v30.8h, v4.8h, v2.h[7]

        subs x14, x14, #2
        bgt Compute12x8
    Compute12x8End:
        cbnz x14, Compute12x8End1
        prfm pldl1keep, [x10, #632]
        ld1 {v1.4h}, [x10], #8
        ld1 {v4.8h}, [x11], #16
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v0.h[1]
        fmla v12.8h, v3.8h, v0.h[2]
        fmla v14.8h, v3.8h, v0.h[3]
        fmla v16.8h, v3.8h, v0.h[4]
        fmla v18.8h, v3.8h, v0.h[5]
        fmla v20.8h, v3.8h, v0.h[6]
        fmla v22.8h, v3.8h, v0.h[7]
        fmla v24.8h, v3.8h, v1.h[0]
        fmla v26.8h, v3.8h, v1.h[1]
        fmla v28.8h, v3.8h, v1.h[2]
        fmla v30.8h, v3.8h, v1.h[3]
        ld1 {v0.8h}, [x10], #16
        mov v3.16b, v4.16b
    Compute12x8End1:
        ld1 {v1.4h}, [x10]
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v0.h[1]
        fmla v12.8h, v3.8h, v0.h[2]
        fmla v14.8h, v3.8h, v0.h[3]
        fmla v16.8h, v3.8h, v0.h[4]
        fmla v18.8h, v3.8h, v0.h[5]
        fmla v20.8h, v3.8h, v0.h[6]
        fmla v22.8h, v3.8h, v0.h[7]
        fmla v24.8h, v3.8h, v1.h[0]
        fmla v26.8h, v3.8h, v1.h[1]
        fmla v28.8h, v3.8h, v1.h[2]
        fmla v30.8h, v3.8h, v1.h[3]
        ret

Compute12x4Unit:
    subs x14, x14, #2
    ble Compute12x4End
    Compute12x4:
        prfm pldl1keep, [x10, #632]
        ld1 {v1.8h, v2.8h}, [x10], #32
        ld1 {v4.4h}, [x11], #8
        fmla v8.4h, v3.4h, v0.h[0]
        fmla v10.4h, v3.4h, v0.h[1]
        fmla v12.4h, v3.4h, v0.h[2]
        fmla v14.4h, v3.4h, v0.h[3]
        fmla v16.4h, v3.4h, v0.h[4]
        fmla v18.4h, v3.4h, v0.h[5]
        fmla v20.4h, v3.4h, v0.h[6]
        fmla v22.4h, v3.4h, v0.h[7]
        fmla v24.4h, v3.4h, v1.h[0]
        fmla v26.4h, v3.4h, v1.h[1]
        fmla v28.4h, v3.4h, v1.h[2]
        fmla v30.4h, v3.4h, v1.h[3]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.4h}, [x11], #8
        fmla v8.4h, v4.4h, v1.h[4]
        fmla v10.4h, v4.4h, v1.h[5]
        fmla v12.4h, v4.4h, v1.h[6]
        fmla v14.4h, v4.4h, v1.h[7]
        ld1 {v0.8h}, [x10], #16
        fmla v16.4h, v4.4h, v2.h[0]
        fmla v18.4h, v4.4h, v2.h[1]
        fmla v20.4h, v4.4h, v2.h[2]
        fmla v22.4h, v4.4h, v2.h[3]
        fmla v24.4h, v4.4h, v2.h[4]
        fmla v26.4h, v4.4h, v2.h[5]
        fmla v28.4h, v4.4h, v2.h[6]
        fmla v30.4h, v4.4h, v2.h[7]

        subs x14, x14, #2
        bgt Compute12x4
    Compute12x4End:
        cbnz x14, Compute12x4End1
        prfm pldl1keep, [x10, #632]
        ld1 {v1.4h}, [x10], #8
        ld1 {v4.4h}, [x11], #8
        fmla v8.4h, v3.4h, v0.h[0]
        fmla v10.4h, v3.4h, v0.h[1]
        fmla v12.4h, v3.4h, v0.h[2]
        fmla v14.4h, v3.4h, v0.h[3]
        fmla v16.4h, v3.4h, v0.h[4]
        fmla v18.4h, v3.4h, v0.h[5]
        fmla v20.4h, v3.4h, v0.h[6]
        fmla v22.4h, v3.4h, v0.h[7]
        fmla v24.4h, v3.4h, v1.h[0]
        fmla v26.4h, v3.4h, v1.h[1]
        fmla v28.4h, v3.4h, v1.h[2]
        fmla v30.4h, v3.4h, v1.h[3]
        ld1 {v0.8h}, [x10], #16
        mov v3.8b, v4.8b
    Compute12x4End1:
        ld1 {v1.4h}, [x10]
        fmla v8.4h, v3.4h, v0.h[0]
        fmla v10.4h, v3.4h, v0.h[1]
        fmla v12.4h, v3.4h, v0.h[2]
        fmla v14.4h, v3.4h, v0.h[3]
        fmla v16.4h, v3.4h, v0.h[4]
        fmla v18.4h, v3.4h, v0.h[5]
        fmla v20.4h, v3.4h, v0.h[6]
        fmla v22.4h, v3.4h, v0.h[7]
        fmla v24.4h, v3.4h, v1.h[0]
        fmla v26.4h, v3.4h, v1.h[1]
        fmla v28.4h, v3.4h, v1.h[2]
        fmla v30.4h, v3.4h, v1.h[3]
        ret

Compute8x16Unit:
    subs x14, x14, #2
    ble Compute8x16End
    Compute8x16:
        prfm pldl1keep, [x10, #632]
        ld1 {v1.8h}, [x10], #16
        ld1 {v4.8h, v5.8h}, [x11], #32
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v0.h[1]
        fmla v12.8h, v3.8h, v0.h[2]
        fmla v14.8h, v3.8h, v0.h[3]
        fmla v16.8h, v3.8h, v0.h[4]
        fmla v18.8h, v3.8h, v0.h[5]
        fmla v20.8h, v3.8h, v0.h[6]
        fmla v22.8h, v3.8h, v0.h[7]
        prfm pldl1strm, [x11, #632]
        ld1 {v6.8h}, [x11], #16
        fmla v9.8h, v4.8h, v0.h[0]
        fmla v11.8h, v4.8h, v0.h[1]
        fmla v13.8h, v4.8h, v0.h[2]
        fmla v15.8h, v4.8h, v0.h[3]
        fmla v17.8h, v4.8h, v0.h[4]
        fmla v19.8h, v4.8h, v0.h[5]
        fmla v21.8h, v4.8h, v0.h[6]
        fmla v23.8h, v4.8h, v0.h[7]

        fmla v8.8h, v5.8h, v1.h[0]
        fmla v10.8h, v5.8h, v1.h[1]
        fmla v12.8h, v5.8h, v1.h[2]
        fmla v14.8h, v5.8h, v1.h[3]
        fmla v16.8h, v5.8h, v1.h[4]
        fmla v18.8h, v5.8h, v1.h[5]
        fmla v20.8h, v5.8h, v1.h[6]
        fmla v22.8h, v5.8h, v1.h[7]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h}, [x11], #16
        fmla v9.8h, v6.8h, v1.h[0]
        fmla v11.8h, v6.8h, v1.h[1]
        fmla v13.8h, v6.8h, v1.h[2]
        fmla v15.8h, v6.8h, v1.h[3]
        prfm pldl1keep, [x10, #632]
        ld1 {v0.8h}, [x10], #16
        fmla v17.8h, v6.8h, v1.h[4]
        fmla v19.8h, v6.8h, v1.h[5]
        fmla v21.8h, v6.8h, v1.h[6]
        fmla v23.8h, v6.8h, v1.h[7]

        subs x14, x14, #2
        bgt Compute8x16
    Compute8x16End:
        cbnz x14, Compute8x16End1
        prfm pldl1keep, [x10, #632]
        ld1 {v1.8h}, [x10]
        ld1 {v4.8h}, [x11], #16
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v0.h[1]
        fmla v12.8h, v3.8h, v0.h[2]
        fmla v14.8h, v3.8h, v0.h[3]
        fmla v16.8h, v3.8h, v0.h[4]
        fmla v18.8h, v3.8h, v0.h[5]
        fmla v20.8h, v3.8h, v0.h[6]
        fmla v22.8h, v3.8h, v0.h[7]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h}, [x11], #16
        fmla v9.8h, v4.8h, v0.h[0]
        fmla v11.8h, v4.8h, v0.h[1]
        fmla v13.8h, v4.8h, v0.h[2]
        fmla v15.8h, v4.8h, v0.h[3]
        fmla v17.8h, v4.8h, v0.h[4]
        fmla v19.8h, v4.8h, v0.h[5]
        fmla v21.8h, v4.8h, v0.h[6]
        fmla v23.8h, v4.8h, v0.h[7]
        mov v0.16b, v1.16b
    Compute8x16End1:
        ld1 {v4.8h}, [x11], #16
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v0.h[1]
        fmla v12.8h, v3.8h, v0.h[2]
        fmla v14.8h, v3.8h, v0.h[3]
        fmla v16.8h, v3.8h, v0.h[4]
        fmla v18.8h, v3.8h, v0.h[5]
        fmla v20.8h, v3.8h, v0.h[6]
        fmla v22.8h, v3.8h, v0.h[7]
        fmla v9.8h, v4.8h, v0.h[0]
        fmla v11.8h, v4.8h, v0.h[1]
        fmla v13.8h, v4.8h, v0.h[2]
        fmla v15.8h, v4.8h, v0.h[3]
        fmla v17.8h, v4.8h, v0.h[4]
        fmla v19.8h, v4.8h, v0.h[5]
        fmla v21.8h, v4.8h, v0.h[6]
        fmla v23.8h, v4.8h, v0.h[7]
        ret

Compute8x8Unit:
    subs x14, x14, #2
    ble Compute8x8End
    Compute8x8:
        prfm pldl1keep, [x10, #632]
        ld1 {v1.8h}, [x10], #16
        ld1 {v4.8h}, [x11], #16
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v0.h[1]
        fmla v12.8h, v3.8h, v0.h[2]
        fmla v14.8h, v3.8h, v0.h[3]
        fmla v16.8h, v3.8h, v0.h[4]
        fmla v18.8h, v3.8h, v0.h[5]
        fmla v20.8h, v3.8h, v0.h[6]
        fmla v22.8h, v3.8h, v0.h[7]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h}, [x11], #16
        fmla v8.8h, v4.8h, v1.h[0]
        fmla v10.8h, v4.8h, v1.h[1]
        fmla v12.8h, v4.8h, v1.h[2]
        fmla v14.8h, v4.8h, v1.h[3]
        ld1 {v0.8h}, [x10], #16
        fmla v16.8h, v4.8h, v1.h[4]
        fmla v18.8h, v4.8h, v1.h[5]
        fmla v20.8h, v4.8h, v1.h[6]
        fmla v22.8h, v4.8h, v1.h[7]

        subs x14, x14, #2
        bgt Compute8x8
    Compute8x8End:
        cbnz x14, Compute8x8End1
        prfm pldl1keep, [x10, #632]
        ld1 {v1.8h}, [x10]
        ld1 {v4.8h}, [x11], #16
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v0.h[1]
        fmla v12.8h, v3.8h, v0.h[2]
        fmla v14.8h, v3.8h, v0.h[3]
        fmla v16.8h, v3.8h, v0.h[4]
        fmla v18.8h, v3.8h, v0.h[5]
        fmla v20.8h, v3.8h, v0.h[6]
        fmla v22.8h, v3.8h, v0.h[7]
        mov v0.16b, v1.16b
        mov v3.16b, v4.16b
    Compute8x8End1:
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v0.h[1]
        fmla v12.8h, v3.8h, v0.h[2]
        fmla v14.8h, v3.8h, v0.h[3]
        fmla v16.8h, v3.8h, v0.h[4]
        fmla v18.8h, v3.8h, v0.h[5]
        fmla v20.8h, v3.8h, v0.h[6]
        fmla v22.8h, v3.8h, v0.h[7]
        ret

Compute8x4Unit:
    subs x14, x14, #2
    ble Compute8x4End
    Compute8x4:
        prfm pldl1keep, [x10, #632]
        ld1 {v1.8h}, [x10], #16
        ld1 {v4.4h}, [x11], #8
        fmla v8.4h, v3.4h, v0.h[0]
        fmla v10.4h, v3.4h, v0.h[1]
        fmla v12.4h, v3.4h, v0.h[2]
        fmla v14.4h, v3.4h, v0.h[3]
        fmla v16.4h, v3.4h, v0.h[4]
        fmla v18.4h, v3.4h, v0.h[5]
        fmla v20.4h, v3.4h, v0.h[6]
        fmla v22.4h, v3.4h, v0.h[7]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.4h}, [x11], #8
        fmla v8.4h, v4.4h, v1.h[0]
        fmla v10.4h, v4.4h, v1.h[1]
        fmla v12.4h, v4.4h, v1.h[2]
        fmla v14.4h, v4.4h, v1.h[3]
        ld1 {v0.8h}, [x10], #16
        fmla v16.4h, v4.4h, v1.h[4]
        fmla v18.4h, v4.4h, v1.h[5]
        fmla v20.4h, v4.4h, v1.h[6]
        fmla v22.4h, v4.4h, v1.h[7]

        subs x14, x14, #2
        bgt Compute8x4
    Compute8x4End:
        cbnz x14, Compute8x4End1
        prfm pldl1keep, [x10, #632]
        ld1 {v1.8h}, [x10]
        ld1 {v4.4h}, [x11], #8
        fmla v8.4h, v3.4h, v0.h[0]
        fmla v10.4h, v3.4h, v0.h[1]
        fmla v12.4h, v3.4h, v0.h[2]
        fmla v14.4h, v3.4h, v0.h[3]
        fmla v16.4h, v3.4h, v0.h[4]
        fmla v18.4h, v3.4h, v0.h[5]
        fmla v20.4h, v3.4h, v0.h[6]
        fmla v22.4h, v3.4h, v0.h[7]
        mov v0.16b, v1.16b
        mov v3.8b, v4.8b
    Compute8x4End1:
        fmla v8.4h, v3.4h, v0.h[0]
        fmla v10.4h, v3.4h, v0.h[1]
        fmla v12.4h, v3.4h, v0.h[2]
        fmla v14.4h, v3.4h, v0.h[3]
        fmla v16.4h, v3.4h, v0.h[4]
        fmla v18.4h, v3.4h, v0.h[5]
        fmla v20.4h, v3.4h, v0.h[6]
        fmla v22.4h, v3.4h, v0.h[7]
        ret

Compute4x16Unit:
    subs x14, x14, #2
    ble Compute4x16End
    Compute4x16:
        prfm pldl1keep, [x10, #632]
        ld1 {v1.4h}, [x10], #8
        ld1 {v4.8h, v5.8h}, [x11], #32
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v0.h[1]
        fmla v12.8h, v3.8h, v0.h[2]
        fmla v14.8h, v3.8h, v0.h[3]
        prfm pldl1strm, [x11, #632]
        ld1 {v6.8h}, [x11], #16
        fmla v9.8h, v4.8h, v0.h[0]
        fmla v11.8h, v4.8h, v0.h[1]
        fmla v13.8h, v4.8h, v0.h[2]
        fmla v15.8h, v4.8h, v0.h[3]

        fmla v8.8h, v5.8h, v1.h[0]
        fmla v10.8h, v5.8h, v1.h[1]
        fmla v12.8h, v5.8h, v1.h[2]
        fmla v14.8h, v5.8h, v1.h[3]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h}, [x11], #16
        fmla v9.8h, v6.8h, v1.h[0]
        fmla v11.8h, v6.8h, v1.h[1]
        fmla v13.8h, v6.8h, v1.h[2]
        fmla v15.8h, v6.8h, v1.h[3]
        ld1 {v0.4h}, [x10], #8

        subs x14, x14, #2
        bgt Compute4x16
    Compute4x16End:
        cbnz x14, Compute4x16End1
        prfm pldl1keep, [x10, #632]
        ld1 {v1.4h}, [x10]
        ld1 {v4.8h}, [x11], #16
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v0.h[1]
        fmla v12.8h, v3.8h, v0.h[2]
        fmla v14.8h, v3.8h, v0.h[3]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h}, [x11], #16
        fmla v9.8h, v4.8h, v0.h[0]
        fmla v11.8h, v4.8h, v0.h[1]
        fmla v13.8h, v4.8h, v0.h[2]
        fmla v15.8h, v4.8h, v0.h[3]
        mov v0.8b, v1.8b
    Compute4x16End1:
        ld1 {v4.8h}, [x11], #16
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v0.h[1]
        fmla v12.8h, v3.8h, v0.h[2]
        fmla v14.8h, v3.8h, v0.h[3]
        fmla v9.8h, v4.8h, v0.h[0]
        fmla v11.8h, v4.8h, v0.h[1]
        fmla v13.8h, v4.8h, v0.h[2]
        fmla v15.8h, v4.8h, v0.h[3]
        ret

Compute4x8Unit:
    subs x14, x14, #2
    ble Compute4x8End
    Compute4x8:
        prfm pldl1keep, [x10, #632]
        ld1 {v1.4h}, [x10], #8
        ld1 {v4.8h}, [x11], #16
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v0.h[1]
        fmla v12.8h, v3.8h, v0.h[2]
        fmla v14.8h, v3.8h, v0.h[3]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h}, [x11], #16
        fmla v8.8h, v4.8h, v1.h[0]
        fmla v10.8h, v4.8h, v1.h[1]
        fmla v12.8h, v4.8h, v1.h[2]
        fmla v14.8h, v4.8h, v1.h[3]
        ld1 {v0.4h}, [x10], #8

        subs x14, x14, #2
        bgt Compute4x8
    Compute4x8End:
        cbnz x14, Compute4x8End1
        prfm pldl1keep, [x10, #632]
        ld1 {v1.4h}, [x10]
        ld1 {v4.8h}, [x11], #16
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v0.h[1]
        fmla v12.8h, v3.8h, v0.h[2]
        fmla v14.8h, v3.8h, v0.h[3]
        mov v0.8b, v1.8b
        mov v3.16b, v4.16b
    Compute4x8End1:
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v0.h[1]
        fmla v12.8h, v3.8h, v0.h[2]
        fmla v14.8h, v3.8h, v0.h[3]
        ret

Compute4x4Unit:
    subs x14, x14, #2
    ble Compute4x4End
    Compute4x4:
        prfm pldl1keep, [x10, #632]
        ld1 {v1.4h}, [x10], #8
        ld1 {v4.4h}, [x11], #8
        fmla v8.4h, v3.4h, v0.h[0]
        fmla v10.4h, v3.4h, v0.h[1]
        fmla v12.4h, v3.4h, v0.h[2]
        fmla v14.4h, v3.4h, v0.h[3]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.4h}, [x11], #8
        fmla v8.4h, v4.4h, v1.h[0]
        fmla v10.4h, v4.4h, v1.h[1]
        fmla v12.4h, v4.4h, v1.h[2]
        fmla v14.4h, v4.4h, v1.h[3]
        ld1 {v0.4h}, [x10], #8

        subs x14, x14, #2
        bgt Compute4x4
    Compute4x4End:
        cbnz x14, Compute4x4End1
        prfm pldl1keep, [x10, #632]
        ld1 {v1.4h}, [x10]
        ld1 {v4.4h}, [x11], #8
        fmla v8.4h, v3.4h, v0.h[0]
        fmla v10.4h, v3.4h, v0.h[1]
        fmla v12.4h, v3.4h, v0.h[2]
        fmla v14.4h, v3.4h, v0.h[3]
        mov v0.8b, v1.8b
        mov v3.8b, v4.8b
    Compute4x4End1:
        fmla v8.4h, v3.4h, v0.h[0]
        fmla v10.4h, v3.4h, v0.h[1]
        fmla v12.4h, v3.4h, v0.h[2]
        fmla v14.4h, v3.4h, v0.h[3]
        ret

Compute3x16Unit:
    add x19, x10, x16
    add x20, x10, x16, lsl #1
    subs x14, x14, #8
    blt Compute3x16End4
    Compute3x16:
        ld1 {v0.8h}, [x10], #16
        ld1 {v1.8h}, [x19], #16
        ld1 {v2.8h}, [x20], #16
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v1.h[0]
        fmla v12.8h, v3.8h, v2.h[0]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v9.8h, v4.8h, v0.h[0]
        fmla v11.8h, v4.8h, v1.h[0]
        fmla v13.8h, v4.8h, v2.h[0]
        fmla v8.8h, v5.8h, v0.h[1]
        fmla v10.8h, v5.8h, v1.h[1]
        fmla v12.8h, v5.8h, v2.h[1]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v9.8h, v6.8h, v0.h[1]
        fmla v11.8h, v6.8h, v1.h[1]
        fmla v13.8h, v6.8h, v2.h[1]
        fmla v8.8h, v3.8h, v0.h[2]
        fmla v10.8h, v3.8h, v1.h[2]
        fmla v12.8h, v3.8h, v2.h[2]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v9.8h, v4.8h, v0.h[2]
        fmla v11.8h, v4.8h, v1.h[2]
        fmla v13.8h, v4.8h, v2.h[2]
        fmla v8.8h, v5.8h, v0.h[3]
        fmla v10.8h, v5.8h, v1.h[3]
        fmla v12.8h, v5.8h, v2.h[3]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v9.8h, v6.8h, v0.h[3]
        fmla v11.8h, v6.8h, v1.h[3]
        fmla v13.8h, v6.8h, v2.h[3]

        fmla v8.8h, v3.8h, v0.h[4]
        fmla v10.8h, v3.8h, v1.h[4]
        fmla v12.8h, v3.8h, v2.h[4]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v9.8h, v4.8h, v0.h[4]
        fmla v11.8h, v4.8h, v1.h[4]
        fmla v13.8h, v4.8h, v2.h[4]
        fmla v8.8h, v5.8h, v0.h[5]
        fmla v10.8h, v5.8h, v1.h[5]
        fmla v12.8h, v5.8h, v2.h[5]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v9.8h, v6.8h, v0.h[5]
        fmla v11.8h, v6.8h, v1.h[5]
        fmla v13.8h, v6.8h, v2.h[5]
        fmla v8.8h, v3.8h, v0.h[6]
        fmla v10.8h, v3.8h, v1.h[6]
        fmla v12.8h, v3.8h, v2.h[6]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v9.8h, v4.8h, v0.h[6]
        fmla v11.8h, v4.8h, v1.h[6]
        fmla v13.8h, v4.8h, v2.h[6]
        fmla v8.8h, v5.8h, v0.h[7]
        fmla v10.8h, v5.8h, v1.h[7]
        fmla v12.8h, v5.8h, v2.h[7]
        fmla v9.8h, v6.8h, v0.h[7]
        fmla v11.8h, v6.8h, v1.h[7]
        fmla v13.8h, v6.8h, v2.h[7]

        subs x14, x14, #8
        bge Compute3x16
    Compute3x16End4:
        adds x14, x14, #8
        cbz x14, Compute3x16Return
        subs x14, x14, #4
        blt Compute3x16EndTail
        ld1 {v0.4h}, [x10], #8
        ld1 {v1.4h}, [x19], #8
        ld1 {v2.4h}, [x20], #8
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v1.h[0]
        fmla v12.8h, v3.8h, v2.h[0]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v9.8h, v4.8h, v0.h[0]
        fmla v11.8h, v4.8h, v1.h[0]
        fmla v13.8h, v4.8h, v2.h[0]
        fmla v8.8h, v5.8h, v0.h[1]
        fmla v10.8h, v5.8h, v1.h[1]
        fmla v12.8h, v5.8h, v2.h[1]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v9.8h, v6.8h, v0.h[1]
        fmla v11.8h, v6.8h, v1.h[1]
        fmla v13.8h, v6.8h, v2.h[1]
        fmla v8.8h, v3.8h, v0.h[2]
        fmla v10.8h, v3.8h, v1.h[2]
        fmla v12.8h, v3.8h, v2.h[2]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v9.8h, v4.8h, v0.h[2]
        fmla v11.8h, v4.8h, v1.h[2]
        fmla v13.8h, v4.8h, v2.h[2]
        fmla v8.8h, v5.8h, v0.h[3]
        fmla v10.8h, v5.8h, v1.h[3]
        fmla v12.8h, v5.8h, v2.h[3]
        fmla v9.8h, v6.8h, v0.h[3]
        fmla v11.8h, v6.8h, v1.h[3]
        fmla v13.8h, v6.8h, v2.h[3]
        subs x14, x14, #4
    Compute3x16EndTail:
        adds x14, x14, #4
        cbz x14, Compute3x16Return
        cmp x14, #1
        beq Compute3x16EndTail1
        cmp x14, #2
        beq Compute3x16EndTail2
        ld1 {v0.4h}, [x10]
        ld1 {v1.4h}, [x19]
        ld1 {v2.s}[0], [x20], #4
        ld1 {v2.h}[2], [x20]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v1.h[0]
        fmla v12.8h, v3.8h, v2.h[0]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v9.8h, v4.8h, v0.h[0]
        fmla v11.8h, v4.8h, v1.h[0]
        fmla v13.8h, v4.8h, v2.h[0]
        fmla v8.8h, v5.8h, v0.h[1]
        fmla v10.8h, v5.8h, v1.h[1]
        fmla v12.8h, v5.8h, v2.h[1]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v9.8h, v6.8h, v0.h[1]
        fmla v11.8h, v6.8h, v1.h[1]
        fmla v13.8h, v6.8h, v2.h[1]
        fmla v8.8h, v3.8h, v0.h[2]
        fmla v10.8h, v3.8h, v1.h[2]
        fmla v12.8h, v3.8h, v2.h[2]
        fmla v9.8h, v4.8h, v0.h[2]
        fmla v11.8h, v4.8h, v1.h[2]
        fmla v13.8h, v4.8h, v2.h[2]
        b Compute3x16Return
    Compute3x16EndTail2:
        ld1 {v0.4h}, [x10]
        ld1 {v1.4h}, [x19]
        ld1 {v2.s}[0], [x20]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v1.h[0]
        fmla v12.8h, v3.8h, v2.h[0]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v9.8h, v4.8h, v0.h[0]
        fmla v11.8h, v4.8h, v1.h[0]
        fmla v13.8h, v4.8h, v2.h[0]
        fmla v8.8h, v5.8h, v0.h[1]
        fmla v10.8h, v5.8h, v1.h[1]
        fmla v12.8h, v5.8h, v2.h[1]
        fmla v9.8h, v6.8h, v0.h[1]
        fmla v11.8h, v6.8h, v1.h[1]
        fmla v13.8h, v6.8h, v2.h[1]
        b Compute3x16Return
    Compute3x16EndTail1:
        ld1 {v0.h}[0], [x10]
        ld1 {v1.h}[0], [x19]
        ld1 {v2.h}[0], [x20]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v1.h[0]
        fmla v12.8h, v3.8h, v2.h[0]
        fmla v9.8h, v4.8h, v0.h[0]
        fmla v11.8h, v4.8h, v1.h[0]
        fmla v13.8h, v4.8h, v2.h[0]
    Compute3x16Return:
        ret

Compute3x8Unit:
    add x19, x10, x16
    add x20, x10, x16, lsl #1
    subs x14, x14, #8
    blt Compute3x8End4
    Compute3x8:
        ld1 {v0.8h}, [x10], #16
        ld1 {v1.8h}, [x19], #16
        ld1 {v2.8h}, [x20], #16
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v1.h[0]
        fmla v12.8h, v3.8h, v2.h[0]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v8.8h, v4.8h, v0.h[1]
        fmla v10.8h, v4.8h, v1.h[1]
        fmla v12.8h, v4.8h, v2.h[1]
        fmla v8.8h, v5.8h, v0.h[2]
        fmla v10.8h, v5.8h, v1.h[2]
        fmla v12.8h, v5.8h, v2.h[2]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v8.8h, v6.8h, v0.h[3]
        fmla v10.8h, v6.8h, v1.h[3]
        fmla v12.8h, v6.8h, v2.h[3]
        fmla v8.8h, v3.8h, v0.h[4]
        fmla v10.8h, v3.8h, v1.h[4]
        fmla v12.8h, v3.8h, v2.h[4]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v8.8h, v4.8h, v0.h[5]
        fmla v10.8h, v4.8h, v1.h[5]
        fmla v12.8h, v4.8h, v2.h[5]
        fmla v8.8h, v5.8h, v0.h[6]
        fmla v10.8h, v5.8h, v1.h[6]
        fmla v12.8h, v5.8h, v2.h[6]
        fmla v8.8h, v6.8h, v0.h[7]
        fmla v10.8h, v6.8h, v1.h[7]
        fmla v12.8h, v6.8h, v2.h[7]

        subs x14, x14, #8
        bge Compute3x8
    Compute3x8End4:
        adds x14, x14, #8
        cbz x14, Compute3x8Return
        subs x14, x14, #4
        blt Compute3x8EndTail
        ld1 {v0.4h}, [x10], #8
        ld1 {v1.4h}, [x19], #8
        ld1 {v2.4h}, [x20], #8
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v1.h[0]
        fmla v12.8h, v3.8h, v2.h[0]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v8.8h, v4.8h, v0.h[1]
        fmla v10.8h, v4.8h, v1.h[1]
        fmla v12.8h, v4.8h, v2.h[1]
        fmla v8.8h, v5.8h, v0.h[2]
        fmla v10.8h, v5.8h, v1.h[2]
        fmla v12.8h, v5.8h, v2.h[2]
        fmla v8.8h, v6.8h, v0.h[3]
        fmla v10.8h, v6.8h, v1.h[3]
        fmla v12.8h, v6.8h, v2.h[3]
        subs x14, x14, #4
    Compute3x8EndTail:
        adds x14, x14, #4
        cbz x14, Compute3x8Return
        cmp x14, #1
        beq Compute3x8EndTail1
        cmp x14, #2
        beq Compute3x8EndTail2
        ld1 {v0.4h}, [x10]
        ld1 {v1.4h}, [x19]
        ld1 {v2.s}[0], [x20], #4
        ld1 {v2.h}[2], [x20]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v1.h[0]
        fmla v12.8h, v3.8h, v2.h[0]
        ld1 {v5.8h}, [x11], #16
        fmla v8.8h, v4.8h, v0.h[1]
        fmla v10.8h, v4.8h, v1.h[1]
        fmla v12.8h, v4.8h, v2.h[1]
        fmla v8.8h, v5.8h, v0.h[2]
        fmla v10.8h, v5.8h, v1.h[2]
        fmla v12.8h, v5.8h, v2.h[2]
        b Compute3x8Return
    Compute3x8EndTail2:
        ld1 {v0.4h}, [x10]
        ld1 {v1.4h}, [x19]
        ld2 {v2.h, v3.h}[0], [x20]
        prfm pldl1strm, [x11, #632]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v8.8h, v5.8h, v0.h[0]
        fmla v10.8h, v5.8h, v1.h[0]
        fmla v12.8h, v5.8h, v2.h[0]
        fmla v8.8h, v6.8h, v0.h[1]
        fmla v10.8h, v6.8h, v1.h[1]
        fmla v12.8h, v6.8h, v3.h[0]
        b Compute3x8Return
    Compute3x8EndTail1:
        ld1 {v0.h}[0], [x10]
        ld1 {v1.h}[0], [x19]
        ld1 {v2.h}[0], [x20]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h}, [x11], #16
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v1.h[0]
        fmla v12.8h, v3.8h, v2.h[0]
    Compute3x8Return:
        ret

Compute3x4Unit:
    add x19, x10, x16
    add x20, x10, x16, lsl #1
    subs x14, x14, #8
    blt Compute3x4End4
    Compute3x4:
        ld1 {v0.8h}, [x10], #16
        ld1 {v1.8h}, [x19], #16
        ld1 {v2.8h}, [x20], #16
        prfm pldl1strm, [x11, #632]
        ld1 {v3.4h, v4.4h}, [x11], #16
        fmla v8.4h, v3.4h, v0.h[0]
        fmla v10.4h, v3.4h, v1.h[0]
        fmla v12.4h, v3.4h, v2.h[0]
        ld1 {v5.4h, v6.4h}, [x11], #16
        fmla v8.4h, v4.4h, v0.h[1]
        fmla v10.4h, v4.4h, v1.h[1]
        fmla v12.4h, v4.4h, v2.h[1]
        fmla v8.4h, v5.4h, v0.h[2]
        fmla v10.4h, v5.4h, v1.h[2]
        fmla v12.4h, v5.4h, v2.h[2]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.4h, v4.4h}, [x11], #16
        fmla v8.4h, v6.4h, v0.h[3]
        fmla v10.4h, v6.4h, v1.h[3]
        fmla v12.4h, v6.4h, v2.h[3]
        fmla v8.4h, v3.4h, v0.h[4]
        fmla v10.4h, v3.4h, v1.h[4]
        fmla v12.4h, v3.4h, v2.h[4]
        ld1 {v5.4h, v6.4h}, [x11], #16
        fmla v8.4h, v4.4h, v0.h[5]
        fmla v10.4h, v4.4h, v1.h[5]
        fmla v12.4h, v4.4h, v2.h[5]
        fmla v8.4h, v5.4h, v0.h[6]
        fmla v10.4h, v5.4h, v1.h[6]
        fmla v12.4h, v5.4h, v2.h[6]
        fmla v8.4h, v6.4h, v0.h[7]
        fmla v10.4h, v6.4h, v1.h[7]
        fmla v12.4h, v6.4h, v2.h[7]

        subs x14, x14, #8
        bge Compute3x4
    Compute3x4End4:
        adds x14, x14, #8
        cbz x14, Compute3x4Return
        subs x14, x14, #4
        blt Compute3x4EndTail
        ld1 {v0.4h}, [x10], #8
        ld1 {v1.4h}, [x19], #8
        ld1 {v2.4h}, [x20], #8
        prfm pldl1strm, [x11, #632]
        ld1 {v3.4h, v4.4h}, [x11], #16
        fmla v8.4h, v3.4h, v0.h[0]
        fmla v10.4h, v3.4h, v1.h[0]
        fmla v12.4h, v3.4h, v2.h[0]
        ld1 {v5.4h, v6.4h}, [x11], #16
        fmla v8.4h, v4.4h, v0.h[1]
        fmla v10.4h, v4.4h, v1.h[1]
        fmla v12.4h, v4.4h, v2.h[1]
        fmla v8.4h, v5.4h, v0.h[2]
        fmla v10.4h, v5.4h, v1.h[2]
        fmla v12.4h, v5.4h, v2.h[2]
        fmla v8.4h, v6.4h, v0.h[3]
        fmla v10.4h, v6.4h, v1.h[3]
        fmla v12.4h, v6.4h, v2.h[3]
        subs x14, x14, #4
    Compute3x4EndTail:
        adds x14, x14, #4
        cbz x14, Compute3x4Return
        cmp x14, #1
        beq Compute3x4EndTail1
        cmp x14, #2
        beq Compute3x4EndTail2
        ld1 {v0.4h}, [x10]
        ld1 {v1.4h}, [x19]
        ld1 {v2.s}[0], [x20], #4
        ld1 {v2.h}[2], [x20]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.4h, v4.4h}, [x11], #16
        fmla v8.4h, v3.4h, v0.h[0]
        fmla v10.4h, v3.4h, v1.h[0]
        fmla v12.4h, v3.4h, v2.h[0]
        ld1 {v5.4h}, [x11], #8
        fmla v8.4h, v4.4h, v0.h[1]
        fmla v10.4h, v4.4h, v1.h[1]
        fmla v12.4h, v4.4h, v2.h[1]
        fmla v8.4h, v5.4h, v0.h[2]
        fmla v10.4h, v5.4h, v1.h[2]
        fmla v12.4h, v5.4h, v2.h[2]
        b Compute3x4Return
    Compute3x4EndTail2:
        ld1 {v0.4h}, [x10]
        ld1 {v1.4h}, [x19]
        ld2 {v2.h, v3.h}[0], [x20]
        prfm pldl1strm, [x11, #632]
        ld1 {v5.4h, v6.4h}, [x11], #16
        fmla v8.4h, v5.4h, v0.h[0]
        fmla v10.4h, v5.4h, v1.h[0]
        fmla v12.4h, v5.4h, v2.h[0]
        fmla v8.4h, v6.4h, v0.h[1]
        fmla v10.4h, v6.4h, v1.h[1]
        fmla v12.4h, v6.4h, v3.h[0]
        b Compute3x4Return
    Compute3x4EndTail1:
        ld1 {v0.h}[0], [x10]
        ld1 {v1.h}[0], [x19]
        ld1 {v2.h}[0], [x20]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.4h}, [x11], #8
        fmla v8.4h, v3.4h, v0.h[0]
        fmla v10.4h, v3.4h, v1.h[0]
        fmla v12.4h, v3.4h, v2.h[0]
    Compute3x4Return:
        ret

Compute2x16Unit:
    add x19, x10, x16
    subs x14, x14, #8
    blt Compute2x16End4
    Compute2x16:
        ld1 {v0.8h}, [x10], #16
        ld1 {v1.8h}, [x19], #16
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v1.h[0]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v9.8h, v4.8h, v0.h[0]
        fmla v11.8h, v4.8h, v1.h[0]
        fmla v8.8h, v5.8h, v0.h[1]
        fmla v10.8h, v5.8h, v1.h[1]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v9.8h, v6.8h, v0.h[1]
        fmla v11.8h, v6.8h, v1.h[1]
        fmla v8.8h, v3.8h, v0.h[2]
        fmla v10.8h, v3.8h, v1.h[2]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v9.8h, v4.8h, v0.h[2]
        fmla v11.8h, v4.8h, v1.h[2]
        fmla v8.8h, v5.8h, v0.h[3]
        fmla v10.8h, v5.8h, v1.h[3]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v9.8h, v6.8h, v0.h[3]
        fmla v11.8h, v6.8h, v1.h[3]

        fmla v8.8h, v3.8h, v0.h[4]
        fmla v10.8h, v3.8h, v1.h[4]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v9.8h, v4.8h, v0.h[4]
        fmla v11.8h, v4.8h, v1.h[4]
        fmla v8.8h, v5.8h, v0.h[5]
        fmla v10.8h, v5.8h, v1.h[5]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v9.8h, v6.8h, v0.h[5]
        fmla v11.8h, v6.8h, v1.h[5]
        fmla v8.8h, v3.8h, v0.h[6]
        fmla v10.8h, v3.8h, v1.h[6]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v9.8h, v4.8h, v0.h[6]
        fmla v11.8h, v4.8h, v1.h[6]
        fmla v8.8h, v5.8h, v0.h[7]
        fmla v10.8h, v5.8h, v1.h[7]
        fmla v9.8h, v6.8h, v0.h[7]
        fmla v11.8h, v6.8h, v1.h[7]

        subs x14, x14, #8
        bge Compute2x16
    Compute2x16End4:
        adds x14, x14, #8
        cbz x14, Compute2x16Return
        subs x14, x14, #4
        blt Compute2x16EndTail
        ld1 {v0.4h}, [x10], #8
        ld1 {v1.4h}, [x19], #8
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v1.h[0]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v9.8h, v4.8h, v0.h[0]
        fmla v11.8h, v4.8h, v1.h[0]
        fmla v8.8h, v5.8h, v0.h[1]
        fmla v10.8h, v5.8h, v1.h[1]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v9.8h, v6.8h, v0.h[1]
        fmla v11.8h, v6.8h, v1.h[1]
        fmla v8.8h, v3.8h, v0.h[2]
        fmla v10.8h, v3.8h, v1.h[2]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v9.8h, v4.8h, v0.h[2]
        fmla v11.8h, v4.8h, v1.h[2]
        fmla v8.8h, v5.8h, v0.h[3]
        fmla v10.8h, v5.8h, v1.h[3]
        fmla v9.8h, v6.8h, v0.h[3]
        fmla v11.8h, v6.8h, v1.h[3]
        subs x14, x14, #4
    Compute2x16EndTail:
        adds x14, x14, #4
        cbz x14, Compute2x16Return
        cmp x14, #1
        beq Compute2x16EndTail1
        cmp x14, #2
        beq Compute2x16EndTail2
        ld1 {v0.4h}, [x10]
        ld1 {v1.s}[0], [x19], #4
        ld1 {v1.h}[2], [x19]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v1.h[0]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v9.8h, v4.8h, v0.h[0]
        fmla v11.8h, v4.8h, v1.h[0]
        fmla v8.8h, v5.8h, v0.h[1]
        fmla v10.8h, v5.8h, v1.h[1]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v9.8h, v6.8h, v0.h[1]
        fmla v11.8h, v6.8h, v1.h[1]
        fmla v8.8h, v3.8h, v0.h[2]
        fmla v10.8h, v3.8h, v1.h[2]
        fmla v9.8h, v4.8h, v0.h[2]
        fmla v11.8h, v4.8h, v1.h[2]
        b Compute2x16Return
    Compute2x16EndTail2:
        ld1 {v0.4h}, [x10]
        ld2 {v1.h, v2.h}[0], [x19]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v1.h[0]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v9.8h, v4.8h, v0.h[0]
        fmla v11.8h, v4.8h, v1.h[0]
        fmla v8.8h, v5.8h, v0.h[1]
        fmla v10.8h, v5.8h, v2.h[0]
        fmla v9.8h, v6.8h, v0.h[1]
        fmla v11.8h, v6.8h, v2.h[0]
        b Compute2x16Return
    Compute2x16EndTail1:
        ld1 {v0.h}[0], [x10]
        ld1 {v1.h}[0], [x19]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v1.h[0]
        fmla v9.8h, v4.8h, v0.h[0]
        fmla v11.8h, v4.8h, v1.h[0]
    Compute2x16Return:
        ret

Compute2x8Unit:
    add x19, x10, x16
    subs x14, x14, #8
    blt Compute2x8End4
    Compute2x8:
        ld1 {v0.8h}, [x10], #16
        ld1 {v1.8h}, [x19], #16
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v1.h[0]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v8.8h, v4.8h, v0.h[1]
        fmla v10.8h, v4.8h, v1.h[1]
        fmla v8.8h, v5.8h, v0.h[2]
        fmla v10.8h, v5.8h, v1.h[2]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v8.8h, v6.8h, v0.h[3]
        fmla v10.8h, v6.8h, v1.h[3]
        fmla v8.8h, v3.8h, v0.h[4]
        fmla v10.8h, v3.8h, v1.h[4]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v8.8h, v4.8h, v0.h[5]
        fmla v10.8h, v4.8h, v1.h[5]
        fmla v8.8h, v5.8h, v0.h[6]
        fmla v10.8h, v5.8h, v1.h[6]
        fmla v8.8h, v6.8h, v0.h[7]
        fmla v10.8h, v6.8h, v1.h[7]

        subs x14, x14, #8
        bge Compute2x8
    Compute2x8End4:
        adds x14, x14, #8
        cbz x14, Compute2x8Return
        subs x14, x14, #4
        blt Compute2x8EndTail
        ld1 {v0.4h}, [x10], #8
        ld1 {v1.4h}, [x19], #8
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v1.h[0]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v8.8h, v4.8h, v0.h[1]
        fmla v10.8h, v4.8h, v1.h[1]
        fmla v8.8h, v5.8h, v0.h[2]
        fmla v10.8h, v5.8h, v1.h[2]
        fmla v8.8h, v6.8h, v0.h[3]
        fmla v10.8h, v6.8h, v1.h[3]
        subs x14, x14, #4
    Compute2x8EndTail:
        adds x14, x14, #4
        cbz x14, Compute2x8Return
        cmp x14, #1
        beq Compute2x8EndTail1
        cmp x14, #2
        beq Compute2x8EndTail2
        ld1 {v0.4h}, [x10]
        ld3 {v1.h, v2.h, v3.h}[0], [x19]
        prfm pldl1strm, [x11, #632]
        ld1 {v4.8h, v5.8h}, [x11], #32
        fmla v8.8h, v4.8h, v0.h[0]
        fmla v10.8h, v4.8h, v1.h[0]
        ld1 {v6.8h}, [x11], #16
        fmla v8.8h, v5.8h, v0.h[1]
        fmla v10.8h, v5.8h, v2.h[0]
        fmla v8.8h, v6.8h, v0.h[2]
        fmla v10.8h, v6.8h, v3.h[0]
        b Compute2x8Return
    Compute2x8EndTail2:
        ld1 {v0.4h}, [x10]
        ld2 {v1.h, v2.h}[0], [x19]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v1.h[0]
        fmla v8.8h, v4.8h, v0.h[1]
        fmla v10.8h, v4.8h, v2.h[0]
        b Compute2x8Return
    Compute2x8EndTail1:
        ld1 {v0.h}[0], [x10]
        ld1 {v1.h}[0], [x19]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h}, [x11], #16
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v10.8h, v3.8h, v1.h[0]
    Compute2x8Return:
        ret

Compute2x4Unit:
    add x19, x10, x16
    subs x14, x14, #8
    blt Compute2x4End4
    Compute2x4:
        ld1 {v0.8h}, [x10], #16
        ld1 {v1.8h}, [x19], #16
        prfm pldl1strm, [x11, #632]
        ld1 {v3.4h, v4.4h}, [x11], #16
        fmla v8.4h, v3.4h, v0.h[0]
        fmla v10.4h, v3.4h, v1.h[0]
        ld1 {v5.4h, v6.4h}, [x11], #16
        fmla v8.4h, v4.4h, v0.h[1]
        fmla v10.4h, v4.4h, v1.h[1]
        fmla v8.4h, v5.4h, v0.h[2]
        fmla v10.4h, v5.4h, v1.h[2]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.4h, v4.4h}, [x11], #16
        fmla v8.4h, v6.4h, v0.h[3]
        fmla v10.4h, v6.4h, v1.h[3]
        fmla v8.4h, v3.4h, v0.h[4]
        fmla v10.4h, v3.4h, v1.h[4]
        ld1 {v5.4h, v6.4h}, [x11], #16
        fmla v8.4h, v4.4h, v0.h[5]
        fmla v10.4h, v4.4h, v1.h[5]
        fmla v8.4h, v5.4h, v0.h[6]
        fmla v10.4h, v5.4h, v1.h[6]
        fmla v8.4h, v6.4h, v0.h[7]
        fmla v10.4h, v6.4h, v1.h[7]

        subs x14, x14, #8
        bge Compute2x4
    Compute2x4End4:
        adds x14, x14, #8
        cbz x14, Compute2x4Return
        subs x14, x14, #4
        blt Compute2x4EndTail
        ld1 {v0.4h}, [x10], #8
        ld1 {v1.4h}, [x19], #8
        prfm pldl1strm, [x11, #632]
        ld1 {v3.4h, v4.4h}, [x11], #16
        fmla v8.4h, v3.4h, v0.h[0]
        fmla v10.4h, v3.4h, v1.h[0]
        ld1 {v5.4h, v6.4h}, [x11], #16
        fmla v8.4h, v4.4h, v0.h[1]
        fmla v10.4h, v4.4h, v1.h[1]
        fmla v8.4h, v5.4h, v0.h[2]
        fmla v10.4h, v5.4h, v1.h[2]
        fmla v8.4h, v6.4h, v0.h[3]
        fmla v10.4h, v6.4h, v1.h[3]
        subs x14, x14, #4
    Compute2x4EndTail:
        adds x14, x14, #4
        cbz x14, Compute2x4Return
        cmp x14, #1
        beq Compute2x4EndTail1
        cmp x14, #2
        beq Compute2x4EndTail2
        ld1 {v0.4h}, [x10]
        ld3 {v1.h, v2.h, v3.h}[0], [x19]
        prfm pldl1strm, [x11, #632]
        ld1 {v4.4h, v5.4h}, [x11], #16
        fmla v8.4h, v4.4h, v0.h[0]
        fmla v10.4h, v4.4h, v1.h[0]
        ld1 {v6.4h}, [x11], #8
        fmla v8.4h, v5.4h, v0.h[1]
        fmla v10.4h, v5.4h, v2.h[0]
        fmla v8.4h, v6.4h, v0.h[2]
        fmla v10.4h, v6.4h, v3.h[0]
        b Compute2x4Return
    Compute2x4EndTail2:
        ld1 {v0.4h}, [x10]
        ld2 {v1.h, v2.h}[0], [x19]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.4h, v4.4h}, [x11], #16
        fmla v8.4h, v3.4h, v0.h[0]
        fmla v10.4h, v3.4h, v1.h[0]
        fmla v8.4h, v4.4h, v0.h[1]
        fmla v10.4h, v4.4h, v2.h[0]
        b Compute2x4Return
    Compute2x4EndTail1:
        ld1 {v0.h}[0], [x10]
        ld1 {v1.h}[0], [x19]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.4h}, [x11], #8
        fmla v8.4h, v3.4h, v0.h[0]
        fmla v10.4h, v3.4h, v1.h[0]
    Compute2x4Return:
        ret

Compute1x16Unit:
    subs x14, x14, #8
    blt Compute1x16End4
    Compute1x16:
        ld1 {v0.8h}, [x10], #16
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v8.8h, v3.8h, v0.h[0]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v9.8h, v4.8h, v0.h[0]
        fmla v8.8h, v5.8h, v0.h[1]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v9.8h, v6.8h, v0.h[1]
        fmla v8.8h, v3.8h, v0.h[2]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v9.8h, v4.8h, v0.h[2]
        fmla v8.8h, v5.8h, v0.h[3]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v9.8h, v6.8h, v0.h[3]

        fmla v8.8h, v3.8h, v0.h[4]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v9.8h, v4.8h, v0.h[4]
        fmla v8.8h, v5.8h, v0.h[5]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v9.8h, v6.8h, v0.h[5]
        fmla v8.8h, v3.8h, v0.h[6]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v9.8h, v4.8h, v0.h[6]
        fmla v8.8h, v5.8h, v0.h[7]
        fmla v9.8h, v6.8h, v0.h[7]

        subs x14, x14, #8
        bge Compute1x16
    Compute1x16End4:
        adds x14, x14, #8
        cbz x14, Compute1x16Return
        subs x14, x14, #4
        blt Compute1x16EndTail
        ld1 {v0.4h}, [x10], #8
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v8.8h, v3.8h, v0.h[0]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v9.8h, v4.8h, v0.h[0]
        fmla v8.8h, v5.8h, v0.h[1]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v9.8h, v6.8h, v0.h[1]
        fmla v8.8h, v3.8h, v0.h[2]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v9.8h, v4.8h, v0.h[2]
        fmla v8.8h, v5.8h, v0.h[3]
        fmla v9.8h, v6.8h, v0.h[3]
        subs x14, x14, #4
    Compute1x16EndTail:
        adds x14, x14, #4
        cbz x14, Compute1x16Return
        cmp x14, #1
        beq Compute1x16EndTail1
        cmp x14, #2
        beq Compute1x16EndTail2
        ld3 {v0.h, v1.h, v2.h}[0], [x10]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v8.8h, v3.8h, v0.h[0]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v9.8h, v4.8h, v0.h[0]
        fmla v8.8h, v5.8h, v1.h[0]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v9.8h, v6.8h, v1.h[0]
        fmla v8.8h, v3.8h, v2.h[0]
        fmla v9.8h, v4.8h, v2.h[0]
        b Compute1x16Return
    Compute1x16EndTail2:
        ld2 {v0.h, v1.h}[0], [x10]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v8.8h, v3.8h, v0.h[0]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v9.8h, v4.8h, v0.h[0]
        fmla v8.8h, v5.8h, v1.h[0]
        fmla v9.8h, v6.8h, v1.h[0]
        b Compute1x16Return
    Compute1x16EndTail1:
        ld1 {v0.h}[0], [x10]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v9.8h, v4.8h, v0.h[0]
    Compute1x16Return:
        ret

Compute1x8Unit:
    subs x14, x14, #8
    blt Compute1x8End4
    Compute1x8:
        ld1 {v0.8h}, [x10], #16
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v8.8h, v3.8h, v0.h[0]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v8.8h, v4.8h, v0.h[1]
        fmla v8.8h, v5.8h, v0.h[2]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v8.8h, v6.8h, v0.h[3]
        fmla v8.8h, v3.8h, v0.h[4]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v8.8h, v4.8h, v0.h[5]
        fmla v8.8h, v5.8h, v0.h[6]
        fmla v8.8h, v6.8h, v0.h[7]

        subs x14, x14, #8
        bge Compute1x8
    Compute1x8End4:
        adds x14, x14, #8
        cbz x14, Compute1x8Return
        subs x14, x14, #4
        blt Compute1x8EndTail
        ld1 {v0.4h}, [x10], #8
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v8.8h, v3.8h, v0.h[0]
        ld1 {v5.8h, v6.8h}, [x11], #32
        fmla v8.8h, v4.8h, v0.h[1]
        fmla v8.8h, v5.8h, v0.h[2]
        fmla v8.8h, v6.8h, v0.h[3]
        subs x14, x14, #4
    Compute1x8EndTail:
        adds x14, x14, #4
        cbz x14, Compute1x8Return
        cmp x14, #1
        beq Compute1x8EndTail1
        cmp x14, #2
        beq Compute1x8EndTail2
        ld3 {v0.h, v1.h, v2.h}[0], [x10]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v8.8h, v3.8h, v0.h[0]
        ld1 {v5.8h}, [x11], #16
        fmla v8.8h, v4.8h, v1.h[0]
        fmla v8.8h, v5.8h, v2.h[0]
        b Compute1x8Return
    Compute1x8EndTail2:
        ld2 {v0.h, v1.h}[0], [x10]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h, v4.8h}, [x11], #32
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v8.8h, v4.8h, v1.h[0]
        b Compute1x8Return
    Compute1x8EndTail1:
        ld1 {v0.h}[0], [x10]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.8h}, [x11], #16
        fmla v8.8h, v3.8h, v0.h[0]
    Compute1x8Return:
        ret

Compute1x4Unit:
    subs x14, x14, #8
    blt Compute1x4End4
    Compute1x4:
        ld1 {v0.8h}, [x10], #16
        prfm pldl1strm, [x11, #632]
        ld1 {v3.4h, v4.4h}, [x11], #16
        fmla v8.4h, v3.4h, v0.h[0]
        ld1 {v5.4h, v6.4h}, [x11], #16
        fmla v8.4h, v4.4h, v0.h[1]
        fmla v8.4h, v5.4h, v0.h[2]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.4h, v4.4h}, [x11], #16
        fmla v8.4h, v6.4h, v0.h[3]
        fmla v8.4h, v3.4h, v0.h[4]
        ld1 {v5.4h, v6.4h}, [x11], #16
        fmla v8.4h, v4.4h, v0.h[5]
        fmla v8.4h, v5.4h, v0.h[6]
        fmla v8.4h, v6.4h, v0.h[7]

        subs x14, x14, #8
        bge Compute1x4
    Compute1x4End4:
        adds x14, x14, #8
        cbz x14, Compute1x4Return
        subs x14, x14, #4
        blt Compute1x4EndTail
        ld1 {v0.4h}, [x10], #8
        prfm pldl1strm, [x11, #632]
        ld1 {v3.4h, v4.4h}, [x11], #16
        fmla v8.4h, v3.4h, v0.h[0]
        ld1 {v5.4h, v6.4h}, [x11], #16
        fmla v8.4h, v4.4h, v0.h[1]
        fmla v8.4h, v5.4h, v0.h[2]
        fmla v8.4h, v6.4h, v0.h[3]
        subs x14, x14, #4
    Compute1x4EndTail:
        adds x14, x14, #4
        cbz x14, Compute1x4Return
        cmp x14, #1
        beq Compute1x4EndTail1
        cmp x14, #2
        beq Compute1x4EndTail2
        ld3 {v0.h, v1.h, v2.h}[0], [x10]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.4h, v4.4h}, [x11], #16
        fmla v8.4h, v3.4h, v0.h[0]
        ld1 {v5.4h}, [x11], #8
        fmla v8.4h, v4.4h, v1.h[0]
        fmla v8.4h, v5.4h, v2.h[0]
        b Compute1x4Return
    Compute1x4EndTail2:
        ld2 {v0.h, v1.h}[0], [x10]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.4h, v4.4h}, [x11], #16
        fmla v8.4h, v3.4h, v0.h[0]
        fmla v8.4h, v4.4h, v1.h[0]
        b Compute1x4Return
    Compute1x4EndTail1:
        ld1 {v0.h}[0], [x10]
        prfm pldl1strm, [x11, #632]
        ld1 {v3.4h}, [x11], #8
        fmla v8.4h, v3.4h, v0.h[0]
    Compute1x4Return:
        ret

End:
  sub sp, sp, #192
  ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
  ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
  ldp x19, x20, [sp], #16
  ldp x21, x22, [sp], #16
  ldp x23, x24, [sp], #16
  ldp x29, x30, [sp], #16
  ret
#endif
