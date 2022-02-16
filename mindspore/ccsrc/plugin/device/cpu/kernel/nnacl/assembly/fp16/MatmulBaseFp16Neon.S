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

// void MatmulBaseFp16Neon(const float16_t *a, const float16_t *b, float16_t *c, const float16_t *bias, int act_type,
//                          int depth, int row, int col, size_t stride, size_t writeMode)
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

asm_function MatmulBaseFp16Neon
    sub sp, sp, #96
    st1 {v8.8h, v9.8h, v10.8h, v11.8h}, [sp], #64
    stp x19, x20, [sp], #16
    stp x21, x22, [sp], #16

    ldr x8, [sp]
    ldr x9, [sp, #8]  // act
    add x8, x8, x8  // stride * sizeof(float16_t)

    add x16, x7, x7 // col * sizeof(float16_t)
    add x17, x5, x5  // depth * zieof(float16_t)
    mov x11, x2
    dup v12.8h, wzr
    movi v13.8h, #0x46, lsl #8
LoopRowStart:
    cmp x6, #16
    bge LoopRow16
    cmp x6, #8
    bge LoopRow8
    b LoopRow4

LoopRow16:
    mov x15, #16
    mov x14, x1 // reload rhs ptr
    mov x13, x7 // reload rhs col
    mov x12, x3 // reload bias

    LoopCol16:
        mov x11, x2
        mov x10, x0 // reload lhs ptr
        mov x19, x5 // reload depth

        ld1 {v16.8h}, [x12], #16
        mov v17.16b, v16.16b
        mov v18.16b, v16.16b
        mov v19.16b, v16.16b
        mov v20.16b, v16.16b
        mov v21.16b, v16.16b
        mov v22.16b, v16.16b
        mov v23.16b, v16.16b
        mov v24.16b, v16.16b
        mov v25.16b, v16.16b
        mov v26.16b, v16.16b
        mov v27.16b, v16.16b
        mov v28.16b, v16.16b
        mov v29.16b, v16.16b
        mov v30.16b, v16.16b
        mov v31.16b, v16.16b

        cmp x19, #4
        blt LoopDepth16One

    LoopDepth16:
        ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x10], #64
        ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x14], #64
        fmla v16.8h, v8.8h, v0.h[0]
        fmla v17.8h, v8.8h, v0.h[1]
        fmla v18.8h, v8.8h, v0.h[2]
        fmla v19.8h, v8.8h, v0.h[3]
        fmla v20.8h, v8.8h, v0.h[4]
        fmla v21.8h, v8.8h, v0.h[5]
        fmla v22.8h, v8.8h, v0.h[6]
        fmla v23.8h, v8.8h, v0.h[7]
        fmla v24.8h, v8.8h, v1.h[0]
        fmla v25.8h, v8.8h, v1.h[1]
        fmla v26.8h, v8.8h, v1.h[2]
        fmla v27.8h, v8.8h, v1.h[3]
        fmla v28.8h, v8.8h, v1.h[4]
        fmla v29.8h, v8.8h, v1.h[5]
        fmla v30.8h, v8.8h, v1.h[6]
        fmla v31.8h, v8.8h, v1.h[7]
        ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x10], #64
        fmla v16.8h, v9.8h, v2.h[0]
        fmla v17.8h, v9.8h, v2.h[1]
        fmla v18.8h, v9.8h, v2.h[2]
        fmla v19.8h, v9.8h, v2.h[3]
        fmla v20.8h, v9.8h, v2.h[4]
        fmla v21.8h, v9.8h, v2.h[5]
        fmla v22.8h, v9.8h, v2.h[6]
        fmla v23.8h, v9.8h, v2.h[7]
        fmla v24.8h, v9.8h, v3.h[0]
        fmla v25.8h, v9.8h, v3.h[1]
        fmla v26.8h, v9.8h, v3.h[2]
        fmla v27.8h, v9.8h, v3.h[3]
        fmla v28.8h, v9.8h, v3.h[4]
        fmla v29.8h, v9.8h, v3.h[5]
        fmla v30.8h, v9.8h, v3.h[6]
        fmla v31.8h, v9.8h, v3.h[7]
        fmla v16.8h, v10.8h, v4.h[0]
        fmla v17.8h, v10.8h, v4.h[1]
        fmla v18.8h, v10.8h, v4.h[2]
        fmla v19.8h, v10.8h, v4.h[3]
        fmla v20.8h, v10.8h, v4.h[4]
        fmla v21.8h, v10.8h, v4.h[5]
        fmla v22.8h, v10.8h, v4.h[6]
        fmla v23.8h, v10.8h, v4.h[7]
        fmla v24.8h, v10.8h, v5.h[0]
        fmla v25.8h, v10.8h, v5.h[1]
        fmla v26.8h, v10.8h, v5.h[2]
        fmla v27.8h, v10.8h, v5.h[3]
        fmla v28.8h, v10.8h, v5.h[4]
        fmla v29.8h, v10.8h, v5.h[5]
        fmla v30.8h, v10.8h, v5.h[6]
        fmla v31.8h, v10.8h, v5.h[7]
        fmla v16.8h, v11.8h, v6.h[0]
        fmla v17.8h, v11.8h, v6.h[1]
        fmla v18.8h, v11.8h, v6.h[2]
        fmla v19.8h, v11.8h, v6.h[3]
        fmla v20.8h, v11.8h, v6.h[4]
        fmla v21.8h, v11.8h, v6.h[5]
        fmla v22.8h, v11.8h, v6.h[6]
        fmla v23.8h, v11.8h, v6.h[7]
        fmla v24.8h, v11.8h, v7.h[0]
        fmla v25.8h, v11.8h, v7.h[1]
        fmla v26.8h, v11.8h, v7.h[2]
        fmla v27.8h, v11.8h, v7.h[3]
        fmla v28.8h, v11.8h, v7.h[4]
        fmla v29.8h, v11.8h, v7.h[5]
        fmla v30.8h, v11.8h, v7.h[6]
        fmla v31.8h, v11.8h, v7.h[7]

        subs x19, x19, #4
        beq Activation16
        cmp x19, #4
        bge LoopDepth16

    LoopDepth16One:
        ld1 {v0.8h, v1.8h}, [x10], #32
        ld1 {v2.8h}, [x14], #16
        fmla v16.8h, v2.8h, v0.h[0]
        fmla v17.8h, v2.8h, v0.h[1]
        fmla v18.8h, v2.8h, v0.h[2]
        fmla v19.8h, v2.8h, v0.h[3]
        fmla v20.8h, v2.8h, v0.h[4]
        fmla v21.8h, v2.8h, v0.h[5]
        fmla v22.8h, v2.8h, v0.h[6]
        fmla v23.8h, v2.8h, v0.h[7]
        fmla v24.8h, v2.8h, v1.h[0]
        fmla v25.8h, v2.8h, v1.h[1]
        fmla v26.8h, v2.8h, v1.h[2]
        fmla v27.8h, v2.8h, v1.h[3]
        fmla v28.8h, v2.8h, v1.h[4]
        fmla v29.8h, v2.8h, v1.h[5]
        fmla v30.8h, v2.8h, v1.h[6]
        fmla v31.8h, v2.8h, v1.h[7]
        subs x19, x19, #1
        bgt LoopDepth16One

    Activation16:
        cmp x4, #3
        beq Relu616
        cmp x4, #1
        beq Relu16
        b Write16
        Relu616:
            fmin v16.8h, v16.8h, v13.8h
            fmin v17.8h, v17.8h, v13.8h
            fmin v18.8h, v18.8h, v13.8h
            fmin v19.8h, v19.8h, v13.8h
            fmin v20.8h, v20.8h, v13.8h
            fmin v21.8h, v21.8h, v13.8h
            fmin v22.8h, v22.8h, v13.8h
            fmin v23.8h, v23.8h, v13.8h
            fmin v24.8h, v24.8h, v13.8h
            fmin v25.8h, v25.8h, v13.8h
            fmin v26.8h, v26.8h, v13.8h
            fmin v27.8h, v27.8h, v13.8h
            fmin v28.8h, v28.8h, v13.8h
            fmin v29.8h, v29.8h, v13.8h
            fmin v30.8h, v30.8h, v13.8h
            fmin v31.8h, v31.8h, v13.8h
        Relu16:
            fmax v16.8h, v16.8h, v12.8h
            fmax v17.8h, v17.8h, v12.8h
            fmax v18.8h, v18.8h, v12.8h
            fmax v19.8h, v19.8h, v12.8h
            fmax v20.8h, v20.8h, v12.8h
            fmax v21.8h, v21.8h, v12.8h
            fmax v22.8h, v22.8h, v12.8h
            fmax v23.8h, v23.8h, v12.8h
            fmax v24.8h, v24.8h, v12.8h
            fmax v25.8h, v25.8h, v12.8h
            fmax v26.8h, v26.8h, v12.8h
            fmax v27.8h, v27.8h, v12.8h
            fmax v28.8h, v28.8h, v12.8h
            fmax v29.8h, v29.8h, v12.8h
            fmax v30.8h, v30.8h, v12.8h
            fmax v31.8h, v31.8h, v12.8h
    Write16:
        cmp x13, #8
        bge Write16x8
        b Write
        Write16x8:
            add x2, x2, #16
            st1 {v16.8h}, [x11], x8
            st1 {v17.8h}, [x11], x8
            st1 {v18.8h}, [x11], x8
            st1 {v19.8h}, [x11], x8
            st1 {v20.8h}, [x11], x8
            st1 {v21.8h}, [x11], x8
            st1 {v22.8h}, [x11], x8
            st1 {v23.8h}, [x11], x8
            st1 {v24.8h}, [x11], x8
            st1 {v25.8h}, [x11], x8
            st1 {v26.8h}, [x11], x8
            st1 {v27.8h}, [x11], x8
            st1 {v28.8h}, [x11], x8
            st1 {v29.8h}, [x11], x8
            st1 {v30.8h}, [x11], x8
            st1 {v31.8h}, [x11], x8
            b WriteEnd

LoopRow8:
    mov x15, #8
    mov x14, x1 // reload rhs ptr
    mov x13, x7 // reload rhs col
    mov x12, x3 // reload bias

    LoopCol8:
        mov x11, x2
        mov x10, x0 // reload lhs ptr
        mov x19, x5 // reload depth

        ld1 {v16.8h}, [x12], #16
        mov v17.16b, v16.16b
        mov v18.16b, v16.16b
        mov v19.16b, v16.16b
        mov v20.16b, v16.16b
        mov v21.16b, v16.16b
        mov v22.16b, v16.16b
        mov v23.16b, v16.16b

        cmp x19, #4
        blt LoopDepth8One

    LoopDepth8:
        ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x10], #64
        ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x14], #64
        fmla v16.8h, v8.8h, v0.h[0]
        fmla v17.8h, v8.8h, v0.h[1]
        fmla v18.8h, v8.8h, v0.h[2]
        fmla v19.8h, v8.8h, v0.h[3]
        fmla v20.8h, v8.8h, v0.h[4]
        fmla v21.8h, v8.8h, v0.h[5]
        fmla v22.8h, v8.8h, v0.h[6]
        fmla v23.8h, v8.8h, v0.h[7]
        fmla v16.8h, v9.8h, v1.h[0]
        fmla v17.8h, v9.8h, v1.h[1]
        fmla v18.8h, v9.8h, v1.h[2]
        fmla v19.8h, v9.8h, v1.h[3]
        fmla v20.8h, v9.8h, v1.h[4]
        fmla v21.8h, v9.8h, v1.h[5]
        fmla v22.8h, v9.8h, v1.h[6]
        fmla v23.8h, v9.8h, v1.h[7]
        fmla v16.8h, v10.8h, v2.h[0]
        fmla v17.8h, v10.8h, v2.h[1]
        fmla v18.8h, v10.8h, v2.h[2]
        fmla v19.8h, v10.8h, v2.h[3]
        fmla v20.8h, v10.8h, v2.h[4]
        fmla v21.8h, v10.8h, v2.h[5]
        fmla v22.8h, v10.8h, v2.h[6]
        fmla v23.8h, v10.8h, v2.h[7]
        fmla v16.8h, v11.8h, v3.h[0]
        fmla v17.8h, v11.8h, v3.h[1]
        fmla v18.8h, v11.8h, v3.h[2]
        fmla v19.8h, v11.8h, v3.h[3]
        fmla v20.8h, v11.8h, v3.h[4]
        fmla v21.8h, v11.8h, v3.h[5]
        fmla v22.8h, v11.8h, v3.h[6]
        fmla v23.8h, v11.8h, v3.h[7]
        subs x19, x19, #4
        beq Activation8
        cmp x19, #4
        bge LoopDepth8
    LoopDepth8One:
        ld1 {v0.8h}, [x10], #16
        ld1 {v2.8h}, [x14], #16
        fmla v16.8h, v2.8h, v0.h[0]
        fmla v17.8h, v2.8h, v0.h[1]
        fmla v18.8h, v2.8h, v0.h[2]
        fmla v19.8h, v2.8h, v0.h[3]
        fmla v20.8h, v2.8h, v0.h[4]
        fmla v21.8h, v2.8h, v0.h[5]
        fmla v22.8h, v2.8h, v0.h[6]
        fmla v23.8h, v2.8h, v0.h[7]
        subs x19, x19, #1
        bgt LoopDepth8One
    Activation8:
        cmp x4, #3
        beq Relu68
        cmp x4, #1
        beq Relu8
        b Write8_Row
        Relu68:
            fmin v16.8h, v16.8h, v13.8h
            fmin v17.8h, v17.8h, v13.8h
            fmin v18.8h, v18.8h, v13.8h
            fmin v19.8h, v19.8h, v13.8h
            fmin v20.8h, v20.8h, v13.8h
            fmin v21.8h, v21.8h, v13.8h
            fmin v22.8h, v22.8h, v13.8h
            fmin v23.8h, v23.8h, v13.8h
        Relu8:
            fmax v16.8h, v16.8h, v12.8h
            fmax v17.8h, v17.8h, v12.8h
            fmax v18.8h, v18.8h, v12.8h
            fmax v19.8h, v19.8h, v12.8h
            fmax v20.8h, v20.8h, v12.8h
            fmax v21.8h, v21.8h, v12.8h
            fmax v22.8h, v22.8h, v12.8h
            fmax v23.8h, v23.8h, v12.8h
    Write8_Row:
        cmp x13, #8  // row
        bge Write8x8
        b Write
        Write8x8:
            add x2, x2, #16
            st1 {v16.8h}, [x11], x8
            st1 {v17.8h}, [x11], x8
            st1 {v18.8h}, [x11], x8
            st1 {v19.8h}, [x11], x8
            st1 {v20.8h}, [x11], x8
            st1 {v21.8h}, [x11], x8
            st1 {v22.8h}, [x11], x8
            st1 {v23.8h}, [x11], x8
            b WriteEnd

LoopRow4:
    mov x15, #4
    mov x14, x1 // reload rhs ptr
    mov x13, x7 // reload rhs col
    mov x12, x3 // reload bias

    LoopCol4:
        mov x11, x2
        mov x10, x0 // reload lhs ptr
        mov x19, x5 // reload depth
        ld1 {v16.8h}, [x12], #16
        mov v17.16b, v16.16b
        mov v18.16b, v16.16b
        mov v19.16b, v16.16b
        cmp x19, #4
        blt LoopDepth4One
    LoopDepth4:
        ld1 {v0.8h, v1.8h}, [x10], #32
        ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x14], #64
        fmla v16.8h, v8.8h, v0.h[0]
        fmla v17.8h, v8.8h, v0.h[1]
        fmla v18.8h, v8.8h, v0.h[2]
        fmla v19.8h, v8.8h, v0.h[3]
        fmla v16.8h, v9.8h, v0.h[4]
        fmla v17.8h, v9.8h, v0.h[5]
        fmla v18.8h, v9.8h, v0.h[6]
        fmla v19.8h, v9.8h, v0.h[7]
        fmla v16.8h, v10.8h, v1.h[0]
        fmla v17.8h, v10.8h, v1.h[1]
        fmla v18.8h, v10.8h, v1.h[2]
        fmla v19.8h, v10.8h, v1.h[3]
        fmla v16.8h, v11.8h, v1.h[4]
        fmla v17.8h, v11.8h, v1.h[5]
        fmla v18.8h, v11.8h, v1.h[6]
        fmla v19.8h, v11.8h, v1.h[7]
        subs x19, x19, #4
        beq Activation4
        cmp x19, #4
        bge LoopDepth4
    LoopDepth4One:
        ld1 {v0.4h}, [x10], #8
        ld1 {v2.8h}, [x14], #16
        fmla v16.8h, v2.8h, v0.h[0]
        fmla v17.8h, v2.8h, v0.h[1]
        fmla v18.8h, v2.8h, v0.h[2]
        fmla v19.8h, v2.8h, v0.h[3]
        subs x19, x19, #1
        bgt LoopDepth4One
    Activation4:
        cmp x4, #3
        beq Relu64
        cmp x4, #1
        beq Relu4
        b Write4_Row
        Relu64:
            fmin v16.8h, v16.8h, v13.8h
            fmin v17.8h, v17.8h, v13.8h
            fmin v18.8h, v18.8h, v13.8h
            fmin v19.8h, v19.8h, v13.8h
        Relu4:
            fmax v16.8h, v16.8h, v12.8h
            fmax v17.8h, v17.8h, v12.8h
            fmax v18.8h, v18.8h, v12.8h
            fmax v19.8h, v19.8h, v12.8h
    Write4_Row:
        cmp x6, #4
        bge Write4x8
        b Write
        Write4x8:
            cmp x13, #8
            blt Write
            add x2, x2, #16
            st1 {v16.8h}, [x11], x8
            st1 {v17.8h}, [x11], x8
            st1 {v18.8h}, [x11], x8
            st1 {v19.8h}, [x11], x8
            b WriteEnd

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
        b Write8

    Write1:
        add x2, x2, #2
        st1 {v16.h}[0], [x11], x8
        cmp x6, #1
        beq WriteEnd
        st1 {v17.h}[0], [x11], x8
        cmp x6, #2
        beq WriteEnd
        st1 {v18.h}[0], [x11], x8
        cmp x6, #3
        beq WriteEnd
        st1 {v19.h}[0], [x11], x8
        cmp x6, #4
        beq WriteEnd
        st1 {v20.h}[0], [x11], x8
        cmp x6, #5
        beq WriteEnd
        st1 {v21.h}[0], [x11], x8
        cmp x6, #6
        beq WriteEnd
        st1 {v22.h}[0], [x11], x8
        cmp x6, #7
        beq WriteEnd
        st1 {v23.h}[0], [x11], x8
        cmp x6, #8
        beq WriteEnd
        st1 {v24.h}[0], [x11], x8
        cmp x6, #9
        beq WriteEnd
        st1 {v25.h}[0], [x11], x8
        cmp x6, #10
        beq WriteEnd
        st1 {v26.h}[0], [x11], x8
        cmp x6, #11
        beq WriteEnd
        st1 {v27.h}[0], [x11], x8
        cmp x6, #12
        beq WriteEnd
        st1 {v28.h}[0], [x11], x8
        cmp x6, #13
        beq WriteEnd
        st1 {v29.h}[0], [x11], x8
        cmp x6, #14
        beq WriteEnd
        st1 {v30.h}[0], [x11], x8
        cmp x6, #15
        beq WriteEnd
        st1 {v31.h}[0], [x11], x8
        b WriteEnd
    Write2:
        add x2, x2, #4
        st1 {v16.s}[0], [x11], x8
        cmp x6, #1
        beq WriteEnd
        st1 {v17.s}[0], [x11], x8
        cmp x6, #2
        beq WriteEnd
        st1 {v18.s}[0], [x11], x8
        cmp x6, #3
        beq WriteEnd
        st1 {v19.s}[0], [x11], x8
        cmp x6, #4
        beq WriteEnd
        st1 {v20.s}[0], [x11], x8
        cmp x6, #5
        beq WriteEnd
        st1 {v21.s}[0], [x11], x8
        cmp x6, #6
        beq WriteEnd
        st1 {v22.s}[0], [x11], x8
        cmp x6, #7
        beq WriteEnd
        st1 {v23.s}[0], [x11], x8
        cmp x6, #8
        beq WriteEnd
        st1 {v24.s}[0], [x11], x8
        cmp x6, #9
        beq WriteEnd
        st1 {v25.s}[0], [x11], x8
        cmp x6, #10
        beq WriteEnd
        st1 {v26.s}[0], [x11], x8
        cmp x6, #11
        beq WriteEnd
        st1 {v27.s}[0], [x11], x8
        cmp x6, #12
        beq WriteEnd
        st1 {v28.s}[0], [x11], x8
        cmp x6, #13
        beq WriteEnd
        st1 {v29.s}[0], [x11], x8
        cmp x6, #14
        beq WriteEnd
        st1 {v30.s}[0], [x11], x8
        cmp x6, #15
        beq WriteEnd
        st1 {v31.s}[0], [x11], x8
        b WriteEnd
    Write3:
        add x2, x2, #6
        add x19, x11, #4
        st1 {v16.s}[0], [x11], x8
        st1 {v16.h}[2], [x19], x8
        cmp x6, #1
        beq WriteEnd
        st1 {v17.s}[0], [x11], x8
        st1 {v17.h}[2], [x19], x8
        cmp x6, #2
        beq WriteEnd
        st1 {v18.s}[0], [x11], x8
        st1 {v18.h}[2], [x19], x8
        cmp x6, #3
        beq WriteEnd
        st1 {v19.s}[0], [x11], x8
        st1 {v19.h}[2], [x19], x8
        cmp x6, #4
        beq WriteEnd
        st1 {v20.s}[0], [x11], x8
        st1 {v20.h}[2], [x19], x8
        cmp x6, #5
        beq WriteEnd
        st1 {v21.s}[0], [x11], x8
        st1 {v21.h}[2], [x19], x8
        cmp x6, #6
        beq WriteEnd
        st1 {v22.s}[0], [x11], x8
        st1 {v22.h}[2], [x19], x8
        cmp x6, #7
        beq WriteEnd
        st1 {v23.s}[0], [x11], x8
        st1 {v23.h}[2], [x19], x8
        cmp x6, #8
        beq WriteEnd
        st1 {v24.s}[0], [x11], x8
        st1 {v24.h}[2], [x19], x8
        cmp x6, #9
        beq WriteEnd
        st1 {v25.s}[0], [x11], x8
        st1 {v25.h}[2], [x19], x8
        cmp x6, #10
        beq WriteEnd
        st1 {v26.s}[0], [x11], x8
        st1 {v26.h}[2], [x19], x8
        cmp x6, #11
        beq WriteEnd
        st1 {v27.s}[0], [x11], x8
        st1 {v27.h}[2], [x19], x8
        cmp x6, #12
        beq WriteEnd
        st1 {v28.s}[0], [x11], x8
        st1 {v28.h}[2], [x19], x8
        cmp x6, #13
        beq WriteEnd
        st1 {v29.s}[0], [x11], x8
        st1 {v29.h}[2], [x19], x8
        cmp x6, #14
        beq WriteEnd
        st1 {v30.s}[0], [x11], x8
        st1 {v30.h}[2], [x19], x8
        cmp x6, #15
        beq WriteEnd
        st1 {v31.s}[0], [x11], x8
        st1 {v31.h}[2], [x19]
        b WriteEnd
    Write4:
        add x2, x2, #8
        st1 {v16.4h}, [x11], x8
        cmp x6, #1
        beq WriteEnd
        st1 {v17.4h}, [x11], x8
        cmp x6, #2
        beq WriteEnd
        st1 {v18.4h}, [x11], x8
        cmp x6, #3
        beq WriteEnd
        st1 {v19.4h}, [x11], x8
        cmp x6, #4
        beq WriteEnd
        st1 {v20.4h}, [x11], x8
        cmp x6, #5
        beq WriteEnd
        st1 {v21.4h}, [x11], x8
        cmp x6, #6
        beq WriteEnd
        st1 {v22.4h}, [x11], x8
        cmp x6, #7
        beq WriteEnd
        st1 {v23.4h}, [x11], x8
        cmp x6, #8
        beq WriteEnd
        st1 {v24.4h}, [x11], x8
        cmp x6, #9
        beq WriteEnd
        st1 {v25.4h}, [x11], x8
        cmp x6, #10
        beq WriteEnd
        st1 {v26.4h}, [x11], x8
        cmp x6, #11
        beq WriteEnd
        st1 {v27.4h}, [x11], x8
        cmp x6, #12
        beq WriteEnd
        st1 {v28.4h}, [x11], x8
        cmp x6, #13
        beq WriteEnd
        st1 {v29.4h}, [x11], x8
        cmp x6, #14
        beq WriteEnd
        st1 {v30.4h}, [x11], x8
        cmp x6, #15
        beq WriteEnd
        st1 {v31.4h}, [x11], x8
        b WriteEnd
    Write5:
        add x2, x2, #10
        add x19, x11, #8
        st1 {v16.4h}, [x11], x8
        st1 {v16.h}[4], [x19], x8
        cmp x6, #1
        beq WriteEnd
        st1 {v17.4h}, [x11], x8
        st1 {v17.h}[4], [x19], x8
        cmp x6, #2
        beq WriteEnd
        st1 {v18.4h}, [x11], x8
        st1 {v18.h}[4], [x19], x8
        cmp x6, #3
        beq WriteEnd
        st1 {v19.4h}, [x11], x8
        st1 {v19.h}[4], [x19], x8
        cmp x6, #4
        beq WriteEnd
        st1 {v20.4h}, [x11], x8
        st1 {v20.h}[4], [x19], x8
        cmp x6, #5
        beq WriteEnd
        st1 {v21.4h}, [x11], x8
        st1 {v21.h}[4], [x19], x8
        cmp x6, #6
        beq WriteEnd
        st1 {v22.4h}, [x11], x8
        st1 {v22.h}[4], [x19], x8
        cmp x6, #7
        beq WriteEnd
        st1 {v23.4h}, [x11], x8
        st1 {v23.h}[4], [x19], x8
        cmp x6, #8
        beq WriteEnd
        st1 {v24.4h}, [x11], x8
        st1 {v24.h}[4], [x19], x8
        cmp x6, #9
        beq WriteEnd
        st1 {v25.4h}, [x11], x8
        st1 {v25.h}[4], [x19], x8
        cmp x6, #10
        beq WriteEnd
        st1 {v26.4h}, [x11], x8
        st1 {v26.h}[4], [x19], x8
        cmp x6, #11
        beq WriteEnd
        st1 {v27.4h}, [x11], x8
        st1 {v27.h}[4], [x19], x8
        cmp x6, #12
        beq WriteEnd
        st1 {v28.4h}, [x11], x8
        st1 {v28.h}[4], [x19], x8
        cmp x6, #13
        beq WriteEnd
        st1 {v29.4h}, [x11], x8
        st1 {v29.h}[4], [x19], x8
        cmp x6, #14
        beq WriteEnd
        st1 {v30.4h}, [x11], x8
        st1 {v30.h}[4], [x19], x8
        cmp x6, #15
        beq WriteEnd
        st1 {v31.4h}, [x11], x8
        st1 {v31.h}[4], [x19]
        b WriteEnd
    Write6:
        add x2, x2, #12
        add x19, x11, #8
        st1 {v16.4h}, [x11], x8
        st1 {v16.s}[2], [x19], x8
        cmp x6, #1
        beq WriteEnd
        st1 {v17.4h}, [x11], x8
        st1 {v17.s}[2], [x19], x8
        cmp x6, #2
        beq WriteEnd
        st1 {v18.4h}, [x11], x8
        st1 {v18.s}[2], [x19], x8
        cmp x6, #3
        beq WriteEnd
        st1 {v19.4h}, [x11], x8
        st1 {v19.s}[2], [x19], x8
        cmp x6, #4
        beq WriteEnd
        st1 {v20.4h}, [x11], x8
        st1 {v20.s}[2], [x19], x8
        cmp x6, #5
        beq WriteEnd
        st1 {v21.4h}, [x11], x8
        st1 {v21.s}[2], [x19], x8
        cmp x6, #6
        beq WriteEnd
        st1 {v22.4h}, [x11], x8
        st1 {v22.s}[2], [x19], x8
        cmp x6, #7
        beq WriteEnd
        st1 {v23.4h}, [x11], x8
        st1 {v23.s}[2], [x19], x8
        cmp x6, #8
        beq WriteEnd
        st1 {v24.4h}, [x11], x8
        st1 {v24.s}[2], [x19], x8
        cmp x6, #9
        beq WriteEnd
        st1 {v25.4h}, [x11], x8
        st1 {v25.s}[2], [x19], x8
        cmp x6, #10
        beq WriteEnd
        st1 {v26.4h}, [x11], x8
        st1 {v26.s}[2], [x19], x8
        cmp x6, #11
        beq WriteEnd
        st1 {v27.4h}, [x11], x8
        st1 {v27.s}[2], [x19], x8
        cmp x6, #12
        beq WriteEnd
        st1 {v28.4h}, [x11], x8
        st1 {v28.s}[2], [x19], x8
        cmp x6, #13
        beq WriteEnd
        st1 {v29.4h}, [x11], x8
        st1 {v29.s}[2], [x19], x8
        cmp x6, #14
        beq WriteEnd
        st1 {v30.4h}, [x11], x8
        st1 {v30.s}[2], [x19], x8
        cmp x6, #15
        beq WriteEnd
        st1 {v31.4h}, [x11], x8
        st1 {v31.s}[2], [x19]
        b WriteEnd
    Write7:
        add x2, x2, #14
        add x19, x11, #8
        add x10, x11, #12
        st1 {v16.4h}, [x11], x8
        st1 {v16.s}[2], [x19], x8
        st1 {v16.h}[6], [x10], x8
        cmp x6, #1
        beq WriteEnd
        st1 {v17.4h}, [x11], x8
        st1 {v17.s}[2], [x19], x8
        st1 {v17.h}[6], [x10], x8
        cmp x6, #2
        beq WriteEnd
        st1 {v18.4h}, [x11], x8
        st1 {v18.s}[2], [x19], x8
        st1 {v18.h}[6], [x10], x8
        cmp x6, #3
        beq WriteEnd
        st1 {v19.4h}, [x11], x8
        st1 {v19.s}[2], [x19], x8
        st1 {v19.h}[6], [x10], x8
        cmp x6, #4
        beq WriteEnd
        st1 {v20.4h}, [x11], x8
        st1 {v20.s}[2], [x19], x8
        st1 {v20.h}[6], [x10], x8
        cmp x6, #5
        beq WriteEnd
        st1 {v21.4h}, [x11], x8
        st1 {v21.s}[2], [x19], x8
        st1 {v21.h}[6], [x10], x8
        cmp x6, #6
        beq WriteEnd
        st1 {v22.4h}, [x11], x8
        st1 {v22.s}[2], [x19], x8
        st1 {v22.h}[6], [x10], x8
        cmp x6, #7
        beq WriteEnd
        st1 {v23.4h}, [x11], x8
        st1 {v23.s}[2], [x19], x8
        st1 {v23.h}[6], [x10], x8
        cmp x6, #8
        beq WriteEnd
        st1 {v24.4h}, [x11], x8
        st1 {v24.s}[2], [x19], x8
        st1 {v24.h}[6], [x10], x8
        cmp x6, #9
        beq WriteEnd
        st1 {v25.4h}, [x11], x8
        st1 {v25.s}[2], [x19], x8
        st1 {v25.h}[6], [x10], x8
        cmp x6, #10
        beq WriteEnd
        st1 {v26.4h}, [x11], x8
        st1 {v26.s}[2], [x19], x8
        st1 {v26.h}[6], [x10], x8
        cmp x6, #11
        beq WriteEnd
        st1 {v27.4h}, [x11], x8
        st1 {v27.s}[2], [x19], x8
        st1 {v27.h}[6], [x10], x8
        cmp x6, #12
        beq WriteEnd
        st1 {v28.4h}, [x11], x8
        st1 {v28.s}[2], [x19], x8
        st1 {v28.h}[6], [x10], x8
        cmp x6, #13
        beq WriteEnd
        st1 {v29.4h}, [x11], x8
        st1 {v29.s}[2], [x19], x8
        st1 {v29.h}[6], [x10], x8
        cmp x6, #14
        beq WriteEnd
        st1 {v30.4h}, [x11], x8
        st1 {v30.s}[2], [x19], x8
        st1 {v30.h}[6], [x10], x8
        cmp x6, #15
        beq WriteEnd
        st1 {v31.4h}, [x11], x8
        st1 {v31.s}[2], [x19]
        st1 {v31.h}[6], [x10]
        b WriteEnd
    Write8:
        add x2, x2, #16
        st1 {v16.8h}, [x11], x8
        cmp x6, #1
        beq WriteEnd
        st1 {v17.8h}, [x11], x8
        cmp x6, #2
        beq WriteEnd
        st1 {v18.8h}, [x11], x8
        cmp x6, #3
        beq WriteEnd
        st1 {v19.8h}, [x11], x8
        cmp x6, #4
        beq WriteEnd
        st1 {v20.8h}, [x11], x8
        cmp x6, #5
        beq WriteEnd
        st1 {v21.8h}, [x11], x8
        cmp x6, #6
        beq WriteEnd
        st1 {v22.8h}, [x11], x8
        cmp x6, #7
        beq WriteEnd
        st1 {v23.8h}, [x11], x8
        cmp x6, #8
        beq WriteEnd
        st1 {v24.8h}, [x11], x8
        cmp x6, #9
        beq WriteEnd
        st1 {v25.8h}, [x11], x8
        cmp x6, #10
        beq WriteEnd
        st1 {v26.8h}, [x11], x8
        cmp x6, #11
        beq WriteEnd
        st1 {v27.8h}, [x11], x8
        cmp x6, #12
        beq WriteEnd
        st1 {v28.8h}, [x11], x8
        cmp x6, #13
        beq WriteEnd
        st1 {v29.8h}, [x11], x8
        cmp x6, #14
        beq WriteEnd
        st1 {v30.8h}, [x11], x8
        cmp x6, #15
        beq WriteEnd
        st1 {v31.8h}, [x11], x8

    WriteEnd:
        subs x13, x13, #8 // rhs col - 8
        ble LoopColEnd
        cmp x6, #16
        bge LoopCol16
        cmp x6, #8
        bge LoopCol8
        b LoopCol4

LoopColEnd:
    sub x2, x2, x16  // dst - col * 2
    mul x21, x8, x15  // row_block * col * 2
    add x2, x2, x21
    subs x6, x6, x15
    mul x15, x15, x17
    add x0, x0, x15
    bgt LoopRowStart

    sub sp, sp, #96
    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [sp], #64
    ldp x19, x20, [sp], #16
    ldp x21, x22, [sp], #16
    ret
#endif
