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
#ifdef ENABLE_ARM64
#include "nnacl/assembly_global.h"

.text
.align 5

// void BigMatmulFloatNeon64(const float *a, const float *b, float *c, const float *bias, int act_type, int depth
//                           int row, int col, size_t stride)
// x0: a
// x1: b
// x2: c
// x3: bias
// x4: act_type
// x5: depth
// x6: row
// x7: col
// x8: stride

asm_function BigMatmulFloatNeon64Opt
    sub sp, sp, #224
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
    stp x19, x20, [sp], #16
    stp x21, x22, [sp], #16
    stp x23, x24, [sp], #16
    stp x25, x26, [sp], #16
    stp x27, x28, [sp], #16
    stp x29, x30, [sp], #16

    ldr x8, [sp]
    mov x20, #1
    mov x22, #32
    mov x23, #48
    mul x26, x5, x23  // stride for lhs
    mul x24, x8, x23  // stride for out
    lsl x27, x23, #9  // stride by depth for lhs
    lsl x28, x22, #9  // stride by depth for rhs
    lsl x22, x5, #5  // stride for rhs
    lsl x8, x8, #2
    subs x5, x5, #512
    ble DepthTail
Depth512:
    mov x25, x0  // restore lhs
    mov x13, x2  // out
    mov x10, x6  // restore row
    RowStart:
        mov x12, x1  // rhs
        mov x14, x13  // out
        mov x15, x3  // restore bias
        mov x9, x7  // restore col
        cmp x10, #4
        ble LoopRow4
        cmp x10, #8
        ble LoopRow8

        LoopRow12:
            mov x11, x25  // lhs
            mov x23, x12 // rhs
            mov x21, #512  // depth unit
            prfm pldl1strm, [x23, #632]
            ld1 {v3.4s}, [x23], #16
            ld1 {v0.4s}, [x11], #16
            cmp x9, #4
            ble LoopCol12x4

            LoopCol12x8:
                cbz x20, Reload12x8
                cbnz x15, InitFromBias12x8
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
                b Compute12x8Enter
            InitFromBias12x8:
                ld1 {v8.4s, v9.4s}, [x15]
                ld1 {v10.4s, v11.4s}, [x15]
                ld1 {v12.4s, v13.4s}, [x15]
                ld1 {v14.4s, v15.4s}, [x15]
                ld1 {v16.4s, v17.4s}, [x15]
                ld1 {v18.4s, v19.4s}, [x15]
                ld1 {v20.4s, v21.4s}, [x15]
                ld1 {v22.4s, v23.4s}, [x15]
                ld1 {v24.4s, v25.4s}, [x15]
                ld1 {v26.4s, v27.4s}, [x15]
                ld1 {v28.4s, v29.4s}, [x15]
                ld1 {v30.4s, v31.4s}, [x15]
                add x15, x15, #32
                b Compute12x8Enter
            Reload12x8:
                bl Reload
            Compute12x8Enter:
                cbz x21, Write
                bl Compute12x8Unit
                b Write

            LoopCol12x4:
                cbz x20, Reload12x4
                cbnz x15, InitFromBias12x4
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
                b Compute12x4Enter
            InitFromBias12x4:
                ld1 {v8.4s}, [x15]
                ld1 {v10.4s}, [x15]
                ld1 {v12.4s}, [x15]
                ld1 {v14.4s}, [x15]
                ld1 {v16.4s}, [x15]
                ld1 {v18.4s}, [x15]
                ld1 {v20.4s}, [x15]
                ld1 {v22.4s}, [x15]
                ld1 {v24.4s}, [x15]
                ld1 {v26.4s}, [x15]
                ld1 {v28.4s}, [x15]
                ld1 {v30.4s}, [x15]
                b Compute12x4Enter
            Reload12x4:
                bl Reload
            Compute12x4Enter:
                cbz x21, Write
                bl Compute12x4Unit
                b Write

        LoopRow8:
            mov x11, x25  // lhs
            mov x23, x12 // rhs
            mov x21, #512  // depth unit
            prfm pldl1strm, [x23, #632]
            ld1 {v3.4s}, [x23], #16
            ld1 {v0.4s}, [x11], #16
            cmp x9, #4
            ble LoopCol8x4

            LoopCol8x8:
                cbz x20, Reload8x8
                cbnz x15, InitFromBias8x8
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
                b Compute8x8Enter
            InitFromBias8x8:
                ld1 {v8.4s, v9.4s}, [x15]
                ld1 {v10.4s, v11.4s}, [x15]
                ld1 {v12.4s, v13.4s}, [x15]
                ld1 {v14.4s, v15.4s}, [x15]
                ld1 {v16.4s, v17.4s}, [x15]
                ld1 {v18.4s, v19.4s}, [x15]
                ld1 {v20.4s, v21.4s}, [x15]
                ld1 {v22.4s, v23.4s}, [x15]
                add x15, x15, #32
                b Compute8x8Enter
            Reload8x8:
                bl Reload
            Compute8x8Enter:
                cbz x21, Write
                bl Compute8x8Unit
                b Write

            LoopCol8x4:
                cbz x20, Reload8x4
                cbnz x15, InitFromBias8x4
                dup v8.2d, xzr
                dup v10.2d, xzr
                dup v12.2d, xzr
                dup v14.2d, xzr
                dup v16.2d, xzr
                dup v18.2d, xzr
                dup v20.2d, xzr
                dup v22.2d, xzr
                b Compute8x4Enter
            InitFromBias8x4:
                ld1 {v8.4s}, [x15]
                ld1 {v10.4s}, [x15]
                ld1 {v12.4s}, [x15]
                ld1 {v14.4s}, [x15]
                ld1 {v16.4s}, [x15]
                ld1 {v18.4s}, [x15]
                ld1 {v20.4s}, [x15]
                ld1 {v22.4s}, [x15]
                b Compute8x4Enter
            Reload8x4:
                bl Reload
            Compute8x4Enter:
                cbz x21, Write
                bl Compute8x4Unit
                b Write

        LoopRow4:
            mov x11, x25  // lhs
            mov x23, x12 // rhs
            mov x21, #512  // depth unit
            prfm pldl1strm, [x23, #632]
            ld1 {v3.4s}, [x23], #16
            ld1 {v0.4s}, [x11], #16
            cmp x9, #4
            ble LoopCol4x4

            LoopCol4x8:
                cbz x20, Reload4x8
                cbnz x15, InitFromBias4x8
                dup v8.2d, xzr
                dup v9.2d, xzr
                dup v10.2d, xzr
                dup v11.2d, xzr
                dup v12.2d, xzr
                dup v13.2d, xzr
                dup v14.2d, xzr
                dup v15.2d, xzr
                b Compute4x8Enter
            InitFromBias4x8:
                ld1 {v8.4s, v9.4s}, [x15]
                ld1 {v10.4s, v11.4s}, [x15]
                ld1 {v12.4s, v13.4s}, [x15]
                ld1 {v14.4s, v15.4s}, [x15]
                add x15, x15, #32
                b Compute4x8Enter
            Reload4x8:
                bl Reload
            Compute4x8Enter:
                cbz x21, Write
                bl Compute4x8Unit
                b Write

            LoopCol4x4:
                cbz x20, Reload4x4
                cbnz x15, InitFromBias4x4
                dup v8.2d, xzr
                dup v10.2d, xzr
                dup v12.2d, xzr
                dup v14.2d, xzr
                b Compute4x4Enter
            InitFromBias4x4:
                ld1 {v8.4s}, [x15]
                ld1 {v10.4s}, [x15]
                ld1 {v12.4s}, [x15]
                ld1 {v14.4s}, [x15]
                b Compute4x4Enter
            Reload4x4:
                bl Reload
            Compute4x4Enter:
                cbz x21, Write
                bl Compute4x4Unit

Write:
    mov x21, x14
    cmp x9, #1
    beq Write1
    cmp x9, #2
    beq Write2
    cmp x9, #3
    beq Write3
    cmp x9, #4
    beq Write4
    cmp x9, #5
    beq Write5
    cmp x9, #6
    beq Write6
    cmp x9, #7
    beq Write7
    b Write8

    Write1:
        str s8, [x21]
        cmp x10, #1
        beq LoopCol
        add x21, x21, x8
        str s10, [x21]
        cmp x10, #2
        beq LoopCol
        add x21, x21, x8
        str s12, [x21]
        cmp x10, #3
        beq LoopCol
        add x21, x21, x8
        str s14, [x21]
        cmp x10, #4
        beq LoopCol
        add x21, x21, x8
        str s16, [x21]
        cmp x10, #5
        beq LoopCol
        add x21, x21, x8
        str s18, [x21]
        cmp x10, #6
        beq LoopCol
        add x21, x21, x8
        str s20, [x21]
        cmp x10, #7
        beq LoopCol
        add x21, x21, x8
        str s22, [x21]
        cmp x10, #8
        beq LoopCol
        add x21, x21, x8
        str s24, [x21]
        cmp x10, #9
        beq LoopCol
        add x21, x21, x8
        str s26, [x21]
        cmp x10, #10
        beq LoopCol
        add x21, x21, x8
        str s28, [x21]
        cmp x10, #11
        beq LoopCol
        add x21, x21, x8
        str s30, [x21]
        b LoopCol
    Write2:
        st1 {v8.2s}, [x21], x8
        cmp x10, #1
        beq LoopCol
        st1 {v10.2s}, [x21], x8
        cmp x10, #2
        beq LoopCol
        st1 {v12.2s}, [x21], x8
        cmp x10, #3
        beq LoopCol
        st1 {v14.2s}, [x21], x8
        cmp x10, #4
        beq LoopCol
        st1 {v16.2s}, [x21], x8
        cmp x10, #5
        beq LoopCol
        st1 {v18.2s}, [x21], x8
        cmp x10, #6
        beq LoopCol
        st1 {v20.2s}, [x21], x8
        cmp x10, #7
        beq LoopCol
        st1 {v22.2s}, [x21], x8
        cmp x10, #8
        beq LoopCol
        st1 {v24.2s}, [x21], x8
        cmp x10, #9
        beq LoopCol
        st1 {v26.2s}, [x21], x8
        cmp x10, #10
        beq LoopCol
        st1 {v28.2s}, [x21], x8
        cmp x10, #11
        beq LoopCol
        st1 {v30.2s}, [x21], x8
        add x21, x21, #8
        b LoopCol
    Write3:
        add x11, x21, #8
        st1 {v8.2s}, [x21], x8
        st1 {v8.s}[2], [x11], x8
        cmp x10, #1
        beq LoopCol
        st1 {v10.2s}, [x21], x8
        st1 {v10.s}[2], [x11], x8
        cmp x10, #2
        beq LoopCol
        st1 {v12.2s}, [x21], x8
        st1 {v12.s}[2], [x11], x8
        cmp x10, #3
        beq LoopCol
        st1 {v14.2s}, [x21], x8
        st1 {v14.s}[2], [x11], x8
        cmp x10, #4
        beq LoopCol
        st1 {v16.2s}, [x21], x8
        st1 {v16.s}[2], [x11], x8
        cmp x10, #5
        beq LoopCol
        st1 {v18.2s}, [x21], x8
        st1 {v18.s}[2], [x11], x8
        cmp x10, #6
        beq LoopCol
        st1 {v20.2s}, [x21], x8
        st1 {v20.s}[2], [x11], x8
        cmp x10, #7
        beq LoopCol
        st1 {v22.2s}, [x21], x8
        st1 {v22.s}[2], [x11], x8
        cmp x10, #8
        beq LoopCol
        st1 {v24.2s}, [x21], x8
        st1 {v24.s}[2], [x11], x8
        cmp x10, #9
        beq LoopCol
        st1 {v26.2s}, [x21], x8
        st1 {v26.s}[2], [x11], x8
        cmp x10, #10
        beq LoopCol
        st1 {v28.2s}, [x21], x8
        st1 {v28.s}[2], [x11], x8
        cmp x10, #11
        beq LoopCol
        st1 {v30.2s}, [x21], x8
        st1 {v30.s}[2], [x11]
        add x21, x21, #12
        b LoopCol
    Write4:
        st1 {v8.4s}, [x21], x8
        cmp x10, #1
        beq LoopCol
        st1 {v10.4s}, [x21], x8
        cmp x10, #2
        beq LoopCol
        st1 {v12.4s}, [x21], x8
        cmp x10, #3
        beq LoopCol
        st1 {v14.4s}, [x21], x8
        cmp x10, #4
        beq LoopCol
        st1 {v16.4s}, [x21], x8
        cmp x10, #5
        beq LoopCol
        st1 {v18.4s}, [x21], x8
        cmp x10, #6
        beq LoopCol
        st1 {v20.4s}, [x21], x8
        cmp x10, #7
        beq LoopCol
        st1 {v22.4s}, [x21], x8
        cmp x10, #8
        beq LoopCol
        st1 {v24.4s}, [x21], x8
        cmp x10, #9
        beq LoopCol
        st1 {v26.4s}, [x21], x8
        cmp x10, #10
        beq LoopCol
        st1 {v28.4s}, [x21], x8
        cmp x10, #11
        beq LoopCol
        st1 {v30.4s}, [x21], x8
        add x21, x21, #16
        b LoopCol
    Write5:
        add x11, x21, #16
        st1 {v8.4s}, [x21], x8
        str s9, [x11]
        cmp x10, #1
        beq LoopCol
        add x11, x11, x8
        st1 {v10.4s}, [x21], x8
        str s11, [x11]
        cmp x10, #2
        beq LoopCol
        add x11, x11, x8
        st1 {v12.4s}, [x21], x8
        str s13, [x11]
        cmp x10, #3
        beq LoopCol
        add x11, x11, x8
        st1 {v14.4s}, [x21], x8
        str s15, [x11]
        cmp x10, #4
        beq LoopCol
        add x11, x11, x8
        st1 {v16.4s}, [x21], x8
        str s17, [x11]
        cmp x10, #5
        beq LoopCol
        add x11, x11, x8
        st1 {v18.4s}, [x21], x8
        str s19, [x11]
        cmp x10, #6
        beq LoopCol
        add x11, x11, x8
        st1 {v20.4s}, [x21], x8
        str s21, [x11]
        cmp x10, #7
        beq LoopCol
        add x11, x11, x8
        st1 {v22.4s}, [x21], x8
        str s23, [x11]
        cmp x10, #8
        beq LoopCol
        add x11, x11, x8
        st1 {v24.4s}, [x21], x8
        str s25, [x11]
        cmp x10, #9
        beq LoopCol
        add x11, x11, x8
        st1 {v26.4s}, [x21], x8
        str s27, [x11]
        cmp x10, #10
        beq LoopCol
        add x11, x11, x8
        st1 {v28.4s}, [x21], x8
        str s29, [x11]
        cmp x10, #11
        beq LoopCol
        add x11, x11, x8
        st1 {v30.4s}, [x21], x8
        str s31, [x11]
        add x21, x21, #20
        b LoopCol
    Write6:
        add x11, x21, #16
        st1 {v8.4s}, [x21], x8
        st1 {v9.2s}, [x11], x8
        cmp x10, #1
        beq LoopCol
        st1 {v10.4s}, [x21], x8
        st1 {v11.2s}, [x11], x8
        cmp x10, #2
        beq LoopCol
        st1 {v12.4s}, [x21], x8
        st1 {v13.2s}, [x11], x8
        cmp x10, #3
        beq LoopCol
        st1 {v14.4s}, [x21], x8
        st1 {v15.2s}, [x11], x8
        cmp x10, #4
        beq LoopCol
        st1 {v16.4s}, [x21], x8
        st1 {v17.2s}, [x11], x8
        cmp x10, #5
        beq LoopCol
        st1 {v18.4s}, [x21], x8
        st1 {v19.2s}, [x11], x8
        cmp x10, #6
        beq LoopCol
        st1 {v20.4s}, [x21], x8
        st1 {v21.2s}, [x11], x8
        cmp x10, #7
        beq LoopCol
        st1 {v22.4s}, [x21], x8
        st1 {v23.2s}, [x11], x8
        cmp x10, #8
        beq LoopCol
        st1 {v24.4s}, [x21], x8
        st1 {v25.2s}, [x11], x8
        cmp x10, #9
        beq LoopCol
        st1 {v26.4s}, [x21], x8
        st1 {v27.2s}, [x11], x8
        cmp x10, #10
        beq LoopCol
        st1 {v28.4s}, [x21], x8
        st1 {v29.2s}, [x11], x8
        cmp x10, #11
        beq LoopCol
        st1 {v30.4s}, [x21], x8
        st1 {v31.2s}, [x11]
        add x21, x21, #24
        b LoopCol
    Write7:
        add x11, x21, #16
        add x23, x21, #24
        st1 {v8.4s}, [x21], x8
        st1 {v9.2s}, [x11], x8
        st1 {v9.s}[2], [x23], x8
        cmp x10, #1
        beq LoopCol
        st1 {v10.4s}, [x21], x8
        st1 {v11.2s}, [x11], x8
        st1 {v11.s}[2], [x23], x8
        cmp x10, #2
        beq LoopCol
        st1 {v12.4s}, [x21], x8
        st1 {v13.2s}, [x11], x8
        st1 {v13.s}[2], [x23], x8
        cmp x10, #3
        beq LoopCol
        st1 {v14.4s}, [x21], x8
        st1 {v15.2s}, [x11], x8
        st1 {v15.s}[2], [x23], x8
        cmp x10, #4
        beq LoopCol
        st1 {v16.4s}, [x21], x8
        st1 {v17.2s}, [x11], x8
        st1 {v17.s}[2], [x23], x8
        cmp x10, #5
        beq LoopCol
        st1 {v18.4s}, [x21], x8
        st1 {v19.2s}, [x11], x8
        st1 {v19.s}[2], [x23], x8
        cmp x10, #6
        beq LoopCol
        st1 {v20.4s}, [x21], x8
        st1 {v21.2s}, [x11], x8
        st1 {v21.s}[2], [x23], x8
        cmp x10, #7
        beq LoopCol
        st1 {v22.4s}, [x21], x8
        st1 {v23.2s}, [x11], x8
        st1 {v23.s}[2], [x23], x8
        cmp x10, #8
        beq LoopCol
        st1 {v24.4s}, [x21], x8
        st1 {v25.2s}, [x11], x8
        st1 {v25.s}[2], [x23], x8
        cmp x10, #9
        beq LoopCol
        st1 {v26.4s}, [x21], x8
        st1 {v27.2s}, [x11], x8
        st1 {v27.s}[2], [x23], x8
        cmp x10, #10
        beq LoopCol
        st1 {v28.4s}, [x21], x8
        st1 {v29.2s}, [x11], x8
        st1 {v29.s}[2], [x23], x8
        cmp x10, #11
        beq LoopCol
        st1 {v30.4s}, [x21], x8
        st1 {v31.2s}, [x11]
        st1 {v31.s}[2], [x23]
        add x21, x21, #28
        b LoopCol

    Write8:
        st1 {v8.4s, v9.4s}, [x21], x8
        cmp x10, #1
        beq LoopCol
        st1 {v10.4s, v11.4s}, [x21], x8
        cmp x10, #2
        beq LoopCol
        st1 {v12.4s, v13.4s}, [x21], x8
        cmp x10, #3
        beq LoopCol
        st1 {v14.4s, v15.4s}, [x21], x8
        cmp x10, #4
        beq LoopCol
        st1 {v16.4s, v17.4s}, [x21], x8
        cmp x10, #5
        beq LoopCol
        st1 {v18.4s, v19.4s}, [x21], x8
        cmp x10, #6
        beq LoopCol
        st1 {v20.4s, v21.4s}, [x21], x8
        cmp x10, #7
        beq LoopCol
        st1 {v22.4s, v23.4s}, [x21], x8
        cmp x10, #8
        beq LoopCol
        st1 {v24.4s, v25.4s}, [x21], x8
        cmp x10, #9
        beq LoopCol
        st1 {v26.4s, v27.4s}, [x21], x8
        cmp x10, #10
        beq LoopCol
        st1 {v28.4s, v29.4s}, [x21], x8
        cmp x10, #11
        beq LoopCol
        st1 {v30.4s, v31.4s}, [x21], x8
        add x21, x21, #32
        b LoopCol

LoopCol:
    subs x9, x9, #8
    ble LoopColEnd
    add x12, x12, x22  // update rhs
    add x14, x14, #32  // update out
    cmp x10, #4
    ble LoopRow4
    cmp x10, #8
    ble LoopRow8
    b LoopRow12

LoopColEnd:
    add x25, x25, x26    // update lhs
    add x13, x13, x24  // update out
    subs x10, x10, #12  // update row
    bgt  RowStart
    mov x20, #0
    add x0, x0, x27  // update lhs by depth
    add x1, x1, x28  // update rhs by depth
    subs x5, x5, #512
    bgt Depth512

///////////////////////////////////////////////////////

DepthTail:
    add x5, x5, #512
    mov x13, x2  // out
    mov x10, x6
    TailRowStart:
        mov x12, x1  // rhs
        mov x14, x13  // out
        mov x15, x3  // restore bias
        mov x9, x7  // restore col
        cmp x10, #4
        ble LoopTailRow4
        cmp x10, #8
        ble LoopTailRow8

        LoopTailRow12:
            mov x11, x0  // lhs
            mov x23, x12 // rhs
            mov x21, x5  // depth unit
            prfm pldl1strm, [x23, #632]
            ld1 {v3.4s}, [x23], #16
            ld1 {v0.4s}, [x11], #16
            cmp x9, #4
            ble LoopTailCol12x4

            LoopTailCol12x8:
                cbz x20, ReloadTail12x8
                cbnz x15, InitTailFromBias12x8
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
                b ComputeTail12x8Enter
            InitTailFromBias12x8:
                ld1 {v8.4s, v9.4s}, [x15]
                ld1 {v10.4s, v11.4s}, [x15]
                ld1 {v12.4s, v13.4s}, [x15]
                ld1 {v14.4s, v15.4s}, [x15]
                ld1 {v16.4s, v17.4s}, [x15]
                ld1 {v18.4s, v19.4s}, [x15]
                ld1 {v20.4s, v21.4s}, [x15]
                ld1 {v22.4s, v23.4s}, [x15]
                ld1 {v24.4s, v25.4s}, [x15]
                ld1 {v26.4s, v27.4s}, [x15]
                ld1 {v28.4s, v29.4s}, [x15]
                ld1 {v30.4s, v31.4s}, [x15]
                add x15, x15, #32
                b ComputeTail12x8Enter
            ReloadTail12x8:
                bl Reload
            ComputeTail12x8Enter:
                cbz x21, Activation12x8
                bl Compute12x8Unit
            Activation12x8:
                cmp x4, #3
                beq Relu612x8
                cmp x4, #1
                beq Relu12x8
                b WriteTail

                Relu612x8:
                    mov w19, #6
                    dup v2.4s, w19
                    scvtf v2.4s, v2.4s
                    fmin v8.4s, v8.4s, v2.4s
                    fmin v9.4s, v9.4s, v2.4s
                    fmin v10.4s, v10.4s, v2.4s
                    fmin v11.4s, v11.4s, v2.4s
                    fmin v12.4s, v12.4s, v2.4s
                    fmin v13.4s, v13.4s, v2.4s
                    fmin v14.4s, v14.4s, v2.4s
                    fmin v15.4s, v15.4s, v2.4s
                    fmin v16.4s, v16.4s, v2.4s
                    fmin v17.4s, v17.4s, v2.4s
                    fmin v18.4s, v18.4s, v2.4s
                    fmin v19.4s, v19.4s, v2.4s
                    fmin v20.4s, v20.4s, v2.4s
                    fmin v21.4s, v21.4s, v2.4s
                    fmin v22.4s, v22.4s, v2.4s
                    fmin v23.4s, v23.4s, v2.4s
                    fmin v24.4s, v24.4s, v2.4s
                    fmin v25.4s, v25.4s, v2.4s
                    fmin v26.4s, v26.4s, v2.4s
                    fmin v27.4s, v27.4s, v2.4s
                    fmin v28.4s, v28.4s, v2.4s
                    fmin v29.4s, v29.4s, v2.4s
                    fmin v30.4s, v30.4s, v2.4s
                    fmin v31.4s, v31.4s, v2.4s

                Relu12x8:
                    dup v3.4s, wzr
                    fmax v8.4s, v8.4s, v3.4s
                    fmax v9.4s, v9.4s, v3.4s
                    fmax v10.4s, v10.4s, v3.4s
                    fmax v11.4s, v11.4s, v3.4s
                    fmax v12.4s, v12.4s, v3.4s
                    fmax v13.4s, v13.4s, v3.4s
                    fmax v14.4s, v14.4s, v3.4s
                    fmax v15.4s, v15.4s, v3.4s
                    fmax v16.4s, v16.4s, v3.4s
                    fmax v17.4s, v17.4s, v3.4s
                    fmax v18.4s, v18.4s, v3.4s
                    fmax v19.4s, v19.4s, v3.4s
                    fmax v20.4s, v20.4s, v3.4s
                    fmax v21.4s, v21.4s, v3.4s
                    fmax v22.4s, v22.4s, v3.4s
                    fmax v23.4s, v23.4s, v3.4s
                    fmax v24.4s, v24.4s, v3.4s
                    fmax v25.4s, v25.4s, v3.4s
                    fmax v26.4s, v26.4s, v3.4s
                    fmax v27.4s, v27.4s, v3.4s
                    fmax v28.4s, v28.4s, v3.4s
                    fmax v29.4s, v29.4s, v3.4s
                    fmax v30.4s, v30.4s, v3.4s
                    fmax v31.4s, v31.4s, v3.4s
                    b WriteTail

            LoopTailCol12x4:
                cbz x20, ReloadTail12x4
                cbnz x15, InitTailFromBias12x4
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
                b ComputeTail12x4Enter
            InitTailFromBias12x4:
                ld1 {v8.4s}, [x15]
                ld1 {v10.4s}, [x15]
                ld1 {v12.4s}, [x15]
                ld1 {v14.4s}, [x15]
                ld1 {v16.4s}, [x15]
                ld1 {v18.4s}, [x15]
                ld1 {v20.4s}, [x15]
                ld1 {v22.4s}, [x15]
                ld1 {v24.4s}, [x15]
                ld1 {v26.4s}, [x15]
                ld1 {v28.4s}, [x15]
                ld1 {v30.4s}, [x15]
                b ComputeTail12x4Enter
            ReloadTail12x4:
                bl Reload
            ComputeTail12x4Enter:
                cbz x21, Activation12x4
                bl Compute12x4Unit
            Activation12x4:
                cmp x4, #3
                beq Relu612x4
                cmp x4, #1
                beq Relu12x4
                b WriteTail

                Relu612x4:
                    mov w19, #6
                    dup v2.4s, w19
                    scvtf v2.4s, v2.4s
                    fmin v8.4s, v8.4s, v2.4s
                    fmin v10.4s, v10.4s, v2.4s
                    fmin v12.4s, v12.4s, v2.4s
                    fmin v14.4s, v14.4s, v2.4s
                    fmin v16.4s, v16.4s, v2.4s
                    fmin v18.4s, v18.4s, v2.4s
                    fmin v20.4s, v20.4s, v2.4s
                    fmin v22.4s, v22.4s, v2.4s
                    fmin v24.4s, v24.4s, v2.4s
                    fmin v26.4s, v26.4s, v2.4s
                    fmin v28.4s, v28.4s, v2.4s
                    fmin v30.4s, v30.4s, v2.4s

                Relu12x4:
                    dup v3.4s, wzr
                    fmax v8.4s, v8.4s, v3.4s
                    fmax v10.4s, v10.4s, v3.4s
                    fmax v12.4s, v12.4s, v3.4s
                    fmax v14.4s, v14.4s, v3.4s
                    fmax v16.4s, v16.4s, v3.4s
                    fmax v18.4s, v18.4s, v3.4s
                    fmax v20.4s, v20.4s, v3.4s
                    fmax v22.4s, v22.4s, v3.4s
                    fmax v24.4s, v24.4s, v3.4s
                    fmax v26.4s, v26.4s, v3.4s
                    fmax v28.4s, v28.4s, v3.4s
                    fmax v30.4s, v30.4s, v3.4s
                    b WriteTail

        LoopTailRow8:
            mov x11, x0  // lhs
            mov x23, x12 // rhs
            mov x21, x5  // depth unit
            prfm pldl1strm, [x23, #632]
            ld1 {v3.4s}, [x23], #16
            ld1 {v0.4s}, [x11], #16
            cmp x9, #4
            ble LoopTailCol8x4

            LoopTailCol8x8:
                cbz x20, ReloadTail8x8
                cbnz x15, InitTailFromBias8x8
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
                b ComputeTail8x8Enter
            InitTailFromBias8x8:
                ld1 {v8.4s, v9.4s}, [x15]
                ld1 {v10.4s, v11.4s}, [x15]
                ld1 {v12.4s, v13.4s}, [x15]
                ld1 {v14.4s, v15.4s}, [x15]
                ld1 {v16.4s, v17.4s}, [x15]
                ld1 {v18.4s, v19.4s}, [x15]
                ld1 {v20.4s, v21.4s}, [x15]
                ld1 {v22.4s, v23.4s}, [x15]
                add x15, x15, #32
                b ComputeTail8x8Enter
            ReloadTail8x8:
                bl Reload
            ComputeTail8x8Enter:
                cbz x21, Activation8x8
                bl Compute8x8Unit
            Activation8x8:
                cmp x4, #3
                beq Relu68x8
                cmp x4, #1
                beq Relu8x8
                b WriteTail

                Relu68x8:
                    mov w19, #6
                    dup v2.4s, w19
                    scvtf v2.4s, v2.4s
                    fmin v8.4s, v8.4s, v2.4s
                    fmin v9.4s, v9.4s, v2.4s
                    fmin v10.4s, v10.4s, v2.4s
                    fmin v11.4s, v11.4s, v2.4s
                    fmin v12.4s, v12.4s, v2.4s
                    fmin v13.4s, v13.4s, v2.4s
                    fmin v14.4s, v14.4s, v2.4s
                    fmin v15.4s, v15.4s, v2.4s
                    fmin v16.4s, v16.4s, v2.4s
                    fmin v17.4s, v17.4s, v2.4s
                    fmin v18.4s, v18.4s, v2.4s
                    fmin v19.4s, v19.4s, v2.4s
                    fmin v20.4s, v20.4s, v2.4s
                    fmin v21.4s, v21.4s, v2.4s
                    fmin v22.4s, v22.4s, v2.4s
                    fmin v23.4s, v23.4s, v2.4s

                Relu8x8:
                    dup v3.4s, wzr
                    fmax v8.4s, v8.4s, v3.4s
                    fmax v9.4s, v9.4s, v3.4s
                    fmax v10.4s, v10.4s, v3.4s
                    fmax v11.4s, v11.4s, v3.4s
                    fmax v12.4s, v12.4s, v3.4s
                    fmax v13.4s, v13.4s, v3.4s
                    fmax v14.4s, v14.4s, v3.4s
                    fmax v15.4s, v15.4s, v3.4s
                    fmax v16.4s, v16.4s, v3.4s
                    fmax v17.4s, v17.4s, v3.4s
                    fmax v18.4s, v18.4s, v3.4s
                    fmax v19.4s, v19.4s, v3.4s
                    fmax v20.4s, v20.4s, v3.4s
                    fmax v21.4s, v21.4s, v3.4s
                    fmax v22.4s, v22.4s, v3.4s
                    fmax v23.4s, v23.4s, v3.4s
                    b WriteTail

            LoopTailCol8x4:
                cbz x20, ReloadTail8x4
                cbnz x15, InitTailFromBias8x4
                dup v8.2d, xzr
                dup v10.2d, xzr
                dup v12.2d, xzr
                dup v14.2d, xzr
                dup v16.2d, xzr
                dup v18.2d, xzr
                dup v20.2d, xzr
                dup v22.2d, xzr
                b ComputeTail8x4Enter
            InitTailFromBias8x4:
                ld1 {v8.4s}, [x15]
                ld1 {v10.4s}, [x15]
                ld1 {v12.4s}, [x15]
                ld1 {v14.4s}, [x15]
                ld1 {v16.4s}, [x15]
                ld1 {v18.4s}, [x15]
                ld1 {v20.4s}, [x15]
                ld1 {v22.4s}, [x15]
                b ComputeTail8x4Enter
            ReloadTail8x4:
                bl Reload
            ComputeTail8x4Enter:
                cbz x21, Activation8x4
                bl Compute8x4Unit
            Activation8x4:
                cmp x4, #3
                beq Relu68x4
                cmp x4, #1
                beq Relu8x4
                b WriteTail

                Relu68x4:
                    mov w19, #6
                    dup v2.4s, w19
                    scvtf v2.4s, v2.4s
                    fmin v8.4s, v8.4s, v2.4s
                    fmin v10.4s, v10.4s, v2.4s
                    fmin v12.4s, v12.4s, v2.4s
                    fmin v14.4s, v14.4s, v2.4s
                    fmin v16.4s, v16.4s, v2.4s
                    fmin v18.4s, v18.4s, v2.4s
                    fmin v20.4s, v20.4s, v2.4s
                    fmin v22.4s, v22.4s, v2.4s

                Relu8x4:
                    dup v3.4s, wzr
                    fmax v8.4s, v8.4s, v3.4s
                    fmax v10.4s, v10.4s, v3.4s
                    fmax v12.4s, v12.4s, v3.4s
                    fmax v14.4s, v14.4s, v3.4s
                    fmax v16.4s, v16.4s, v3.4s
                    fmax v18.4s, v18.4s, v3.4s
                    fmax v20.4s, v20.4s, v3.4s
                    fmax v22.4s, v22.4s, v3.4s
                    b WriteTail

        LoopTailRow4:
            mov x11, x0  // lhs
            mov x23, x12 // rhs
            mov x21, x5  // depth unit
            prfm pldl1strm, [x23, #632]
            ld1 {v3.4s}, [x23], #16
            ld1 {v0.4s}, [x11], #16
            cmp x9, #4
            ble LoopTailCol4x4

            LoopTailCol4x8:
                cbz x20, ReloadTail4x8
                cbnz x15, InitTailFromBias4x8
                dup v8.2d, xzr
                dup v9.2d, xzr
                dup v10.2d, xzr
                dup v11.2d, xzr
                dup v12.2d, xzr
                dup v13.2d, xzr
                dup v14.2d, xzr
                dup v15.2d, xzr
                b ComputeTail4x8Enter
            InitTailFromBias4x8:
                ld1 {v8.4s, v9.4s}, [x15]
                ld1 {v10.4s, v11.4s}, [x15]
                ld1 {v12.4s, v13.4s}, [x15]
                ld1 {v14.4s, v15.4s}, [x15]
                add x15, x15, #32
                b ComputeTail4x8Enter
            ReloadTail4x8:
                bl Reload
            ComputeTail4x8Enter:
                cbz x21, Activation4x8
                bl Compute4x8Unit
            Activation4x8:
                cmp x4, #3
                beq Relu64x8
                cmp x4, #1
                beq Relu4x8
                b WriteTail

                Relu64x8:
                    mov w19, #6
                    dup v2.4s, w19
                    scvtf v2.4s, v2.4s
                    fmin v8.4s, v8.4s, v2.4s
                    fmin v9.4s, v9.4s, v2.4s
                    fmin v10.4s, v10.4s, v2.4s
                    fmin v11.4s, v11.4s, v2.4s
                    fmin v12.4s, v12.4s, v2.4s
                    fmin v13.4s, v13.4s, v2.4s
                    fmin v14.4s, v14.4s, v2.4s
                    fmin v15.4s, v15.4s, v2.4s

                Relu4x8:
                    dup v3.4s, wzr
                    fmax v8.4s, v8.4s, v3.4s
                    fmax v9.4s, v9.4s, v3.4s
                    fmax v10.4s, v10.4s, v3.4s
                    fmax v11.4s, v11.4s, v3.4s
                    fmax v12.4s, v12.4s, v3.4s
                    fmax v13.4s, v13.4s, v3.4s
                    fmax v14.4s, v14.4s, v3.4s
                    fmax v15.4s, v15.4s, v3.4s
                    b WriteTail

            LoopTailCol4x4:
                cbz x20, ReloadTail4x4
                cbnz x15, InitTailFromBias4x4
                dup v8.2d, xzr
                dup v10.2d, xzr
                dup v12.2d, xzr
                dup v14.2d, xzr
                b ComputeTail4x4Enter
            InitTailFromBias4x4:
                ld1 {v8.4s}, [x15]
                ld1 {v10.4s}, [x15]
                ld1 {v12.4s}, [x15]
                ld1 {v14.4s}, [x15]
                b ComputeTail4x4Enter
            ReloadTail4x4:
                bl Reload
            ComputeTail4x4Enter:
                cbz x21, Activation4x4
                bl Compute4x4Unit
            Activation4x4:
                cmp x4, #3
                beq Relu64x4
                cmp x4, #1
                beq Relu4x4
                b WriteTail

                Relu64x4:
                    mov w19, #6
                    dup v2.4s, w19
                    scvtf v2.4s, v2.4s
                    fmin v8.4s, v8.4s, v2.4s
                    fmin v10.4s, v10.4s, v2.4s
                    fmin v12.4s, v12.4s, v2.4s
                    fmin v14.4s, v14.4s, v2.4s

                Relu4x4:
                    dup v3.4s, wzr
                    fmax v8.4s, v8.4s, v3.4s
                    fmax v10.4s, v10.4s, v3.4s
                    fmax v12.4s, v12.4s, v3.4s
                    fmax v14.4s, v14.4s, v3.4s

WriteTail:
    mov x21, x14
    cmp x9, #1
    beq WriteTail1
    cmp x9, #2
    beq WriteTail2
    cmp x9, #3
    beq WriteTail3
    cmp x9, #4
    beq WriteTail4
    cmp x9, #5
    beq WriteTail5
    cmp x9, #6
    beq WriteTail6
    cmp x9, #7
    beq WriteTail7
    b WriteTail8

    WriteTail1:
        str s8, [x21]
        cmp x10, #1
        beq LoopTailCol
        add x21, x21, x8
        str s10, [x21]
        cmp x10, #2
        beq LoopTailCol
        add x21, x21, x8
        str s12, [x21]
        cmp x10, #3
        beq LoopTailCol
        add x21, x21, x8
        str s14, [x21]
        cmp x10, #4
        beq LoopTailCol
        add x21, x21, x8
        str s16, [x21]
        cmp x10, #5
        beq LoopTailCol
        add x21, x21, x8
        str s18, [x21]
        cmp x10, #6
        beq LoopTailCol
        add x21, x21, x8
        str s20, [x21]
        cmp x10, #7
        beq LoopTailCol
        add x21, x21, x8
        str s22, [x21]
        cmp x10, #8
        beq LoopTailCol
        add x21, x21, x8
        str s24, [x21]
        cmp x10, #9
        beq LoopTailCol
        add x21, x21, x8
        str s26, [x21]
        cmp x10, #10
        beq LoopTailCol
        add x21, x21, x8
        str s28, [x21]
        cmp x10, #11
        beq LoopTailCol
        add x21, x21, x8
        str s30, [x21]
        b LoopTailCol
    WriteTail2:
        st1 {v8.2s}, [x21], x8
        cmp x10, #1
        beq LoopTailCol
        st1 {v10.2s}, [x21], x8
        cmp x10, #2
        beq LoopTailCol
        st1 {v12.2s}, [x21], x8
        cmp x10, #3
        beq LoopTailCol
        st1 {v14.2s}, [x21], x8
        cmp x10, #4
        beq LoopTailCol
        st1 {v16.2s}, [x21], x8
        cmp x10, #5
        beq LoopTailCol
        st1 {v18.2s}, [x21], x8
        cmp x10, #6
        beq LoopTailCol
        st1 {v20.2s}, [x21], x8
        cmp x10, #7
        beq LoopTailCol
        st1 {v22.2s}, [x21], x8
        cmp x10, #8
        beq LoopTailCol
        st1 {v24.2s}, [x21], x8
        cmp x10, #9
        beq LoopTailCol
        st1 {v26.2s}, [x21], x8
        cmp x10, #10
        beq LoopTailCol
        st1 {v28.2s}, [x21], x8
        cmp x10, #11
        beq LoopTailCol
        st1 {v30.2s}, [x21], x8
        add x21, x21, #8
        b LoopTailCol
    WriteTail3:
        add x11, x21, #8
        st1 {v8.2s}, [x21], x8
        st1 {v8.s}[2], [x11], x8
        cmp x10, #1
        beq LoopTailCol
        st1 {v10.2s}, [x21], x8
        st1 {v10.s}[2], [x11], x8
        cmp x10, #2
        beq LoopTailCol
        st1 {v12.2s}, [x21], x8
        st1 {v12.s}[2], [x11], x8
        cmp x10, #3
        beq LoopTailCol
        st1 {v14.2s}, [x21], x8
        st1 {v14.s}[2], [x11], x8
        cmp x10, #4
        beq LoopTailCol
        st1 {v16.2s}, [x21], x8
        st1 {v16.s}[2], [x11], x8
        cmp x10, #5
        beq LoopTailCol
        st1 {v18.2s}, [x21], x8
        st1 {v18.s}[2], [x11], x8
        cmp x10, #6
        beq LoopTailCol
        st1 {v20.2s}, [x21], x8
        st1 {v20.s}[2], [x11], x8
        cmp x10, #7
        beq LoopTailCol
        st1 {v22.2s}, [x21], x8
        st1 {v22.s}[2], [x11], x8
        cmp x10, #8
        beq LoopTailCol
        st1 {v24.2s}, [x21], x8
        st1 {v24.s}[2], [x11], x8
        cmp x10, #9
        beq LoopTailCol
        st1 {v26.2s}, [x21], x8
        st1 {v26.s}[2], [x11], x8
        cmp x10, #10
        beq LoopTailCol
        st1 {v28.2s}, [x21], x8
        st1 {v28.s}[2], [x11], x8
        cmp x10, #11
        beq LoopTailCol
        st1 {v30.2s}, [x21], x8
        st1 {v30.s}[2], [x11]
        add x21, x21, #12
        b LoopTailCol
    WriteTail4:
        st1 {v8.4s}, [x21], x8
        cmp x10, #1
        beq LoopTailCol
        st1 {v10.4s}, [x21], x8
        cmp x10, #2
        beq LoopTailCol
        st1 {v12.4s}, [x21], x8
        cmp x10, #3
        beq LoopTailCol
        st1 {v14.4s}, [x21], x8
        cmp x10, #4
        beq LoopTailCol
        st1 {v16.4s}, [x21], x8
        cmp x10, #5
        beq LoopTailCol
        st1 {v18.4s}, [x21], x8
        cmp x10, #6
        beq LoopTailCol
        st1 {v20.4s}, [x21], x8
        cmp x10, #7
        beq LoopTailCol
        st1 {v22.4s}, [x21], x8
        cmp x10, #8
        beq LoopTailCol
        st1 {v24.4s}, [x21], x8
        cmp x10, #9
        beq LoopTailCol
        st1 {v26.4s}, [x21], x8
        cmp x10, #10
        beq LoopTailCol
        st1 {v28.4s}, [x21], x8
        cmp x10, #11
        beq LoopTailCol
        st1 {v30.4s}, [x21], x8
        add x21, x21, #16
        b LoopTailCol
    WriteTail5:
        add x11, x21, #16
        st1 {v8.4s}, [x21], x8
        str s9, [x11]
        cmp x10, #1
        beq LoopTailCol
        add x11, x11, x8
        st1 {v10.4s}, [x21], x8
        str s11, [x11]
        cmp x10, #2
        beq LoopTailCol
        add x11, x11, x8
        st1 {v12.4s}, [x21], x8
        str s13, [x11]
        cmp x10, #3
        beq LoopTailCol
        add x11, x11, x8
        st1 {v14.4s}, [x21], x8
        str s15, [x11]
        cmp x10, #4
        beq LoopTailCol
        add x11, x11, x8
        st1 {v16.4s}, [x21], x8
        str s17, [x11]
        cmp x10, #5
        beq LoopTailCol
        add x11, x11, x8
        st1 {v18.4s}, [x21], x8
        str s19, [x11]
        cmp x10, #6
        beq LoopTailCol
        add x11, x11, x8
        st1 {v20.4s}, [x21], x8
        str s21, [x11]
        cmp x10, #7
        beq LoopTailCol
        add x11, x11, x8
        st1 {v22.4s}, [x21], x8
        str s23, [x11]
        cmp x10, #8
        beq LoopTailCol
        add x11, x11, x8
        st1 {v24.4s}, [x21], x8
        str s25, [x11]
        cmp x10, #9
        beq LoopTailCol
        add x11, x11, x8
        st1 {v26.4s}, [x21], x8
        str s27, [x11]
        cmp x10, #10
        beq LoopTailCol
        add x11, x11, x8
        st1 {v28.4s}, [x21], x8
        str s29, [x11]
        cmp x10, #11
        beq LoopTailCol
        add x11, x11, x8
        st1 {v30.4s}, [x21], x8
        str s31, [x11]
        add x21, x21, #20
        b LoopTailCol
    WriteTail6:
        add x11, x21, #16
        st1 {v8.4s}, [x21], x8
        st1 {v9.2s}, [x11], x8
        cmp x10, #1
        beq LoopTailCol
        st1 {v10.4s}, [x21], x8
        st1 {v11.2s}, [x11], x8
        cmp x10, #2
        beq LoopTailCol
        st1 {v12.4s}, [x21], x8
        st1 {v13.2s}, [x11], x8
        cmp x10, #3
        beq LoopTailCol
        st1 {v14.4s}, [x21], x8
        st1 {v15.2s}, [x11], x8
        cmp x10, #4
        beq LoopTailCol
        st1 {v16.4s}, [x21], x8
        st1 {v17.2s}, [x11], x8
        cmp x10, #5
        beq LoopTailCol
        st1 {v18.4s}, [x21], x8
        st1 {v19.2s}, [x11], x8
        cmp x10, #6
        beq LoopTailCol
        st1 {v20.4s}, [x21], x8
        st1 {v21.2s}, [x11], x8
        cmp x10, #7
        beq LoopTailCol
        st1 {v22.4s}, [x21], x8
        st1 {v23.2s}, [x11], x8
        cmp x10, #8
        beq LoopTailCol
        st1 {v24.4s}, [x21], x8
        st1 {v25.2s}, [x11], x8
        cmp x10, #9
        beq LoopTailCol
        st1 {v26.4s}, [x21], x8
        st1 {v27.2s}, [x11], x8
        cmp x10, #10
        beq LoopTailCol
        st1 {v28.4s}, [x21], x8
        st1 {v29.2s}, [x11], x8
        cmp x10, #11
        beq LoopTailCol
        st1 {v30.4s}, [x21], x8
        st1 {v31.2s}, [x11]
        add x21, x21, #24
        b LoopTailCol
    WriteTail7:
        add x11, x21, #16
        add x23, x21, #24
        st1 {v8.4s}, [x21], x8
        st1 {v9.2s}, [x11], x8
        st1 {v9.s}[2], [x23], x8
        cmp x10, #1
        beq LoopTailCol
        st1 {v10.4s}, [x21], x8
        st1 {v11.2s}, [x11], x8
        st1 {v11.s}[2], [x23], x8
        cmp x10, #2
        beq LoopTailCol
        st1 {v12.4s}, [x21], x8
        st1 {v13.2s}, [x11], x8
        st1 {v13.s}[2], [x23], x8
        cmp x10, #3
        beq LoopTailCol
        st1 {v14.4s}, [x21], x8
        st1 {v15.2s}, [x11], x8
        st1 {v15.s}[2], [x23], x8
        cmp x10, #4
        beq LoopTailCol
        st1 {v16.4s}, [x21], x8
        st1 {v17.2s}, [x11], x8
        st1 {v17.s}[2], [x23], x8
        cmp x10, #5
        beq LoopTailCol
        st1 {v18.4s}, [x21], x8
        st1 {v19.2s}, [x11], x8
        st1 {v19.s}[2], [x23], x8
        cmp x10, #6
        beq LoopTailCol
        st1 {v20.4s}, [x21], x8
        st1 {v21.2s}, [x11], x8
        st1 {v21.s}[2], [x23], x8
        cmp x10, #7
        beq LoopTailCol
        st1 {v22.4s}, [x21], x8
        st1 {v23.2s}, [x11], x8
        st1 {v23.s}[2], [x23], x8
        cmp x10, #8
        beq LoopTailCol
        st1 {v24.4s}, [x21], x8
        st1 {v25.2s}, [x11], x8
        st1 {v25.s}[2], [x23], x8
        cmp x10, #9
        beq LoopTailCol
        st1 {v26.4s}, [x21], x8
        st1 {v27.2s}, [x11], x8
        st1 {v27.s}[2], [x23], x8
        cmp x10, #10
        beq LoopTailCol
        st1 {v28.4s}, [x21], x8
        st1 {v29.2s}, [x11], x8
        st1 {v29.s}[2], [x23], x8
        cmp x10, #11
        beq LoopTailCol
        st1 {v30.4s}, [x21], x8
        st1 {v31.2s}, [x11]
        st1 {v31.s}[2], [x23]
        add x21, x21, #28
        b LoopTailCol

    WriteTail8:
        st1 {v8.4s, v9.4s}, [x21], x8
        cmp x10, #1
        beq LoopTailCol
        st1 {v10.4s, v11.4s}, [x21], x8
        cmp x10, #2
        beq LoopTailCol
        st1 {v12.4s, v13.4s}, [x21], x8
        cmp x10, #3
        beq LoopTailCol
        st1 {v14.4s, v15.4s}, [x21], x8
        cmp x10, #4
        beq LoopTailCol
        st1 {v16.4s, v17.4s}, [x21], x8
        cmp x10, #5
        beq LoopTailCol
        st1 {v18.4s, v19.4s}, [x21], x8
        cmp x10, #6
        beq LoopTailCol
        st1 {v20.4s, v21.4s}, [x21], x8
        cmp x10, #7
        beq LoopTailCol
        st1 {v22.4s, v23.4s}, [x21], x8
        cmp x10, #8
        beq LoopTailCol
        st1 {v24.4s, v25.4s}, [x21], x8
        cmp x10, #9
        beq LoopTailCol
        st1 {v26.4s, v27.4s}, [x21], x8
        cmp x10, #10
        beq LoopTailCol
        st1 {v28.4s, v29.4s}, [x21], x8
        cmp x10, #11
        beq LoopTailCol
        st1 {v30.4s, v31.4s}, [x21], x8
        add x21, x21, #32
        b LoopTailCol

LoopTailCol:
    subs x9, x9, #8
    ble LoopTailEnd
    add x12, x12, x22  // update rhs
    add x14, x14, #32
    cmp x10, #4
    ble LoopTailRow4
    cmp x10, #8
    ble LoopTailRow8
    b LoopTailRow12

LoopTailEnd:
    add x0, x0, x26    // update lhs
    add x13, x13, x24  // update out
    subs x10, x10, #12  // update row
    bgt  TailRowStart
    b End

Reload:
    mov x15, x14
    cmp x9, #1
    beq Reload1
    cmp x9, #2
    beq Reload2
    cmp x9, #3
    beq Reload3
    cmp x9, #4
    beq Reload4
    cmp x9, #5
    beq Reload5
    cmp x9, #6
    beq Reload6
    cmp x9, #7
    beq Reload7
    b Reload8

    Reload1:
        ldr s8, [x15]
        cmp x10, #1
        beq ReloadEnd
        add x15, x15, x8
        ldr s10, [x15]
        cmp x10, #2
        beq ReloadEnd
        add x15, x15, x8
        ldr s12, [x15]
        cmp x10, #3
        beq ReloadEnd
        add x15, x15, x8
        ldr s14, [x15]
        cmp x10, #4
        beq ReloadEnd
        add x15, x15, x8
        ldr s16, [x15]
        cmp x10, #5
        beq ReloadEnd
        add x15, x15, x8
        ldr s18, [x15]
        cmp x10, #6
        beq ReloadEnd
        add x15, x15, x8
        ldr s20, [x15]
        cmp x10, #7
        beq ReloadEnd
        add x15, x15, x8
        ldr s22, [x15]
        cmp x10, #8
        beq ReloadEnd
        add x15, x15, x8
        ldr s24, [x15]
        cmp x10, #9
        beq ReloadEnd
        add x15, x15, x8
        ldr s26, [x15]
        cmp x10, #10
        beq ReloadEnd
        add x15, x15, x8
        ldr s28, [x15]
        cmp x10, #11
        beq ReloadEnd
        add x15, x15, x8
        ldr s30, [x15]
        b ReloadEnd
    Reload2:
        ld1 {v8.2s}, [x15], x8
        cmp x10, #1
        beq ReloadEnd
        ld1 {v10.2s}, [x15], x8
        cmp x10, #2
        beq ReloadEnd
        ld1 {v12.2s}, [x15], x8
        cmp x10, #3
        beq ReloadEnd
        ld1 {v14.2s}, [x15], x8
        cmp x10, #4
        beq ReloadEnd
        ld1 {v16.2s}, [x15], x8
        cmp x10, #5
        beq ReloadEnd
        ld1 {v18.2s}, [x15], x8
        cmp x10, #6
        beq ReloadEnd
        ld1 {v20.2s}, [x15], x8
        cmp x10, #7
        beq ReloadEnd
        ld1 {v22.2s}, [x15], x8
        cmp x10, #8
        beq ReloadEnd
        ld1 {v24.2s}, [x15], x8
        cmp x10, #9
        beq ReloadEnd
        ld1 {v26.2s}, [x15], x8
        cmp x10, #10
        beq ReloadEnd
        ld1 {v28.2s}, [x15], x8
        cmp x10, #11
        beq ReloadEnd
        ld1 {v30.2s}, [x15], x8
        add x15, x15, #8
        b ReloadEnd
    Reload3:
        add x19, x15, #8
        ld1 {v8.2s}, [x15], x8
        ld1 {v8.s}[2], [x19], x8
        cmp x10, #1
        beq ReloadEnd
        ld1 {v10.2s}, [x15], x8
        ld1 {v10.s}[2], [x19], x8
        cmp x10, #2
        beq ReloadEnd
        ld1 {v12.2s}, [x15], x8
        ld1 {v12.s}[2], [x19], x8
        cmp x10, #3
        beq ReloadEnd
        ld1 {v14.2s}, [x15], x8
        ld1 {v14.s}[2], [x19], x8
        cmp x10, #4
        beq ReloadEnd
        ld1 {v16.2s}, [x15], x8
        ld1 {v16.s}[2], [x19], x8
        cmp x10, #5
        beq ReloadEnd
        ld1 {v18.2s}, [x15], x8
        ld1 {v18.s}[2], [x19], x8
        cmp x10, #6
        beq ReloadEnd
        ld1 {v20.2s}, [x15], x8
        ld1 {v20.s}[2], [x19], x8
        cmp x10, #7
        beq ReloadEnd
        ld1 {v22.2s}, [x15], x8
        ld1 {v22.s}[2], [x19], x8
        cmp x10, #8
        beq ReloadEnd
        ld1 {v24.2s}, [x15], x8
        ld1 {v24.s}[2], [x19], x8
        cmp x10, #9
        beq ReloadEnd
        ld1 {v26.2s}, [x15], x8
        ld1 {v26.s}[2], [x19], x8
        cmp x10, #10
        beq ReloadEnd
        ld1 {v28.2s}, [x15], x8
        ld1 {v28.s}[2], [x19], x8
        cmp x10, #11
        beq ReloadEnd
        ld1 {v30.2s}, [x15], x8
        ld1 {v30.s}[2], [x19]
        add x15, x15, #12
        b ReloadEnd
    Reload4:
        ld1 {v8.4s}, [x15], x8
        cmp x10, #1
        beq ReloadEnd
        ld1 {v10.4s}, [x15], x8
        cmp x10, #2
        beq ReloadEnd
        ld1 {v12.4s}, [x15], x8
        cmp x10, #3
        beq ReloadEnd
        ld1 {v14.4s}, [x15], x8
        cmp x10, #4
        beq ReloadEnd
        ld1 {v16.4s}, [x15], x8
        cmp x10, #5
        beq ReloadEnd
        ld1 {v18.4s}, [x15], x8
        cmp x10, #6
        beq ReloadEnd
        ld1 {v20.4s}, [x15], x8
        cmp x10, #7
        beq ReloadEnd
        ld1 {v22.4s}, [x15], x8
        cmp x10, #8
        beq ReloadEnd
        ld1 {v24.4s}, [x15], x8
        cmp x10, #9
        beq ReloadEnd
        ld1 {v26.4s}, [x15], x8
        cmp x10, #10
        beq ReloadEnd
        ld1 {v28.4s}, [x15], x8
        cmp x10, #11
        beq ReloadEnd
        ld1 {v30.4s}, [x15], x8
        add x15, x15, #16
        b ReloadEnd
    Reload5:
        add x19, x15, #16
        ld1 {v8.4s}, [x15], x8
        ldr s9, [x19]
        cmp x10, #1
        beq ReloadEnd
        add x19, x19, x8
        ld1 {v10.4s}, [x15], x8
        ldr s11, [x19]
        cmp x10, #2
        beq ReloadEnd
        add x19, x19, x8
        ld1 {v12.4s}, [x15], x8
        ldr s13, [x19]
        cmp x10, #3
        beq ReloadEnd
        add x19, x19, x8
        ld1 {v14.4s}, [x15], x8
        ldr s15, [x19]
        cmp x10, #4
        beq ReloadEnd
        add x19, x19, x8
        ld1 {v16.4s}, [x15], x8
        ldr s17, [x19]
        cmp x10, #5
        beq ReloadEnd
        add x19, x19, x8
        ld1 {v18.4s}, [x15], x8
        ldr s19, [x19]
        cmp x10, #6
        beq ReloadEnd
        add x19, x19, x8
        ld1 {v20.4s}, [x15], x8
        ldr s21, [x19]
        cmp x10, #7
        beq ReloadEnd
        add x19, x19, x8
        ld1 {v22.4s}, [x15], x8
        ldr s23, [x19]
        cmp x10, #8
        beq ReloadEnd
        add x19, x19, x8
        ld1 {v24.4s}, [x15], x8
        ldr s25, [x19]
        cmp x10, #9
        beq ReloadEnd
        add x19, x19, x8
        ld1 {v26.4s}, [x15], x8
        ldr s27, [x19]
        cmp x10, #10
        beq ReloadEnd
        add x19, x19, x8
        ld1 {v28.4s}, [x15], x8
        ldr s29, [x19]
        cmp x10, #11
        beq ReloadEnd
        add x19, x19, x8
        ld1 {v30.4s}, [x15], x8
        ldr s31, [x19]
        add x15, x15, #20
        b ReloadEnd
    Reload6:
        add x19, x15, #16
        ld1 {v8.4s}, [x15], x8
        ld1 {v9.2s}, [x19], x8
        cmp x10, #1
        beq ReloadEnd
        ld1 {v10.4s}, [x15], x8
        ld1 {v11.2s}, [x19], x8
        cmp x10, #2
        beq ReloadEnd
        ld1 {v12.4s}, [x15], x8
        ld1 {v13.2s}, [x19], x8
        cmp x10, #3
        beq ReloadEnd
        ld1 {v14.4s}, [x15], x8
        ld1 {v15.2s}, [x19], x8
        cmp x10, #4
        beq ReloadEnd
        ld1 {v16.4s}, [x15], x8
        ld1 {v17.2s}, [x19], x8
        cmp x10, #5
        beq ReloadEnd
        ld1 {v18.4s}, [x15], x8
        ld1 {v19.2s}, [x19], x8
        cmp x10, #6
        beq ReloadEnd
        ld1 {v20.4s}, [x15], x8
        ld1 {v21.2s}, [x19], x8
        cmp x10, #7
        beq ReloadEnd
        ld1 {v22.4s}, [x15], x8
        ld1 {v23.2s}, [x19], x8
        cmp x10, #8
        beq ReloadEnd
        ld1 {v24.4s}, [x15], x8
        ld1 {v25.2s}, [x19], x8
        cmp x10, #9
        beq ReloadEnd
        ld1 {v26.4s}, [x15], x8
        ld1 {v27.2s}, [x19], x8
        cmp x10, #10
        beq ReloadEnd
        ld1 {v28.4s}, [x15], x8
        ld1 {v29.2s}, [x19], x8
        cmp x10, #11
        beq ReloadEnd
        ld1 {v30.4s}, [x15], x8
        ld1 {v31.2s}, [x19]
        add x15, x15, #24
        b ReloadEnd
    Reload7:
        add x19, x15, #16
        add x16, x15, #24
        ld1 {v8.4s}, [x15], x8
        ld1 {v9.2s}, [x19], x8
        ld1 {v9.s}[2], [x16], x8
        cmp x10, #1
        beq ReloadEnd
        ld1 {v10.4s}, [x15], x8
        ld1 {v11.2s}, [x19], x8
        ld1 {v11.s}[2], [x16], x8
        cmp x10, #2
        beq ReloadEnd
        ld1 {v12.4s}, [x15], x8
        ld1 {v13.2s}, [x19], x8
        ld1 {v13.s}[2], [x16], x8
        cmp x10, #3
        beq ReloadEnd
        ld1 {v14.4s}, [x15], x8
        ld1 {v15.2s}, [x19], x8
        ld1 {v15.s}[2], [x16], x8
        cmp x10, #4
        beq ReloadEnd
        ld1 {v16.4s}, [x15], x8
        ld1 {v17.2s}, [x19], x8
        ld1 {v17.s}[2], [x16], x8
        cmp x10, #5
        beq ReloadEnd
        ld1 {v18.4s}, [x15], x8
        ld1 {v19.2s}, [x19], x8
        ld1 {v19.s}[2], [x16], x8
        cmp x10, #6
        beq ReloadEnd
        ld1 {v20.4s}, [x15], x8
        ld1 {v21.2s}, [x19], x8
        ld1 {v21.s}[2], [x16], x8
        cmp x10, #7
        beq ReloadEnd
        ld1 {v22.4s}, [x15], x8
        ld1 {v23.2s}, [x19], x8
        ld1 {v23.s}[2], [x16], x8
        cmp x10, #8
        beq ReloadEnd
        ld1 {v24.4s}, [x15], x8
        ld1 {v25.2s}, [x19], x8
        ld1 {v25.s}[2], [x16], x8
        cmp x10, #9
        beq ReloadEnd
        ld1 {v26.4s}, [x15], x8
        ld1 {v27.2s}, [x19], x8
        ld1 {v27.s}[2], [x16], x8
        cmp x10, #10
        beq ReloadEnd
        ld1 {v28.4s}, [x15], x8
        ld1 {v29.2s}, [x19], x8
        ld1 {v29.s}[2], [x16], x8
        cmp x10, #11
        beq ReloadEnd
        ld1 {v30.4s}, [x15], x8
        ld1 {v31.2s}, [x19]
        ld1 {v31.s}[2], [x16]
        add x15, x15, #28
        b ReloadEnd

    Reload8:
        ld1 {v8.4s, v9.4s}, [x15], x8
        cmp x10, #1
        beq ReloadEnd
        ld1 {v10.4s, v11.4s}, [x15], x8
        cmp x10, #2
        beq ReloadEnd
        ld1 {v12.4s, v13.4s}, [x15], x8
        cmp x10, #3
        beq ReloadEnd
        ld1 {v14.4s, v15.4s}, [x15], x8
        cmp x10, #4
        beq ReloadEnd
        ld1 {v16.4s, v17.4s}, [x15], x8
        cmp x10, #5
        beq ReloadEnd
        ld1 {v18.4s, v19.4s}, [x15], x8
        cmp x10, #6
        beq ReloadEnd
        ld1 {v20.4s, v21.4s}, [x15], x8
        cmp x10, #7
        beq ReloadEnd
        ld1 {v22.4s, v23.4s}, [x15], x8
        cmp x10, #8
        beq ReloadEnd
        ld1 {v24.4s, v25.4s}, [x15], x8
        cmp x10, #9
        beq ReloadEnd
        ld1 {v26.4s, v27.4s}, [x15], x8
        cmp x10, #10
        beq ReloadEnd
        ld1 {v28.4s, v29.4s}, [x15], x8
        cmp x10, #11
        beq ReloadEnd
        ld1 {v30.4s, v31.4s}, [x15], x8
        add x15, x15, #32
        b ReloadEnd

ReloadEnd:
  ret
  
Compute12x8Unit:
    subs x21, x21, #2
    ble Compute12x8End
    Compute12x8:
        prfm pldl1keep, [x11, #632]
        ld1 {v1.4s, v2.4s}, [x11], #32
        ld1 {v4.4s}, [x23], #16
        fmla v8.4s, v3.4s, v0.s[0]
        fmla v10.4s, v3.4s, v0.s[1]
        fmla v12.4s, v3.4s, v0.s[2]
        fmla v14.4s, v3.4s, v0.s[3]
        fmla v16.4s, v3.4s, v1.s[0]
        fmla v18.4s, v3.4s, v1.s[1]
        fmla v20.4s, v3.4s, v1.s[2]
        fmla v22.4s, v3.4s, v1.s[3]
        fmla v24.4s, v3.4s, v2.s[0]
        fmla v26.4s, v3.4s, v2.s[1]
        fmla v28.4s, v3.4s, v2.s[2]
        fmla v30.4s, v3.4s, v2.s[3]
        prfm pldl1strm, [x23, #632]
        ld1 {v3.4s}, [x23], #16
        fmla v9.4s, v4.4s, v0.s[0]
        fmla v11.4s, v4.4s, v0.s[1]
        fmla v13.4s, v4.4s, v0.s[2]
        fmla v15.4s, v4.4s, v0.s[3]
        ld1 {v0.4s}, [x11], #16
        fmla v17.4s, v4.4s, v1.s[0]
        fmla v19.4s, v4.4s, v1.s[1]
        fmla v21.4s, v4.4s, v1.s[2]
        fmla v23.4s, v4.4s, v1.s[3]
        fmla v25.4s, v4.4s, v2.s[0]
        fmla v27.4s, v4.4s, v2.s[1]
        fmla v29.4s, v4.4s, v2.s[2]
        fmla v31.4s, v4.4s, v2.s[3]

        prfm pldl1keep, [x11, #632]
        ld1 {v1.4s, v2.4s}, [x11], #32
        ld1 {v4.4s}, [x23], #16
        fmla v8.4s, v3.4s, v0.s[0]
        fmla v10.4s, v3.4s, v0.s[1]
        fmla v12.4s, v3.4s, v0.s[2]
        fmla v14.4s, v3.4s, v0.s[3]
        fmla v16.4s, v3.4s, v1.s[0]
        fmla v18.4s, v3.4s, v1.s[1]
        fmla v20.4s, v3.4s, v1.s[2]
        fmla v22.4s, v3.4s, v1.s[3]
        fmla v24.4s, v3.4s, v2.s[0]
        fmla v26.4s, v3.4s, v2.s[1]
        fmla v28.4s, v3.4s, v2.s[2]
        fmla v30.4s, v3.4s, v2.s[3]
        prfm pldl1strm, [x23, #632]
        ld1 {v3.4s}, [x23], #16
        fmla v9.4s, v4.4s, v0.s[0]
        fmla v11.4s, v4.4s, v0.s[1]
        fmla v13.4s, v4.4s, v0.s[2]
        fmla v15.4s, v4.4s, v0.s[3]
        ld1 {v0.4s}, [x11], #16
        fmla v17.4s, v4.4s, v1.s[0]
        fmla v19.4s, v4.4s, v1.s[1]
        fmla v21.4s, v4.4s, v1.s[2]
        fmla v23.4s, v4.4s, v1.s[3]
        fmla v25.4s, v4.4s, v2.s[0]
        fmla v27.4s, v4.4s, v2.s[1]
        fmla v29.4s, v4.4s, v2.s[2]
        fmla v31.4s, v4.4s, v2.s[3]

        subs x21, x21, #2
        bgt Compute12x8

    Compute12x8End:
        cbnz x21, Compute12x8End1
        prfm pldl1keep, [x11, #632]
        ld1 {v1.4s, v2.4s}, [x11], #32
        ld1 {v4.4s}, [x23], #16
        fmla v8.4s, v3.4s, v0.s[0]
        fmla v10.4s, v3.4s, v0.s[1]
        fmla v12.4s, v3.4s, v0.s[2]
        fmla v14.4s, v3.4s, v0.s[3]
        fmla v16.4s, v3.4s, v1.s[0]
        fmla v18.4s, v3.4s, v1.s[1]
        fmla v20.4s, v3.4s, v1.s[2]
        fmla v22.4s, v3.4s, v1.s[3]
        fmla v24.4s, v3.4s, v2.s[0]
        fmla v26.4s, v3.4s, v2.s[1]
        fmla v28.4s, v3.4s, v2.s[2]
        fmla v30.4s, v3.4s, v2.s[3]
        prfm pldl1strm, [x23, #632]
        ld1 {v3.4s}, [x23], #16
        fmla v9.4s, v4.4s, v0.s[0]
        fmla v11.4s, v4.4s, v0.s[1]
        fmla v13.4s, v4.4s, v0.s[2]
        fmla v15.4s, v4.4s, v0.s[3]
        ld1 {v0.4s}, [x11], #16
        fmla v17.4s, v4.4s, v1.s[0]
        fmla v19.4s, v4.4s, v1.s[1]
        fmla v21.4s, v4.4s, v1.s[2]
        fmla v23.4s, v4.4s, v1.s[3]
        fmla v25.4s, v4.4s, v2.s[0]
        fmla v27.4s, v4.4s, v2.s[1]
        fmla v29.4s, v4.4s, v2.s[2]
        fmla v31.4s, v4.4s, v2.s[3]
    Compute12x8End1:
        ld1 {v1.4s, v2.4s}, [x11]
        ld1 {v4.4s}, [x23]
        fmla v8.4s, v3.4s, v0.s[0]
        fmla v10.4s, v3.4s, v0.s[1]
        fmla v12.4s, v3.4s, v0.s[2]
        fmla v14.4s, v3.4s, v0.s[3]
        fmla v16.4s, v3.4s, v1.s[0]
        fmla v18.4s, v3.4s, v1.s[1]
        fmla v20.4s, v3.4s, v1.s[2]
        fmla v22.4s, v3.4s, v1.s[3]
        fmla v24.4s, v3.4s, v2.s[0]
        fmla v26.4s, v3.4s, v2.s[1]
        fmla v28.4s, v3.4s, v2.s[2]
        fmla v30.4s, v3.4s, v2.s[3]
        fmla v9.4s, v4.4s, v0.s[0]
        fmla v11.4s, v4.4s, v0.s[1]
        fmla v13.4s, v4.4s, v0.s[2]
        fmla v15.4s, v4.4s, v0.s[3]
        fmla v17.4s, v4.4s, v1.s[0]
        fmla v19.4s, v4.4s, v1.s[1]
        fmla v21.4s, v4.4s, v1.s[2]
        fmla v23.4s, v4.4s, v1.s[3]
        fmla v25.4s, v4.4s, v2.s[0]
        fmla v27.4s, v4.4s, v2.s[1]
        fmla v29.4s, v4.4s, v2.s[2]
        fmla v31.4s, v4.4s, v2.s[3]
        ret

Compute12x4Unit:
    subs x21, x21, #2
    ble Compute12x4End
    Compute12x4:
        prfm pldl1keep, [x11, #632]
        ld1 {v1.4s, v2.4s}, [x11], #32
        add x23, x23, #16
        fmla v8.4s, v3.4s, v0.s[0]
        fmla v10.4s, v3.4s, v0.s[1]
        fmla v12.4s, v3.4s, v0.s[2]
        fmla v14.4s, v3.4s, v0.s[3]
        ld1 {v0.4s}, [x11], #16
        fmla v16.4s, v3.4s, v1.s[0]
        fmla v18.4s, v3.4s, v1.s[1]
        fmla v20.4s, v3.4s, v1.s[2]
        fmla v22.4s, v3.4s, v1.s[3]
        fmla v24.4s, v3.4s, v2.s[0]
        fmla v26.4s, v3.4s, v2.s[1]
        fmla v28.4s, v3.4s, v2.s[2]
        fmla v30.4s, v3.4s, v2.s[3]
        prfm pldl1strm, [x23, #632]
        ld1 {v3.4s}, [x23], #16

        prfm pldl1keep, [x11, #632]
        ld1 {v1.4s, v2.4s}, [x11], #32
        add x23, x23, #16
        fmla v8.4s, v3.4s, v0.s[0]
        fmla v10.4s, v3.4s, v0.s[1]
        fmla v12.4s, v3.4s, v0.s[2]
        fmla v14.4s, v3.4s, v0.s[3]
        ld1 {v0.4s}, [x11], #16
        fmla v16.4s, v3.4s, v1.s[0]
        fmla v18.4s, v3.4s, v1.s[1]
        fmla v20.4s, v3.4s, v1.s[2]
        fmla v22.4s, v3.4s, v1.s[3]
        fmla v24.4s, v3.4s, v2.s[0]
        fmla v26.4s, v3.4s, v2.s[1]
        fmla v28.4s, v3.4s, v2.s[2]
        fmla v30.4s, v3.4s, v2.s[3]
        prfm pldl1strm, [x23, #632]
        ld1 {v3.4s}, [x23], #16

        subs x21, x21, #2
        bgt Compute12x4

    Compute12x4End:
        cbnz x21, Compute12x4End1
        prfm pldl1keep, [x11, #632]
        ld1 {v1.4s, v2.4s}, [x11], #32
        add x23, x23, #16
        fmla v8.4s, v3.4s, v0.s[0]
        fmla v10.4s, v3.4s, v0.s[1]
        fmla v12.4s, v3.4s, v0.s[2]
        fmla v14.4s, v3.4s, v0.s[3]
        ld1 {v0.4s}, [x11], #16
        fmla v16.4s, v3.4s, v1.s[0]
        fmla v18.4s, v3.4s, v1.s[1]
        fmla v20.4s, v3.4s, v1.s[2]
        fmla v22.4s, v3.4s, v1.s[3]
        fmla v24.4s, v3.4s, v2.s[0]
        fmla v26.4s, v3.4s, v2.s[1]
        fmla v28.4s, v3.4s, v2.s[2]
        fmla v30.4s, v3.4s, v2.s[3]
        prfm pldl1strm, [x23, #632]
        ld1 {v3.4s}, [x23], #16
    Compute12x4End1:
        ld1 {v1.4s, v2.4s}, [x11]
        fmla v8.4s, v3.4s, v0.s[0]
        fmla v10.4s, v3.4s, v0.s[1]
        fmla v12.4s, v3.4s, v0.s[2]
        fmla v14.4s, v3.4s, v0.s[3]
        fmla v16.4s, v3.4s, v1.s[0]
        fmla v18.4s, v3.4s, v1.s[1]
        fmla v20.4s, v3.4s, v1.s[2]
        fmla v22.4s, v3.4s, v1.s[3]
        fmla v24.4s, v3.4s, v2.s[0]
        fmla v26.4s, v3.4s, v2.s[1]
        fmla v28.4s, v3.4s, v2.s[2]
        fmla v30.4s, v3.4s, v2.s[3]
        ret

Compute8x8Unit:
    subs x21, x21, #2
    ble Compute8x8End
    Compute8x8:
        prfm pldl1keep, [x11, #632]
        ld1 {v1.4s}, [x11]
        add x11, x11, #32
        ld1 {v4.4s}, [x23], #16
        fmla v8.4s, v3.4s, v0.s[0]
        fmla v10.4s, v3.4s, v0.s[1]
        fmla v12.4s, v3.4s, v0.s[2]
        fmla v14.4s, v3.4s, v0.s[3]
        fmla v16.4s, v3.4s, v1.s[0]
        fmla v18.4s, v3.4s, v1.s[1]
        fmla v20.4s, v3.4s, v1.s[2]
        fmla v22.4s, v3.4s, v1.s[3]
        prfm pldl1strm, [x23, #632]
        ld1 {v3.4s}, [x23], #16
        fmla v9.4s, v4.4s, v0.s[0]
        fmla v11.4s, v4.4s, v0.s[1]
        fmla v13.4s, v4.4s, v0.s[2]
        fmla v15.4s, v4.4s, v0.s[3]
        ld1 {v0.4s}, [x11], #16
        fmla v17.4s, v4.4s, v1.s[0]
        fmla v19.4s, v4.4s, v1.s[1]
        fmla v21.4s, v4.4s, v1.s[2]
        fmla v23.4s, v4.4s, v1.s[3]

        prfm pldl1keep, [x11, #632]
        ld1 {v1.4s}, [x11]
        add x11, x11, #32
        ld1 {v4.4s}, [x23], #16
        fmla v8.4s, v3.4s, v0.s[0]
        fmla v10.4s, v3.4s, v0.s[1]
        fmla v12.4s, v3.4s, v0.s[2]
        fmla v14.4s, v3.4s, v0.s[3]
        fmla v16.4s, v3.4s, v1.s[0]
        fmla v18.4s, v3.4s, v1.s[1]
        fmla v20.4s, v3.4s, v1.s[2]
        fmla v22.4s, v3.4s, v1.s[3]
        prfm pldl1strm, [x23, #632]
        ld1 {v3.4s}, [x23], #16
        fmla v9.4s, v4.4s, v0.s[0]
        fmla v11.4s, v4.4s, v0.s[1]
        fmla v13.4s, v4.4s, v0.s[2]
        fmla v15.4s, v4.4s, v0.s[3]
        ld1 {v0.4s}, [x11], #16
        fmla v17.4s, v4.4s, v1.s[0]
        fmla v19.4s, v4.4s, v1.s[1]
        fmla v21.4s, v4.4s, v1.s[2]
        fmla v23.4s, v4.4s, v1.s[3]

        subs x21, x21, #2
        bgt Compute8x8

    Compute8x8End:
        cbnz x21, Compute8x8End1
        prfm pldl1keep, [x11, #632]
        ld1 {v1.4s}, [x11]
        add x11, x11, #32
        ld1 {v4.4s}, [x23], #16
        fmla v8.4s, v3.4s, v0.s[0]
        fmla v10.4s, v3.4s, v0.s[1]
        fmla v12.4s, v3.4s, v0.s[2]
        fmla v14.4s, v3.4s, v0.s[3]
        fmla v16.4s, v3.4s, v1.s[0]
        fmla v18.4s, v3.4s, v1.s[1]
        fmla v20.4s, v3.4s, v1.s[2]
        fmla v22.4s, v3.4s, v1.s[3]
        prfm pldl1strm, [x23, #632]
        ld1 {v3.4s}, [x23], #16
        fmla v9.4s, v4.4s, v0.s[0]
        fmla v11.4s, v4.4s, v0.s[1]
        fmla v13.4s, v4.4s, v0.s[2]
        fmla v15.4s, v4.4s, v0.s[3]
        ld1 {v0.4s}, [x11], #16
        fmla v17.4s, v4.4s, v1.s[0]
        fmla v19.4s, v4.4s, v1.s[1]
        fmla v21.4s, v4.4s, v1.s[2]
        fmla v23.4s, v4.4s, v1.s[3]
    Compute8x8End1:
        ld1 {v1.4s}, [x11]
        ld1 {v4.4s}, [x23]
        fmla v8.4s, v3.4s, v0.s[0]
        fmla v10.4s, v3.4s, v0.s[1]
        fmla v12.4s, v3.4s, v0.s[2]
        fmla v14.4s, v3.4s, v0.s[3]
        fmla v16.4s, v3.4s, v1.s[0]
        fmla v18.4s, v3.4s, v1.s[1]
        fmla v20.4s, v3.4s, v1.s[2]
        fmla v22.4s, v3.4s, v1.s[3]
        fmla v9.4s, v4.4s, v0.s[0]
        fmla v11.4s, v4.4s, v0.s[1]
        fmla v13.4s, v4.4s, v0.s[2]
        fmla v15.4s, v4.4s, v0.s[3]
        fmla v17.4s, v4.4s, v1.s[0]
        fmla v19.4s, v4.4s, v1.s[1]
        fmla v21.4s, v4.4s, v1.s[2]
        fmla v23.4s, v4.4s, v1.s[3]
        ret

Compute8x4Unit:
    subs x21, x21, #2
    ble Compute8x4End
    Compute8x4:
        prfm pldl1keep, [x11, #632]
        ld1 {v1.4s}, [x11]
        add x23, x23, #16
        add x11, x11, #32
        fmla v8.4s, v3.4s, v0.s[0]
        fmla v10.4s, v3.4s, v0.s[1]
        fmla v12.4s, v3.4s, v0.s[2]
        fmla v14.4s, v3.4s, v0.s[3]
        ld1 {v0.4s}, [x11], #16
        fmla v16.4s, v3.4s, v1.s[0]
        fmla v18.4s, v3.4s, v1.s[1]
        fmla v20.4s, v3.4s, v1.s[2]
        fmla v22.4s, v3.4s, v1.s[3]
        prfm pldl1strm, [x23, #632]
        ld1 {v3.4s}, [x23], #16

        prfm pldl1keep, [x11, #632]
        ld1 {v1.4s}, [x11]
        add x23, x23, #16
        add x11, x11, #32
        fmla v8.4s, v3.4s, v0.s[0]
        fmla v10.4s, v3.4s, v0.s[1]
        fmla v12.4s, v3.4s, v0.s[2]
        fmla v14.4s, v3.4s, v0.s[3]
        ld1 {v0.4s}, [x11], #16
        fmla v16.4s, v3.4s, v1.s[0]
        fmla v18.4s, v3.4s, v1.s[1]
        fmla v20.4s, v3.4s, v1.s[2]
        fmla v22.4s, v3.4s, v1.s[3]
        prfm pldl1strm, [x23, #632]
        ld1 {v3.4s}, [x23], #16

        subs x21, x21, #2
        bgt Compute8x4

    Compute8x4End:
        cbnz x21, Compute8x4End1
        prfm pldl1keep, [x11, #632]
        ld1 {v1.4s}, [x11]
        add x23, x23, #16
        add x11, x11, #32
        fmla v8.4s, v3.4s, v0.s[0]
        fmla v10.4s, v3.4s, v0.s[1]
        fmla v12.4s, v3.4s, v0.s[2]
        fmla v14.4s, v3.4s, v0.s[3]
        ld1 {v0.4s}, [x11], #16
        fmla v16.4s, v3.4s, v1.s[0]
        fmla v18.4s, v3.4s, v1.s[1]
        fmla v20.4s, v3.4s, v1.s[2]
        fmla v22.4s, v3.4s, v1.s[3]
        prfm pldl1strm, [x23, #632]
        ld1 {v3.4s}, [x23], #16
    Compute8x4End1:
        ld1 {v1.4s}, [x11]
        fmla v8.4s, v3.4s, v0.s[0]
        fmla v10.4s, v3.4s, v0.s[1]
        fmla v12.4s, v3.4s, v0.s[2]
        fmla v14.4s, v3.4s, v0.s[3]
        fmla v16.4s, v3.4s, v1.s[0]
        fmla v18.4s, v3.4s, v1.s[1]
        fmla v20.4s, v3.4s, v1.s[2]
        fmla v22.4s, v3.4s, v1.s[3]
        ret

Compute4x8Unit:
    subs x21, x21, #2
    ble Compute4x8End
    Compute4x8:
        prfm pldl1keep, [x11, #632]
        add x11, x11, #32
        ld1 {v4.4s}, [x23], #16
        fmla v8.4s, v3.4s, v0.s[0]
        fmla v10.4s, v3.4s, v0.s[1]
        fmla v12.4s, v3.4s, v0.s[2]
        fmla v14.4s, v3.4s, v0.s[3]
        prfm pldl1strm, [x23, #632]
        ld1 {v3.4s}, [x23], #16
        fmla v9.4s, v4.4s, v0.s[0]
        fmla v11.4s, v4.4s, v0.s[1]
        fmla v13.4s, v4.4s, v0.s[2]
        fmla v15.4s, v4.4s, v0.s[3]
        ld1 {v0.4s}, [x11], #16

        prfm pldl1keep, [x11, #632]
        add x11, x11, #32
        ld1 {v4.4s}, [x23], #16
        fmla v8.4s, v3.4s, v0.s[0]
        fmla v10.4s, v3.4s, v0.s[1]
        fmla v12.4s, v3.4s, v0.s[2]
        fmla v14.4s, v3.4s, v0.s[3]
        prfm pldl1strm, [x23, #632]
        ld1 {v3.4s}, [x23], #16
        fmla v9.4s, v4.4s, v0.s[0]
        fmla v11.4s, v4.4s, v0.s[1]
        fmla v13.4s, v4.4s, v0.s[2]
        fmla v15.4s, v4.4s, v0.s[3]
        ld1 {v0.4s}, [x11], #16

        subs x21, x21, #2
        bgt Compute4x8

    Compute4x8End:
        cbnz x21, Compute4x8End1
        prfm pldl1keep, [x11, #632]
        add x11, x11, #32
        ld1 {v4.4s}, [x23], #16
        fmla v8.4s, v3.4s, v0.s[0]
        fmla v10.4s, v3.4s, v0.s[1]
        fmla v12.4s, v3.4s, v0.s[2]
        fmla v14.4s, v3.4s, v0.s[3]
        prfm pldl1strm, [x23, #632]
        ld1 {v3.4s}, [x23], #16
        fmla v9.4s, v4.4s, v0.s[0]
        fmla v11.4s, v4.4s, v0.s[1]
        fmla v13.4s, v4.4s, v0.s[2]
        fmla v15.4s, v4.4s, v0.s[3]
        ld1 {v0.4s}, [x11], #16
    Compute4x8End1:
        ld1 {v4.4s}, [x23]
        fmla v8.4s, v3.4s, v0.s[0]
        fmla v10.4s, v3.4s, v0.s[1]
        fmla v12.4s, v3.4s, v0.s[2]
        fmla v14.4s, v3.4s, v0.s[3]
        fmla v9.4s, v4.4s, v0.s[0]
        fmla v11.4s, v4.4s, v0.s[1]
        fmla v13.4s, v4.4s, v0.s[2]
        fmla v15.4s, v4.4s, v0.s[3]
        ret

Compute4x4Unit:
    subs x21, x21, #2
    ble Compute4x4End
    Compute4x4:
        prfm pldl1keep, [x11, #632]
        add x23, x23, #16
        add x11, x11, #32
        fmla v8.4s, v3.4s, v0.s[0]
        fmla v10.4s, v3.4s, v0.s[1]
        fmla v12.4s, v3.4s, v0.s[2]
        fmla v14.4s, v3.4s, v0.s[3]
        prfm pldl1strm, [x23, #632]
        ld1 {v3.4s}, [x23], #16
        ld1 {v0.4s}, [x11], #16

        prfm pldl1keep, [x11, #632]
        add x23, x23, #16
        add x11, x11, #32
        fmla v8.4s, v3.4s, v0.s[0]
        fmla v10.4s, v3.4s, v0.s[1]
        fmla v12.4s, v3.4s, v0.s[2]
        fmla v14.4s, v3.4s, v0.s[3]
        prfm pldl1strm, [x23, #632]
        ld1 {v3.4s}, [x23], #16
        ld1 {v0.4s}, [x11], #16

        subs x21, x21, #2
        bgt Compute4x4

    Compute4x4End:
        cbnz x21, Compute4x4End1
        prfm pldl1keep, [x11, #632]
        add x23, x23, #16
        add x11, x11, #32
        fmla v8.4s, v3.4s, v0.s[0]
        fmla v10.4s, v3.4s, v0.s[1]
        fmla v12.4s, v3.4s, v0.s[2]
        fmla v14.4s, v3.4s, v0.s[3]
        prfm pldl1strm, [x23, #632]
        ld1 {v3.4s}, [x23], #16
        ld1 {v0.4s}, [x11], #16
    Compute4x4End1:
        fmla v8.4s, v3.4s, v0.s[0]
        fmla v10.4s, v3.4s, v0.s[1]
        fmla v12.4s, v3.4s, v0.s[2]
        fmla v14.4s, v3.4s, v0.s[3]
        ret

End:
  sub sp, sp, #224
  ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
  ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
  ldp x19, x20, [sp], #16
  ldp x21, x22, [sp], #16
  ldp x23, x24, [sp], #16
  ldp x25, x26, [sp], #16
  ldp x27, x28, [sp], #16
  ldp x29, x30, [sp], #16
  ret
#endif
