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

// void MatVecMulPackFp32(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int col)
// x0: a
// x1: b
// x2: c
// x3: bias
// w4: act_type
// w5: depth
// w6: col

asm_default_function MatVecMulPackFp32
    sub sp, sp, #16
    stp x29, x30, [sp], #16
  
    dup v1.2d, xzr
    mov w7, #6
    dup v2.4s, w7
    scvtf v2.4s, v2.4s
    subs w6, w6, #8
    blt Loop1xNStart
    Loop1x8Start:
        bl Compute1x8Unit
        st1 {v24.4s, v25.4s}, [x2], #32
        subs w6, w6, #8
        bge Loop1x8Start
                
    Loop1xNStart:
        add w6, w6, #8
        cbz w6, End
        subs w6, w6, #4
        ble Loop1x4Start
        bl Compute1x8Unit
        st1 {v24.4s}, [x2], #16
        st1 {v25.s}[0], [x2], #4
        cmp w6, #1
        beq End
        st1 {v25.s}[1], [x2], #4
        cmp w6, #2
        beq End
        st1 {v25.s}[2], [x2]
        b End
            
    Loop1x4Start:
        add w6, w6, #4
        cbz w6, End
        bl Compute1x4Unit
        st1 {v24.s}[0], [x2], #4
        cmp w6, #1
        beq End
        st1 {v24.s}[1], [x2], #4
        cmp w6, #2
        beq End
        st1 {v24.s}[2], [x2], #4
        cmp w6, #3
        beq End
        st1 {v24.s}[3], [x2], #4
        b End
        
    Compute1x8Unit:
        mov x7, x0     // reload a-ptr
        mov w8, w5     // reset depth
        dup v24.2d, xzr
        dup v25.2d, xzr
        dup v26.2d, xzr
        dup v27.2d, xzr
        dup v28.2d, xzr
        dup v29.2d, xzr
        dup v30.2d, xzr
        dup v31.2d, xzr
        cbz x3, Compute1x8Enter
        ld1 {v24.4s, v25.4s}, [x3], #32
        Compute1x8Enter:
            subs w8, w8, #4
            blt Compute1x8Tail
            Compute1x8:
                ld1 {v0.4s}, [x7], #16
                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x1], #64
                ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x1], #64
                fmla v24.4s, v16.4s, v0.s[0]
                fmla v25.4s, v17.4s, v0.s[0]
                fmla v26.4s, v18.4s, v0.s[1]
                fmla v27.4s, v19.4s, v0.s[1]
                fmla v28.4s, v20.4s, v0.s[2]
                fmla v29.4s, v21.4s, v0.s[2]
                fmla v30.4s, v22.4s, v0.s[3]
                fmla v31.4s, v23.4s, v0.s[3]
                subs w8, w8, #4
                bge Compute1x8
            Compute1x8Tail:
                add w8, w8, #4
                cbz w8, Compute1x8UnionTail
                Compute1x8DepthTail:
                    ld1 {v0.s}[0], [x7], #4
                    ld1 {v16.4s, v17.4s}, [x1], #32
                    fmla v24.4s, v16.4s, v0.s[0]
                    fmla v25.4s, v17.4s, v0.s[0]
                    subs w8, w8, #1
                    bgt Compute1x8DepthTail
                Compute1x8UnionTail:
                    fadd v24.4s, v24.4s, v26.4s
                    fadd v25.4s, v25.4s, v27.4s
                    fadd v28.4s, v28.4s, v30.4s
                    fadd v29.4s, v29.4s, v31.4s
                    fadd v24.4s, v24.4s, v28.4s
                    fadd v25.4s, v25.4s, v29.4s
                Act1x8:
                    cmp x4, #3
                    beq Relu61x8
                    cmp x4, #1
                    beq Relu1x8
                    b Return1x8
                Relu61x8:
                    fmin v24.4s, v24.4s, v2.4s
                    fmin v25.4s, v25.4s, v2.4s
                    fmax v24.4s, v24.4s, v1.4s
                    fmax v25.4s, v25.4s, v1.4s
                    b Return1x8
                Relu1x8:
                    fmax v24.4s, v24.4s, v1.4s
                    fmax v25.4s, v25.4s, v1.4s
                Return1x8:
                    ret
  
    Compute1x4Unit:
        mov x7, x0     // reload a-ptr
        mov w8, w5     // reset depth
        dup v24.2d, xzr
        dup v26.2d, xzr
        dup v28.2d, xzr
        dup v30.2d, xzr
        cbz x3, Compute1x4Enter
        ld1 {v24.4s}, [x3]
        Compute1x4Enter:
            subs w8, w8, #4
            blt Compute1x4Tail
            Compute1x4:
                ld1 {v0.4s}, [x7], #16
                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x1], #64
                ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x1], #64
                fmla v24.4s, v16.4s, v0.s[0]
                fmla v26.4s, v18.4s, v0.s[1]
                fmla v28.4s, v20.4s, v0.s[2]
                fmla v30.4s, v22.4s, v0.s[3]
                subs w8, w8, #4
                bge Compute1x4
            Compute1x4Tail:
                add w8, w8, #4
                cbz w8, Compute1x4UnionTail
                Compute1x4DepthTail:
                    ld1 {v0.s}[0], [x7], #4
                    ld1 {v16.4s}, [x1]
                    add x1, x1, #32
                    fmla v24.4s, v16.4s, v0.s[0]
                    subs w8, w8, #1
                    bgt Compute1x4DepthTail
                Compute1x4UnionTail:
                    fadd v24.4s, v24.4s, v26.4s
                    fadd v28.4s, v28.4s, v30.4s
                    fadd v24.4s, v24.4s, v28.4s
                Act1x4:
                    cmp x4, #3
                    beq Relu61x4
                    cmp x4, #1
                    beq Relu1x4
                    b Return1x4
                Relu61x4:
                    fmin v24.4s, v24.4s, v2.4s
                    fmax v24.4s, v24.4s, v1.4s
                    b Return1x8
                Relu1x4:
                    fmax v24.4s, v24.4s, v1.4s
                Return1x4:
                    ret
  
    End:
        sub sp, sp, #16
        ldp x29, x30, [sp], #16
        ret
#endif
