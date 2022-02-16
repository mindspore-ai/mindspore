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
#ifdef ENABLE_ARM32
#include "nnacl/assembly_global.h"
    .text
    .align 5
    .global MatMul12x8A32Fp16
#ifndef __APPLE__
    .type MatMul12x8A32Fp16, %function
#endif

// void MatMul12x8A32Fp16(const float16_t *a, const float16_t *b, float16_t *dst, const float16_t *bias, ActType act_type,
//                     int deep, int row, int col, int stride, bool write_mode);
// r0: a
// r1: b
// r2: dst
// r3: bias
// #4: depth
// #8: row
// #12: col
// #16: stride
// #20: writeNhwc/writeWino

asm_function MatMul12x8A32Fp16
    // r13(sp) and r15(pc) can not be used!!
    // r9 r4 is tmp register
    // r4-r8 and q4-q7 must be saved according to https://static.docs.arm.com/ihi0042/i/aapcs32.pdf
    push {r3-r11, lr}
    vpush {q4-q7}
    add sp, sp, #104

    ldr r5, [sp, #4]
    ldr r6, [sp, #8]
    ldr r7, [sp, #12]
    ldr r8, [sp, #16]
    ldr lr, [sp, #20]

    mov r10, r1  // b
    mov r11, r0  // a
    mov r12, r2  // dst

    cmp lr, #2
    bne NoWinograd
    mul r4, r8, r7  // stride * col
    add r4, r4, r4  // r4 * sizeof(float16_t)
    mov r9, #16
    mul r9, r8, r9  // stride * 8 * sizeof(float16_t)
NoWinograd:
    add r8, r8, r8  // stride * sizeof(float16_t)

a       .req    r0
weight  .req    r1
dst     .req    r2
bias    .req    r3
depth   .req    r5
row     .req    r6
col     .req    r7
stride  .req    r8
b_tmp   .req    r10
a_tmp   .req    r11
dst_tmp .req    r12

.macro STORE_12x8 p1
    vst1.16 {\p1}, [dst]
    add dst, dst, stride
.endm

.macro STORE_12x7 p1, p2, p3
    add r4, dst, #8
    add r9, dst, #12
    vst1.16 {\p1}, [dst]
    vst1.32 {\p2}, [r4]
    vst1.16 {\p3}, [r9]
    add dst, dst, stride
.endm

.macro STORE_12x6 p1, p2
    add r4, dst, #8
    vst1.16 {\p1}, [dst]
    vst1.32 {\p2}, [r4]
    add dst, dst, stride
.endm

.macro STORE_12x5 p1, p2
    add r4, dst, #8
    vst1.16 {\p1}, [dst]
    vst1.16 {\p2}, [r4]
    add dst, dst, stride
.endm

.macro STORE_12x4 p1
    vst1.16 {\p1}, [dst]
    add dst, dst, stride
.endm

.macro STORE_12x3 p1, p2
    add r4, dst, #4
    vst1.32 {\p1}, [dst]
    vst1.16 {\p2}, [r4]
    add dst, dst, stride
.endm

.macro STORE_12x2 p1
    vst1.32 {\p1}, [dst]
    add dst, dst, stride
.endm

.macro STORE_12x1 p1
    vst1.16 {\p1}, [dst]
    add dst, dst, stride
.endm

.macro STORE_C8 p1, p2
    vst1.16 {\p1}, [dst]
    cmp row, \p2
    add dst, dst, stride
    beq WriteEnd
.endm

.macro STORE_C7 p1, p2, p3, p4
    add r4, dst, #8
    add r9, dst, #12
    vst1.16 {\p1}, [dst]
    vst1.32 {\p2}, [r4]
    vst1.16 {\p3}, [r9]
    add dst, dst, stride
    cmp row, \p4
    beq WriteEnd
.endm

.macro STORE_C6 p1, p2, p3
    add r4, dst, #8
    vst1.16 {\p1}, [dst]
    vst1.32 {\p2}, [r4]
    add dst, dst, stride
    cmp row, \p3
    beq WriteEnd
.endm

.macro STORE_C5 p1, p2, p3
    add r4, dst, #8
    vst1.16 {\p1}, [dst]
    vst1.16 {\p2}, [r4]
    add dst, dst, stride
    cmp row, \p3
    beq WriteEnd
.endm

.macro STORE_C4 p1, p2
    vst1.16 {\p1}, [dst]
    cmp row, \p2
    add dst, dst, stride
    beq WriteEnd
.endm

.macro STORE_C3 p1, p2, p3
    add r4, dst, #4
    vst1.32 {\p1}, [dst]
    vst1.16 {\p2}, [r4]
    add dst, dst, stride
    cmp row, \p3
    beq WriteEnd
.endm

.macro STORE_C2 p1, p2
    vst1.32 {\p1}, [dst]
    add dst, dst, stride
    cmp row, \p2
    beq WriteEnd
.endm

.macro STORE_C1 p1, p2
    vst1.16 {\p1}, [dst]
    add dst, dst, stride
    cmp row, \p2
    beq WriteEnd
.endm

LoopRow12:
    ldr bias, [sp, #-40]
    LoopCol8:
        mov dst, dst_tmp
        mov a, a_tmp
        ldr depth, [sp, #4]
        veor q4, q4, q4
        veor q5, q5, q5
        veor q6, q6, q6
        veor q7, q7, q7
        veor q8, q8, q8
        veor q9, q9, q9
        veor q10, q10, q10
        veor q11, q11, q11
        veor q12, q12, q12
        veor q13, q13, q13
        veor q14, q14, q14
        veor q15, q15, q15
        LoopDepth:
            vld1.16 {q0, d2}, [a]!
            vld1.16 {q2}, [weight]!
            vmla.f16 q4, q2, d0[0]
            vmla.f16 q5, q2, d0[1]
            vmla.f16 q6, q2, d0[2]
            vmla.f16 q7, q2, d0[3]
            vmla.f16 q8, q2, d1[0]
            vmla.f16 q9, q2, d1[1]
            vmla.f16 q10, q2, d1[2]
            vmla.f16 q11, q2, d1[3]
            vmla.f16 q12, q2, d2[0]
            vmla.f16 q13, q2, d2[1]
            vmla.f16 q14, q2, d2[2]
            vmla.f16 q15, q2, d2[3]

            subs depth, depth, #1
            bne LoopDepth

        Bias:
            cmp bias, #0
            beq Activation
            vld1.16 {q0}, [bias]!
            vadd.f16 q4, q4, q0
            vadd.f16 q5, q5, q0
            vadd.f16 q6, q6, q0
            vadd.f16 q7, q7, q0
            vadd.f16 q8, q8, q0
            vadd.f16 q9, q9, q0
            vadd.f16 q10, q10, q0
            vadd.f16 q11, q11, q0
            vadd.f16 q12, q12, q0
            vadd.f16 q13, q13, q0
            vadd.f16 q14, q14, q0
            vadd.f16 q15, q15, q0

        Activation:
            ldr lr, [sp]
            cmp lr, #3
            beq Relu6
            cmp lr, #1
            beq Relu
            b Write

        Relu6:
            vmov.i16 q2, #0x4600
            vadd.f16 q4, q4, q2
            vadd.f16 q5, q5, q2
            vadd.f16 q6, q6, q2
            vadd.f16 q7, q7, q2
            vmin.f16 q8, q8, q2
            vmin.f16 q9, q9, q2
            vmin.f16 q10, q10, q2
            vmin.f16 q11, q11, q2
            vmin.f16 q12, q12, q2
            vmin.f16 q13, q13, q2
            vmin.f16 q14, q14, q2
            vmin.f16 q15, q15, q2

        Relu:
            veor q3, q3, q3
            vmax.f16 q4, q4, q3
            vmax.f16 q5, q5, q3
            vmax.f16 q6, q6, q3
            vmax.f16 q7, q7, q3
            vmax.f16 q8, q8, q3
            vmax.f16 q9, q9, q3
            vmax.f16 q10, q10, q3
            vmax.f16 q11, q11, q3
            vmax.f16 q12, q12, q3
            vmax.f16 q13, q13, q3
            vmax.f16 q14, q14, q3
            vmax.f16 q15, q15, q3

        Write:
            ldr lr, [sp, #20]
            cmp lr, #2
            beq WriteWinograd
            cmp row, #12
            bge Write12xCol
            b WriteRowxCol

        WriteWinograd:
            vst1.16 {q4}, [dst]
            add dst, dst, r4
            vst1.16 {q5}, [dst]
            add dst, dst, r4
            vst1.16 {q6}, [dst]
            add dst, dst, r4
            vst1.16 {q7}, [dst]
            add dst, dst, r4
            vst1.16 {q8}, [dst]
            add dst, dst, r4
            vst1.16 {q9}, [dst]
            add dst, dst, r4
            vst1.16 {q10}, [dst]
            add dst, dst, r4
            vst1.16 {q11}, [dst]
            add dst, dst, r4
            vst1.16 {q12}, [dst]
            add dst, dst, r4
            vst1.16 {q13}, [dst]
            add dst, dst, r4
            vst1.16 {q14}, [dst]
            add dst, dst, r4
            vst1.16 {q15}, [dst]
            add dst_tmp, dst_tmp, r9
            b WriteEnd
        Write12xCol:
            cmp col, #8
            bge Write12x8
            cmp col, #1
            beq Write12x1
            cmp col, #2
            beq Write12x2
            cmp col, #3
            beq Write12x3
            cmp col, #4
            beq Write12x4
            cmp col, #5
            beq Write12x5
            cmp col, #6
            beq Write12x6
            b Write12x7

        WriteRowxCol:
            cmp col, #8
            bge WriteRowx8
            cmp col, #1
            beq WriteRowx1
            cmp col, #2
            beq WriteRowx2
            cmp col, #3
            beq WriteRowx3
            cmp col, #4
            beq WriteRowx4
            cmp col, #5
            beq WriteRowx5
            cmp col, #6
            beq WriteRowx6
            b WriteRowx7

        Write12x8:
            STORE_12x8 q4
            STORE_12x8 q5
            STORE_12x8 q6
            STORE_12x8 q7
            STORE_12x8 q8
            STORE_12x8 q9
            STORE_12x8 q10
            STORE_12x8 q11
            STORE_12x8 q12
            STORE_12x8 q13
            STORE_12x8 q14
            STORE_12x8 q15
            b WriteEnd
        WriteRowx8:
            STORE_C8 q4, #1
            STORE_C8 q5, #2
            STORE_C8 q6, #3
            STORE_C8 q7, #4
            STORE_C8 q8, #5
            STORE_C8 q9, #6
            STORE_C8 q10, #7
            STORE_C8 q11, #8
            STORE_C8 q12, #9
            STORE_C8 q13, #10
            STORE_C8 q14, #11
            STORE_C8 q15, #12
            b WriteEnd

        Write12x1:
            STORE_12x1 d8[0]
            STORE_12x1 d10[0]
            STORE_12x1 d12[0]
            STORE_12x1 d14[0]
            STORE_12x1 d16[0]
            STORE_12x1 d18[0]
            STORE_12x1 d20[0]
            STORE_12x1 d22[0]
            STORE_12x1 d24[0]
            STORE_12x1 d26[0]
            STORE_12x1 d28[0]
            STORE_12x1 d30[0]
            b WriteEnd
        WriteRowx1:
            STORE_C1 d8[0], #1
            STORE_C1 d10[0], #2
            STORE_C1 d12[0], #3
            STORE_C1 d14[0], #4
            STORE_C1 d16[0], #5
            STORE_C1 d18[0], #6
            STORE_C1 d20[0], #7
            STORE_C1 d22[0], #8
            STORE_C1 d24[0], #9
            STORE_C1 d26[0], #10
            STORE_C1 d28[0], #11
            STORE_C1 d30[0], #12
            b WriteEnd
        
        Write12x2:
            STORE_12x2 d8[0]
            STORE_12x2 d10[0]
            STORE_12x2 d12[0]
            STORE_12x2 d14[0]
            STORE_12x2 d16[0]
            STORE_12x2 d18[0]
            STORE_12x2 d20[0]
            STORE_12x2 d22[0]
            STORE_12x2 d24[0]
            STORE_12x2 d26[0]
            STORE_12x2 d28[0]
            STORE_12x2 d30[0]
            b WriteEnd
        WriteRowx2:
            STORE_C2 d8[0], #1
            STORE_C2 d10[0], #2
            STORE_C2 d12[0], #3
            STORE_C2 d14[0], #4
            STORE_C2 d16[0], #5
            STORE_C2 d18[0], #6
            STORE_C2 d20[0], #7
            STORE_C2 d22[0], #8
            STORE_C2 d24[0], #9
            STORE_C2 d26[0], #10
            STORE_C2 d28[0], #11
            STORE_C2 d30[0], #12
            b WriteEnd

        Write12x3:
            STORE_12x3 d8[0], d8[2]
            STORE_12x3 d10[0], d10[2]
            STORE_12x3 d12[0], d12[2]
            STORE_12x3 d14[0], d14[2]
            STORE_12x3 d16[0], d16[2]
            STORE_12x3 d18[0], d18[2]
            STORE_12x3 d20[0], d20[2]
            STORE_12x3 d22[0], d22[2]
            STORE_12x3 d24[0], d24[2]
            STORE_12x3 d26[0], d26[2]
            STORE_12x3 d28[0], d28[2]
            STORE_12x3 d30[0], d30[2]
            b WriteEnd
        WriteRowx3:
            STORE_C3 d8[0], d8[2], #1
            STORE_C3 d10[0], d10[2], #2
            STORE_C3 d12[0], d12[2], #3
            STORE_C3 d14[0], d14[2], #4
            STORE_C3 d16[0], d16[2], #5
            STORE_C3 d18[0], d18[2], #6
            STORE_C3 d20[0], d20[2], #7
            STORE_C3 d22[0], d22[2], #8
            STORE_C3 d24[0], d24[2], #9
            STORE_C3 d26[0], d26[2], #10
            STORE_C3 d28[0], d28[2], #11
            STORE_C3 d30[0], d30[2], #12
            b WriteEnd

        Write12x4:
            STORE_12x4 d8
            STORE_12x4 d10
            STORE_12x4 d12
            STORE_12x4 d14
            STORE_12x4 d16
            STORE_12x4 d18
            STORE_12x4 d20
            STORE_12x4 d22
            STORE_12x4 d24
            STORE_12x4 d26
            STORE_12x4 d28
            STORE_12x4 d30
            b WriteEnd
        WriteRowx4:
            STORE_C4 d8, #1
            STORE_C4 d10, #2
            STORE_C4 d12, #3
            STORE_C4 d14, #4
            STORE_C4 d16, #5
            STORE_C4 d18, #6
            STORE_C4 d20, #7
            STORE_C4 d22, #8
            STORE_C4 d24, #9
            STORE_C4 d26, #10
            STORE_C4 d28, #11
            STORE_C4 d30, #12
            b WriteEnd

        Write12x5:
            STORE_12x5 d8, d9[0]
            STORE_12x5 d10, d11[0]
            STORE_12x5 d12, d13[0]
            STORE_12x5 d14, d15[0]
            STORE_12x5 d16, d17[0]
            STORE_12x5 d18, d19[0]
            STORE_12x5 d20, d21[0]
            STORE_12x5 d22, d23[0]
            STORE_12x5 d24, d25[0]
            STORE_12x5 d26, d27[0]
            STORE_12x5 d28, d29[0]
            STORE_12x5 d30, d31[0]
            b WriteEnd
        WriteRowx5:
            STORE_C5 d8, d9[0], #1
            STORE_C5 d10, d11[0], #2
            STORE_C5 d12, d13[0], #3
            STORE_C5 d14, d15[0], #4
            STORE_C5 d16, d17[0], #5
            STORE_C5 d18, d19[0], #6
            STORE_C5 d20, d21[0], #7
            STORE_C5 d22, d23[0], #8
            STORE_C5 d24, d25[0], #9
            STORE_C5 d26, d27[0], #10
            STORE_C5 d28, d29[0], #11
            STORE_C5 d30, d31[0], #12
            b WriteEnd

        Write12x6:
            STORE_12x6 d8, d9[0]
            STORE_12x6 d10, d11[0]
            STORE_12x6 d12, d13[0]
            STORE_12x6 d14, d15[0]
            STORE_12x6 d16, d17[0]
            STORE_12x6 d18, d19[0]
            STORE_12x6 d20, d21[0]
            STORE_12x6 d22, d23[0]
            STORE_12x6 d24, d25[0]
            STORE_12x6 d26, d27[0]
            STORE_12x6 d28, d29[0]
            STORE_12x6 d30, d31[0]
            b WriteEnd
        WriteRowx6:
            STORE_C6 d8, d9[0], #1
            STORE_C6 d10, d11[0], #2
            STORE_C6 d12, d13[0], #3
            STORE_C6 d14, d15[0], #4
            STORE_C6 d16, d17[0], #5
            STORE_C6 d18, d19[0], #6
            STORE_C6 d20, d21[0], #7
            STORE_C6 d22, d23[0], #8
            STORE_C6 d24, d25[0], #9
            STORE_C6 d26, d27[0], #10
            STORE_C6 d28, d29[0], #11
            STORE_C6 d30, d31[0], #12
            b WriteEnd

        Write12x7:
            STORE_12x7 d8, d9[0], d9[2]
            STORE_12x7 d10, d11[0], d11[2]
            STORE_12x7 d12, d13[0], d13[2]
            STORE_12x7 d14, d15[0], d15[2]
            STORE_12x7 d16, d17[0], d17[2]
            STORE_12x7 d18, d19[0], d19[2]
            STORE_12x7 d20, d21[0], d21[2]
            STORE_12x7 d22, d23[0], d23[2]
            STORE_12x7 d24, d25[0], d25[2]
            STORE_12x7 d26, d27[0], d27[2]
            STORE_12x7 d28, d29[0], d29[2]
            STORE_12x7 d30, d31[0], d31[2]
            b WriteEnd
        WriteRowx7:
            STORE_C7 d8, d9[0], d9[2], #1
            STORE_C7 d10, d11[0], d11[2], #2
            STORE_C7 d12, d13[0], d13[2], #3
            STORE_C7 d14, d15[0], d15[2], #4
            STORE_C7 d16, d17[0], d17[2], #5
            STORE_C7 d18, d19[0], d19[2], #6
            STORE_C7 d20, d21[0], d21[2], #7
            STORE_C7 d22, d23[0], d23[2], #8
            STORE_C7 d24, d25[0], d25[2], #9
            STORE_C7 d26, d27[0], d27[2], #10
            STORE_C7 d28, d29[0], d29[2], #11
            STORE_C7 d30, d31[0], d31[2], #12
            b WriteEnd

        WriteEnd:
            cmp col, #8
            ble LoopColEnd
            sub col, col, #8
            ldr lr, [sp, #20]
            cmp lr, #2
            beq LoopCol8
            add dst_tmp, dst_tmp, #16
            b LoopCol8
    LoopColEnd:
        cmp row, #12
        ble LoopRowEnd
        sub row, row, #12
        mov a_tmp, a
        mov weight, b_tmp
        ldr lr, [sp, #20]
        cmp lr, #2
        beq WinogradDst
        ldr lr, [sp, #12]
        sub lr, lr, col
        add lr, lr, lr  // col *= 2
        sub dst_tmp, dst, lr
        b LoopRow
    WinogradDst:
        add dst_tmp, dst, r9
    LoopRow:
        mov dst, dst_tmp
        ldr col, [sp, #12]
        b LoopRow12
LoopRowEnd:
    sub sp, sp, #104
    vpop {q4-q7}
    pop {r3-r11, pc}
#endif
