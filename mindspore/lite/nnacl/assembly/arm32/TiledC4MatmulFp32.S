#ifdef ENABLE_ARM32
#include "nnacl/assembly_global.h"
    .text
    .align 5
    .global TiledC4MatmulFp32
#ifndef __APPLE__
    .type TiledC4MatmulFp32, %function
#endif

asm_function TiledC4MatmulFp32
//void TiledC4MatmulFp32(float* dst, const float* src, const float* weight, size_t cal_num, size_t ic4, size_t oc4)
//x0: dst
//x1: src
//x2: weight
//x3: cal_num
//x4: ic4
//x5: oc4

push {r4-r8, lr}
ldr r4, [sp, #24]
ldr r5, [sp, #28]
//step multi by sizeof(float)
mov r8, #4
mul r3, r8, r3

vpush {q4-q7}

LoopOc:
    mov r6, r1
    mov r8, r0
    subs r7, r4, #1
    vld1.32 {q0, q1}, [r1]!
    vld1.32 {q2, q3}, [r1]!
    vld1.32 {q4, q5}, [r2]!
    vld1.32 {q6, q7}, [r2]!

    vmul.f32 q8, q4, d0[0]
    vmul.f32 q9, q4, d2[0]
    vmul.f32 q10, q4, d4[0]
    vmul.f32 q11, q4, d6[0]

    vmla.f32 q8, q5, d0[1]
    vmla.f32 q9, q5, d2[1]
    vmla.f32 q10, q5, d4[1]
    vmla.f32 q11, q5, d6[1]

    vmla.f32 q8, q6, d1[0]
    vmla.f32 q9, q6, d3[0]
    vmla.f32 q10, q6, d5[0]
    vmla.f32 q11, q6, d7[0]

    vmla.f32 q8, q7, d1[1]
    vmla.f32 q9, q7, d3[1]
    vmla.f32 q10, q7, d5[1]
    vmla.f32 q11, q7, d7[1]

    vld1.32 {q0, q1}, [r1]!
    vld1.32 {q2, q3}, [r1]!

    vmul.f32 q12, q4, d0[0]
    vmul.f32 q13, q4, d2[0]
    vmul.f32 q14, q4, d4[0]
    vmul.f32 q15, q4, d6[0]

    vmla.f32 q12, q5, d0[1]
    vmla.f32 q13, q5, d2[1]
    vmla.f32 q14, q5, d4[1]
    vmla.f32 q15, q5, d6[1]

    vmla.f32 q12, q6, d1[0]
    vmla.f32 q13, q6, d3[0]
    vmla.f32 q14, q6, d5[0]
    vmla.f32 q15, q6, d7[0]

    vmla.f32 q12, q7, d1[1]
    vmla.f32 q13, q7, d3[1]
    vmla.f32 q14, q7, d5[1]
    vmla.f32 q15, q7, d7[1]
    beq LoopIcEnd

    subs r7, r7, #1

    vld1.32 {q4, q5}, [r2]!
    vld1.32 {q0, q1}, [r1]!
    vld1.32 {q2, q3}, [r1]!

    vmla.f32 q8, q4, d0[0]
    vmla.f32 q9, q4, d2[0]
    beq LoopIcEndHalf

    LoopIc:
        vmla.f32 q10, q4, d4[0]
        vmla.f32 q11, q4, d6[0]

        vmla.f32 q8, q5, d0[1]
        vmla.f32 q9, q5, d2[1]
        vld1.32 {q6, q7}, [r2]!
        vmla.f32 q10, q5, d4[1]
        vmla.f32 q11, q5, d6[1]

        vmla.f32 q8, q6, d1[0]
        vmla.f32 q9, q6, d3[0]
        vmla.f32 q10, q6, d5[0]
        vmla.f32 q11, q6, d7[0]

        vmla.f32 q8, q7, d1[1]
        vmla.f32 q9, q7, d3[1]
        vmla.f32 q10, q7, d5[1]
        vld1.32 {q0, q1}, [r1]!
        vmla.f32 q11, q7, d7[1]

        vld1.32 {q2, q3}, [r1]!

        vmla.f32 q12, q4, d0[0]
        vmla.f32 q13, q4, d2[0]
        vmla.f32 q14, q4, d4[0]
        vmla.f32 q15, q4, d6[0]

        vmla.f32 q12, q5, d0[1]
        vmla.f32 q13, q5, d2[1]
        vmla.f32 q14, q5, d4[1]
        vmla.f32 q15, q5, d6[1]

        vmla.f32 q12, q6, d1[0]
        vmla.f32 q13, q6, d3[0]
        vmla.f32 q14, q6, d5[0]
        vld1.32 {q4, q5}, [r2]!
        vmla.f32 q15, q6, d7[0]

        vmla.f32 q12, q7, d1[1]
        vmla.f32 q13, q7, d3[1]
        vmla.f32 q14, q7, d5[1]
        vld1.32 {q0, q1}, [r1]!
        vmla.f32 q15, q7, d7[1]

        vld1.32 {q2, q3}, [r1]!

        vmla.f32 q8, q4, d0[0]
        vmla.f32 q9, q4, d2[0]

        subs r7, r7, #1
        bne LoopIc
    LoopIcEndHalf:
        vmla.f32 q10, q4, d4[0]
        vmla.f32 q11, q4, d6[0]

        vmla.f32 q8, q5, d0[1]
        vmla.f32 q9, q5, d2[1]
        vld1.32 {q6, q7}, [r2]!
        vmla.f32 q10, q5, d4[1]
        vmla.f32 q11, q5, d6[1]

        vmla.f32 q8, q6, d1[0]
        vmla.f32 q9, q6, d3[0]
        vmla.f32 q10, q6, d5[0]
        vmla.f32 q11, q6, d7[0]

        vmla.f32 q8, q7, d1[1]
        vmla.f32 q9, q7, d3[1]
        vmla.f32 q10, q7, d5[1]
        vld1.32 {q0, q1}, [r1]!
        vmla.f32 q11, q7, d7[1]

        vld1.32 {q2, q3}, [r1]!

        vmla.f32 q12, q4, d0[0]
        vmla.f32 q13, q4, d2[0]
        vmla.f32 q14, q4, d4[0]
        vmla.f32 q15, q4, d6[0]

        vmla.f32 q12, q5, d0[1]
        vmla.f32 q13, q5, d2[1]
        vmla.f32 q14, q5, d4[1]
        vmla.f32 q15, q5, d6[1]

        vmla.f32 q12, q6, d1[0]
        vmla.f32 q13, q6, d3[0]
        vmla.f32 q14, q6, d5[0]
        vmla.f32 q15, q6, d7[0]

        vmla.f32 q12, q7, d1[1]
        vmla.f32 q13, q7, d3[1]
        vmla.f32 q14, q7, d5[1]
        vmla.f32 q15, q7, d7[1]
    LoopIcEnd:
        vst1.32 {q8, q9}, [r0]!
        vst1.32 {q10, q11}, [r0]!
        vst1.32 {q12, q13}, [r0]!
        vst1.32 {q14, q15}, [r0]!
        mov r1, r6

    subs r5, r5, #1
    add r0, r8, r3
    bne LoopOc

    vpop {q4-q7}
    pop {r4-r8, pc}

#endif
