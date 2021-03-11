#ifdef ENABLE_ARM32
#include "nnacl/assembly_global.h"

.text
.align 5
.global IndirectGemmInt16to32_8x4
#ifndef __APPLE__
.type IndirectGemmInt16to32_8x4, %function
#endif

// void IndirectGemmInt16to32_8x4(int *output, short *input, short *weight, size_t kszie, size_t ic8, size_t oc4, size_t offset);
// r0: output, r1: input, r2: weight, r3: kszie, r4: ic8, r5: oc4, r6: offset
asm_function IndirectGemmInt16to32_8x4

    .macro INIT_ZERO
        // we could also use "vmov.s32 q12, #0" to initialize q12 by 0
        veor q12, q12, q12
        veor q13, q13, q13
        veor q14, q14, q14
        veor q15, q15, q15
    .endm

    // at return, clang generates "push {lr}, pop {pc}"" while gcc will generate "bx lr"
    // according to https://stackoverflow.com/questions/53625807
    // even if we jump to link register instead of saving it, we still have to save it in subroutine calls anyway
    // clang's rule seems more simple, though there are no subroutine calls here
    // r4-r8 and q4-q7 must be saved according to https://static.docs.arm.com/ihi0042/i/aapcs32.pdf
    push {r4-r8, r10, lr}

    ldr r4, [sp, #28]
    ldr r5, [sp, #32]
    ldr r6, [sp, #36]

    vpush {q4-q7}

    LoopOc:

        mov r7, r3
        mov r8, r1

        LoopKsize:
            mov r10, r0
            INIT_ZERO

            // load input
            vld1.16 {q0, q1}, [r8]!
            // load weight
            vld1.16 {q4}, [r2]!
            vmull.s16 q8, d8, d0[0]
            vmull.s16 q9, d8, d2[0]
            // load weight
            vld1.16 {q5}, [r2]!
            vmlal.s16 q8, d9, d0[1]
            vmlal.s16 q9, d9, d2[1]
            // load input
            vld1.16 {q2, q3}, [r8]!
            vmlal.s16 q8, d10, d0[2]
            vmlal.s16 q9, d10, d2[2]
            vmlal.s16 q8, d11, d0[3]
            vmlal.s16 q9, d11, d2[3]
            // load weight
            vld1.16 {q6, q7}, [r2]!
            vmull.s16 q10, d8, d4[0]
            vmull.s16 q11, d8, d6[0]

            subs r12, r4, #1
            beq LoopIcEnd

            LoopIc:

                vmlal.s16 q10, d9, d4[1]
                vmlal.s16 q11, d9, d6[1]
                vmlal.s16 q10, d10, d4[2]
                vmlal.s16 q11, d10, d6[2]
                vmlal.s16 q10, d11, d4[3]
                vmlal.s16 q11, d11, d6[3]

                vmlal.s16 q8, d12, d1[0]
                vmlal.s16 q9, d12, d3[0]
                vmlal.s16 q8, d13, d1[1]
                vmlal.s16 q9, d13, d3[1]
                vmlal.s16 q8, d14, d1[2]
                vmlal.s16 q9, d14, d3[2]
                vmlal.s16 q8, d15, d1[3]
                vmlal.s16 q9, d15, d3[3]
                // load input
                vld1.16 {q0, q1}, [r8]!
                vmlal.s16 q10, d12, d5[0]
                vmlal.s16 q11, d12, d7[0]
                vmlal.s16 q10, d13, d5[1]
                vmlal.s16 q11, d13, d7[1]
                vmlal.s16 q10, d14, d5[2]
                vmlal.s16 q11, d14, d7[2]
                vmlal.s16 q10, d15, d5[3]
                vmlal.s16 q11, d15, d7[3]

                // load input
                vld1.16 {q2, q3}, [r8]!
                vmlal.s16 q12, d8, d0[0]
                vmlal.s16 q13, d8, d2[0]
                vmlal.s16 q12, d9, d0[1]
                vmlal.s16 q13, d9, d2[1]
                vmlal.s16 q12, d10, d0[2]
                vmlal.s16 q13, d10, d2[2]
                vmlal.s16 q12, d11, d0[3]
                vmlal.s16 q13, d11, d2[3]

                vmlal.s16 q14, d8, d4[0]
                vmlal.s16 q15, d8, d6[0]
                vmlal.s16 q14, d9, d4[1]
                vmlal.s16 q15, d9, d6[1]
                vmlal.s16 q14, d10, d4[2]
                vmlal.s16 q15, d10, d6[2]
                vmlal.s16 q14, d11, d4[3]
                vmlal.s16 q15, d11, d6[3]
                // load weight
                vld1.16 {q4, q5}, [r2]!
                vmlal.s16 q12, d12, d1[0]
                vmlal.s16 q13, d12, d3[0]
                vmlal.s16 q12, d13, d1[1]
                vmlal.s16 q13, d13, d3[1]
                vmlal.s16 q12, d14, d1[2]
                vmlal.s16 q13, d14, d3[2]
                vmlal.s16 q12, d15, d1[3]
                vmlal.s16 q13, d15, d3[3]
                // load input
                vld1.16 {q0, q1}, [r8]!
                vmlal.s16 q14, d12, d5[0]
                vmlal.s16 q15, d12, d7[0]
                vmlal.s16 q14, d13, d5[1]
                vmlal.s16 q15, d13, d7[1]
                vmlal.s16 q14, d14, d5[2]
                vmlal.s16 q15, d14, d7[2]
                vmlal.s16 q14, d15, d5[3]
                vmlal.s16 q15, d15, d7[3]
                // load input
                vld1.16 {q2, q3}, [r8]!
                vmlal.s16 q8, d8, d0[0]
                vmlal.s16 q9, d8, d2[0]
                vmlal.s16 q8, d9, d0[1]
                vmlal.s16 q9, d9, d2[1]
                // load weight
                vld1.16 {q6, q7}, [r2]!
                vmlal.s16 q8, d10, d0[2]
                vmlal.s16 q9, d10, d2[2]
                vmlal.s16 q8, d11, d0[3]
                vmlal.s16 q9, d11, d2[3]
                vmlal.s16 q10, d8, d4[0]
                vmlal.s16 q11, d8, d6[0]

                subs r12, r12, #1
                bne LoopIc

            LoopIcEnd:

                vmlal.s16 q10, d9, d4[1]
                vmlal.s16 q11, d9, d6[1]
                vmlal.s16 q10, d10, d4[2]
                vmlal.s16 q11, d10, d6[2]
                vmlal.s16 q10, d11, d4[3]
                vmlal.s16 q11, d11, d6[3]

                vmlal.s16 q8, d12, d1[0]
                vmlal.s16 q9, d12, d3[0]
                vmlal.s16 q8, d13, d1[1]
                vmlal.s16 q9, d13, d3[1]
                vmlal.s16 q8, d14, d1[2]
                vmlal.s16 q9, d14, d3[2]
                vmlal.s16 q8, d15, d1[3]
                vmlal.s16 q9, d15, d3[3]
                // load input
                vld1.16 {q0, q1}, [r8]!
                vmlal.s16 q10, d12, d5[0]
                vmlal.s16 q11, d12, d7[0]
                vmlal.s16 q10, d13, d5[1]
                vst1.32 {q8}, [r10], r6
                vmlal.s16 q11, d13, d7[1]
                vmlal.s16 q10, d14, d5[2]
                vst1.32 {q9}, [r10], r6
                vmlal.s16 q11, d14, d7[2]
                vmlal.s16 q10, d15, d5[3]
                vmlal.s16 q11, d15, d7[3]

                // load input
                vld1.s16 {q2, q3}, [r8]!
                vmlal.s16 q12, d8, d0[0]
                vmlal.s16 q13, d8, d2[0]
                vmlal.s16 q12, d9, d0[1]
                vst1.32 {q10}, [r10], r6
                vmlal.s16 q13, d9, d2[1]
                vmlal.s16 q12, d10, d0[2]
                vst1.32 {q11}, [r10], r6
                vmlal.s16 q13, d10, d2[2]
                vmlal.s16 q12, d11, d0[3]
                vmlal.s16 q13, d11, d2[3]

                vmlal.s16 q14, d8, d4[0]
                vmlal.s16 q15, d8, d6[0]
                vmlal.s16 q14, d9, d4[1]
                vmlal.s16 q15, d9, d6[1]
                vmlal.s16 q14, d10, d4[2]
                vmlal.s16 q15, d10, d6[2]
                vmlal.s16 q14, d11, d4[3]
                vmlal.s16 q15, d11, d6[3]

                vmlal.s16 q12, d12, d1[0]
                vmlal.s16 q13, d12, d3[0]
                vmlal.s16 q12, d13, d1[1]
                vmlal.s16 q13, d13, d3[1]
                vmlal.s16 q12, d14, d1[2]
                vmlal.s16 q13, d14, d3[2]
                vmlal.s16 q12, d15, d1[3]
                vmlal.s16 q13, d15, d3[3]
                vst1.32 {q12}, [r10], r6
                vmlal.s16 q14, d12, d5[0]
                vmlal.s16 q15, d12, d7[0]
                vmlal.s16 q14, d13, d5[1]
                vmlal.s16 q15, d13, d7[1]
                vmlal.s16 q14, d14, d5[2]
                vst1.32 {q13}, [r10], r6
                vmlal.s16 q15, d14, d7[2]
                vmlal.s16 q14, d15, d5[3]
                vmlal.s16 q15, d15, d7[3]

                vst1.32 {q14}, [r10], r6
                vst1.32 {q15}, [r10]

            subs r7, r7, #1
            add r0, r0, #16
            bne LoopKsize

        subs r5, r5, #1
        bne LoopOc

    vpop {q4-q7}
    pop {r4-r8, r10, pc}

#endif

