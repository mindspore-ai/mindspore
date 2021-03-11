#ifdef ENABLE_ARM32
#include "nnacl/assembly_global.h"

.text
.align 5
.global MatrixMultiplyWinograd
#ifndef __APPLE__
.type MatrixMultiplyWinograd, %function
#endif

// MatrixMultiplyWinograd(float *matix_a, float *matrix_b, float *matrix_c, int m, int k, int n, int in_channel, int c4_channel)
    // r0: matrix_a, r1: matrix_b, r2: matrix_c, r3: m, r4: k, r5: n, r6: in_channel, r7: c4_channel * 4
    // #-56: matrix_a, #-52: matrix_b, #-48: matrix_c, #-44: m, #0: k, #4: n, #8: in_channel, #12: c4_channel * 4
asm_function MatrixMultiplyWinograd
    // at return, clang generates "push {lr}, pop {pc}"" while gcc will generate "bx lr"
    // according to https://stackoverflow.com/questions/53625807
    // even if we jump to link register instead of saving it, we still have to save it in subroutine calls anyway
    // clang's rule seems more simple, though there are no subroutine calls here
    // r4-r8 and q4-q7 must be saved according to https://static.docs.arm.com/ihi0042/i/aapcs32.pdf
    push {r0-r12, lr}
    vpush {q4-q7}
    add sp, sp, #120

    mov r0, #4
    ldr r4, [sp, #4]  // n
    ldr r5, [sp, #8]  // in_channel
    ldr r6, [sp]  // k
    mul r5, r5, r0   // in_channel * 4
    mul r4, r4, r0  // n * 4
    mul r6, r6, r5  // in_channel * 4 * k

    // r3 = m
    // r2 = dst
    LoopM:
        ldr r7, [sp, #4]   // n
        ldr r8, [sp, #-52]  // matrix_b
        LoopN:
            ldr r0, [sp, #4]  // n
            ldr r1, [sp, #-44]  // m
            sub r0, r0, r7   // ni
            mul r0, r0, r1  // ni * m
            sub r1, r1, r3  // mi
            add r0, r0, r1  // ni * m + mi
            ldr r1, [sp, #12]
            mul r9, r0, r1  // (ni * m + mi) * c4_channel * 4
            add r11, r2, r9  // dst + offset

            ldr r10, [sp, #8]  // in_channel
            ldr r9, [sp, #-56]  // src
            cmp r10, #16
            bge LoopC16
            cmp r10, #8
            bge LoopC8
            cmp r10, #4
            bge LoopC4
            cmp r10, #1
            bge LoopC
            b EndLoopC
            
            LoopC16:
                mov r0, r8  // mat_b1
                ldr r12, [sp] // k
                veor q5, q5, q5
                veor q6, q6, q6
                veor q7, q7, q7
                veor q8, q8, q8
                LoopK16:
                    vld1.32 {q0, q1}, [r9]!
                    vld1.32 {q2, q3}, [r9]!
                    add r9, r9, r5
                    sub r9, r9, #64
                    vld1.32 d8[0], [r0], r4
                    vmla.f32 q5, q0, d8[0]
                    vmla.f32 q6, q1, d8[0]
                    vmla.f32 q7, q2, d8[0]
                    vmla.f32 q8, q3, d8[0]
                    subs r12, r12, #1
                    bne LoopK16
                Write16:
                    vst1.32 {q5, q6}, [r11]!
                    vst1.32 {q7, q8}, [r11]!
                subs r10, r10, #16
                beq EndLoopC
                sub r9, r9, r6
                add r9, r9, #64
                cmp r10, #16
                bge LoopC16
                cmp r10, #8
                bge LoopC8
                cmp r10, #4
                bge LoopC4
                cmp r10, #1
                bge LoopC

            LoopC8:
                veor q5, q5, q5
                veor q6, q6, q6
                mov r0, r8  // mat_b1
                ldr r12, [sp] // k
                LoopK8:
                    vld1.32 {q0, q1}, [r9], r5
                    vld1.32 d8[0], [r0], r4
                    vmla.f32 q5, q0, d8[0]
                    vmla.f32 q6, q1, d8[0]
                    subs r12, r12, #1
                    bne LoopK8
                Write8:
                    vst1.32 {q5, q6}, [r11]!
                subs r10, r10, #8
                beq EndLoopC
                sub r9, r9, r6
                add r9, r9, #32
                cmp r10, #8
                bge LoopC8
                cmp r10, #4
                bge LoopC4
                cmp r10, #1
                bge LoopC

            LoopC4:
                veor q5, q5, q5
                mov r0, r8  // mat_b1
                ldr r12, [sp] // k
                LoopK4:
                   vld1.32 {q0}, [r9], r5
                   vld1.32 d8[0], [r0], r4
                   vmla.f32 q5, q0, d8[0]
                   subs r12, r12, #1
                   bne LoopK4
                Write4:
                   vst1.32 {q5}, [r11]!
                subs r10, r10, #4
                beq EndLoopC
                sub r9, r9, r6
                add r9, r9, #16
                cmp r10, #4
                bge LoopC4
                cmp r10, #1
                bge LoopC

            LoopC:
                veor q2, q2, q2
                mov r0, r8  // mat_b1
                ldr r12, [sp] // k
                LoopK:
                    vld1.32 {s0}, [r9], r5
                    vld1.32 d2[0], [r0], r4
                    vmla.f32 s8, s0, s4
                    subs r12, r12, #1
                    bne LoopK
                Write:
                    vst1.32 d4[0], [r11]!
                subs r10, r10, #1
                beq EndLoopC
                sub r9, r9, r6
                add r9, r9, #4
                b LoopC

            EndLoopC:
                subs r7, r7, #1
                beq EndLoopN
                add r8, r8, #4
                b LoopN
        EndLoopN:
            subs r3, r3, #1
            beq EndLoopM
            ldr r0, [sp, #-56]
            add r0, r0, r6
            str r0, [sp, #-56]
            b LoopM
    EndLoopM:
        sub sp, sp, #120
        vpop {q4-q7}
        pop {r0-r12, pc}
#endif

            