#ifdef ENABLE_ARM32
#include "nnacl/assembly_global.h"

    .text
    .align 5
    .global WinogradTransRight
#ifndef __APPLE__
    .type WinogradTransRight, %function
#endif

//void WinogradTransRight(const float* S, const float* B, float* M, size_t w, size_t h, size_t k, size_t length);
//x0: S
//x1: B
//x2: M
//x3: w
//x4: h
//x5: k
//x6: length
asm_function WinogradTransRight
    push {r4-r11, lr}
    ldr r4, [sp, #36]
    ldr r5, [sp, #40]
    ldr r6, [sp, #44]

    mov r8, #16 // 4 * sizeof(float)
    mul r8, r6, r8
    mul r9, r5, r8 // step for S
    mov r10, #4
    mul r10, r4, r10 // step for B

LoopH:
    push {r1, r3}
    LoopW:
        push {r0, r1}
        vmov.i32 q14, #0
        mov r11, r6
        InitZero:
            vst1.32 {q14}, [r2]!
            subs r11, r11, #1
            bne InitZero

        sub r2, r2, r8
        mov r12, r5
        LoopKStart7:
            cmp r12, #7
            blt LoopKStart4
            push {r3-r7}
        LoopK7:
            vld1.32 {d0[0]}, [r1], r10
            vld1.32 {d0[1]}, [r1], r10
            vld1.32 {d1[0]}, [r1], r10
            vld1.32 {d1[1]}, [r1], r10
            vld1.32 {d2[0]}, [r1], r10
            vld1.32 {d2[1]}, [r1], r10
            vld1.32 {d3[0]}, [r1], r10
            mov r11, r6
            vmov.32 d30[0], r1

            add r1, r0, r8
            add r3, r1, r8
            add r4, r3, r8
            add r5, r4, r8
            add r6, r5, r8
            add r7, r6, r8
            LoopLength7:
                vld1.32 {q8}, [r2]
                vld1.32 {q12}, [r0]!
                vmla.f32 q8, q12, d0[0]
                vld1.32 {q13}, [r1]!
                vmul.f32 q9, q13, d0[1]
                vld1.32 {q12}, [r3]!
                vmla.f32 q8, q12, d1[0]
                vld1.32 {q13}, [r4]!
                vmla.f32 q9, q13, d1[1]
                vld1.32 {q12}, [r5]!
                vmla.f32 q8, q12, d2[0]
                vld1.32 {q13}, [r6]!
                vmla.f32 q9, q13, d2[1]
                vld1.32 {q12}, [r7]!
                vmla.f32 q8, q12, d3[0]

                vadd.f32 q9, q8, q9
                vst1.32 {q9}, [r2]!
                subs r11, r11, #1
                bne LoopLength7

            sub r2, r2, r8
            sub r12, r12, #7
            mov r0, r7
            vmov.32 r1, d30[0]
            cmp r12, #7
            bge LoopK7

        pop {r3-r7}

        LoopKStart4:
            cmp r12, #4
            blt LoopKStart3
            vmov.32 d30[1], r3
            vmov.32 d31[0], r4
        LoopK4:
            vld1.32 {d0[0]}, [r1], r10
            vld1.32 {d0[1]}, [r1], r10
            vld1.32 {d1[0]}, [r1], r10
            vld1.32 {d1[1]}, [r1], r10
            mov r11, r6
            vmov.32 d30[0], r1

            add r1, r0, r8
            add r3, r1, r8
            add r4, r3, r8

            LoopLength4:
                vld1.32 {q8}, [r2]
                vld1.32 {q12}, [r0]!
                vmla.f32 q8, q12, d0[0]
                vld1.32 {q13}, [r1]!
                vmul.f32 q9, q13, d0[1]
                vld1.32 {q12}, [r3]!
                vmla.f32 q8, q12, d1[0]
                vld1.32 {q13}, [r4]!
                vmla.f32 q9, q13, d1[1]

                vadd.f32 q9, q8, q9
                vst1.32 {q9}, [r2]!
                subs r11, r11, #1
                bne LoopLength4

            sub r2, r2, r8
            sub r12, r12, #4
            mov r0, r4
            vmov.32 r1, d30[0]
            cmp r12, #4
            bge LoopK4

        vmov.32 r3, d30[1]
        vmov.32 r4, d31[0]

        LoopKStart3:
            cmp r12, #3
            blt LoopKStart
            vmov.32 d30[1], r3
        LoopK3:
            vld1.32 {d0[0]}, [r1], r10
            vld1.32 {d0[1]}, [r1], r10
            vld1.32 {d1[0]}, [r1], r10
            mov r11, r6
            vmov.32 d30[0], r1

            add r1, r0, r8
            add r3, r1, r8

            LoopLength3:
                vld1.32 {q8}, [r2]
                vld1.32 {q12}, [r0]!
                vmla.f32 q8, q12, d0[0]
                vld1.32 {q13}, [r1]!
                vmul.f32 q9, q13, d0[1]
                vld1.32 {q12}, [r3]!
                vmla.f32 q8, q12, d1[0]

                vadd.f32 q9, q8, q9
                vst1.32 {q9}, [r2]!
                subs r11, r11, #1
                bne LoopLength3

            sub r2, r2, r8
            sub r12, r12, #3
            mov r0, r3
            vmov.32 r1, d30[0]
            cmp r12, #3
            bge LoopK3

        vmov.32 r3, d30[1]

        LoopKStart:
            cmp r12, #0
            beq LoopKEnd
        LoopK:
            vld1.32 {d30[0]}, [r1], r10
            vdup.32 q15, d30[0]
            mov r11, r6
            LoopLength:
                vld1.32 {q0}, [r2]
                vld1.32 {q1}, [r0]!
                vmla.f32 q0, q1, q15

                vst1.32 {q0}, [r2]!
                subs r11, r11, #1
                bne LoopLength

            subs r12, r12, #1
            sub r2, r2, r8
            bne LoopK
        LoopKEnd:
            pop {r0, r1}
            subs r3, r3, #1
            add r2, r2, r8
            add r1, r1, #4 //sizeof(float)
            bne LoopW

    pop {r1, r3}
    add r0, r0, r9
    subs r4, r4, #1
    bne LoopH

    pop {r4-r11, pc}

#endif
