#ifdef ENABLE_ARM32
#include "nnacl/assembly_global.h"

.text
.align 5
.global ConvDwFp32Row
#ifndef __APPLE__
.type ConvDwFp32Row, %function
#endif

// voidConvDwFp32Row(float* output_ptr, const float* input_ptr, const float* filter_ptr,
//                   size_t num_pixels, size_t input_channel, size_t input_step)
// r0: output_ptr, r1: input_ptr, r2: filter_ptr, r3: num_pixels,
// r4: input_channel, r5: input_step
asm_function ConvDwFp32Row
    // r4-r8 and q4-q7 must be saved according to https://static.docs.arm.com/ihi0042/i/aapcs32.pdf

    push {r4-r6, r8, r10, r11}
    vpush {q4-q7}
    add sp, sp, #88
    mov r11, r0
    ldr r4, [sp]
    ldr r5, [sp, #4]
    mov r6, #4
    mul r5, r5, r6
    cmp r3, #0
    beq End

    LoopNumPixel:
        mov r6, r1   // input_ptr
        mov r8, r2   // filter_ptr
        mov r10, r4  // input_channel

        LoopDepth16In:
            cmp r10, #16
            blt L4
            sub r10, r10, #16

            vld1.32 {q0, q1}, [r6]!
            vld1.32 {q4, q5}, [r8]!
            vld1.32 {q8, q9}, [r0]!

            cmp r10, #16
            blt LoopDepth16Out
            LoopDepth16:
                vmla.f32 q8, q0, q4
                vmla.f32 q9, q1, q5
                vst1.32 {q8, q9}, [r11]!

                vld1.32 {q2, q3}, [r6]!
                vld1.32 {q6, q7}, [r8]!
                vld1.32 {q10, q11}, [r0]!
                vmla.f32 q10, q2, q6
                vmla.f32 q11, q3, q7
                vst1.32 {q10, q11}, [r11]!

                vld1.32 {q0, q1}, [r6]!
                vld1.32 {q4, q5}, [r8]!
                vld1.32 {q8, q9}, [r0]!

                sub r10, r10, #16
                cmp r10, #16
                bge LoopDepth16

        LoopDepth16Out:
            vmla.f32 q8, q0, q4
            vmla.f32 q9, q1, q5
            vst1.32 {q8, q9}, [r11]!

            vld1.32 {q2, q3}, [r6]!
            vld1.32 {q6, q7}, [r8]!
            vld1.32 {q10, q11}, [r0]!
            vmla.f32 q10, q2, q6
            vmla.f32 q11, q3, q7
            vst1.32 {q10, q11}, [r11]!

        L4:
            cmp r10, #4
            blt L0

            LoopDepth4:
                vld1.32 {q0}, [r6]!
                vld1.32 {q4}, [r8]!
                vld1.32 {q8}, [r0]!
                vmla.f32 q8, q0, q4
                vst1.32 {q8}, [r11]!
                sub r10, r10, #4
                cmp r10, #4
                bge LoopDepth4

        L0:
            cmp r10, #0
            beq Loop16LineEnd

            LoopDepth0:
                vld1.32 d0[0], [r6]!
                vld1.32 d2[0], [r8]!
                vld1.32 d4[0], [r0]!
                vmla.f32 s8, s0, s4
                vst1.32 d4[0], [r11]!
                subs r10, r10, #1
                bne LoopDepth0

        Loop16LineEnd:
            subs r3, r3, #1
            add r1, r1, r5
            bne LoopNumPixel

    End:
        sub sp, sp, #88
        vpop {q4-q7}
        pop {r4-r6, r8, r10, r11}
        bx lr
#endif
