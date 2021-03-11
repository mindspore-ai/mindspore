#ifdef __arm__
#ifndef __aarch64__
#include "nnacl/assembly_global.h"

.text
.align 5
.global ConvDwInt8Row
#ifndef __APPLE__
.type ConvDwInt8Row, %function
#endif

// void ConvDwInt8Row(int32_t *output_ptr, const int8_t *input_ptr, const int16_t *weight_ptr, int num_pixels,
//                    int output_channel, int input_step, int8_t input_zp)
// r0: output_ptr, r1: input_ptr, r2: weight_ptr, r3: num_pixels,
// r4: output_channel, r5: input_step, r6: input_zp,

asm_function ConvDwInt8Row
    // at return, clang generates "push {lr}, pop {pc}"" while gcc will generate "bx lr"
    // according to https://stackoverflow.com/questions/53625807
    // even if we jump to link register instead of saving it, we still have to save it in subroutine calls anyway
    // clang's rule seems more simple, though there are no subroutine calls here
    // r4-r8 and q4-q7 must be saved according to https://static.docs.arm.com/ihi0042/i/aapcs32.pdf
    push {r4-r8, r9-r12, lr}
    vpush {q4-q7}
    add sp, sp, #104

    cmp r3, #0
    beq End

    ldr r4, [sp] // channel
    ldr r5, [sp, #4] // input_step
    ldr r6, [sp, #8] // input_zp
    vdup.8 d30, r6

    mov r7, r0

    LoopPixel:
        mov r8, r1 // input
        mov r10, r2 // weight
        mov r11, r4

        LoopDepth16In:
            cmp r11, #16
            blt L8
            sub r11, r11, #16

            vld1.8 {q0}, [r8]!
            vld1.16 {q1, q2}, [r10]! // weight

            vsubl.s8 q3, d0, d30  // -zp
            vld1.32 {q4, q5}, [r0]!
            vmlal.s16 q4, d6, d2
            vmlal.s16 q5, d7, d3

            cmp r11, #16
            blt LoopDepth16Out
            LoopDepth16:
                vst1.32 {q4, q5}, [r7]!

                vsubl.s8 q6, d1, d30
                vld1.32 {q7, q8}, [r0]!
                vmlal.s16 q7, d12, d4
                vmlal.s16 q8, d13, d5
                vst1.32 {q7, q8}, [r7]!

                vld1.8 {q0}, [r8]!
                vld1.16 {q1, q2}, [r10]! // weight

                vsubl.s8 q3, d0, d30  // -zp
                vld1.32 {q4, q5}, [r0]!
                vmlal.s16 q4, d6, d2
                vmlal.s16 q5, d7, d3

                sub r11, r11, #16
                cmp r11, #16
                bge LoopDepth16

            LoopDepth16Out:
                vst1.32 {q4, q5}, [r7]!

                vsubl.s8 q6, d1, d30
                vld1.32 {q7, q8}, [r0]!
                vmlal.s16 q7, d12, d4
                vmlal.s16 q8, d13, d5
                vst1.32 {q7, q8}, [r7]!

        L8:
            cmp r11, #8
            blt L0

            LoopDepth8:
                vld1.8 {d0}, [r8]!
                vld1.16 {d2, d3}, [r10]! // weight

                vsubl.s8 q2, d0, d30  // -zp

                vld1.32 {q3}, [r0]!
                vmlal.s16 q3, d4, d2
                vst1.32 {q3}, [r7]!

                vld1.32 {q4}, [r0]!
                vmlal.s16 q4, d5, d3
                vst1.32 {q4}, [r7]!

                sub r11, r11, #8
                cmp r11, #8
                bge LoopDepth8

        L0:
            cmp r11, #0
            beq LoopDepthEnd

            LoopDepth0:
                ldrsb r12, [r8], #1
                ldrsh r9, [r10], #2
                sub r12, r12, r6

                ldr lr, [r0], #4
                smlabb r12, r12, r9, lr
                str r12, [r7], #4

                subs r11, r11, #1
                bne L0

        LoopDepthEnd:
        add r1, r1, r5
        subs r3, r3, #1
        bne LoopPixel

    End:
    sub sp, sp, #104
    vpop {q4-q7}
    pop {r4-r8, r9-r12, pc}
#endif
#endif
