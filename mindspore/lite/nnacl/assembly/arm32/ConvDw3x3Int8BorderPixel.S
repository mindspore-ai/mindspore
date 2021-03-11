#ifdef __arm__
#ifndef __aarch64__
#include "nnacl/assembly_global.h"

.text
.align 5
.global ConvDw3x3Int8BorderPixel
#ifndef __APPLE__
.type ConvDw3x3Int8BorderPixel, %function
#endif

// void ConvDw3x3Int8BorderPixel(int8_t *dst, const int8_t *src, const int16_t *weight, const int32_t *bias, size_t height,
//                               size_t width, size_t in_kh_step, size_t in_kw_step, size_t channel, size_t in_zp, size_t out_zp,
//                               size_t out_multiplier, size_t left_shift, size_t right_shift, size_t acc_min, size_t acc_max
//                               size_t per_channel) {

// todo: support per channel
// r0: dst, r1: src, r2: weight, r3: bias, r4: height, r5: width, r6: in_kh_step, r7: in_kw_step,
// r8: channel, r9: in_zp,  r10: out_zp, r11: out_multiplier, r12: left_shift, r13: right_shift
// r14: acc_min, r15: acc_max
asm_function ConvDw3x3Int8BorderPixel
    // at return, clang generates "push {lr}, pop {pc}"" while gcc will generate "bx lr"
    // according to https://stackoverflow.com/questions/53625807
    // even if we jump to link register instead of saving it, we still have to save it in subroutine calls anyway
    // clang's rule seems more simple, though there are no subroutine calls here
    // r4-r8 and q4-q7 must be saved according to https://static.docs.arm.com/ihi0042/i/aapcs32.pdf

    push {r4-r8, r9-r12, lr}
    vpush {q4-q7}
    add sp, sp, #104

    ldr r4, [sp]
    ldr r5, [sp, #4]
    ldr r6, [sp, #8]
    ldr r7, [sp, #12]
    ldr r8, [sp, #16]

    ldrb r10, [sp, #20] // in_zp
    vdup.8 d18, r10
    ldr r10, [sp, #24] // out_zp
    vdup.32 q15, r10
    ldr r10, [sp, #28] // out_multiplier
    vdup.32 q14, r10
    ldr r10, [sp, #32] // left_shift
    vdup.32 q13, r10
    ldr r10, [sp, #36] // right_shift
    vdup.32 q12, r10
    ldr r10, [sp, #40] // acc_min
    vdup.32 q11, r10
    ldr r10, [sp, #44] // acc_max
    vdup.32 q10, r10

    mov r4, #2
    mul lr, r8, r4

    LoopC:
        mov r9, r1
        mov r10, r2
        ldr r4, [sp]

        vld1.32 {q3}, [r3]!
        vld1.32 {q4}, [r3]!
        LoopH:
            mov r11, r9
            mov r12, r10
            ldr r5, [sp, #4]
            LoopW:
                vld1.8 {d0}, [r11], r7
                vld1.16 {d2, d3}, [r12], lr // weight
                vsubl.s8 q2, d0, d18  // -zp

                vmlal.s16 q3, d4, d2
                vmlal.s16 q4, d5, d3

                subs r5, r5, #1
                bne LoopW
            subs r4, r4, #1
            add r9, r9, r6
            mov r11, #3
            mul r5, lr, r11
            add r10, r10, r5
            bne LoopH

        vshl.s32 q3, q3, q13
        vqrdmulh.s32 q3, q3, q14
        vand q5, q3, q12
        vshr.s32 q5, q5, #31
        vqadd.s32 q3, q3, q5
        vrshl.s32 q3, q3, q12
        vadd.i32 q3, q3, q15
        vmax.s32 q3, q3, q11
        vmin.s32 q3, q3, q10
        vqmovn.s32 d14, q3

        vshl.s32 q4, q4, q13
        vqrdmulh.s32 q4, q4, q14
        vand q6, q4, q12
        vshr.s32 q6, q6, #31
        vqadd.s32 q4, q4, q6
        vrshl.s32 q4, q4, q12
        vadd.i32 q4, q4, q15
        vmax.s32 q4, q4, q11
        vmin.s32 q4, q4, q10
        vqmovn.s32 d15, q4
        vqmovn.s16 d16, q7

        vst1.8 {d16}, [r0]!
        add r1, r1, #8
        add r2, r2, #16

        sub r8, r8, #8
        cmp r8, #8
        bge LoopC

    sub sp, sp, #104
    vpop {q4-q7}
    pop {r4-r8, r9-r12, pc}
#endif
#endif
