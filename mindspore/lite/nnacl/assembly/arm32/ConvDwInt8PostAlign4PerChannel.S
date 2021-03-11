#ifdef __arm__
#ifndef __aarch64__
#include "nnacl/assembly_global.h"

.text
.align 5
.global ConvDwInt8PostAlign4PerChannel
#ifndef __APPLE__
.type ConvDwInt8PostAlign4PerChannel, %function
#endif

// void ConvDwInt8PostAlign4PerChannel(int8_t *dst, int32_t *buffer, int channel4, int32_t output_zp, int32_t *out_multiplier,
//                                     int32_t *left_shift, int32_t *right_shift, int32_t acc_min, int32_t acc_max);
// r0: dst, r1: buffer, r2: num_pixels, r3: output_zp, r4: out_multiplier,
// r5: left_shift, r6: right_shift, r7: acc_min, r8: acc_max

asm_function ConvDwInt8PostAlign4PerChannel
    // at return, clang generates "push {lr}, pop {pc}"" while gcc will generate "bx lr"
    // according to https://stackoverflow.com/questions/53625807
    // even if we jump to link register instead of saving it, we still have to save it in subroutine calls anyway
    // clang's rule seems more simple, though there are no subroutine calls here
    // r4-r8 and q4-q7 must be saved according to https://static.docs.arm.com/ihi0042/i/aapcs32.pdf
    push {r4-r8, r10}
    vpush {q4-q7}
    add sp, sp, #88

    vdup.32 q15, r3 // output_zp

    ldr r4, [sp] // out_multiplier
    ldr r5, [sp, #4] // left_shift
    ldr r6, [sp, #8] // right_shift

    ldr r7, [sp, #12] // acc_min
    vdup.32 q11, r7

    ldr r8, [sp, #16] // acc_max
    vdup.32 q10, r8

    mov r10, r0

    LoopDepth8:
        cmp r2, #8
        blt End
        vld1.32 {q0}, [r1]!
        vld1.32 {q13}, [r5]!
        vshl.s32 q0, q0, q13
        vld1.32 {q14}, [r4]!
        vqrdmulh.s32 q0, q0, q14
        vld1.32 {q12}, [r6]!
        vand q4, q0, q12
        vshr.s32 q4, q4, #31
        vqadd.s32 q0, q0, q4
        vrshl.s32 q0, q0, q12
        vadd.i32 q0, q0, q15
        vmax.s32 q0, q0, q11
        vmin.s32 q0, q0, q10
        vqmovn.s32 d4, q0

        vld1.32 {q1}, [r1]!
        vld1.32 {q13}, [r5]!
        vshl.s32 q1, q1, q13
        vld1.32 {q14}, [r4]!
        vqrdmulh.s32 q1, q1, q14
        vld1.32 {q12}, [r6]!
        vand q4, q1, q12
        vshr.s32 q4, q4, #31
        vqadd.s32 q1, q1, q4
        vrshl.s32 q1, q1, q12
        vadd.i32 q1, q1, q15
        vmax.s32 q1, q1, q11
        vmin.s32 q1, q1, q10
        vqmovn.s32 d5, q1
        vqmovn.s16 d4, q2

        vst1.8 {d4}, [r10]!

        sub r2, r2, #8
        b LoopDepth8

    LoopDepth4:
        cmp r2, #4
        blt End
        vld1.32 {q0}, [r1]!
        vld1.32 {q13}, [r5]!
        vshl.s32 q0, q0, q13
        vld1.32 {q14}, [r4]!
        vqrdmulh.s32 q0, q0, q14
        vld1.32 {q12}, [r6]!
        vand q4, q0, q12
        vshr.s32 q4, q4, #31
        vqadd.s32 q0, q0, q4
        vrshl.s32 q0, q0, q12
        vadd.i32 q0, q0, q15
        vmax.s32 q0, q0, q11
        vmin.s32 q0, q0, q10

        vqmovn.s32 d0, q0
        vqmovn.s16 d0, q0

        vst1.8 {d0[0]}, [r10]!
        vst1.8 {d0[1]}, [r10]!
        vst1.8 {d0[2]}, [r10]!
        vst1.8 {d0[3]}, [r10]!

        sub r2, r2, #4
        b LoopDepth4
    End:
        sub sp, sp, #88
        vpop {q4-q7}
        pop {r4-r8, r10}
        bx lr

#endif
#endif
