#ifdef __arm__
#ifndef __aarch64__
#include "nnacl/assembly_global.h"

.text
.align 5
.global DeconvDwInt8Post
#ifndef __APPLE__
.type DeconvDwInt8Post, %function
#endif

// void DeconvDwInt8Post(int8_t *dst, int32_t *output_buffer, const int32_t *bias, int block_channel, int pixel_nums,
//                       int out_multiplier, int left_shift, int right_shift, int32_t out_zp, int32_t acc_min,
//                       int32_t acc_max)
// r0: dst, r1: output_buffer, r2: bias, r3: block_channel, r4: pixel_nums, r5: out_multiplier,
// r6: left_shift, r7: right_shift, r8: out_zp, r9: acc_min, r10: acc_max

asm_function DeconvDwInt8Post
    // at return, clang generates "push {lr}, pop {pc}"" while gcc will generate "bx lr"
    // according to https://stackoverflow.com/questions/53625807
    // even if we jump to link register instead of saving it, we still have to save it in subroutine calls anyway
    // clang's rule seems more simple, though there are no subroutine calls here
    // r4-r8 and q4-q7 must be saved according to https://static.docs.arm.com/ihi0042/i/aapcs32.pdf
    push {r4-r8}
    add sp, sp, #20

    vld1.32 {q9}, [r2]
    ldr r4, [sp]
    ldr r5, [sp, #4]
    vdup.32 q14, r5   // out_multiplier
    ldr r6, [sp, #8]
    vdup.32 q13, r6   // left_shift
    ldr r5, [sp, #12]
    vdup.32 q12, r5   // right_shift
    ldr r6, [sp, #16]
    vdup.32 q15, r6   // output_zp
    ldr r7, [sp, #20]
    vdup.32 q11, r7   // acc_min
    ldr r8, [sp, #24]
    vdup.32 q10, r8   // acc_max

    LoopCount:
        mov r8, r0
        vld1.32 {q0}, [r1]!
        vand q0, q0, q9

        vshl.s32 q0, q0, q13
        vqrdmulh.s32 q0, q0, q14
        vand q4, q0, q12
        vshr.s32 q4, q4, #31
        vqadd.s32 q0, q0, q4
        vrshl.s32 q0, q0, q12
        vadd.i32 q0, q0, q15
        vmax.s32 q0, q0, q11
        vmin.s32 q0, q0, q10

        vqmovn.s32 d0, q0
        vqmovn.s16 d0, q0

        vst1.8 {d0[0]}, [r8]!
        vst1.8 {d0[1]}, [r8]!
        vst1.8 {d0[2]}, [r8]!
        vst1.8 {d0[3]}, [r8]!
        add r0, r0, r3

        sub r4, r4, #1
        cmp r4, #1
        bge LoopCount
    End:
        sub sp, sp, #20
        pop {r4-r8}
        bx lr

#endif
#endif
