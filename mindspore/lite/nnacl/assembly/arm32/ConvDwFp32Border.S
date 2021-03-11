#ifdef ENABLE_ARM32
#include "nnacl/assembly_global.h"

.text
.align 5
.global ConvDwFp32Border
#ifndef __APPLE__
.type ConvDwFp32Border, %function
#endif

// void ConvDwFp32Border(float *dst, const float *src, const float *weight, const float *bias, size_t height, size_t width,
//                       size_t in_kh_step, size_t in_kw_step, size_t kernel_w, size_t relu, size_t relu6)
// r0: dst, r1: src, r2: weight, r3: bias, r4: height, r5: width, r6: in_kh_step, r7: in_kw_step,
// r8: kernel_w, r9: relu, r10: relu6
asm_function ConvDwFp32Border
    // r4-r8 and q4-q7 must be saved according to https://static.docs.arm.com/ihi0042/i/aapcs32.pdf
    push {r4-r12, lr}
    vpush {q4-q7}
    add sp, sp, #104

    ldr r4, [sp]       //  height
    ldr r5, [sp, #4]   //  width
    ldr r6, [sp, #8]   //  in_kh_step
    ldr r7, [sp, #12]  //  in_kw_step
    ldr r8, [sp, #16]  //  kernel_w
    ldr r9, [sp, #20]  // relu
    ldr r10, [sp, #24]  // relu6

    vld1.32 {q0}, [r3] // bias
    vmov.i32 q1, #6    // relu6
    vcvt.f32.s32 q1, q1
    veor q2, q2, q2  // relu

    LoopH:
        mov r11, r1
        mov r12, r2
        mov r14, r5
        LoopW:
            vld1.32 {q3}, [r11], r7
            vld1.32 {q4}, [r12]!
            vmla.f32 q0, q3, q4
            subs r14, r14, #1
            bne LoopW
        subs r4, r4, #1
        add r1, r1, r6
        add r2, r2, r8
        bne LoopH

    cmp r10, #0
    bne Relu6
    cmp r9, #0
    bne Relu
    b Write
    Relu6:
        vmin.f32 q0, q0, q1
    Relu:
        vmax.f32 q0, q0, q2
    Write:
        vst1.32 {q0}, [r0]

    sub sp, sp, #104
    vpop {q4-q7}
    pop {r4-r12, pc}
#endif
