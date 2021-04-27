#ifdef ENABLE_ARM32
#include "nnacl/assembly_global.h"
    .text
    .align 5
    .global Float32ToFloat16
#ifndef __APPLE__
    .type Float32ToFloat16, %function
#endif

// void Float32ToFloat16(const float *input, float16_t *output, int number);
// r0: input, r1: output, r2: number
Float32ToFloat16:
    cmp r2, #0
    beq LoopEnd
    cmp r2, #16
    bge Loop16
    cmp r2, #8
    bge Loop8
    b Loop
    Loop16:
        vld1.32 {q0, q1}, [r0]!
        vcvt.f16.f32 d0, q0
        vcvt.f16.f32 d1, q1
        vld1.32 {q2, q3}, [r0]!
        vcvt.f16.f32 d2, q2
        vcvt.f16.f32 d3, q3
        vst1.16 {q0, q1}, [r1]!
        subs r2, r2, #16
        beq LoopEnd
        cmp r2, #16
        bge Loop16
        cmp r2, #8
        bge Loop8
        b Loop
    Loop8:
        vld1.32 {q0, q1}, [r0]!
        vcvt.f16.f32 d0, q0
        vcvt.f16.f32 d1, q1
        vst1.16 {q0}, [r1]!
        subs r2, r2, #8
        beq LoopEnd
        cmp r2, #8
        bge Loop8
        b Loop
    Loop:
        vldr s0, [r0]
        vcvtb.f16.f32 s0, s0
        vstr.16 s0, [r1]
        add r0, r0, #4
        add r1, r1, #2
        subs r2, r2, #1
        bgt Loop
    LoopEnd:
        mov pc, lr
#endif
