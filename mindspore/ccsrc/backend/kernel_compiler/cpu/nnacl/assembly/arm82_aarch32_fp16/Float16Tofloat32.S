#ifdef ENABLE_ARM32
#include "nnacl/assembly_global.h"
    .text
    .align 5
    .global Float16ToFloat32
#ifndef __APPLE__
    .type Float16ToFloat32, %function
#endif

// void Float16ToFloat32(const float16_t *input, float *output, int number);
// r0: input, r1: output, r2: number
Float16ToFloat32:
    cmp r2, #0
    beq LoopEnd
    cmp r2, #16
    bge Loop16
    cmp r2, #8
    bge Loop8
    b Loop
    Loop16:
        vld1.16 {q0, q1}, [r0]!
        vcvt.f32.f16 q3, d0
        vcvt.f32.f16 q4, d1
        vcvt.f32.f16 q5, d2
        vst1.32 {q3, q4}, [r1]!
        vcvt.f32.f16 q6, d3
        subs r2, r2, #16
        vst1.32 {q5, q6}, [r1]!
        beq LoopEnd
        cmp r2, #16
        bge Loop16
        cmp r2, #8
        bge Loop8
        b Loop
    Loop8:
        vld1.16 {q0}, [r0]!
        vcvt.f32.f16 q1, d0
        vcvt.f32.f16 q2, d1
        vst1.32 {q1, q2}, [r1]!
        subs r2, r2, #8
        beq LoopEnd
        cmp r2, #8
        bge Loop8
        b Loop
    Loop:
        vldr.16 s0, [r0]
        vcvtb.f32.f16 s0, s0
        vstr.32 s0, [r1]
        add r0, r0, #2
        add r1, r1, #4
        subs r2, r2, #1
        bgt Loop
    LoopEnd:
        mov pc, lr
#endif
