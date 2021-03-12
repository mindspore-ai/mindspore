#ifdef __arm__
#ifndef __aarch64__
#include "nnacl/assembly_global.h"

.text
.align 5
.global MatVecMulFp32
#ifndef __APPLE__
.type MatVecMulFp32, %function
#endif

// void MatVecMulFp32(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int col)
// r0: a
// r1: b
// r2: c
// r3: bias
// r4: act_type
// r5: depth
// r6: col

asm_function MatVecMulFp32
  // r4-r8 and q4-q7 must be saved according to https://static.docs.arm.com/ihi0042/i/aapcs32.pdf
  push {r0-r8, r9, r10, r11, lr}
  add sp, sp, #52

  ldr r4, [sp]
  ldr r5, [sp, #4]
  ldr r6, [sp, #8]

  mov r10, #4
  mul r10, r10, r5  // stride = depth * sizeof(float)
  mov r11, #4
  mul r11, r11, r10 // stride x 4

  cmp r6, #4
  blt Col1Loop

Col4Loop:
  mov r7, r0    // reload a(vector) ptr  
  mov r9, r1    // reload b(matrix) ptr
  mov r8, r5    // reload depth value

  veor q9, q9, q9
  veor q10, q10, q10
  veor q11, q11, q11
  veor q12, q12, q12
  veor q15, q15, q15

  cmp r8, #4
  blt Col4Depth1

  Col4Depth4:
    vld1.f32 {q8}, [r7]!
    add lr, r9, r10
    vld1.f32 {q0}, [r9]!
    vld1.f32 {q1}, [lr], r10
    vld1.f32 {q2}, [lr], r10
    vld1.f32 {q3}, [lr]

    vmla.f32 q9, q8, q0
    vmla.f32 q10, q8, q1
    vmla.f32 q11, q8, q2
    vmla.f32 q12, q8, q3
    sub r8, r8, #4
    cmp r8, #4
    bge Col4Depth4

  vpadd.f32 d26, d18, d20
  vpadd.f32 d27, d19, d21
  vpadd.f32 d28, d22, d24
  vpadd.f32 d29, d23, d25
  vadd.f32 d30, d26, d27
  vadd.f32 d31, d28, d29
  cmp r8, #0
  beq Col4End

  Col4Depth1:
    vld1.f32 {d0[0]}, [r7]!
    add lr, r9, r10
    vld1.f32 {d2[0]}, [r9]!
    vld1.f32 {d2[1]}, [lr], r10
    vld1.f32 {d3[0]}, [lr], r10
    vld1.f32 {d3[1]}, [lr]

    vmla.f32 q15, q1, d0[0]
    subs r8, r8, #1
    bne Col4Depth1

  Col4End:
    cmp r3, #0
    beq Col4Activation
    vld1.f32 {q13}, [r3]!
    vadd.f32 q15, q15, q13

  Col4Activation:
    cmp r4, #3
    beq Col4Relu6
    cmp r4, #1
    beq Col4Relu
    b Col4Write

  Col4Relu6:
    vmov.i32 q12, #6
    vcvt.f32.s32 q12, q12
    vmin.f32 q15, q15, q12

  Col4Relu:
    veor q13, q13, q13
    vmax.f32 q15, q15, q13

  Col4Write:
    vst1.f32 {q15}, [r2]!
    subs r6, r6, #4
    beq End
    add r1, r1, r11
    cmp r6, #4
    bge Col4Loop

Col1Loop:
  mov r7, r0    // reload a(vector) ptr  
  mov r9, r1    // reload b(matrix) ptr
  mov r8, r5    // reload depth value
  veor q10, q10, q10
  veor q13, q13, q13
  veor q15, q15, q15

  cmp r8, #4
  blt Col1Depth1

  Col1Depth4:
    vld1.f32 {q0}, [r7]!
    vld1.f32 {q1}, [r9]!

    vmla.f32 q10, q1, q0
    sub r8, r8, #4
    cmp r8, #4
    bge Col1Depth4

  vpadd.f32 d24, d20, d22
  vpadd.f32 d25, d21, d23
  vadd.f32 d30, d24, d25
  cmp r8, #0
  beq Col1End

  Col1Depth1:
    vld1.f32 {d0[0]}, [r7]!
    vld1.f32 {d2[0]}, [r9]!

    vmla.f32 d30, d2, d0[0]
    subs r8, r8, #1
    bne Col1Depth1

  Col1End:
    cmp r3, #0
    beq Col1Activation
    vld1.f32 {d28[0]}, [r3]!
    vadd.f32 d30, d30, d28

  Col1Activation:
    cmp r4, #3
    beq Col1Relu6
    cmp r4, #1
    beq Col1Relu
    b Col1Write

  Col1Relu6:
    vmov.i32 d26, #6
    vcvt.f32.s32 d26, d26
    vmin.f32 d30, d30, d26

  Col1Relu:
    veor d24, d24, d24
    vmax.f32 d30, d30, d24

  Col1Write:
    vst1.f32 {d30[0]}, [r2]!
    subs r6, r6, #1
    beq End
    add r1, r1, r10
    b Col1Loop

End:
  sub sp, sp, #52
  pop {r0-r8, r9, r10, r11, pc}
#endif
#endif
