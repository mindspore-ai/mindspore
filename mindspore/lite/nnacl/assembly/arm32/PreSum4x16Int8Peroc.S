#include "nnacl/assembly_global.h"

.text
.align 5
.global PreSum4x16Int8Peroc
#ifndef __APPLE__
.type PreSum4x16Int8Peroc, %function
#endif


//void PreSum4x16Int8Peroc(const int8_t *src, int32_t *sum, int32_t *zp, size_t hw4, size_t ic16, int32_t oc_div2,
//                            size_t oc_res2, size_t stride);

// r0 src
// r1 sum
// r2 zp
// r3 hw4
// r4 ic16
// r5 oc_div2
// r6 oc_res2
// r7 stride

asm_function PreSum4x16Int8Peroc
  push {r4-r11, lr}
  vpush {q4-q7}
  add sp, sp, #100

  ldr r4, [sp]
  ldr r5, [sp, #4]
  ldr r6, [sp, #8]
  ldr r7, [sp, #12]

  mov r8, #0
  mov r10, #8

RowLoop:
  cmp r8, r3
  beq End
  add r8, r8, #4
  vmov.s32 q13, #0
  mov r9, #0
  mov r11, r2

Sum:
  cmp r9, r4
  beq Mul
  add r9, r9, #16

  vld1.8 {q0, q1}, [r0]!
  vld1.8 {q2, q3}, [r0]!

  vpaddl.s8 q4, q0
  vpaddl.s8 q5, q1
  vpaddl.s8 q6, q2
  vpaddl.s8 q7, q3

  vpaddl.s16 q0, q4
  vpaddl.s16 q1, q5
  vpaddl.s16 q2, q6
  vpaddl.s16 q3, q7

  vpaddl.s32 q4, q0
  vpaddl.s32 q5, q1
  vpaddl.s32 q6, q2
  vpaddl.s32 q7, q3

  vqmovn.s64 d0, q4
  vqmovn.s64 d1, q5
  vqmovn.s64 d2, q6
  vqmovn.s64 d3, q7

  vpaddl.s32 q4, q0
  vpaddl.s32 q5, q1

  vqmovn.s64 d0, q4
  vqmovn.s64 d1, q5

  vadd.i32 q13, q13, q0
  b Sum

Mul:
  mov r12, r1
  add r1, r1, #32
  mov r9, #0

  vdup.32 d1, d26[0]
  vdup.32 d2, d26[1]
  vdup.32 d3, d27[0]
  vdup.32 d4, d27[1]

Write:

  cmp r9, r5
  beq OcRes
  add r9, r9, #2
  vld1.32 {d9}, [r11]!

  vmul.i32 d5, d1, d9
  vmul.i32 d6, d2, d9
  vmul.i32 d7, d3, d9
  vmul.i32 d8, d4, d9

  vst1.32 d5, [r12], r10
  vst1.32 d6, [r12], r10
  vst1.32 d7, [r12], r10
  vst1.32 d8, [r12], r10
  add r12, r12, r7
  b Write

OcRes:
  cmp r6, #0
  beq RowLoop

  vmov.s32 d9, #0
  vld1.8 {d9[0]}, [r11]

  vmul.i32 d5, d1, d9
  vmul.i32 d6, d2, d9
  vmul.i32 d7, d3, d9
  vmul.i32 d8, d4, d9

  vst1.32 d5, [r12], r10
  vst1.32 d6, [r12], r10
  vst1.32 d7, [r12], r10
  vst1.32 d8, [r12], r10
  b RowLoop

End:
  sub sp, sp, #100
  vpop {q4-q7}
  pop {r4-r11, pc}
