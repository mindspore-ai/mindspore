/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifdef ENABLE_ARM32
#include "nnacl/assembly_global.h"

.text
.align 5

// void MatVecMulA32NeonFp16(const float16_t *a, const float16_t *b, float16_t *c, const float16_t *bias, int act_type,
//                           int depth, int col) {
// r0: a
// r1: b
// r2: c
// r3: bias
// r4: act_type
// r5: depth
// r6: col

asm_function MatVecMulA32NeonFp16
  // r4-r8 and q4-q7 must be saved according to https://static.docs.arm.com/ihi0042/i/aapcs32.pdf
  push {r0-r8, r9, r10, r11, lr}
  add sp, sp, #52

  ldr r4, [sp]
  ldr r5, [sp, #4]
  ldr r6, [sp, #8]

  add r10, r5, r5  // stride = depth * sizeof(float16_t)
  mov lr, #4
  mul r11, r10, lr // stride x 4

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

  cmp r8, #8
  bge Col4Depth8
  cmp r8, #4
  bge Col4Depth4
  cmp r8, #1
  bge Col4Depth1
  b Col4End

  Col4Depth8:
    vld1.16 {q8}, [r7]!
    add lr, r9, r10
    vld1.16 {q0}, [r9]!
    vld1.16 {q1}, [lr], r10
    vld1.16 {q2}, [lr], r10
    vld1.16 {q3}, [lr]

    vmla.f16 q9, q8, q0
    vmla.f16 q10, q8, q1
    vmla.f16 q11, q8, q2
    vmla.f16 q12, q8, q3
    sub r8, r8, #8
    cmp r8, #8
    bge Col4Depth8
    cmp r8, #4
    bge Col4Depth4
    b AddC4

  Col4Depth4:
    vld1.16 {d16}, [r7]!
    add lr, r9, r10
    vld1.16 {d0}, [r9]!
    vld1.16 {d2}, [lr], r10
    vld1.16 {d4}, [lr], r10
    vld1.16 {d6}, [lr]

    vmla.f16 d18, d16, d0
    vmla.f16 d20, d16, d2
    vmla.f16 d22, d16, d4
    vmla.f16 d24, d16, d6
    sub r8, r8, #4
    cmp r8, #4
    bge Col4Depth4

  AddC4:
    vpadd.f16 d0, d18, d19
    vpadd.f16 d1, d20, d21
    vpadd.f16 d2, d22, d23
    vpadd.f16 d4, d24, d25
    vpadd.f16 d30, d0, d1
    vpadd.f16 d31, d2, d4
    vpadd.f16 d30, d30, d31
    cmp r8, #1
    bge Col4Depth1
    b Col4End

  Col4Depth1:
    vld1.16 {d0[0]}, [r7]!
    add lr, r9, r10
    vld1.16 {d2[0]}, [r9]!
    vld1.16 {d2[1]}, [lr], r10
    vld1.16 {d2[2]}, [lr], r10
    vld1.16 {d2[3]}, [lr]

    vmla.f16 d30, d2, d0[0]
    subs r8, r8, #1
    bne Col4Depth1

  Col4End:
    cmp r3, #0
    beq Col4Activation
    vld1.16 {d26}, [r3]!
    vadd.f16 d30, d30, d26

  Col4Activation:
    cmp r4, #3
    beq Col4Relu6
    cmp r4, #1
    beq Col4Relu
    b Col4Write

  Col4Relu6:
    vmov.i16 q12, #6
    vcvt.f16.s16 q12, q12
    vmin.f16 d30, d30, d24

  Col4Relu:
    veor q13, q13, q13
    vmax.f16 d30, d30, d26

  Col4Write:
    vst1.16 {d30}, [r2]!
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
  veor q15, q15, q15

  cmp r8, #8
  bge Col1Depth8
  cmp r8, #4
  bge Col1Depth4
  cmp r8, #1
  bge Col1Depth1
  b Col1End

  Col1Depth8:
    vld1.16 {q0}, [r7]!
    vld1.16 {q1}, [r9]!
    vmla.f16 q10, q1, q0
    sub r8, r8, #8
    cmp r8, #8
    bge Col1Depth8
    cmp r8, #4
    bge Col1Depth4
    b AddC1

  Col1Depth4:
    vld1.16 {d0}, [r7]!
    vld1.16 {d2}, [r9]!
    vmla.f16 d20, d2, d0
    sub r8, r8, #4
    cmp r8, #4
    bge Col1Depth4

  AddC1:
    vpadd.f16 d30, d20, d21
    vpadd.f16 d30, d30, d20
    vpadd.f16 d30, d30, d20
    cmp r8, #1
    bge Col1Depth1
    b Col1End

  Col1Depth1:
    vld1.16 {d0[0]}, [r7]!
    vld1.16 {d2[0]}, [r9]!
    vmla.f16 d30, d2, d0[0]
    subs r8, r8, #1
    bne Col1Depth1

  Col1End:
    cmp r3, #0
    beq Col1Activation
    vld1.16 {d28[0]}, [r3]!
    vadd.f16 d30, d30, d28

  Col1Activation:
    cmp r4, #3
    beq Col1Relu6
    cmp r4, #1
    beq Col1Relu
    b Col1Write

  Col1Relu6:
    vmov.i16 d26, #6
    vcvt.f16.s16 d26, d26
    vmin.f16 d30, d30, d26

  Col1Relu:
    veor d24, d24, d24
    vmax.f16 d30, d30, d24

  Col1Write:
    vst1.16 {d30[0]}, [r2]!
    subs r6, r6, #1
    beq End
    add r1, r1, r10
    b Col1Loop

End:
  sub sp, sp, #52
  pop {r0-r8, r9, r10, r11, pc}
#endif
