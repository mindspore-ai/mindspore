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

// void PreSum4x16Int8Pert(const int8_t *src, int32_t *sum, size_t row4, size_t col16, int32_t filter_zp);

// r0 src
// r1 sum
// r2 row4
// r3 co16
// r4 filter_zp

asm_function PreSum4x16Int8Pert
  push {r4-r8, r10, r11, lr}
  vpush {q4-q7}
  add sp, sp, #96

  ldr r4, [sp]

  vdup.32 q10, r4
  mov r5, #0
  mov r7, #16

RowLoop:
  cmp r5, r2
  beq End
  add r5, r5, #4
  vmov.s32 q13, #0
  mov r6, #0

CalLoop:
  cmp r6, r3
  beq Write
  add r6, r6, #16

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
  b CalLoop

Write:
  vmul.i32 q13, q13, q10
  vst1.32 {d26, d27}, [r1], r7
  beq RowLoop

End:
  sub sp, sp, #96
  vpop {q4-q7}
  pop {r4-r8, r10, r11, pc}
#endif
