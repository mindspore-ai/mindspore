#include "nnacl/assembly_global.h"

.text
.align 5
//.p2align 5,,15
.global PostFuncBiasReluC4
#ifndef __APPLE__
.type PostFuncBiasReluC4, %function
#endif

asm_function PostFuncBiasReluC4
  push {r4-r8, r10, r11, lr}
  add sp, sp, #32

  ldr r4, [sp]
  ldr r5, [sp, #4]
  ldr r6, [sp, #8]
  ldr r7, [sp, #12]

  vmov.i32 q14, #6
  vcvt.f32.s32 q14, q14
  veor q15, q15, q15

  mov lr, #4
  add r12, r3, r4
  mul r12, r12, lr

  mov lr, #0

Loop_C4:
  cmp lr, r3
  beq Loop_C1
  mov r11,  #4
  mul r10, lr, r11
  add r11, r0, r10
  add lr, lr, #4
  mov r8, r5
  vld1.32 {q12}, [r2]!

Loop_4x4:
  cmp r8, #4
  blt Loop_1x4
  sub r8, r8, #4
  vld1.32 {q0-q1}, [r1]!
  vld1.32 {q2-q3}, [r1]!

  vadd.f32 q0, q0, q12
  vadd.f32 q1, q1, q12
  vadd.f32 q2, q2, q12
  vadd.f32 q3, q3, q12

  cmp r7, #3
  beq Relu6_4x4
  cmp r7, #1
  beq Relu_4x4
  b Write_4x4
Relu6_4x4:
  vmin.f32 q0, q0, q14
  vmin.f32 q1, q1, q14
  vmin.f32 q2, q2, q14
  vmin.f32 q3, q3, q14
Relu_4x4:
  vmax.f32 q0, q0, q15
  vmax.f32 q1, q1, q15
  vmax.f32 q2, q2, q15
  vmax.f32 q3, q3, q15
Write_4x4:
  vst1.32 {q0}, [r11], r12
  vst1.32 {q1}, [r11], r12
  vst1.32 {q2}, [r11], r12
  vst1.32 {q3}, [r11], r12
  b Loop_4x4

Loop_1x4:
  cmp r7, #3
  beq Relu6_1x4
  cmp r7, #1
  beq Relu_1x4
  b Write_1x4
Relu6_1x4:
  cmp r8, #0
  beq HW_Add
  sub r8, r8, #1
  vld1.32 {q0}, [r1]!
  vadd.f32 q0, q0, q12
  vmin.f32 q0, q0, q14
  vmax.f32 q0, q0, q15
  vst1.32 {q0}, [r11], r12
  b Relu6_1x4
Relu_1x4:
  cmp r8, #0
  beq HW_Add
  sub r8, r8, #1
  vld1.32 {q0}, [r1]!
  vadd.f32 q0, q0, q12
  vmax.f32 q0, q0, q15
  vst1.32 {q0}, [r11], r12
  b Relu_1x4
Write_1x4:
  cmp r8, #0
  beq HW_Add
  sub r8, r8, #1
  vld1.32 {q0}, [r1]!
  vadd.f32 q0, q0, q12
  vst1.32 {q0}, [r11], r12
  b Write_1x4

HW_Add:
  add r1, r1, r6
  b Loop_C4

Loop_C1:
  cmp r4, #0
  beq End
  mov r8, r5
  vld1.32 {q12}, [r2]!
  mov r11,  #4
  mul r10, lr, r11
  add r0, r0, r10

  cmp r4, #1
  beq Loop_C1_1
  cmp r4, #2
  beq Loop_C1_2
  cmp r4, #3
  beq Loop_C1_3

Loop_C1_1:
  cmp r7, #3
  beq Loop_C1_1_Relu6
  cmp r7, #1
  beq Loop_C1_1_Relu
  b Loop_C1_1_Write
Loop_C1_1_Relu6:
  cmp r8, #0
  beq End
  sub r8, r8, #1
  vld1.32 {q0}, [r1]!
  vadd.f32 q0, q0, q12
  vmin.f32 q0, q0, q14
  vmax.f32 q0, q0, q15
  vst1.32 {d0[0]}, [r0], r12
  b Loop_C1_1_Relu6
Loop_C1_1_Relu:
  cmp r8, #0
  beq End
  sub r8, r8, #1
  vld1.32 {q0}, [r1]!
  vadd.f32 q0, q0, q12
  vmax.f32 q0, q0, q15
  vst1.32 {d0[0]}, [r0], r12
  b Loop_C1_1_Relu
Loop_C1_1_Write:
  cmp r8, #0
  beq End
  sub r8, r8, #1
  vld1.32 {q0}, [r1]!
  vadd.f32 q0, q0, q12
  vst1.32 {d0[0]}, [r0], r12
  b Loop_C1_1_Write

Loop_C1_2:
  cmp r7, #3
  beq Loop_C1_2_Relu6
  cmp r7, #1
  beq Loop_C1_2_Relu
  b Loop_C1_2_Write
Loop_C1_2_Relu6:
  cmp r8, #0
  beq End
  sub r8, r8, #1
  vld1.32 {q0}, [r1]!
  vadd.f32 q0, q0, q12
  vmin.f32 q0, q0, q14
  vmax.f32 q0, q0, q15
  vst1.32 {d0}, [r0], r12
  b Loop_C1_2_Relu6
Loop_C1_2_Relu:
  cmp r8, #0
  beq End
  sub r8, r8, #1
  vld1.32 {q0}, [r1]!
  vadd.f32 q0, q0, q12
  vmax.f32 q0, q0, q15
  vst1.32 {d0}, [r0], r12
  b Loop_C1_2_Relu
Loop_C1_2_Write:
  cmp r8, #0
  beq End
  sub r8, r8, #1
  vld1.32 {q0}, [r1]!
  vadd.f32 q0, q0, q12
  vst1.32 {d0}, [r0], r12
  b Loop_C1_2_Write

Loop_C1_3:
  add r11, r0, #8
  cmp r7, #3
  beq Loop_C1_3_Relu6
  cmp r7, #1
  beq Loop_C1_3_Relu
  b Loop_C1_3_Write
Loop_C1_3_Relu6:
  cmp r8, #0
  beq End
  sub r8, r8, #1
  vld1.32 {q0}, [r1]!
  vadd.f32 q0, q0, q12
  vmin.f32 q0, q0, q14
  vmax.f32 q0, q0, q15
  vst1.32 {d0}, [r0], r12
  vst1.32 {d1[0]}, [r11], r12
  b Loop_C1_3_Relu6
Loop_C1_3_Relu:
  cmp r8, #0
  beq End
  sub r8, r8, #1
  vld1.32 {q0}, [r1]!
  vadd.f32 q0, q0, q12
  vmax.f32 q0, q0, q15
  vst1.32 {d0}, [r0], r12
  vst1.32 {d1[0]}, [r11], r12
  b Loop_C1_3_Relu
Loop_C1_3_Write:
  cmp r8, #0
  beq End
  sub r8, r8, #1
  vld1.32 {q0}, [r1]!
  vadd.f32 q0, q0, q12
  vst1.32 {d0}, [r0], r12
  vst1.32 {d1[0]}, [r11], r12
  b Loop_C1_3_Write

End:
  sub sp, sp, #32
  pop {r4-r8, r10, r11, pc}
