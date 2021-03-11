#ifdef ENABLE_ARM32
#include "nnacl/assembly_global.h"

.text
.align 5
//.p2align 5,,15
.global PostFuncBiasReluC8
#ifndef __APPLE__
.type PostFuncBiasReluC8, %function
#endif

//void PostFuncBiasReluC8(float *dst, const float *src, const float *bias, size_t oc8div,size_t oc8mod
//                        size_t plane_size, size_t stride, int relu_type);
// r0 dst           r1 srx           r2 bias
// r3 oc8div        r4 oc8mod        r5 plane_size
// r6 stride        r7 relu_type

// v0 ~ v15 value
// v16  v17 bias data
// r10  r11  weite loop tmp buf
// r16  relu6  #6;    r17 relu #0
// lr  oc8 loop control
// r8  hw  loop control

asm_function PostFuncBiasReluC8
  push {r4-r8, r10, r11, lr}
  add sp, sp, #32

  ldr r4, [sp]
  ldr r5, [sp, #4]
  ldr r6, [sp, #8]
  ldr r7, [sp, #12]

  vmov.i32 q14, #6
  vcvt.f32.s32 q14, q14
  veor q15, q15, q15
  mov lr, #0

Loop_C8:
  cmp lr, r3
  beq Loop_C1
  mov r11,  #4
  mul r10, lr, r11
  add r11, r0, r10
  add lr, lr, #8
  mov r8, r5
  vld1.32 {q12-q13}, [r2]!

Loop_4x8:
  cmp r8, #4
  blt Loop_1x8
  sub r8, r8, #4
  vld1.32 {q0-q1}, [r1]!
  vld1.32 {q2-q3}, [r1]!
  vld1.32 {q8-q9}, [r1]!
  vld1.32 {q10-q11}, [r1]!

  vadd.f32 q0, q0, q12
  vadd.f32 q1, q1, q13
  vadd.f32 q2, q2, q12
  vadd.f32 q3, q3, q13
  vadd.f32 q8, q8, q12
  vadd.f32 q9, q9, q13
  vadd.f32 q10, q10, q12
  vadd.f32 q11, q11, q13

  cmp r7, #3
  beq Relu6_4x8
  cmp r7, #1
  beq Relu_4x8
  b Write_4x8
Relu6_4x8:
  vmin.f32 q0, q0, q14
  vmin.f32 q1, q1, q14
  vmin.f32 q2, q2, q14
  vmin.f32 q3, q3, q14
  vmin.f32 q8, q8, q14
  vmin.f32 q9, q9, q14
  vmin.f32 q10, q10, q14
  vmin.f32 q11, q11, q14
Relu_4x8:
  vmax.f32 q0, q0, q15
  vmax.f32 q1, q1, q15
  vmax.f32 q2, q2, q15
  vmax.f32 q3, q3, q15
  vmax.f32 q8, q8, q15
  vmax.f32 q9, q9, q15
  vmax.f32 q10, q10, q15
  vmax.f32 q11, q11, q15
Write_4x8:
  vst1.32 {q0-q1}, [r11], r6
  vst1.32 {q2-q3}, [r11], r6
  vst1.32 {q8-q9}, [r11], r6
  vst1.32 {q10-q11}, [r11], r6
  b Loop_4x8

Loop_1x8:
  cmp r7, #3
  beq Relu6_1x8
  cmp r7, #1
  beq Relu_1x8
  b Write_1x8
Relu6_1x8:
  cmp r8, #0
  beq Loop_C8
  sub r8, r8, #1
  vld1.32 {q0-q1}, [r1]!
  vadd.f32 q0, q0, q12
  vadd.f32 q1, q1, q13
  vmin.f32 q0, q0, q14
  vmin.f32 q1, q1, q14
  vmax.f32 q0, q0, q15
  vmax.f32 q1, q1, q15
  vst1.32 {q0-q1}, [r11], r6
  b Relu6_1x8
Relu_1x8:
  cmp r8, #0
  beq Loop_C8
  sub r8, r8, #1
  vld1.32 {q0-q1}, [r1]!
  vadd.f32 q0, q0, q12
  vadd.f32 q1, q1, q13
  vmax.f32 q0, q0, q15
  vmax.f32 q1, q1, q15
  vst1.32 {q0-q1}, [r11], r6
  b Relu_1x8
Write_1x8:
  cmp r8, #0
  beq Loop_C8
  sub r8, r8, #1
  vld1.32 {q0-q1}, [r1]!
  vadd.f32 q0, q0, q12
  vadd.f32 q1, q1, q13
  vst1.32 {q0-q1}, [r11], r6
  b Write_1x8

Loop_C1:
  cmp r4, #0
  beq End
  mov r8, r5
  vld1.32 {q12-q13}, [r2]!
  mov r11,  #4
  mul r10, lr, r11
  add r0, r0, r10

  cmp r4, #1
  beq Loop_C1_1
  cmp r4, #2
  beq Loop_C1_2
  cmp r4, #3
  beq Loop_C1_3
  cmp r4, #4
  beq Loop_C1_4
  cmp r4, #5
  beq Loop_C1_5
  cmp r4, #6
  beq Loop_C1_6
  cmp r4, #7
  beq Loop_C1_7

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
  vld1.32 {q0-q1}, [r1]!
  vadd.f32 q0, q0, q12
  vmin.f32 q0, q0, q14
  vmax.f32 q0, q0, q15
  vst1.32 {d0[0]}, [r0], r6
  b Loop_C1_1_Relu6
Loop_C1_1_Relu:
  cmp r8, #0
  beq End
  sub r8, r8, #1
  vld1.32 {q0-q1}, [r1]!
  vadd.f32 q0, q0, q12
  vmax.f32 q0, q0, q15
  vst1.32 {d0[0]}, [r0], r6
  b Loop_C1_1_Relu
Loop_C1_1_Write:
  cmp r8, #0
  beq End
  sub r8, r8, #1
  vld1.32 {q0-q1}, [r1]!
  vadd.f32 q0, q0, q12
  vst1.32 {d0[0]}, [r0], r6
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
  vld1.32 {q0-q1}, [r1]!
  vadd.f32 q0, q0, q12
  vmin.f32 q0, q0, q14
  vmax.f32 q0, q0, q15
  vst1.32 {d0}, [r0], r6
  b Loop_C1_2_Relu6
Loop_C1_2_Relu:
  cmp r8, #0
  beq End
  sub r8, r8, #1
  vld1.32 {q0-q1}, [r1]!
  vadd.f32 q0, q0, q12
  vmax.f32 q0, q0, q15
  vst1.32 {d0}, [r0], r6
  b Loop_C1_2_Relu
Loop_C1_2_Write:
  cmp r8, #0
  beq End
  sub r8, r8, #1
  vld1.32 {q0-q1}, [r1]!
  vadd.f32 q0, q0, q12
  vst1.32 {d0}, [r0], r6
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
  vld1.32 {q0-q1}, [r1]!
  vadd.f32 q0, q0, q12
  vmin.f32 q0, q0, q14
  vmax.f32 q0, q0, q15
  vst1.32 {d0}, [r0], r6
  vst1.32 {d1[0]}, [r11], r6
  b Loop_C1_3_Relu6
Loop_C1_3_Relu:
  cmp r8, #0
  beq End
  sub r8, r8, #1
  vld1.32 {q0-q1}, [r1]!
  vadd.f32 q0, q0, q12
  vmax.f32 q0, q0, q15
  vst1.32 {d0}, [r0], r6
  vst1.32 {d1[0]}, [r11], r6
  b Loop_C1_3_Relu
Loop_C1_3_Write:
  cmp r8, #0
  beq End
  sub r8, r8, #1
  vld1.32 {q0-q1}, [r1]!
  vadd.f32 q0, q0, q12
  vst1.32 {d0}, [r0], r6
  vst1.32 {d1[0]}, [r11], r6
  b Loop_C1_3_Write

Loop_C1_4:
  cmp r7, #3
  beq Loop_C1_4_Relu6
  cmp r7, #1
  beq Loop_C1_4_Relu
  b Loop_C1_4_Write
Loop_C1_4_Relu6:
  cmp r8, #0
  beq End
  sub r8, r8, #1
  vld1.32 {q0-q1}, [r1]!
  vadd.f32 q0, q0, q12
  vmin.f32 q0, q0, q14
  vmax.f32 q0, q0, q15
  vst1.32 {q0}, [r0], r6
  b Loop_C1_4_Relu6
Loop_C1_4_Relu:
  cmp r8, #0
  beq End
  sub r8, r8, #1
  vld1.32 {q0-q1}, [r1]!
  vadd.f32 q0, q0, q12
  vmax.f32 q0, q0, q15
  vst1.32 {q0}, [r0], r6
  b Loop_C1_4_Relu6
Loop_C1_4_Write:
  cmp r8, #0
  beq End
  sub r8, r8, #1
  vld1.32 {q0-q1}, [r1]!
  vadd.f32 q0, q0, q12
  vst1.32 {q0}, [r0], r6
  b Loop_C1_4_Write

Loop_C1_5:
  add r11, r0, #16
  cmp r7, #3
  beq Loop_C1_5_Relu6
  cmp r7, #1
  beq Loop_C1_5_Relu
  b Loop_C1_5_Write
Loop_C1_5_Relu6:
  cmp r8, #0
  beq End
  sub r8, r8, #1
  vld1.32 {q0-q1}, [r1]!
  vadd.f32 q0, q0, q12
  vadd.f32 q1, q1, q13
  vmin.f32 q0, q0, q14
  vmin.f32 q1, q1, q14
  vmax.f32 q0, q0, q15
  vmax.f32 q1, q1, q15
  vst1.32 {q0}, [r0], r6
  vst1.32 {d2[0]}, [r11], r6
  b Loop_C1_5_Relu6
Loop_C1_5_Relu:
  cmp r8, #0
  beq End
  sub r8, r8, #1
  vld1.32 {q0-q1}, [r1]!
  vadd.f32 q0, q0, q12
  vadd.f32 q1, q1, q13
  vmax.f32 q0, q0, q15
  vmax.f32 q1, q1, q15
  vst1.32 {q0}, [r0], r6
  vst1.32 {d2[0]}, [r11], r6
  b Loop_C1_5_Relu
Loop_C1_5_Write:
  cmp r8, #0
  beq End
  sub r8, r8, #1
  vld1.32 {q0-q1}, [r1]!
  vadd.f32 q0, q0, q12
  vadd.f32 q1, q1, q13
  vst1.32 {q0}, [r0], r6
  vst1.32 {d2[0]}, [r11], r6
  b Loop_C1_5_Write

Loop_C1_6:
  add r11, r0, #16
  cmp r7, #3
  beq Loop_C1_6_Relu6
  cmp r7, #1
  beq Loop_C1_6_Relu
  b Loop_C1_6_Write
Loop_C1_6_Relu6:
  cmp r8, #0
  beq End
  sub r8, r8, #1
  vld1.32 {q0-q1}, [r1]!
  vadd.f32 q0, q0, q12
  vadd.f32 q1, q1, q13
  vmin.f32 q0, q0, q14
  vmin.f32 q1, q1, q14
  vmax.f32 q0, q0, q15
  vmax.f32 q1, q1, q15
  vst1.32 {q0}, [r0], r6
  vst1.32 {d2}, [r11], r6
  b Loop_C1_6_Relu6
Loop_C1_6_Relu:
  cmp r8, #0
  beq End
  sub r8, r8, #1
  vld1.32 {q0-q1}, [r1]!
  vadd.f32 q0, q0, q12
  vadd.f32 q1, q1, q13
  vmax.f32 q0, q0, q15
  vmax.f32 q1, q1, q15
  vst1.32 {q0}, [r0], r6
  vst1.32 {d2}, [r11], r6
  b Loop_C1_6_Relu
Loop_C1_6_Write:
  cmp r8, #0
  beq End
  sub r8, r8, #1
  vld1.32 {q0-q1}, [r1]!
  vadd.f32 q0, q0, q12
  vadd.f32 q1, q1, q13
  vst1.32 {q0}, [r0], r6
  vst1.32 {d2}, [r11], r6
  b Loop_C1_6_Write

Loop_C1_7:
  add r11, r0, #16
  add r10, r0, #24
  cmp r7, #3
  beq Loop_C1_7_Relu6
  cmp r7, #1
  beq Loop_C1_7_Relu
  b Loop_C1_7_Write
Loop_C1_7_Relu6:
  cmp r8, #0
  beq End
  sub r8, r8, #1
  vld1.32 {q0-q1}, [r1]!
  vadd.f32 q0, q0, q12
  vadd.f32 q1, q1, q13
  vmin.f32 q0, q0, q14
  vmin.f32 q1, q1, q14
  vmax.f32 q0, q0, q15
  vmax.f32 q1, q1, q15
  vst1.32 {q0}, [r0], r6
  vst1.32 {d2}, [r11], r6
  vst1.32 {d3[0]}, [r10], r6
  b Loop_C1_7_Relu6
Loop_C1_7_Relu:
  cmp r8, #0
  beq End
  sub r8, r8, #1
  vld1.32 {q0-q1}, [r1]!
  vadd.f32 q0, q0, q12
  vadd.f32 q1, q1, q13
  vmax.f32 q0, q0, q15
  vmax.f32 q1, q1, q15
  vst1.32 {q0}, [r0], r6
  vst1.32 {d2}, [r11], r6
  vst1.32 {d3[0]}, [r10], r6
  b Loop_C1_7_Relu
Loop_C1_7_Write:
  cmp r8, #0
  beq End
  sub r8, r8, #1
  vld1.32 {q0-q1}, [r1]!
  vadd.f32 q0, q0, q12
  vadd.f32 q1, q1, q13
  vst1.32 {q0}, [r0], r6
  vst1.32 {d2}, [r11], r6
  vst1.32 {d3[0]}, [r10], r6
  b Loop_C1_7_Write

End:
  sub sp, sp, #32
  pop {r4-r8, r10, r11, pc}
#endif
