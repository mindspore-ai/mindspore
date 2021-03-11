#ifdef __aarch64__
#include "nnacl/assembly_global.h"

    .text
    .align 5
    //.p2align 5,,15
    .global PostFuncBiasReluC4
#ifndef __APPLE__
    .type PostFuncBiasReluC4, %function
#endif

//void PostFuncBiasReluC4(float *dst, const float *src, const float *bias, size_t oc4div, size_t oc4mod,
//                        size_t plane_size, size_t plane_stride, size_t relu_type);
// x0 dst           x1 srx           x2 bias
// w3 oc4div        w4 oc4mod        w5 plane_size
// x6 plane_stride  x7 relu_type

// v0 ~ v7 value
// v16  bias data
// x12  oc_stride
// x14 x15   write loop tmp buf
// v26  relu6  #6;    v27 relu #0
// w10  oc4 loop control
// w13  hw  loop control


asm_function PostFuncBiasReluC4

  movi v26.4s, #6
  scvtf v26.4s, v26.4s
  dup v27.4s, wzr

  mov x10, #4
  add x12, x3, x4
  mul x12, x12, x10

  mov w10, #0

Loop_C4:
  cmp w10, w3
  beq Loop_C1
  mov x15, #4
  mul x14, x10, x15
  add x15, x0, x14
  add w10, w10, #4
  mov w13, w5
  ld1 {v16.4s}, [x2], #16

Loop_8x4:
  cmp w13, #8
  blt Loop_4x4
  sub w13, w13, #8
  ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
  ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x1], #64

  fadd v0.4s, v0.4s, v16.4s
  fadd v1.4s, v1.4s, v16.4s
  fadd v2.4s, v2.4s, v16.4s
  fadd v3.4s, v3.4s, v16.4s
  fadd v4.4s, v4.4s, v16.4s
  fadd v5.4s, v5.4s, v16.4s
  fadd v6.4s, v6.4s, v16.4s
  fadd v7.4s, v7.4s, v16.4s

  cmp x7, #3
  beq Relu6_8x4
  cmp x7, #1
  beq Relu_8x4
  b Write_8x4
Relu6_8x4:
  fmin v0.4s, v0.4s, v26.4s
  fmin v1.4s, v1.4s, v26.4s
  fmin v2.4s, v2.4s, v26.4s
  fmin v3.4s, v3.4s, v26.4s
  fmin v4.4s, v4.4s, v26.4s
  fmin v5.4s, v5.4s, v26.4s
  fmin v6.4s, v6.4s, v26.4s
  fmin v7.4s, v7.4s, v26.4s
Relu_8x4:
  fmax v0.4s, v0.4s, v27.4s
  fmax v1.4s, v1.4s, v27.4s
  fmax v2.4s, v2.4s, v27.4s
  fmax v3.4s, v3.4s, v27.4s
  fmax v4.4s, v4.4s, v27.4s
  fmax v5.4s, v5.4s, v27.4s
  fmax v6.4s, v6.4s, v27.4s
  fmax v7.4s, v7.4s, v27.4s
Write_8x4:
  st1 {v0.4s}, [x15], x12
  st1 {v1.4s}, [x15], x12
  st1 {v2.4s}, [x15], x12
  st1 {v3.4s}, [x15], x12
  st1 {v4.4s}, [x15], x12
  st1 {v5.4s}, [x15], x12
  st1 {v6.4s}, [x15], x12
  st1 {v7.4s}, [x15], x12
  b Loop_8x4

Loop_4x4:
  cmp w13, #4
  blt Loop_1x4
  sub w13, w13, #4
  ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
  fadd v0.4s, v0.4s, v16.4s
  fadd v1.4s, v1.4s, v16.4s
  fadd v2.4s, v2.4s, v16.4s
  fadd v3.4s, v3.4s, v16.4s
  cmp x7, #3
  beq Relu6_4x4
  cmp x7, #1
  beq Relu_4x4
  b Write_4x4
Relu6_4x4:
  fmin v0.4s, v0.4s, v26.4s
  fmin v1.4s, v1.4s, v26.4s
  fmin v2.4s, v2.4s, v26.4s
  fmin v3.4s, v3.4s, v26.4s
Relu_4x4:
  fmax v0.4s, v0.4s, v27.4s
  fmax v1.4s, v1.4s, v27.4s
  fmax v2.4s, v2.4s, v27.4s
  fmax v3.4s, v3.4s, v27.4s
Write_4x4:
  st1 {v0.4s}, [x15], x12
  st1 {v1.4s}, [x15], x12
  st1 {v2.4s}, [x15], x12
  st1 {v3.4s}, [x15], x12

Loop_1x4:
  cmp x7, #3
  beq Relu6_1x4
  cmp x7, #1
  beq Relu_1x4
  b Write_1x4
Relu6_1x4:
  cmp w13, #0
  beq HW_Add
  sub w13, w13, #1
  ld1 {v0.4s}, [x1], #16
  fadd v0.4s, v0.4s, v16.4s
  fmin v0.4s, v0.4s, v26.4s
  fmax v0.4s, v0.4s, v27.4s
  st1 {v0.4s}, [x15], x12
  b Relu6_1x4
Relu_1x4:
  cmp w13, #0
  beq HW_Add
  sub w13, w13, #1
  ld1 {v0.4s}, [x1], #16
  fadd v0.4s, v0.4s, v16.4s
  fmax v0.4s, v0.4s, v27.4s
  st1 {v0.4s}, [x15], x12
  b Relu_1x4
Write_1x4:
  cmp w13, #0
  beq HW_Add
  sub w13, w13, #1
  ld1 {v0.4s}, [x1], #16
  fadd v0.4s, v0.4s, v16.4s
  st1 {v0.4s}, [x15], x12
  b Write_1x4

HW_Add:
  add x1, x1, x6
  b Loop_C4

Loop_C1:
  cmp x4, #0
  beq End
  mov w13, w5
  ld1 {v16.4s}, [x2], #16
  mov x15,  #4
  mul x14, x10, x15
  add x0, x0, x14

  cmp x4, #1
  beq Loop_C1_1
  cmp x4, #2
  beq Loop_C1_2
  cmp x4, #3
  beq Loop_C1_3

Loop_C1_1:
  cmp x7, #3
  beq Loop_C1_1_Relu6
  cmp x7, #1
  beq Loop_C1_1_Relu
  b Loop_C1_1_Write
Loop_C1_1_Relu6:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4s}, [x1], #16
  fadd v0.4s, v0.4s, v16.4s
  fmin v0.4s, v0.4s, v26.4s
  fmax v0.4s, v0.4s, v27.4s
  str s0, [x0]
  add x0, x0, x12
  b Loop_C1_1_Relu6
Loop_C1_1_Relu:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4s}, [x1], #16
  fadd v0.4s, v0.4s, v16.4s
  fmax v0.4s, v0.4s, v27.4s
  str s0, [x0]
  add x0, x0, x12
  b Loop_C1_1_Relu
Loop_C1_1_Write:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4s}, [x1], #16
  fadd v0.4s, v0.4s, v16.4s
  str s0, [x0]
  add x0, x0, x12
  b Loop_C1_1_Write

Loop_C1_2:
  cmp x7, #3
  beq Loop_C1_2_Relu6
  cmp x7, #1
  beq Loop_C1_2_Relu
  b Loop_C1_2_Write
Loop_C1_2_Relu6:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4s}, [x1], #16
  fadd v0.4s, v0.4s, v16.4s
  fmin v0.4s, v0.4s, v26.4s
  fmax v0.4s, v0.4s, v27.4s
  dup s1, v0.s[1]
  stp s0, s1, [x0]
  add x0, x0, x12
  b Loop_C1_2_Relu6
Loop_C1_2_Relu:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4s}, [x1], #16
  fadd v0.4s, v0.4s, v16.4s
  fmax v0.4s, v0.4s, v27.4s
  dup s1, v0.s[1]
  stp s0, s1, [x0]
  add x0, x0, x12
  b Loop_C1_2_Relu
Loop_C1_2_Write:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4s}, [x1], #16
  fadd v0.4s, v0.4s, v16.4s
  dup s1, v0.s[1]
  stp s0, s1, [x0]
  add x0, x0, x12
  b Loop_C1_2_Write

Loop_C1_3:
  add x15, x0, #8
  cmp x7, #3
  beq Loop_C1_3_Relu6
  cmp x7, #1
  beq Loop_C1_3_Relu
  b Loop_C1_3_Write
Loop_C1_3_Relu6:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4s}, [x1], #16
  fadd v0.4s, v0.4s, v16.4s
  fmin v0.4s, v0.4s, v26.4s
  fmax v0.4s, v0.4s, v27.4s
  dup s1, v0.s[1]
  stp s0, s1, [x0]
  add x0, x0, x12
  st1 {v0.s}[2], [x15], x12
  b Loop_C1_3_Relu6
Loop_C1_3_Relu:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4s}, [x1], #16
  fadd v0.4s, v0.4s, v16.4s
  fmax v0.4s, v0.4s, v27.4s
  dup s1, v0.s[1]
  stp s0, s1, [x0]
  add x0, x0, x12
  st1 {v0.s}[2], [x15], x12
  b Loop_C1_3_Relu
Loop_C1_3_Write:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4s}, [x1], #16
  fadd v0.4s, v0.4s, v16.4s
  dup s1, v0.s[1]
  stp s0, s1, [x0]
  add x0, x0, x12
  st1 {v0.s}[2], [x15], x12
  b Loop_C1_3_Write

End:
  ret
#endif
