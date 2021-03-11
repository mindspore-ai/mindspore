#include "nnacl/assembly_global.h"

    .text
    .align 5
    //.p2align 5,,15
    .global PostFuncBiasReluC4Fp16
#ifndef __APPLE__
    .type PostFuncBiasReluC4Fp16, %function
#endif

//void PostFuncBiasReluC4Fp16(float16_t *dst, const float16_t *src, const float16_t *bias, size_t oc4div, size_t oc4mod,
//                            size_t plane_size, size_t plane_stride, size_t relu_type);
// x0 dst           x1 srx           x2 bias
// w3 oc4div        w4 oc4mod        w5 plane_size
// x6 plane_stride  x7 relu_type

asm_function PostFuncBiasReluC4Fp16

  movi v26.4h, #6
  scvtf v26.4h, v26.4h
  dup v27.4h, wzr

  mov x10, #2
  add x12, x3, x4
  mul x12, x12, x10

  mov w10, #0

Loop_C4:
  cmp w10, w3
  beq Loop_C1
  mov x15, #2
  mul x14, x10, x15
  add x15, x0, x14
  add w10, w10, #4
  mov w13, w5
  ld1 {v16.4h}, [x2], #8

Loop_8x4:
  cmp w13, #8
  blt Loop_4x4
  sub w13, w13, #8
  ld1 {v0.4h, v1.4h, v2.4h, v3.4h}, [x1], #32
  ld1 {v4.4h, v5.4h, v6.4h, v7.4h}, [x1], #32

  fadd v0.4h, v0.4h, v16.4h
  fadd v1.4h, v1.4h, v16.4h
  fadd v2.4h, v2.4h, v16.4h
  fadd v3.4h, v3.4h, v16.4h
  fadd v4.4h, v4.4h, v16.4h
  fadd v5.4h, v5.4h, v16.4h
  fadd v6.4h, v6.4h, v16.4h
  fadd v7.4h, v7.4h, v16.4h

  cmp x7, #3
  beq Relu6_8x4
  cmp x7, #1
  beq Relu_8x4
  b Write_8x4
Relu6_8x4:
  fmin v0.4h, v0.4h, v26.4h
  fmin v1.4h, v1.4h, v26.4h
  fmin v2.4h, v2.4h, v26.4h
  fmin v3.4h, v3.4h, v26.4h
  fmin v4.4h, v4.4h, v26.4h
  fmin v5.4h, v5.4h, v26.4h
  fmin v6.4h, v6.4h, v26.4h
  fmin v7.4h, v7.4h, v26.4h
Relu_8x4:
  fmax v0.4h, v0.4h, v27.4h
  fmax v1.4h, v1.4h, v27.4h
  fmax v2.4h, v2.4h, v27.4h
  fmax v3.4h, v3.4h, v27.4h
  fmax v4.4h, v4.4h, v27.4h
  fmax v5.4h, v5.4h, v27.4h
  fmax v6.4h, v6.4h, v27.4h
  fmax v7.4h, v7.4h, v27.4h
Write_8x4:
  st1 {v0.4h}, [x15], x12
  st1 {v1.4h}, [x15], x12
  st1 {v2.4h}, [x15], x12
  st1 {v3.4h}, [x15], x12
  st1 {v4.4h}, [x15], x12
  st1 {v5.4h}, [x15], x12
  st1 {v6.4h}, [x15], x12
  st1 {v7.4h}, [x15], x12
  b Loop_8x4

Loop_4x4:
  cmp w13, #4
  blt Loop_1x4
  sub w13, w13, #4
  ld1 {v0.4h, v1.4h, v2.4h, v3.4h}, [x1], #32
  fadd v0.4h, v0.4h, v16.4h
  fadd v1.4h, v1.4h, v16.4h
  fadd v2.4h, v2.4h, v16.4h
  fadd v3.4h, v3.4h, v16.4h
  cmp x7, #3
  beq Relu6_4x4
  cmp x7, #1
  beq Relu_4x4
  b Write_4x4
Relu6_4x4:
  fmin v0.4h, v0.4h, v26.4h
  fmin v1.4h, v1.4h, v26.4h
  fmin v2.4h, v2.4h, v26.4h
  fmin v3.4h, v3.4h, v26.4h
Relu_4x4:
  fmax v0.4h, v0.4h, v27.4h
  fmax v1.4h, v1.4h, v27.4h
  fmax v2.4h, v2.4h, v27.4h
  fmax v3.4h, v3.4h, v27.4h
Write_4x4:
  st1 {v0.4h}, [x15], x12
  st1 {v1.4h}, [x15], x12
  st1 {v2.4h}, [x15], x12
  st1 {v3.4h}, [x15], x12

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
  ld1 {v0.4h}, [x1], #8
  fadd v0.4h, v0.4h, v16.4h
  fmin v0.4h, v0.4h, v26.4h
  fmax v0.4h, v0.4h, v27.4h
  st1 {v0.4h}, [x15], x12
  b Relu6_1x4
Relu_1x4:
  cmp w13, #0
  beq HW_Add
  sub w13, w13, #1
  ld1 {v0.4h}, [x1], #8
  fadd v0.4h, v0.4h, v16.4h
  fmax v0.4h, v0.4h, v27.4h
  st1 {v0.4h}, [x15], x12
  b Relu_1x4
Write_1x4:
  cmp w13, #0
  beq HW_Add
  sub w13, w13, #1
  ld1 {v0.4h}, [x1], #8
  fadd v0.4h, v0.4h, v16.4h
  st1 {v0.4h}, [x15], x12
  b Write_1x4

HW_Add:
  add x1, x1, x6
  b Loop_C4

Loop_C1:
  cmp w4, #0
  beq End
  mov w13, w5
  ld1 {v16.4h}, [x2], #8
  mov x15,  #2
  mul x14, x10, x15
  add x0, x0, x14

  cmp w4, #1
  beq Loop_C1_1
  cmp w4, #2
  beq Loop_C1_2
  cmp w4, #3
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
  ld1 {v0.4h}, [x1], #8
  fadd v0.4h, v0.4h, v16.4h
  fmin v0.4h, v0.4h, v26.4h
  fmax v0.4h, v0.4h, v27.4h
  st1 {v0.h}[0], [x0], x12
  b Loop_C1_1_Relu6
Loop_C1_1_Relu:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4h}, [x1], #8
  fadd v0.4h, v0.4h, v16.4h
  fmax v0.4h, v0.4h, v27.4h
  st1 {v0.h}[0], [x0], x12
  b Loop_C1_1_Relu
Loop_C1_1_Write:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4h}, [x1], #8
  fadd v0.4h, v0.4h, v16.4h
  st1 {v0.h}[0], [x0], x12
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
  ld1 {v0.4h}, [x1], #8
  fadd v0.4h, v0.4h, v16.4h
  fmin v0.4h, v0.4h, v26.4h
  fmax v0.4h, v0.4h, v27.4h
  st1 {v0.s}[0], [x0], x12
  b Loop_C1_2_Relu6
Loop_C1_2_Relu:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4h}, [x1], #8
  fadd v0.4h, v0.4h, v16.4h
  fmax v0.4h, v0.4h, v27.4h
  st1 {v0.s}[0], [x0], x12
  b Loop_C1_2_Relu
Loop_C1_2_Write:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4h}, [x1], #8
  fadd v0.4h, v0.4h, v16.4h
  st1 {v0.s}[0], [x0], x12
  b Loop_C1_2_Write

Loop_C1_3:
  add x15, x0, #4
  cmp x7, #3
  beq Loop_C1_3_Relu6
  cmp x7, #1
  beq Loop_C1_3_Relu
  b Loop_C1_3_Write
Loop_C1_3_Relu6:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4h}, [x1], #8
  fadd v0.4h, v0.4h, v16.4h
  fmin v0.4h, v0.4h, v26.4h
  fmax v0.4h, v0.4h, v27.4h
  st1 {v0.s}[0], [x0], x12
  st1 {v0.h}[2], [x15], x12
  b Loop_C1_3_Relu6
Loop_C1_3_Relu:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4h}, [x1], #8
  fadd v0.4h, v0.4h, v16.4h
  fmax v0.4h, v0.4h, v27.4h
  st1 {v0.s}[0], [x0], x12
  st1 {v0.h}[2], [x15], x12
  b Loop_C1_3_Relu
Loop_C1_3_Write:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4h}, [x1], #8
  fadd v0.4h, v0.4h, v16.4h
  st1 {v0.s}[0], [x0], x12
  st1 {v0.h}[2], [x15], x12
  b Loop_C1_3_Write

End:
  ret
