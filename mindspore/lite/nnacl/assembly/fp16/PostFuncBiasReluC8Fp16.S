#ifdef __aarch64__
#include "nnacl/assembly_global.h"

    .text
    .align 5
    //.p2align 5,,15
    .global PostFuncBiasReluC8Fp16
#ifndef __APPLE__
    .type PostFuncBiasReluC8Fp16, %function
#endif

//void PostFuncBiasReluC8Fp16(float *dst, const float *src, const float *bias, size_t oc8div,size_t oc8mod
//                            size_t plane_size, size_t stride, int relu_type);
// x0 dst           x1 srx           x2 bias
// x3 oc8div        x4 oc8mod        x5 plane_size
// x6 stride        x7 relu_type

// v0 ~ v7 value
// v16  bias data
// x22 x23  x24  x25  write loop tmp buf
// x26  relu6  #6;    x27 relu #0
// w10  oc8 loop control
// w13  hw  loop control

asm_function PostFuncBiasReluC8Fp16
  movi v26.8h, #0x46, lsl #8
  dup v27.8h, wzr
  mov w10, #0

Loop_C8:
  cmp w10, w3
  beq Loop_C1
  mov x25, #2
  mul x24, x10, x25
  add x25, x0, x24
  add w10, w10, #8
  mov w13, w5
  ld1 {v16.8h}, [x2], #16

Loop8x8:
  cmp w13, #8
  blt Loop_4x8
  sub w13, w13, #8
  ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x1], #64
  ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x1], #64

  fadd v0.8h, v0.8h, v16.8h
  fadd v1.8h, v1.8h, v16.8h
  fadd v2.8h, v2.8h, v16.8h
  fadd v3.8h, v3.8h, v16.8h
  fadd v4.8h, v4.8h, v16.8h
  fadd v5.8h, v5.8h, v16.8h
  fadd v6.8h, v6.8h, v16.8h
  fadd v7.8h, v7.8h, v16.8h

  cmp w7, #2
  beq Relu6_8x8
  cmp w7, #1
  beq Relu_8x8
  b Write_8x8
Relu6_8x8:
  fmin v0.8h, v0.8h, v26.8h
  fmin v1.8h, v1.8h, v26.8h
  fmin v2.8h, v2.8h, v26.8h
  fmin v3.8h, v3.8h, v26.8h
  fmin v4.8h, v4.8h, v26.8h
  fmin v5.8h, v5.8h, v26.8h
  fmin v6.8h, v6.8h, v26.8h
  fmin v7.8h, v7.8h, v26.8h
Relu_8x8:
  fmax v0.8h, v0.8h, v27.8h
  fmax v1.8h, v1.8h, v27.8h
  fmax v2.8h, v2.8h, v27.8h
  fmax v3.8h, v3.8h, v27.8h
  fmax v4.8h, v4.8h, v27.8h
  fmax v5.8h, v5.8h, v27.8h
  fmax v6.8h, v6.8h, v27.8h
  fmax v7.8h, v7.8h, v27.8h
Write_8x8:
  st1 {v0.8h}, [x25], x6
  st1 {v1.8h}, [x25], x6
  st1 {v2.8h}, [x25], x6
  st1 {v3.8h}, [x25], x6
  st1 {v4.8h}, [x25], x6
  st1 {v5.8h}, [x25], x6
  st1 {v6.8h}, [x25], x6
  st1 {v7.8h}, [x25], x6
  b Loop8x8

Loop_4x8:
  cmp w13, #4
  blt Loop_1x8
  sub w13, w13, #4
  ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x1], #64

  fadd v0.8h, v0.8h, v16.8h
  fadd v1.8h, v1.8h, v16.8h
  fadd v2.8h, v2.8h, v16.8h
  fadd v3.8h, v3.8h, v16.8h

  cmp w7, #2
  beq Relu6_4x8
  cmp w7, #1
  beq Relu_4x8
  b Write_4x8
Relu6_4x8:
  fmin v0.8h, v0.8h, v26.8h
  fmin v1.8h, v1.8h, v26.8h
  fmin v2.8h, v2.8h, v26.8h
  fmin v3.8h, v3.8h, v26.8h
Relu_4x8:
  fmax v0.8h, v0.8h, v27.8h
  fmax v1.8h, v1.8h, v27.8h
  fmax v2.8h, v2.8h, v27.8h
  fmax v3.8h, v3.8h, v27.8h
Write_4x8:
  st1 {v0.8h}, [x25], x6
  st1 {v1.8h}, [x25], x6
  st1 {v2.8h}, [x25], x6
  st1 {v3.8h}, [x25], x6
  b Loop_4x8

Loop_1x8:
  cmp w7, #2
  beq Relu6_1x8
  cmp w7, #1
  beq Relu_1x8
  b Write_1x8
Relu6_1x8:
  cmp w13, #0
  beq Loop_C8
  sub w13, w13, #1
  ld1 {v0.8h}, [x1], #16
  fadd v0.8h, v0.8h, v16.8h
  fmin v0.8h, v0.8h, v26.8h
  fmax v0.8h, v0.8h, v27.8h
  st1 {v0.8h}, [x25], x6
  b Relu6_1x8
Relu_1x8:
  cmp w13, #0
  beq Loop_C8
  sub w13, w13, #1
  ld1 {v0.8h}, [x1], #16
  fadd v0.8h, v0.8h, v16.8h
  fmax v0.8h, v0.8h, v27.8h
  st1 {v0.8h}, [x25], x6
  b Relu_1x8
Write_1x8:
  cmp w13, #0
  beq Loop_C8
  sub w13, w13, #1
  ld1 {v0.8h}, [x1], #16
  fadd v0.8h, v0.8h, v16.8h
  st1 {v0.8h}, [x25], x6
  b Write_1x8


Loop_C1:
  cmp x4, #0
  beq End
  mov w13, w5
  ld1 {v16.8h}, [x2], #16
  mov x25,  #2
  mul x24, x10, x25
  add x22, x0, x24

  cmp x4, #1
  beq Loop_C1_1
  cmp x4, #2
  beq Loop_C1_2
  cmp x4, #3
  beq Loop_C1_3
  cmp x4, #4
  beq Loop_C1_4
  cmp x4, #5
  beq Loop_C1_5
  cmp x4, #6
  beq Loop_C1_6
  cmp x4, #7
  beq Loop_C1_7

Loop_C1_1:
  cmp w7, #2
  beq Loop_C1_1_Relu6
  cmp w7, #1
  beq Loop_C1_1_Relu
  b Loop_C1_1_Write
Loop_C1_1_Relu6:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.8h}, [x1], #16
  fadd v0.8h, v0.8h, v16.8h
  fmin v0.8h, v0.8h, v26.8h
  fmax v0.8h, v0.8h, v27.8h
  st1 {v0.h}[0], [x22], x6
  b Loop_C1_1_Relu6
Loop_C1_1_Relu:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.8h}, [x1], #16
  fadd v0.8h, v0.8h, v16.8h
  fmax v0.8h, v0.8h, v27.8h
  st1 {v0.h}[0], [x22], x6
  b Loop_C1_1_Relu
Loop_C1_1_Write:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.8h}, [x1], #16
  fadd v0.8h, v0.8h, v16.8h
  st1 {v0.h}[0], [x22], x6
  b Loop_C1_1_Write

Loop_C1_2:
  add x24, x0, #2
  cmp w7, #2
  beq Loop_C1_2_Relu6
  cmp w7, #1
  beq Loop_C1_2_Relu
  b Loop_C1_2_Write
Loop_C1_2_Relu6:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.8h}, [x1], #16
  fadd v0.8h, v0.8h, v16.8h
  fmin v0.8h, v0.8h, v26.8h
  fmax v0.8h, v0.8h, v27.8h
  st1 {v0.h}[0], [x22], x6
  st1 {v0.h}[1], [x24], x6
  b Loop_C1_2_Relu6
Loop_C1_2_Relu:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.8h}, [x1], #16
  fadd v0.8h, v0.8h, v16.8h
  fmax v0.8h, v0.8h, v27.8h
  st1 {v0.h}[0], [x22], x6
  st1 {v0.h}[1], [x24], x6
  b Loop_C1_2_Relu
Loop_C1_2_Write:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.8h}, [x1], #16
  fadd v0.8h, v0.8h, v16.8h
  st1 {v0.h}[0], [x22], x6
  st1 {v0.h}[1], [x24], x6
  b Loop_C1_2_Write


Loop_C1_3:
  add x24, x22, #2
  add x25, x22, #4
  cmp w7, #2
  beq Loop_C1_3_Relu6
  cmp w7, #1
  beq Loop_C1_3_Relu
  b Loop_C1_3_Write
Loop_C1_3_Relu6:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.8h}, [x1], #16
  fadd v0.8h, v0.8h, v16.8h
  fmin v0.8h, v0.8h, v26.8h
  fmax v0.8h, v0.8h, v27.8h
  st1 {v0.h}[0], [x22], x6
  st1 {v0.h}[1], [x24], x6
  st1 {v0.h}[2], [x25], x6
  b Loop_C1_3_Relu6
Loop_C1_3_Relu:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.8h}, [x1], #16
  fadd v0.8h, v0.8h, v16.8h
  fmax v0.8h, v0.8h, v27.8h
  st1 {v0.h}[0], [x22], x6
  st1 {v0.h}[1], [x24], x6
  st1 {v0.h}[2], [x25], x6
  b Loop_C1_3_Relu
Loop_C1_3_Write:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.8h}, [x1], #16
  fadd v0.8h, v0.8h, v16.8h
  st1 {v0.h}[0], [x22], x6
  st1 {v0.h}[1], [x24], x6
  st1 {v0.h}[2], [x25], x6
  b Loop_C1_3_Write

Loop_C1_4:
  cmp w7, #2
  beq Loop_C1_4_Relu6
  cmp w7, #1
  beq Loop_C1_4_Relu
  b Loop_C1_4_Write
Loop_C1_4_Relu6:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.8h}, [x1], #16
  fadd v0.8h, v0.8h, v16.8h
  fmin v0.8h, v0.8h, v26.8h
  fmax v0.8h, v0.8h, v27.8h
  st1 {v0.4h}, [x22], x6
  b Loop_C1_4_Relu6
Loop_C1_4_Relu:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.8h}, [x1], #16
  fadd v0.8h, v0.8h, v16.8h
  fmax v0.8h, v0.8h, v27.8h
  st1 {v0.4h}, [x22], x6
  b Loop_C1_4_Relu6
Loop_C1_4_Write:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.8h}, [x1], #16
  fadd v0.8h, v0.8h, v16.8h
  st1 {v0.4h}, [x22], x6
  b Loop_C1_4_Write

Loop_C1_5:
  add x25, x22, #8
  cmp w7, #2
  beq Loop_C1_5_Relu6
  cmp w7, #1
  beq Loop_C1_5_Relu
  b Loop_C1_5_Write
Loop_C1_5_Relu6:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.8h}, [x1], #16
  fadd v0.8h, v0.8h, v16.8h
  fmin v0.8h, v0.8h, v26.8h
  fmax v0.8h, v0.8h, v27.8h
  st1 {v0.4h}, [x22], x6
  st1 {v0.h}[4], [x25], x6
  b Loop_C1_5_Relu6
Loop_C1_5_Relu:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.8h}, [x1], #16
  fadd v0.8h, v0.8h, v16.8h
  fmax v0.8h, v0.8h, v27.8h
  st1 {v0.4h}, [x22], x6
  st1 {v0.h}[4], [x25], x6
  b Loop_C1_5_Relu
Loop_C1_5_Write:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.8h}, [x1], #16
  fadd v0.8h, v0.8h, v16.8h
  st1 {v0.4h}, [x22], x6
  st1 {v0.h}[4], [x25], x6
  b Loop_C1_5_Write

Loop_C1_6:
  add x23, x22, #8
  add x24, x22, #10
  cmp w7, #2
  beq Loop_C1_6_Relu6
  cmp w7, #1
  beq Loop_C1_6_Relu
  b Loop_C1_6_Write
Loop_C1_6_Relu6:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.8h}, [x1], #16
  fadd v0.8h, v0.8h, v16.8h
  fmin v0.8h, v0.8h, v26.8h
  fmax v0.8h, v0.8h, v27.8h
  st1 {v0.4h}, [x22], x6
  st1 {v0.h}[4], [x23], x6
  st1 {v0.h}[5], [x24], x6
  b Loop_C1_6_Relu6
Loop_C1_6_Relu:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.8h}, [x1], #16
  fadd v0.8h, v0.8h, v16.8h
  fmax v0.8h, v0.8h, v27.8h
  st1 {v0.4h}, [x22], x6
  st1 {v0.h}[4], [x23], x6
  st1 {v0.h}[5], [x24], x6
  b Loop_C1_6_Relu
Loop_C1_6_Write:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.8h}, [x1], #16
  fadd v0.8h, v0.8h, v16.8h
  st1 {v0.4h}, [x22], x6
  st1 {v0.h}[4], [x23], x6
  st1 {v0.h}[5], [x24], x6
  b Loop_C1_6_Write

Loop_C1_7:
  add x23, x22, #8
  add x24, x22, #10
  add x25, x22, #12
  cmp w7, #2
  beq Loop_C1_7_Relu6
  cmp w7, #1
  beq Loop_C1_7_Relu
  b Loop_C1_7_Write
Loop_C1_7_Relu6:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.8h}, [x1], #16
  fadd v0.8h, v0.8h, v16.8h
  fmin v0.8h, v0.8h, v26.8h
  fmax v0.8h, v0.8h, v27.8h
  st1 {v0.4h}, [x22], x6
  st1 {v0.h}[4], [x23], x6
  st1 {v0.h}[5], [x24], x6
  st1 {v0.h}[6], [x25], x6
  b Loop_C1_7_Relu6
Loop_C1_7_Relu:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.8h}, [x1], #16
  fadd v0.8h, v0.8h, v16.8h
  fmax v0.8h, v0.8h, v27.8h
  st1 {v0.4h}, [x22], x6
  st1 {v0.h}[4], [x23], x6
  st1 {v0.h}[5], [x24], x6
  st1 {v0.h}[6], [x25], x6
  b Loop_C1_7_Relu
Loop_C1_7_Write:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.8h}, [x1], #16
  fadd v0.8h, v0.8h, v16.8h
  st1 {v0.4h}, [x22], x6
  st1 {v0.h}[4], [x23], x6
  st1 {v0.h}[5], [x24], x6
  st1 {v0.h}[6], [x25], x6
  b Loop_C1_7_Write

End:
  ret
#endif
