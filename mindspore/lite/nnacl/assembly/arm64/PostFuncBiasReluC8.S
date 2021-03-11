#ifdef __aarch64__
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
// x0 dst           x1 srx           x2 bias
// x3 oc8div        x4 oc8mod        x5 plane_size
// x6 stride        x7 relu_type

// v0 ~ v15 value
// v16  v17 bias data
// x14  x15  weite loop tmp buf
// x16  relu6  #6;    x17 relu #0
// w10  oc8 loop control
// w13  hw  loop control

asm_function PostFuncBiasReluC8
  sub sp, sp, #128
  st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
  st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64

  movi v26.4s, #6
  scvtf v26.4s, v26.4s
  dup v27.4s, wzr
  mov w10, #0

Loop_C8:
  cmp w10, w3
  beq Loop_C1
  mov x15,  #4
  mul x14, x10, x15
  add x15, x0, x14
  add w10, w10, #8
  mov w13, w5
  ld1 {v16.4s, v17.4s}, [x2], #32

Loop_8x8:
  cmp w13, #8
  blt Loop_4x8
  sub w13, w13, #8
  ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
  ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x1], #64
  ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x1], #64
  ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x1], #64

  fadd v0.4s, v0.4s, v16.4s
  fadd v1.4s, v1.4s, v17.4s
  fadd v2.4s, v2.4s, v16.4s
  fadd v3.4s, v3.4s, v17.4s
  fadd v4.4s, v4.4s, v16.4s
  fadd v5.4s, v5.4s, v17.4s
  fadd v6.4s, v6.4s, v16.4s
  fadd v7.4s, v7.4s, v17.4s
  fadd v8.4s, v8.4s, v16.4s
  fadd v9.4s, v9.4s, v17.4s
  fadd v10.4s, v10.4s, v16.4s
  fadd v11.4s, v11.4s, v17.4s
  fadd v12.4s, v12.4s, v16.4s
  fadd v13.4s, v13.4s, v17.4s
  fadd v14.4s, v14.4s, v16.4s
  fadd v15.4s, v15.4s, v17.4s

  cmp x7, #3
  beq Relu6_8x8
  cmp x7, #1
  beq Relu_8x8
  b Write_8x8
Relu6_8x8:
  fmin v0.4s, v0.4s, v26.4s
  fmin v1.4s, v1.4s, v26.4s
  fmin v2.4s, v2.4s, v26.4s
  fmin v3.4s, v3.4s, v26.4s
  fmin v4.4s, v4.4s, v26.4s
  fmin v5.4s, v5.4s, v26.4s
  fmin v6.4s, v6.4s, v26.4s
  fmin v7.4s, v7.4s, v26.4s
  fmin v8.4s, v8.4s, v26.4s
  fmin v9.4s, v9.4s, v26.4s
  fmin v10.4s, v10.4s, v26.4s
  fmin v11.4s, v11.4s, v26.4s
  fmin v12.4s, v12.4s, v26.4s
  fmin v13.4s, v13.4s, v26.4s
  fmin v14.4s, v14.4s, v26.4s
  fmin v15.4s, v15.4s, v26.4s
Relu_8x8:
  fmax v0.4s, v0.4s, v27.4s
  fmax v1.4s, v1.4s, v27.4s
  fmax v2.4s, v2.4s, v27.4s
  fmax v3.4s, v3.4s, v27.4s
  fmax v4.4s, v4.4s, v27.4s
  fmax v5.4s, v5.4s, v27.4s
  fmax v6.4s, v6.4s, v27.4s
  fmax v7.4s, v7.4s, v27.4s
  fmax v8.4s, v8.4s, v27.4s
  fmax v9.4s, v9.4s, v27.4s
  fmax v10.4s, v10.4s, v27.4s
  fmax v11.4s, v11.4s, v27.4s
  fmax v12.4s, v12.4s, v27.4s
  fmax v13.4s, v13.4s, v27.4s
  fmax v14.4s, v14.4s, v27.4s
  fmax v15.4s, v15.4s, v27.4s
Write_8x8:
  st1 {v0.4s, v1.4s}, [x15], x6
  st1 {v2.4s, v3.4s}, [x15], x6
  st1 {v4.4s, v5.4s}, [x15], x6
  st1 {v6.4s, v7.4s}, [x15], x6
  st1 {v8.4s, v9.4s}, [x15], x6
  st1 {v10.4s, v11.4s}, [x15], x6
  st1 {v12.4s, v13.4s}, [x15], x6
  st1 {v14.4s, v15.4s}, [x15], x6
  b Loop_8x8

Loop_4x8:
  cmp w13, #4
  blt Loop_1x8
  sub w13, w13, #4
  ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
  ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x1], #64

  fadd v0.4s, v0.4s, v16.4s
  fadd v1.4s, v1.4s, v17.4s
  fadd v2.4s, v2.4s, v16.4s
  fadd v3.4s, v3.4s, v17.4s
  fadd v4.4s, v4.4s, v16.4s
  fadd v5.4s, v5.4s, v17.4s
  fadd v6.4s, v6.4s, v16.4s
  fadd v7.4s, v7.4s, v17.4s

  cmp x7, #3
  beq Relu6_4x8
  cmp x7, #1
  beq Relu_4x8
  b Write_4x8
Relu6_4x8:
  fmin v0.4s, v0.4s, v26.4s
  fmin v1.4s, v1.4s, v26.4s
  fmin v2.4s, v2.4s, v26.4s
  fmin v3.4s, v3.4s, v26.4s
  fmin v4.4s, v4.4s, v26.4s
  fmin v5.4s, v5.4s, v26.4s
  fmin v6.4s, v6.4s, v26.4s
  fmin v7.4s, v7.4s, v26.4s
Relu_4x8:
  fmax v0.4s, v0.4s, v27.4s
  fmax v1.4s, v1.4s, v27.4s
  fmax v2.4s, v2.4s, v27.4s
  fmax v3.4s, v3.4s, v27.4s
  fmax v4.4s, v4.4s, v27.4s
  fmax v5.4s, v5.4s, v27.4s
  fmax v6.4s, v6.4s, v27.4s
  fmax v7.4s, v7.4s, v27.4s
Write_4x8:
  st1 {v0.4s, v1.4s}, [x15], x6
  st1 {v2.4s, v3.4s}, [x15], x6
  st1 {v4.4s, v5.4s}, [x15], x6
  st1 {v6.4s, v7.4s}, [x15], x6

Loop_1x8:
  cmp x7, #3
  beq Relu6_1x8
  cmp x7, #1
  beq Relu_1x8
  b Write_1x8
Relu6_1x8:
  cmp w13, #0
  beq Loop_C8
  sub w13, w13, #1
  ld1 {v0.4s, v1.4s}, [x1], #32
  fadd v0.4s, v0.4s, v16.4s
  fadd v1.4s, v1.4s, v17.4s
  fmin v0.4s, v0.4s, v26.4s
  fmin v1.4s, v1.4s, v26.4s
  fmax v0.4s, v0.4s, v27.4s
  fmax v1.4s, v1.4s, v27.4s
  st1 {v0.4s, v1.4s}, [x15], x6
  b Relu6_1x8
Relu_1x8:
  cmp w13, #0
  beq Loop_C8
  sub w13, w13, #1
  ld1 {v0.4s, v1.4s}, [x1], #32
  fadd v0.4s, v0.4s, v16.4s
  fadd v1.4s, v1.4s, v17.4s
  fmax v0.4s, v0.4s, v27.4s
  fmax v1.4s, v1.4s, v27.4s
  st1 {v0.4s, v1.4s}, [x15], x6
  b Relu_1x8
Write_1x8:
  cmp w13, #0
  beq Loop_C8
  sub w13, w13, #1
  ld1 {v0.4s, v1.4s}, [x1], #32
  fadd v0.4s, v0.4s, v16.4s
  fadd v1.4s, v1.4s, v17.4s
  st1 {v0.4s, v1.4s}, [x15], x6
  b Write_1x8


Loop_C1:
  cmp x4, #0
  beq End
  mov w13, w5
  ld1 {v16.4s, v17.4s}, [x2], #32
  mov x15,  #4
  mul x14, x10, x15
  add x0, x0, x14

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
  cmp x7, #3
  beq Loop_C1_1_Relu6
  cmp x7, #1
  beq Loop_C1_1_Relu
  b Loop_C1_1_Write
Loop_C1_1_Relu6:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4s, v1.4s}, [x1], #32
  fadd v0.4s, v0.4s, v16.4s
  fmin v0.4s, v0.4s, v26.4s
  fmax v0.4s, v0.4s, v27.4s
  str s0, [x0]
  add x0, x0, x6
  b Loop_C1_1_Relu6
Loop_C1_1_Relu:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4s, v1.4s}, [x1], #32
  fadd v0.4s, v0.4s, v16.4s
  fmax v0.4s, v0.4s, v27.4s
  str s0, [x0]
  add x0, x0, x6
  b Loop_C1_1_Relu
Loop_C1_1_Write:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4s, v1.4s}, [x1], #32
  fadd v0.4s, v0.4s, v16.4s
  str s0, [x0]
  add x0, x0, x6
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
  ld1 {v0.4s, v1.4s}, [x1], #32
  fadd v0.4s, v0.4s, v16.4s
  fmin v0.4s, v0.4s, v26.4s
  fmax v0.4s, v0.4s, v27.4s
  dup s1, v0.s[1]
  stp s0, s1, [x0]
  add x0, x0, x6
  b Loop_C1_2_Relu6
Loop_C1_2_Relu:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4s, v1.4s}, [x1], #32
  fadd v0.4s, v0.4s, v16.4s
  fmax v0.4s, v0.4s, v27.4s
  dup s1, v0.s[1]
  stp s0, s1, [x0]
  add x0, x0, x6
  b Loop_C1_2_Relu
Loop_C1_2_Write:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4s, v1.4s}, [x1], #32
  fadd v0.4s, v0.4s, v16.4s
  dup s1, v0.s[1]
  stp s0, s1, [x0]
  add x0, x0, x6
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
  ld1 {v0.4s, v1.4s}, [x1], #32
  fadd v0.4s, v0.4s, v16.4s
  fmin v0.4s, v0.4s, v26.4s
  fmax v0.4s, v0.4s, v27.4s
  dup s1, v0.s[1]
  stp s0, s1, [x0]
  add x0, x0, x6
  st1 {v0.s}[2], [x15], x6
  b Loop_C1_3_Relu6
Loop_C1_3_Relu:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4s, v1.4s}, [x1], #32
  fadd v0.4s, v0.4s, v16.4s
  fmax v0.4s, v0.4s, v27.4s
  dup s1, v0.s[1]
  stp s0, s1, [x0]
  add x0, x0, x6
  st1 {v0.s}[2], [x15], x6
  b Loop_C1_3_Relu
Loop_C1_3_Write:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4s, v1.4s}, [x1], #32
  fadd v0.4s, v0.4s, v16.4s
  dup s1, v0.s[1]
  stp s0, s1, [x0]
  add x0, x0, x6
  st1 {v0.s}[2], [x15], x6
  b Loop_C1_3_Write

Loop_C1_4:
  cmp x7, #3
  beq Loop_C1_4_Relu6
  cmp x7, #1
  beq Loop_C1_4_Relu
  b Loop_C1_4_Write
Loop_C1_4_Relu6:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4s, v1.4s}, [x1], #32
  fadd v0.4s, v0.4s, v16.4s
  fmin v0.4s, v0.4s, v26.4s
  fmax v0.4s, v0.4s, v27.4s
  st1 {v0.4s}, [x0], x6
  b Loop_C1_4_Relu6
Loop_C1_4_Relu:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4s, v1.4s}, [x1], #32
  fadd v0.4s, v0.4s, v16.4s
  fmax v0.4s, v0.4s, v27.4s
  st1 {v0.4s}, [x0], x6
  b Loop_C1_4_Relu6
Loop_C1_4_Write:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4s, v1.4s}, [x1], #32
  fadd v0.4s, v0.4s, v16.4s
  st1 {v0.4s}, [x0], x6
  b Loop_C1_4_Write

Loop_C1_5:
  add x15, x0, #16
  cmp x7, #3
  beq Loop_C1_5_Relu6
  cmp x7, #1
  beq Loop_C1_5_Relu
  b Loop_C1_5_Write
Loop_C1_5_Relu6:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4s, v1.4s}, [x1], #32
  fadd v0.4s, v0.4s, v16.4s
  fadd v1.4s, v1.4s, v17.4s
  fmin v0.4s, v0.4s, v26.4s
  fmin v1.4s, v1.4s, v26.4s
  fmax v0.4s, v0.4s, v27.4s
  fmax v1.4s, v1.4s, v27.4s
  st1 {v0.4s}, [x0], x6
  str s1, [x15]
  add x15, x15, x6
  b Loop_C1_5_Relu6
Loop_C1_5_Relu:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4s, v1.4s}, [x1], #32
  fadd v0.4s, v0.4s, v16.4s
  fadd v1.4s, v1.4s, v17.4s
  fmax v0.4s, v0.4s, v27.4s
  fmax v1.4s, v1.4s, v27.4s
  st1 {v0.4s}, [x0], x6
  str s1, [x15]
  add x15, x15, x6
  b Loop_C1_5_Relu
Loop_C1_5_Write:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4s, v1.4s}, [x1], #32
  fadd v0.4s, v0.4s, v16.4s
  fadd v1.4s, v1.4s, v17.4s
  st1 {v0.4s}, [x0], x6
  str s1, [x15]
  add x15, x15, x6
  b Loop_C1_5_Write

Loop_C1_6:
  add x15, x0, #16
  cmp x7, #3
  beq Loop_C1_6_Relu6
  cmp x7, #1
  beq Loop_C1_6_Relu
  b Loop_C1_6_Write
Loop_C1_6_Relu6:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4s, v1.4s}, [x1], #32
  fadd v0.4s, v0.4s, v16.4s
  fadd v1.4s, v1.4s, v17.4s
  fmin v0.4s, v0.4s, v26.4s
  fmin v1.4s, v1.4s, v26.4s
  fmax v0.4s, v0.4s, v27.4s
  fmax v1.4s, v1.4s, v27.4s
  st1 {v0.4s}, [x0], x6
  dup s0, v1.s[1]
  stp s1, s0, [x15]
  add x15, x15, x6
  b Loop_C1_6_Relu6
Loop_C1_6_Relu:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4s, v1.4s}, [x1], #32
  fadd v0.4s, v0.4s, v16.4s
  fadd v1.4s, v1.4s, v17.4s
  fmax v0.4s, v0.4s, v27.4s
  fmax v1.4s, v1.4s, v27.4s
  st1 {v0.4s}, [x0], x6
  dup s0, v1.s[1]
  stp s1, s0, [x15]
  add x15, x15, x6
  b Loop_C1_6_Relu
Loop_C1_6_Write:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4s, v1.4s}, [x1], #32
  fadd v0.4s, v0.4s, v16.4s
  fadd v1.4s, v1.4s, v17.4s
  st1 {v0.4s}, [x0], x6
  dup s0, v1.s[1]
  stp s1, s0, [x15]
  add x15, x15, x6
  b Loop_C1_6_Write

Loop_C1_7:
  add x15, x0, #16
  add x14, x0, #24
  cmp x7, #3
  beq Loop_C1_7_Relu6
  cmp x7, #1
  beq Loop_C1_7_Relu
  b Loop_C1_7_Write
Loop_C1_7_Relu6:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4s, v1.4s}, [x1], #32
  fadd v0.4s, v0.4s, v16.4s
  fadd v1.4s, v1.4s, v17.4s
  fmin v0.4s, v0.4s, v26.4s
  fmin v1.4s, v1.4s, v26.4s
  fmax v0.4s, v0.4s, v27.4s
  fmax v1.4s, v1.4s, v27.4s
  st1 {v0.4s}, [x0], x6
  dup s0, v1.s[1]
  stp s1, s0, [x15]
  add x15, x15, x6
  st1 {v1.s}[2], [x14], x6
  b Loop_C1_7_Relu6
Loop_C1_7_Relu:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4s, v1.4s}, [x1], #32
  fadd v0.4s, v0.4s, v16.4s
  fadd v1.4s, v1.4s, v17.4s
  fmax v0.4s, v0.4s, v27.4s
  fmax v1.4s, v1.4s, v27.4s
  st1 {v0.4s}, [x0], x6
  dup s0, v1.s[1]
  stp s1, s0, [x15]
  add x15, x15, x6
  st1 {v1.s}[2], [x14], x6
  b Loop_C1_7_Relu
Loop_C1_7_Write:
  cmp w13, #0
  beq End
  sub w13, w13, #1
  ld1 {v0.4s, v1.4s}, [x1], #32
  fadd v0.4s, v0.4s, v16.4s
  fadd v1.4s, v1.4s, v17.4s
  st1 {v0.4s}, [x0], x6
  dup s0, v1.s[1]
  stp s1, s0, [x15]
  add x15, x15, x6
  st1 {v1.s}[2], [x14], x6
  b Loop_C1_7_Write

End:
  sub sp, sp, #128
  ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
  ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
  ret
#endif
