#ifdef __aarch64__
#include "nnacl/assembly_global.h"
    .text
    .align 5
    .global MatVecMulFp32
#ifndef __APPLE__
    .type MatVecMulFp32, %function
#endif

// void MatVecMulFp32(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int col)
// x0: a
// x1: b
// x2: c
// x3: bias
// w4: act_type
// w5: depth
// w6: col

asm_function MatVecMulFp32
  sub sp, sp, #128
  st1 {v8.8h, v9.8h, v10.8h, v11.8h}, [sp], #64
  st1 {v12.8h, v13.8h, v14.8h, v15.8h}, [sp], #64

  mov w14, #4      // sizeof(float)
  mul w8, w14, w5  // rhs depthx1 block stride
  mov w14, #4
  mul w13, w8, w14 // rhs depthx4 block stride 

Loop:
  mov x15, x0     // reload a ptr
  mov x7, x1      // reload b ptr
  mov w9, w5      // reload depth
  cmp w6, #4
  blt Loop1x1  

Loop1x4: 
  dup v10.8h, wzr  
  dup v11.8h, wzr  
  dup v12.8h, wzr
  dup v13.8h, wzr
  dup v14.8h, wzr

  add x10, x7, x8
  add x11, x10, x8
  add x12, x11, x8

Depth8_1x4:
  cmp w9, #8
  blt Depth4_1x4
  sub w9, w9, #8
  ld1 {v0.4s, v1.4s}, [x15], #32
  ld1 {v2.4s, v3.4s}, [x7], #32
  ld1 {v4.4s, v5.4s}, [x10], #32
  cmp w9, #8
  blt Depth8_1x4_Loop_End

Depth8_1x4_Loop:
  fmla v10.4s, v0.4s, v2.4s
  fmla v10.4s, v1.4s, v3.4s
  ld1 {v6.4s, v7.4s}, [x11], #32
  fmla v11.4s, v0.4s, v4.4s
  fmla v11.4s, v1.4s, v5.4s
  ld1 {v8.4s, v9.4s}, [x12], #32
  fmla v12.4s, v0.4s, v6.4s
  fmla v12.4s, v1.4s, v7.4s
  ld1 {v2.4s, v3.4s}, [x7], #32
  fmla v13.4s, v0.4s, v8.4s
  fmla v13.4s, v1.4s, v9.4s
  ld1 {v0.4s, v1.4s}, [x15], #32
  ld1 {v4.4s, v5.4s}, [x10], #32
  sub w9, w9, #8
  cmp w9, #8
  bge Depth8_1x4_Loop

Depth8_1x4_Loop_End:
  fmla v10.4s, v0.4s, v2.4s
  fmla v10.4s, v1.4s, v3.4s
  ld1 {v6.4s, v7.4s}, [x11], #32
  fmla v11.4s, v0.4s, v4.4s
  fmla v11.4s, v1.4s, v5.4s
  ld1 {v8.4s, v9.4s}, [x12], #32
  fmla v12.4s, v0.4s, v6.4s
  fmla v12.4s, v1.4s, v7.4s
  fmla v13.4s, v0.4s, v8.4s
  fmla v13.4s, v1.4s, v9.4s

Depth4_1x4:
  cmp w9, #4
  blt Depth1_1x4
  sub w9, w9, #4
  ld1 {v0.4s}, [x15], #16
  ld1 {v1.4s}, [x7], #16
  ld1 {v2.4s}, [x10], #16
  cmp w9, #4
  blt Depth4_1x4_Loop_End

Depth4_1x4_Loop:
  fmla v10.4s, v1.4s, v0.4s
  ld1 {v3.4s}, [x11], #16
  fmla v11.4s, v2.4s, v0.4s
  ld1 {v4.4s}, [x12], #16
  fmla v12.4s, v3.4s, v0.4s
  ld1 {v1.4s}, [x7], #16
  fmla v13.4s, v4.4s, v0.4s
  ld1 {v0.4s}, [x15], #16
  ld1 {v2.4s}, [x10], #16
  sub w9, w9, #4
  cmp w9, #4
  bge Depth4_1x4_Loop

Depth4_1x4_Loop_End:
  fmla v10.4s, v1.4s, v0.4s
  ld1 {v3.4s}, [x11], #16
  fmla v11.4s, v2.4s, v0.4s
  ld1 {v4.4s}, [x12], #16
  fmla v12.4s, v3.4s, v0.4s
  fmla v13.4s, v4.4s, v0.4s

Depth1_1x4:
  cmp w9, #0
  beq End1x4
  ld1 {v0.s}[0], [x15], #4
  ld1 {v1.s}[0], [x7], #4
  ld1 {v1.s}[1], [x10], #4
  ld1 {v1.s}[2], [x11], #4
  ld1 {v1.s}[3], [x12], #4

  fmla v14.4s, v1.4s, v0.s[0]
  sub w9, w9, #1
  cbz w9, End1x4
  b Depth1_1x4

End1x4:
  faddp v15.4s, v10.4s, v11.4s
  faddp v16.4s, v12.4s, v13.4s
  faddp v17.4s, v15.4s, v16.4s
  fadd v14.4s, v14.4s, v17.4s

  cbz x3, Act1x4
  ld1 {v15.4s}, [x3], #16
  fadd v14.4s, v14.4s, v15.4s   // add bias 

Act1x4:
  cmp w4, #3
  beq Relu6_1x4
  cmp w4, #1
  beq Relu1x4
  b Write1x4

Relu6_1x4:
  movi v15.4s, #0x46, lsl #8
  fmin v14.4s, v14.4s, v15.4s

Relu1x4:
  dup v15.4s, wzr
  fmax v14.4s, v14.4s, v15.4s

Write1x4:
  st1 {v14.4s}, [x2], #16
  sub w6, w6, #4
  cbz w6, End
  add x1, x1, x13
  b Loop


Loop1x1:
  dup v4.4s, wzr
  dup v5.4s, wzr

Depth8_1x1:
  cmp w9, #8
  blt Depth4_1x1

  ld1 {v0.4s, v1.4s}, [x15], #32
  ld1 {v2.4s, v3.4s}, [x7], #32

  fmla v4.4s, v2.4s, v0.4s
  fmla v4.4s, v3.4s, v1.4s
  sub w9, w9, #8
  cbz w9, End1x1
  b Depth8_1x1

Depth4_1x1:
  cmp w9, #4
  blt Depth1_1x1

  ld1 {v0.4s}, [x15], #16
  ld1 {v1.4s}, [x7], #16

  fmla v4.4s, v1.4s, v0.4s
  sub w9, w9, #4
  cbz w9, End1x1
  b Depth8_1x1

Depth1_1x1:
  ld1 {v0.s}[0], [x15], #4
  ld1 {v1.s}[0], [x7], #4

  fmla v5.4s, v1.4s, v0.s[0]
  sub w9, w9, #1
  cbz w9, End1x1
  b Depth1_1x1

End1x1:
  faddp v6.4s, v4.4s, v4.4s  
  faddp v7.4s, v6.4s, v6.4s  
  fadd v7.4s, v7.4s, v5.4s

  cbz x3, Act1x1
  ld1 {v8.s}[0], [x3], #4
  fadd v7.4s, v7.4s, v8.4s    // add bias

Act1x1:
  cmp w4, #3
  beq Relu6_1x1
  cmp w4, #1
  beq Relu1x1
  b Write1x1

Relu6_1x1:
  movi v8.4s, #0x46, lsl #8
  fmin v7.4s, v7.4s, v8.4s

Relu1x1:
  dup v8.4s, wzr
  fmax v7.4s, v7.4s, v8.4s

Write1x1:
  st1 {v7.s}[0], [x2], #4
  sub w6, w6, #1
  cbz w6, End
  add x1, x1, x8
  b Loop

End:
  sub sp, sp, #128
  ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [sp], #64
  ld1 {v12.8h, v13.8h, v14.8h, v15.8h}, [sp], #64
  ret
#endif