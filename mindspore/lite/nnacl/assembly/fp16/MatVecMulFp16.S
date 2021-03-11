#ifdef __aarch64__
#include "nnacl/assembly_global.h"
    .text
    .align 5
    .global MatVecMulFp16Neon64
#ifndef __APPLE__
    .type MatVecMulFp16Neon64, %function
#endif

// void MatVecMulFp16Neon64(const float16_t *a, const float16_t *b, float16_t *c, const float16_t *bias, int act_type, int depth, int col)
// x0: a
// x1: b
// x2: c
// x3: bias
// w4: act_type
// w5: depth
// w6: col

asm_function MatVecMulFp16Neon64
  sub sp, sp, #128
  st1 {v8.8h, v9.8h, v10.8h, v11.8h}, [sp], #64
  st1 {v12.8h, v13.8h, v14.8h, v15.8h}, [sp], #64

  mov w14, #2      // sizeof(float16)
  mul w8, w14, w5  // rhs depthx1 block stride
  mov w14, #4
  mul w13, w8, w14 // rhs depthx4 block stride 

Loop:
  mov x15, x0    // reload a ptr
  mov x7, x1    // reload b ptr
  mov w9, w5    // reload depth
  cmp w6, #4
  blt Loop1x1  

Loop1x4:
  dup v5.8h, wzr  
  dup v6.8h, wzr  
  dup v7.8h, wzr  
  dup v8.8h, wzr  
  dup v9.8h, wzr  
  dup v10.8h, wzr  
  dup v11.8h, wzr  
  dup v12.8h, wzr
  dup v13.8h, wzr

  add x10, x7, x8
  add x11, x10, x8
  add x12, x11, x8

Depth8_1x4:
  cmp w9, #8
  blt Depth1_1x4

  ld1 {v0.8h}, [x15], #16
  ld1 {v1.8h}, [x7], #16
  ld1 {v2.8h}, [x10], #16
  ld1 {v3.8h}, [x11], #16
  ld1 {v4.8h}, [x12], #16

  fmla v5.8h, v1.8h, v0.8h
  fmla v6.8h, v2.8h, v0.8h
  fmla v7.8h, v3.8h, v0.8h
  fmla v8.8h, v4.8h, v0.8h
  sub w9, w9, #8
  cbz w9, End1x4
  b Depth8_1x4

Depth1_1x4:
  ld1 {v0.h}[0], [x15], #2
  ld1 {v1.h}[0], [x7], #2
  ld1 {v1.h}[1], [x10], #2
  ld1 {v1.h}[2], [x11], #2
  ld1 {v1.h}[3], [x12], #2

  fmla v9.8h, v1.8h, v0.h[0]
  sub w9, w9, #1
  cbz w9, End1x4
  b Depth1_1x4

End1x4:
  faddp v10.8h, v5.8h, v6.8h
  faddp v11.8h, v7.8h, v8.8h
  faddp v12.8h, v10.8h, v11.8h
  faddp v13.8h, v12.8h, v12.8h
  fadd v13.8h, v13.8h, v9.8h

  cbz x3, Act1x4
  ld1 {v14.4h}, [x3], #8
  fadd v13.8h, v13.8h, v14.8h 

Act1x4:
  cmp w4, #3
  beq Relu6_1x4
  cmp w4, #1
  beq Relu1x4
  b Write1x4

Relu6_1x4:
  movi v14.8h, #0x46, lsl #8
  fmin v13.8h, v13.8h, v14.8h

Relu1x4:
  dup v14.8h, wzr
  fmax v13.8h, v13.8h, v14.8h

Write1x4:
  st1 {v13.4h}, [x2], #8
  sub w6, w6, #4
  cbz w6, End
  add x1, x1, x13
  b Loop

Loop1x1:
  dup v2.8h, wzr
  dup v3.8h, wzr
  dup v4.8h, wzr
  dup v5.8h, wzr
  dup v6.8h, wzr

Depth8_1x1:
  cmp w9, #8
  blt Depth1_1x1

  ld1 {v0.8h}, [x15], #16
  ld1 {v1.8h}, [x7], #16

  fmla v2.8h, v1.8h, v0.8h
  sub w9, w9, #8
  cbz w9, End1x1
  b Depth8_1x1

Depth1_1x1:
  ld1 {v0.h}[0], [x15], #2
  ld1 {v1.h}[0], [x7], #2

  fmla v3.8h, v1.8h, v0.h[0]
  sub w9, w9, #1
  cbz w9, End1x1
  b Depth1_1x1

End1x1:
  faddp v4.8h, v2.8h, v2.8h  
  faddp v5.8h, v4.8h, v4.8h  
  faddp v6.8h, v5.8h, v5.8h  
  fadd v6.8h, v6.8h, v3.8h

  cbz x3, Act1x1
  ld1 {v7.h}[0], [x3], #2
  fadd v6.8h, v6.8h, v7.8h 

Act1x1:
  cmp w4, #3
  beq Relu6_1x1
  cmp w4, #1
  beq Relu1x1
  b Write1x1

Relu6_1x1:
  movi v7.8h, #0x46, lsl #8
  fmin v6.8h, v6.8h, v7.8h

Relu1x1:
  dup v7.8h, wzr
  fmax v6.8h, v6.8h, v7.8h

Write1x1:
  st1 {v6.h}[0], [x2], #2
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