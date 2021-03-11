#ifdef __aarch64__
#include "nnacl/assembly_global.h"
    .text
    .align 5
    .global MatmulFp16Neon64
#ifndef __APPLE__
    .type MatmulFp16Neon64, %function
#endif

// void MatmulFp16Neon64(const float16_t *a, const float16_t *b, float16_t *c, const float16_t *bias, int act_type,
//                        int depth, int row, int col, int stride, bool write_nhwc)
// x0: a
// x1: b
// x2: c
// x3: bias
// w4: act_type
// w5: depth
// w6: row
// w7: col
// w17: stride
// w13: writeC8

asm_function MatmulFp16Neon64
  sub sp, sp, #128
  st1 {v8.8h, v9.8h, v10.8h, v11.8h}, [sp], #64
  st1 {v12.8h, v13.8h, v14.8h, v15.8h}, [sp], #64

  mov w18, #16 // sizeof(float16) * 8
  mul w15, w5, w18 // block stride of lhs/rhs: sizeof(float16) * 8 * depth
  mov x11, x3 // bias flag
  mov x18, #2
  ldr x17, [sp]
  mul x17, x17, x18

L1:
  mov w10, w6 // reload lhs row
  mov x12, x0 // reload lhs ptr
  mov x18, x2 // reload dst ptr

L2:
  mov x16, x1 // reload rhs ptr
  mov w13, w5 // reload depth
  mov x14, x3 // reload bias ptr
  dup v16.4s, wzr
  dup v17.4s, wzr
  dup v18.4s, wzr
  dup v19.4s, wzr
  dup v20.4s, wzr
  dup v21.4s, wzr
  dup v22.4s, wzr
  dup v23.4s, wzr
  dup v24.4s, wzr
  dup v25.4s, wzr
  dup v26.4s, wzr
  dup v27.4s, wzr
  dup v28.4s, wzr
  dup v29.4s, wzr
  dup v30.4s, wzr
  dup v31.4s, wzr

  cmp w13, #8
  blt CommLoopMul

OptLoopMul8:
  ld1 {v0.8h, v1.8h}, [x12], #32
  ld1 {v8.8h, v9.8h}, [x16], #32
  fmla v16.8h, v8.8h, v0.h[0]
  fmla v17.8h, v8.8h, v0.h[1]
  fmla v18.8h, v8.8h, v0.h[2]
  fmla v19.8h, v8.8h, v0.h[3]
  fmla v20.8h, v8.8h, v0.h[4]
  fmla v21.8h, v8.8h, v0.h[5]
  fmla v22.8h, v8.8h, v0.h[6]
  fmla v23.8h, v8.8h, v0.h[7]
  ld1 {v2.8h, v3.8h}, [x12], #32
  fmla v24.8h, v8.8h, v1.h[0]
  fmla v25.8h, v8.8h, v1.h[1]
  fmla v26.8h, v8.8h, v1.h[2]
  fmla v27.8h, v8.8h, v1.h[3]
  fmla v28.8h, v8.8h, v1.h[4]
  fmla v29.8h, v8.8h, v1.h[5]
  fmla v30.8h, v8.8h, v1.h[6]
  fmla v31.8h, v8.8h, v1.h[7]
  ld1 {v10.8h, v11.8h}, [x16], #32
  fmla v16.8h, v9.8h, v2.h[0]
  fmla v17.8h, v9.8h, v2.h[1]
  fmla v18.8h, v9.8h, v2.h[2]
  fmla v19.8h, v9.8h, v2.h[3]
  fmla v20.8h, v9.8h, v2.h[4]
  fmla v21.8h, v9.8h, v2.h[5]
  fmla v22.8h, v9.8h, v2.h[6]
  fmla v23.8h, v9.8h, v2.h[7]
  ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x12], #64
  fmla v24.8h, v9.8h, v3.h[0]
  fmla v25.8h, v9.8h, v3.h[1]
  fmla v26.8h, v9.8h, v3.h[2]
  fmla v27.8h, v9.8h, v3.h[3]
  fmla v28.8h, v9.8h, v3.h[4]
  fmla v29.8h, v9.8h, v3.h[5]
  fmla v30.8h, v9.8h, v3.h[6]
  fmla v31.8h, v9.8h, v3.h[7]
  ld1 {v12.8h, v13.8h, v14.8h, v15.8h}, [x16], #64
  fmla v16.8h, v10.8h, v4.h[0]
  fmla v17.8h, v10.8h, v4.h[1]
  fmla v18.8h, v10.8h, v4.h[2]
  fmla v19.8h, v10.8h, v4.h[3]
  fmla v20.8h, v10.8h, v4.h[4]
  fmla v21.8h, v10.8h, v4.h[5]
  fmla v22.8h, v10.8h, v4.h[6]
  fmla v23.8h, v10.8h, v4.h[7]
  ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x12], #64
  fmla v24.8h, v10.8h, v5.h[0]
  fmla v25.8h, v10.8h, v5.h[1]
  fmla v26.8h, v10.8h, v5.h[2]
  fmla v27.8h, v10.8h, v5.h[3]
  fmla v28.8h, v10.8h, v5.h[4]
  fmla v29.8h, v10.8h, v5.h[5]
  fmla v30.8h, v10.8h, v5.h[6]
  fmla v31.8h, v10.8h, v5.h[7]
  ld1 {v4.8h, v5.8h}, [x12], #32
  fmla v16.8h, v11.8h, v6.h[0]
  fmla v17.8h, v11.8h, v6.h[1]
  fmla v18.8h, v11.8h, v6.h[2]
  fmla v19.8h, v11.8h, v6.h[3]
  fmla v20.8h, v11.8h, v6.h[4]
  fmla v21.8h, v11.8h, v6.h[5]
  fmla v22.8h, v11.8h, v6.h[6]
  fmla v23.8h, v11.8h, v6.h[7]
  fmla v24.8h, v11.8h, v7.h[0]
  fmla v25.8h, v11.8h, v7.h[1]
  fmla v26.8h, v11.8h, v7.h[2]
  fmla v27.8h, v11.8h, v7.h[3]
  fmla v28.8h, v11.8h, v7.h[4]
  fmla v29.8h, v11.8h, v7.h[5]
  fmla v30.8h, v11.8h, v7.h[6]
  fmla v31.8h, v11.8h, v7.h[7]
  ld1 {v6.8h, v7.8h}, [x12], #32
  fmla v16.8h, v12.8h, v0.h[0]
  fmla v17.8h, v12.8h, v0.h[1]
  fmla v18.8h, v12.8h, v0.h[2]
  fmla v19.8h, v12.8h, v0.h[3]
  fmla v20.8h, v12.8h, v0.h[4]
  fmla v21.8h, v12.8h, v0.h[5]
  fmla v22.8h, v12.8h, v0.h[6]
  fmla v23.8h, v12.8h, v0.h[7]
  fmla v24.8h, v12.8h, v1.h[0]
  fmla v25.8h, v12.8h, v1.h[1]
  fmla v26.8h, v12.8h, v1.h[2]
  fmla v27.8h, v12.8h, v1.h[3]
  fmla v28.8h, v12.8h, v1.h[4]
  fmla v29.8h, v12.8h, v1.h[5]
  fmla v30.8h, v12.8h, v1.h[6]
  fmla v31.8h, v12.8h, v1.h[7]
  fmla v16.8h, v13.8h, v2.h[0]
  fmla v17.8h, v13.8h, v2.h[1]
  fmla v18.8h, v13.8h, v2.h[2]
  fmla v19.8h, v13.8h, v2.h[3]
  fmla v20.8h, v13.8h, v2.h[4]
  fmla v21.8h, v13.8h, v2.h[5]
  fmla v22.8h, v13.8h, v2.h[6]
  fmla v23.8h, v13.8h, v2.h[7]
  fmla v24.8h, v13.8h, v3.h[0]
  fmla v25.8h, v13.8h, v3.h[1]
  fmla v26.8h, v13.8h, v3.h[2]
  fmla v27.8h, v13.8h, v3.h[3]
  fmla v28.8h, v13.8h, v3.h[4]
  fmla v29.8h, v13.8h, v3.h[5]
  fmla v30.8h, v13.8h, v3.h[6]
  fmla v31.8h, v13.8h, v3.h[7]
  fmla v16.8h, v14.8h, v4.h[0]
  fmla v17.8h, v14.8h, v4.h[1]
  fmla v18.8h, v14.8h, v4.h[2]
  fmla v19.8h, v14.8h, v4.h[3]
  fmla v20.8h, v14.8h, v4.h[4]
  fmla v21.8h, v14.8h, v4.h[5]
  fmla v22.8h, v14.8h, v4.h[6]
  fmla v23.8h, v14.8h, v4.h[7]
  fmla v24.8h, v14.8h, v5.h[0]
  fmla v25.8h, v14.8h, v5.h[1]
  fmla v26.8h, v14.8h, v5.h[2]
  fmla v27.8h, v14.8h, v5.h[3]
  fmla v28.8h, v14.8h, v5.h[4]
  fmla v29.8h, v14.8h, v5.h[5]
  fmla v30.8h, v14.8h, v5.h[6]
  fmla v31.8h, v14.8h, v5.h[7]
  fmla v16.8h, v15.8h, v6.h[0]
  fmla v17.8h, v15.8h, v6.h[1]
  fmla v18.8h, v15.8h, v6.h[2]
  fmla v19.8h, v15.8h, v6.h[3]
  fmla v20.8h, v15.8h, v6.h[4]
  fmla v21.8h, v15.8h, v6.h[5]
  fmla v22.8h, v15.8h, v6.h[6]
  fmla v23.8h, v15.8h, v6.h[7]
  fmla v24.8h, v15.8h, v7.h[0]
  fmla v25.8h, v15.8h, v7.h[1]
  fmla v26.8h, v15.8h, v7.h[2]
  fmla v27.8h, v15.8h, v7.h[3]
  fmla v28.8h, v15.8h, v7.h[4]
  fmla v29.8h, v15.8h, v7.h[5]
  fmla v30.8h, v15.8h, v7.h[6]
  fmla v31.8h, v15.8h, v7.h[7]

  sub w13, w13, #8
  cmp w13, #0
  ble Bias
  cmp w13, #8
  bge OptLoopMul8

CommLoopMul:
  ld1 {v0.8h, v1.8h}, [x12], #32
  ld1 {v8.8h}, [x16], #16
  fmla v16.8h, v8.8h, v0.h[0]
  fmla v17.8h, v8.8h, v0.h[1]
  fmla v18.8h, v8.8h, v0.h[2]
  fmla v19.8h, v8.8h, v0.h[3]
  fmla v20.8h, v8.8h, v0.h[4]
  fmla v21.8h, v8.8h, v0.h[5]
  fmla v22.8h, v8.8h, v0.h[6]
  fmla v23.8h, v8.8h, v0.h[7]
  fmla v24.8h, v8.8h, v1.h[0]
  fmla v25.8h, v8.8h, v1.h[1]
  fmla v26.8h, v8.8h, v1.h[2]
  fmla v27.8h, v8.8h, v1.h[3]
  fmla v28.8h, v8.8h, v1.h[4]
  fmla v29.8h, v8.8h, v1.h[5]
  fmla v30.8h, v8.8h, v1.h[6]
  fmla v31.8h, v8.8h, v1.h[7]

  subs w13, w13, #1
  bgt CommLoopMul

Bias:
  cbz x11, Activation
  ld1 {v0.8h}, [x14], #16
  fadd v16.8h, v16.8h, v0.8h
  fadd v17.8h, v17.8h, v0.8h
  fadd v18.8h, v18.8h, v0.8h
  fadd v19.8h, v19.8h, v0.8h
  fadd v20.8h, v20.8h, v0.8h
  fadd v21.8h, v21.8h, v0.8h
  fadd v22.8h, v22.8h, v0.8h
  fadd v23.8h, v23.8h, v0.8h
  fadd v24.8h, v24.8h, v0.8h
  fadd v25.8h, v25.8h, v0.8h
  fadd v26.8h, v26.8h, v0.8h
  fadd v27.8h, v27.8h, v0.8h
  fadd v28.8h, v28.8h, v0.8h
  fadd v29.8h, v29.8h, v0.8h
  fadd v30.8h, v30.8h, v0.8h
  fadd v31.8h, v31.8h, v0.8h

Activation:
  cmp w4, #3
  beq Relu6
  cmp w4, #1
  beq Relu
  b Write

Relu6:
  movi v15.8h, #0x46, lsl #8
  fmin v16.8h, v16.8h, v15.8h
  fmin v17.8h, v17.8h, v15.8h
  fmin v18.8h, v18.8h, v15.8h
  fmin v19.8h, v19.8h, v15.8h
  fmin v20.8h, v20.8h, v15.8h
  fmin v21.8h, v21.8h, v15.8h
  fmin v22.8h, v22.8h, v15.8h
  fmin v23.8h, v23.8h, v15.8h
  fmin v24.8h, v24.8h, v15.8h
  fmin v25.8h, v25.8h, v15.8h
  fmin v26.8h, v26.8h, v15.8h
  fmin v27.8h, v27.8h, v15.8h
  fmin v28.8h, v28.8h, v15.8h
  fmin v29.8h, v29.8h, v15.8h
  fmin v30.8h, v30.8h, v15.8h
  fmin v31.8h, v31.8h, v15.8h

Relu:
  dup v14.4s, wzr
  fmax v16.8h, v16.8h, v14.8h
  fmax v17.8h, v17.8h, v14.8h
  fmax v18.8h, v18.8h, v14.8h
  fmax v19.8h, v19.8h, v14.8h
  fmax v20.8h, v20.8h, v14.8h
  fmax v21.8h, v21.8h, v14.8h
  fmax v22.8h, v22.8h, v14.8h
  fmax v23.8h, v23.8h, v14.8h
  fmax v24.8h, v24.8h, v14.8h
  fmax v25.8h, v25.8h, v14.8h
  fmax v26.8h, v26.8h, v14.8h
  fmax v27.8h, v27.8h, v14.8h
  fmax v28.8h, v28.8h, v14.8h
  fmax v29.8h, v29.8h, v14.8h
  fmax v30.8h, v30.8h, v14.8h
  fmax v31.8h, v31.8h, v14.8h

Write:
  ldrb w13, [sp, #8]
  cbz w13, WriteC8
  cmp w7, #1
  beq Write1
  cmp w7, #2
  beq Write2
  cmp w7, #3
  beq Write3
  cmp w7, #4
  beq Write4
  cmp w7, #5
  beq Write5
  cmp w7, #6
  beq Write6
  cmp w7, #7
  beq Write7
  b Write8

Write1:
  st1 {v16.h}[0], [x18], x17
  cmp w10, #1
  beq WriteEnd
  st1 {v17.h}[0], [x18], x17
  cmp w10, #2
  beq WriteEnd
  st1 {v18.h}[0], [x18], x17
  cmp w10, #3
  beq WriteEnd
  st1 {v19.h}[0], [x18], x17
  cmp w10, #4
  beq WriteEnd
  st1 {v20.h}[0], [x18], x17
  cmp w10, #5
  beq WriteEnd
  st1 {v21.h}[0], [x18], x17
  cmp w10, #6
  beq WriteEnd
  st1 {v22.h}[0], [x18], x17
  cmp w10, #7
  beq WriteEnd
  st1 {v23.h}[0], [x18], x17
  cmp w10, #8
  beq WriteEnd
  st1 {v24.h}[0], [x18], x17
  cmp w10, #9
  beq WriteEnd
  st1 {v25.h}[0], [x18], x17
  cmp w10, #10
  beq WriteEnd
  st1 {v26.h}[0], [x18], x17
  cmp w10, #11
  beq WriteEnd
  st1 {v27.h}[0], [x18], x17
  cmp w10, #12
  beq WriteEnd
  st1 {v28.h}[0], [x18], x17
  cmp w10, #13
  beq WriteEnd
  st1 {v29.h}[0], [x18], x17
  cmp w10, #14
  beq WriteEnd
  st1 {v30.h}[0], [x18], x17
  cmp w10, #15
  beq WriteEnd
  st1 {v31.h}[0], [x18], x17
  b WriteEnd
Write2:
  add x13, x18, #2
  st1 {v16.h}[0], [x18], x17
  st1 {v16.h}[1], [x13], x17
  cmp w10, #1
  beq WriteEnd
  st1 {v17.h}[0], [x18], x17
  st1 {v17.h}[1], [x13], x17
  cmp w10, #2
  beq WriteEnd
  st1 {v18.h}[0], [x18], x17
  st1 {v18.h}[1], [x13], x17
  cmp w10, #3
  beq WriteEnd
  st1 {v19.h}[0], [x18], x17
  st1 {v19.h}[1], [x13], x17
  cmp w10, #4
  beq WriteEnd
  st1 {v20.h}[0], [x18], x17
  st1 {v20.h}[1], [x13], x17
  cmp w10, #5
  beq WriteEnd
  st1 {v21.h}[0], [x18], x17
  st1 {v21.h}[1], [x13], x17
  cmp w10, #6
  beq WriteEnd
  st1 {v22.h}[0], [x18], x17
  st1 {v22.h}[1], [x13], x17
  cmp w10, #7
  beq WriteEnd
  st1 {v23.h}[0], [x18], x17
  st1 {v23.h}[1], [x13], x17
  cmp w10, #8
  beq WriteEnd
  st1 {v24.h}[0], [x18], x17
  st1 {v24.h}[1], [x13], x17
  cmp w10, #9
  beq WriteEnd
  st1 {v25.h}[0], [x18], x17
  st1 {v25.h}[1], [x13], x17
  cmp w10, #10
  beq WriteEnd
  st1 {v26.h}[0], [x18], x17
  st1 {v26.h}[1], [x13], x17
  cmp w10, #11
  beq WriteEnd
  st1 {v27.h}[0], [x18], x17
  st1 {v27.h}[1], [x13], x17
  cmp w10, #12
  beq WriteEnd
  st1 {v28.h}[0], [x18], x17
  st1 {v28.h}[1], [x13], x17
  cmp w10, #13
  beq WriteEnd
  st1 {v29.h}[0], [x18], x17
  st1 {v29.h}[1], [x13], x17
  cmp w10, #14
  beq WriteEnd
  st1 {v30.h}[0], [x18], x17
  st1 {v30.h}[1], [x13], x17
  cmp w10, #15
  beq WriteEnd
  st1 {v31.h}[0], [x18], x17
  st1 {v31.h}[1], [x13], x17
  b WriteEnd
Write3:
  add x13, x18, #2
  add x14, x18, #4
  st1 {v16.h}[0], [x18], x17
  st1 {v16.h}[1], [x13], x17
  st1 {v16.h}[2], [x14], x17
  cmp w10, #1
  beq WriteEnd
  st1 {v17.h}[0], [x18], x17
  st1 {v17.h}[1], [x13], x17
  st1 {v17.h}[2], [x14], x17
  cmp w10, #2
  beq WriteEnd
  st1 {v18.h}[0], [x18], x17
  st1 {v18.h}[1], [x13], x17
  st1 {v18.h}[2], [x14], x17
  cmp w10, #3
  beq WriteEnd
  st1 {v19.h}[0], [x18], x17
  st1 {v19.h}[1], [x13], x17
  st1 {v19.h}[2], [x14], x17
  cmp w10, #4
  beq WriteEnd
  st1 {v20.h}[0], [x18], x17
  st1 {v20.h}[1], [x13], x17
  st1 {v20.h}[2], [x14], x17
  cmp w10, #5
  beq WriteEnd
  st1 {v21.h}[0], [x18], x17
  st1 {v21.h}[1], [x13], x17
  st1 {v21.h}[2], [x14], x17
  cmp w10, #6
  beq WriteEnd
  st1 {v22.h}[0], [x18], x17
  st1 {v22.h}[1], [x13], x17
  st1 {v22.h}[2], [x14], x17
  cmp w10, #7
  beq WriteEnd
  st1 {v23.h}[0], [x18], x17
  st1 {v23.h}[1], [x13], x17
  st1 {v23.h}[2], [x14], x17
  cmp w10, #8
  beq WriteEnd
  st1 {v24.h}[0], [x18], x17
  st1 {v24.h}[1], [x13], x17
  st1 {v24.h}[2], [x14], x17
  cmp w10, #9
  beq WriteEnd
  st1 {v25.h}[0], [x18], x17
  st1 {v25.h}[1], [x13], x17
  st1 {v25.h}[2], [x14], x17
  cmp w10, #10
  beq WriteEnd
  st1 {v26.h}[0], [x18], x17
  st1 {v26.h}[1], [x13], x17
  st1 {v26.h}[2], [x14], x17
  cmp w10, #11
  beq WriteEnd
  st1 {v27.h}[0], [x18], x17
  st1 {v27.h}[1], [x13], x17
  st1 {v27.h}[2], [x14], x17
  cmp w10, #12
  beq WriteEnd
  st1 {v28.h}[0], [x18], x17
  st1 {v28.h}[1], [x13], x17
  st1 {v28.h}[2], [x14], x17
  cmp w10, #13
  beq WriteEnd
  st1 {v29.h}[0], [x18], x17
  st1 {v29.h}[1], [x13], x17
  st1 {v29.h}[2], [x14], x17
  cmp w10, #14
  beq WriteEnd
  st1 {v30.h}[0], [x18], x17
  st1 {v30.h}[1], [x13], x17
  st1 {v30.h}[2], [x14], x17
  cmp w10, #15
  beq WriteEnd
  st1 {v31.h}[0], [x18], x17
  st1 {v31.h}[1], [x13], x17
  st1 {v31.h}[2], [x14], x17
  b WriteEnd
Write4:
  st1 {v16.4h}, [x18], x17
  cmp w10, #1
  beq WriteEnd
  st1 {v17.4h}, [x18], x17
  cmp w10, #2
  beq WriteEnd
  st1 {v18.4h}, [x18], x17
  cmp w10, #3
  beq WriteEnd
  st1 {v19.4h}, [x18], x17
  cmp w10, #4
  beq WriteEnd
  st1 {v20.4h}, [x18], x17
  cmp w10, #5
  beq WriteEnd
  st1 {v21.4h}, [x18], x17
  cmp w10, #6
  beq WriteEnd
  st1 {v22.4h}, [x18], x17
  cmp w10, #7
  beq WriteEnd
  st1 {v23.4h}, [x18], x17
  cmp w10, #8
  beq WriteEnd
  st1 {v24.4h}, [x18], x17
  cmp w10, #9
  beq WriteEnd
  st1 {v25.4h}, [x18], x17
  cmp w10, #10
  beq WriteEnd
  st1 {v26.4h}, [x18], x17
  cmp w10, #11
  beq WriteEnd
  st1 {v27.4h}, [x18], x17
  cmp w10, #12
  beq WriteEnd
  st1 {v28.4h}, [x18], x17
  cmp w10, #13
  beq WriteEnd
  st1 {v29.4h}, [x18], x17
  cmp w10, #14
  beq WriteEnd
  st1 {v30.4h}, [x18], x17
  cmp w10, #15
  beq WriteEnd
  st1 {v31.4h}, [x18], x17
  b WriteEnd
Write5:
  add x13, x18, #8
  st1 {v16.4h}, [x18], x17
  st1 {v16.h}[4], [x13], x17
  cmp w10, #1
  beq WriteEnd
  st1 {v17.4h}, [x18], x17
  st1 {v17.h}[4], [x13], x17
  cmp w10, #2
  beq WriteEnd
  st1 {v18.4h}, [x18], x17
  st1 {v18.h}[4], [x13], x17
  cmp w10, #3
  beq WriteEnd
  st1 {v19.4h}, [x18], x17
  st1 {v19.h}[4], [x13], x17
  cmp w10, #4
  beq WriteEnd
  st1 {v20.4h}, [x18], x17
  st1 {v20.h}[4], [x13], x17
  cmp w10, #5
  beq WriteEnd
  st1 {v21.4h}, [x18], x17
  st1 {v21.h}[4], [x13], x17
  cmp w10, #6
  beq WriteEnd
  st1 {v22.4h}, [x18], x17
  st1 {v22.h}[4], [x13], x17
  cmp w10, #7
  beq WriteEnd
  st1 {v23.4h}, [x18], x17
  st1 {v23.h}[4], [x13], x17
  cmp w10, #8
  beq WriteEnd
  st1 {v24.4h}, [x18], x17
  st1 {v24.h}[4], [x13], x17
  cmp w10, #9
  beq WriteEnd
  st1 {v25.4h}, [x18], x17
  st1 {v25.h}[4], [x13], x17
  cmp w10, #10
  beq WriteEnd
  st1 {v26.4h}, [x18], x17
  st1 {v26.h}[4], [x13], x17
  cmp w10, #11
  beq WriteEnd
  st1 {v27.4h}, [x18], x17
  st1 {v27.h}[4], [x13], x17
  cmp w10, #12
  beq WriteEnd
  st1 {v28.4h}, [x18], x17
  st1 {v28.h}[4], [x13], x17
  cmp w10, #13
  beq WriteEnd
  st1 {v29.4h}, [x18], x17
  st1 {v29.h}[4], [x13], x17
  cmp w10, #14
  beq WriteEnd
  st1 {v30.4h}, [x18], x17
  st1 {v30.h}[4], [x13], x17
  cmp w10, #15
  beq WriteEnd
  st1 {v31.4h}, [x18], x17
  st1 {v31.h}[4], [x13], x17
  b WriteEnd
Write6:
  add x13, x18, #8
  add x14, x18, #10
  st1 {v16.4h}, [x18], x17
  st1 {v16.h}[4], [x13], x17
  st1 {v16.h}[5], [x14], x17
  cmp w10, #1
  beq WriteEnd
  st1 {v17.4h}, [x18], x17
  st1 {v17.h}[4], [x13], x17
  st1 {v17.h}[5], [x14], x17
  cmp w10, #2
  beq WriteEnd
  st1 {v18.4h}, [x18], x17
  st1 {v18.h}[4], [x13], x17
  st1 {v18.h}[5], [x14], x17
  cmp w10, #3
  beq WriteEnd
  st1 {v19.4h}, [x18], x17
  st1 {v19.h}[4], [x13], x17
  st1 {v19.h}[5], [x14], x17
  cmp w10, #4
  beq WriteEnd
  st1 {v20.4h}, [x18], x17
  st1 {v20.h}[4], [x13], x17
  st1 {v20.h}[5], [x14], x17
  cmp w10, #5
  beq WriteEnd
  st1 {v21.4h}, [x18], x17
  st1 {v21.h}[4], [x13], x17
  st1 {v21.h}[5], [x14], x17
  cmp w10, #6
  beq WriteEnd
  st1 {v22.4h}, [x18], x17
  st1 {v22.h}[4], [x13], x17
  st1 {v22.h}[5], [x14], x17
  cmp w10, #7
  beq WriteEnd
  st1 {v23.4h}, [x18], x17
  st1 {v23.h}[4], [x13], x17
  st1 {v23.h}[5], [x14], x17
  cmp w10, #8
  beq WriteEnd
  st1 {v24.4h}, [x18], x17
  st1 {v24.h}[4], [x13], x17
  st1 {v24.h}[5], [x14], x17
  cmp w10, #9
  beq WriteEnd
  st1 {v25.4h}, [x18], x17
  st1 {v25.h}[4], [x13], x17
  st1 {v25.h}[5], [x14], x17
  cmp w10, #10
  beq WriteEnd
  st1 {v26.4h}, [x18], x17
  st1 {v26.h}[4], [x13], x17
  st1 {v26.h}[5], [x14], x17
  cmp w10, #11
  beq WriteEnd
  st1 {v27.4h}, [x18], x17
  st1 {v27.h}[4], [x13], x17
  st1 {v27.h}[5], [x14], x17
  cmp w10, #12
  beq WriteEnd
  st1 {v28.4h}, [x18], x17
  st1 {v28.h}[4], [x13], x17
  st1 {v28.h}[5], [x14], x17
  cmp w10, #13
  beq WriteEnd
  st1 {v29.4h}, [x18], x17
  st1 {v29.h}[4], [x13], x17
  st1 {v29.h}[5], [x14], x17
  cmp w10, #14
  beq WriteEnd
  st1 {v30.4h}, [x18], x17
  st1 {v30.h}[4], [x13], x17
  st1 {v30.h}[5], [x14], x17
  cmp w10, #15
  beq WriteEnd
  st1 {v31.4h}, [x18], x17
  st1 {v31.h}[4], [x13], x17
  st1 {v31.h}[5], [x14], x17
  b WriteEnd
Write7:
  add x13, x18, #8
  add x14, x18, #10
  add x16, x18, #12
  st1 {v16.4h}, [x18], x17
  st1 {v16.h}[4], [x13], x17
  st1 {v16.h}[5], [x14], x17
  st1 {v16.h}[6], [x16], x17
  cmp w10, #1
  beq WriteEnd
  st1 {v17.4h}, [x18], x17
  st1 {v17.h}[4], [x13], x17
  st1 {v17.h}[5], [x14], x17
  st1 {v17.h}[6], [x16], x17
  cmp w10, #2
  beq WriteEnd
  st1 {v18.4h}, [x18], x17
  st1 {v18.h}[4], [x13], x17
  st1 {v18.h}[5], [x14], x17
  st1 {v18.h}[6], [x16], x17
  cmp w10, #3
  beq WriteEnd
  st1 {v19.4h}, [x18], x17
  st1 {v19.h}[4], [x13], x17
  st1 {v19.h}[5], [x14], x17
  st1 {v19.h}[6], [x16], x17
  cmp w10, #4
  beq WriteEnd
  st1 {v20.4h}, [x18], x17
  st1 {v20.h}[4], [x13], x17
  st1 {v20.h}[5], [x14], x17
  st1 {v20.h}[6], [x16], x17
  cmp w10, #5
  beq WriteEnd
  st1 {v21.4h}, [x18], x17
  st1 {v21.h}[4], [x13], x17
  st1 {v21.h}[5], [x14], x17
  st1 {v21.h}[6], [x16], x17
  cmp w10, #6
  beq WriteEnd
  st1 {v22.4h}, [x18], x17
  st1 {v22.h}[4], [x13], x17
  st1 {v22.h}[5], [x14], x17
  st1 {v22.h}[6], [x16], x17
  cmp w10, #7
  beq WriteEnd
  st1 {v23.4h}, [x18], x17
  st1 {v23.h}[4], [x13], x17
  st1 {v23.h}[5], [x14], x17
  st1 {v23.h}[6], [x16], x17
  cmp w10, #8
  beq WriteEnd
  st1 {v24.4h}, [x18], x17
  st1 {v24.h}[4], [x13], x17
  st1 {v24.h}[5], [x14], x17
  st1 {v24.h}[6], [x16], x17
  cmp w10, #9
  beq WriteEnd
  st1 {v25.4h}, [x18], x17
  st1 {v25.h}[4], [x13], x17
  st1 {v25.h}[5], [x14], x17
  st1 {v25.h}[6], [x16], x17
  cmp w10, #10
  beq WriteEnd
  st1 {v26.4h}, [x18], x17
  st1 {v26.h}[4], [x13], x17
  st1 {v26.h}[5], [x14], x17
  st1 {v26.h}[6], [x16], x17
  cmp w10, #11
  beq WriteEnd
  st1 {v27.4h}, [x18], x17
  st1 {v27.h}[4], [x13], x17
  st1 {v27.h}[5], [x14], x17
  st1 {v27.h}[6], [x16], x17
  cmp w10, #12
  beq WriteEnd
  st1 {v28.4h}, [x18], x17
  st1 {v28.h}[4], [x13], x17
  st1 {v28.h}[5], [x14], x17
  st1 {v28.h}[6], [x16], x17
  cmp w10, #13
  beq WriteEnd
  st1 {v29.4h}, [x18], x17
  st1 {v29.h}[4], [x13], x17
  st1 {v29.h}[5], [x14], x17
  st1 {v29.h}[6], [x16], x17
  cmp w10, #14
  beq WriteEnd
  st1 {v30.4h}, [x18], x17
  st1 {v30.h}[4], [x13], x17
  st1 {v30.h}[5], [x14], x17
  st1 {v30.h}[6], [x16], x17
  cmp w10, #15
  beq WriteEnd
  st1 {v31.4h}, [x18], x17
  st1 {v31.h}[4], [x13], x17
  st1 {v31.h}[5], [x14], x17
  st1 {v31.h}[6], [x16], x17
  b WriteEnd
WriteC8:
  st1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x2], #64
  st1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x2], #64
  st1 {v24.8h, v25.8h, v26.8h, v27.8h}, [x2], #64
  st1 {v28.8h, v29.8h, v30.8h, v31.8h}, [x2], #64
  b WriteEnd
Write8:
  st1 {v16.8h}, [x18], x17
  cmp w10, #1
  beq WriteEnd
  st1 {v17.8h}, [x18], x17
  cmp w10, #2
  beq WriteEnd
  st1 {v18.8h}, [x18], x17
  cmp w10, #3
  beq WriteEnd
  st1 {v19.8h}, [x18], x17
  cmp w10, #4
  beq WriteEnd
  st1 {v20.8h}, [x18], x17
  cmp w10, #5
  beq WriteEnd
  st1 {v21.8h}, [x18], x17
  cmp w10, #6
  beq WriteEnd
  st1 {v22.8h}, [x18], x17
  cmp w10, #7
  beq WriteEnd
  st1 {v23.8h}, [x18], x17
  cmp w10, #8
  beq WriteEnd
  st1 {v24.8h}, [x18], x17
  cmp w10, #9
  beq WriteEnd
  st1 {v25.8h}, [x18], x17
  cmp w10, #10
  beq WriteEnd
  st1 {v26.8h}, [x18], x17
  cmp w10, #11
  beq WriteEnd
  st1 {v27.8h}, [x18], x17
  cmp w10, #12
  beq WriteEnd
  st1 {v28.8h}, [x18], x17
  cmp w10, #13
  beq WriteEnd
  st1 {v29.8h}, [x18], x17
  cmp w10, #14
  beq WriteEnd
  st1 {v30.8h}, [x18], x17
  cmp w10, #15
  beq WriteEnd
  st1 {v31.8h}, [x18], x17

WriteEnd:
  subs w10, w10, #16 // lhs row - 8
  bgt L2

End2:
  subs w7, w7, #8 // rhs col - 8
  add x1, x1, x15 // rhs ptr + stride
  add x3, x3, #16 // bias ptr + stride
  ldrb w13, [sp, #8]
  cbz w13, NoDstStep
  add x2, x2, #16 // dst ptr + stride
NoDstStep:
  bgt L1

End1:
  sub sp, sp, #128
  ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [sp], #64
  ld1 {v12.8h, v13.8h, v14.8h, v15.8h}, [sp], #64
  ret
#endif
