#ifdef __aarch64__
    .text
    .align 5
    .global MatmulFloatNeon64
#ifndef __APPLE__
    .type MatmulFloatNeon64, %function
#endif

// A: LM  [row_8 * depth] col_8_major
// B: RM  [depth * col_8] row_8_major
// C: A*B [row_8 * col_8] col_8x8_major
// A * B -> [8 * depth] * [depth * 8] -> [8 * 4] * [4 * 8] or [8 * 1] * [1 * 8]
///////////////////////////////////////////////////////////////////////////////
//CommLoopMul                                          RM 1x8 block
//                           /-----------------------------------------\
//                           |v2.s[0] ... v2.s[3]   v3.s[0] ... v3.s[3]|
//                           \-----------------------------------------/
//        LM 8x1 block
//  /---------------------\  /-----------------------------------------\
//  |        v0.s[0]      |  |v16.s[0]...v16.s[3]   v17.s[0]...v17.s[3]|
//  |         ...         |  |  ...                              ...   |
//  |        v0.s[3]      |  |v22.s[0]...v22.s[3]   v23.s[0]...v23.s[3]|
//  |        v1.s[0]      |  |v24.s[0]...v24.s[3]   v25.s[0]...v25.s[3]|
//  |         ...         |  |  ...                              ...   |
//  |        v1.s[3]      |  |v30.s[0]...v30.s[3]   v31.s[0]...v31.s[3]|
//  \---------------------/  \-----------------------------------------/
//                                      accumulators 8x8 block
//
///////////////////////////////////////////////////////////////////////////////
//OptLoopMul4                                          RM 1x8 block
//                                       /--------------------------------------------\
//                                       |v8.s[0]  ... v8.s[3]   v9.s[0]  ... v9.s[3] |
//                                       |v10.s[0] ... v10.s[3]  v11.s[0] ... v11.s[3]|
//                                       |v12.s[0] ... v12.s[3]  v13.s[0] ... v13.s[3]|
//                                       |v14.s[0] ... v14.s[3]  v15.s[0] ... v15.s[3]|
//                                       \--------------------------------------------/
//        LM 8x4 block
//  /---------------------------------\  /--------------------------------------------\
//  | v0.s[0] v2.s[0] v4.s[0] v6.s[0] |  |v16.s[0]...v16.s[3]    v17.s[0]...v17.s[3]  |
//  |  ...     ...     ...     ...    |  |  ...                                 ...   |
//  | v0.s[3] v2.s[3] v4.s[3] v6.s[3] |  |v22.s[0]...v22.s[3]    v23.s[0]...v23.s[3]  |
//  | v1.s[0] v3.s[0] v5.s[0] v7.s[0] |  |v24.s[0]...v24.s[3]    v25.s[0]...v25.s[3]  |
//  |  ...     ...     ...     ...    |  |  ...                                 ...   |
//  | v1.s[3] v3.s[3] v5.s[3] v7.s[3] |  |v30.s[0]...v30.s[3]    v31.s[0]...v31.s[3]  |
//  \---------------------------------/  \--------------------------------------------/
//                                                  accumulators 8x8 block
/////////////////////////////////////////////////////////////////////////////////
//
// void MatmulFloatNeon64(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int row, int col)
// x0: a
// x1: b
// x2: c
// x3: bias
// w4: act_type
// w5: depth
// w6: row
// w7: col

MatmulFloatNeon64:
  sub sp, sp, #128
  st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
  st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64

  mov w9, #0     // rm col offset
  mov w10, #0    // lm row offset
  mov w18, #32  // sizeof(float)*8
  mul w15, w5, w18  // the stride of lm/rm: sizeof(float)*8*depth
  mov x11, x3       // bias flag
L1:
  cmp w9, w7
  beq End1

  mov w10, #0    // reset lm row offset
  mov x12, x0   // reload lm ptr
L2:
  cmp w10, w6
  beq End2

  mov x16, x1   // reload rm ptr
  mov w13, w5   // reload depth
  mov x14, x3     // reload bias ptr
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

OptLoopMul4:
  cmp w13, #4
  blt CommLoopMul

  ld1 {v0.4s, v1.4s}, [x12], #32
  ld1 {v8.4s, v9.4s}, [x16], #32
  fmla v16.4s, v8.4s, v0.s[0]
  fmla v17.4s, v9.4s, v0.s[0]
  fmla v18.4s, v8.4s, v0.s[1]
  fmla v19.4s, v9.4s, v0.s[1]
  fmla v20.4s, v8.4s, v0.s[2]
  fmla v21.4s, v9.4s, v0.s[2]
  fmla v22.4s, v8.4s, v0.s[3]
  fmla v23.4s, v9.4s, v0.s[3]
  ld1 {v10.4s, v11.4s}, [x16], #32
  fmla v24.4s, v8.4s, v1.s[0]
  fmla v25.4s, v9.4s, v1.s[0]
  fmla v26.4s, v8.4s, v1.s[1]
  fmla v27.4s, v9.4s, v1.s[1]
  ld1 {v2.4s, v3.4s}, [x12], #32
  fmla v28.4s, v8.4s, v1.s[2]
  fmla v29.4s, v9.4s, v1.s[2]
  fmla v30.4s, v8.4s, v1.s[3]
  fmla v31.4s, v9.4s, v1.s[3]
  fmla v16.4s, v10.4s, v2.s[0]
  fmla v17.4s, v11.4s, v2.s[0]
  fmla v18.4s, v10.4s, v2.s[1]
  fmla v19.4s, v11.4s, v2.s[1]
  fmla v20.4s, v10.4s, v2.s[2]
  fmla v21.4s, v11.4s, v2.s[2]
  fmla v22.4s, v10.4s, v2.s[3]
  fmla v23.4s, v11.4s, v2.s[3]
  ld1 {v12.4s, v13.4s}, [x16], #32
  fmla v24.4s, v10.4s, v3.s[0]
  fmla v25.4s, v11.4s, v3.s[0]
  fmla v26.4s, v10.4s, v3.s[1]
  fmla v27.4s, v11.4s, v3.s[1]
  ld1 {v4.4s, v5.4s}, [x12], #32
  fmla v28.4s, v10.4s, v3.s[2]
  fmla v29.4s, v11.4s, v3.s[2]
  fmla v30.4s, v10.4s, v3.s[3]
  fmla v31.4s, v11.4s, v3.s[3]
  fmla v16.4s, v12.4s, v4.s[0]
  fmla v17.4s, v13.4s, v4.s[0]
  fmla v18.4s, v12.4s, v4.s[1]
  fmla v19.4s, v13.4s, v4.s[1]
  fmla v20.4s, v12.4s, v4.s[2]
  fmla v21.4s, v13.4s, v4.s[2]
  fmla v22.4s, v12.4s, v4.s[3]
  fmla v23.4s, v13.4s, v4.s[3]
  ld1 {v6.4s,v7.4s}, [x12], #32
  fmla v24.4s, v12.4s, v5.s[0]
  fmla v25.4s, v13.4s, v5.s[0]
  fmla v26.4s, v12.4s, v5.s[1]
  fmla v27.4s, v13.4s, v5.s[1]
  ld1 {v14.4s, v15.4s}, [x16], #32
  fmla v28.4s, v12.4s, v5.s[2]
  fmla v29.4s, v13.4s, v5.s[2]
  fmla v30.4s, v12.4s, v5.s[3]
  fmla v31.4s, v13.4s, v5.s[3]
  fmla v16.4s, v14.4s, v6.s[0]
  fmla v17.4s, v15.4s, v6.s[0]
  fmla v18.4s, v14.4s, v6.s[1]
  fmla v19.4s, v15.4s, v6.s[1]
  fmla v20.4s, v14.4s, v6.s[2]
  fmla v21.4s, v15.4s, v6.s[2]
  fmla v22.4s, v14.4s, v6.s[3]
  fmla v23.4s, v15.4s, v6.s[3]
  fmla v24.4s, v14.4s, v7.s[0]
  fmla v25.4s, v15.4s, v7.s[0]
  fmla v26.4s, v14.4s, v7.s[1]
  fmla v27.4s, v15.4s, v7.s[1]
  fmla v28.4s, v14.4s, v7.s[2]
  fmla v29.4s, v15.4s, v7.s[2]
  fmla v30.4s, v14.4s, v7.s[3]
  fmla v31.4s, v15.4s, v7.s[3]
  subs w13, w13, #4
  b OptLoopMul4

CommLoopMul:
  cmp w13, #1
  blt Bias

  ld1 {v0.4s, v1.4s}, [x12], #32
  ld1 {v2.4s, v3.4s}, [x16], #32
  fmla v16.4s, v2.4s, v0.s[0]
  fmla v17.4s, v3.4s, v0.s[0]
  fmla v18.4s, v2.4s, v0.s[1]
  fmla v19.4s, v3.4s, v0.s[1]
  fmla v20.4s, v2.4s, v0.s[2]
  fmla v21.4s, v3.4s, v0.s[2]
  fmla v22.4s, v2.4s, v0.s[3]
  fmla v23.4s, v3.4s, v0.s[3]
  fmla v24.4s, v2.4s, v1.s[0]
  fmla v25.4s, v3.4s, v1.s[0]
  fmla v26.4s, v2.4s, v1.s[1]
  fmla v27.4s, v3.4s, v1.s[1]
  fmla v28.4s, v2.4s, v1.s[2]
  fmla v29.4s, v3.4s, v1.s[2]
  fmla v30.4s, v2.4s, v1.s[3]
  fmla v31.4s, v3.4s, v1.s[3]
  subs w13, w13, #1
  b CommLoopMul

Bias:
  cbz x11, Activation
  ld1 {v0.4s}, [x14], #16
  ld1 {v1.4s}, [x14], #16
  fadd v16.4s, v16.4s, v0.4s
  fadd v17.4s, v17.4s, v1.4s
  fadd v18.4s, v18.4s, v0.4s
  fadd v19.4s, v19.4s, v1.4s
  fadd v20.4s, v20.4s, v0.4s
  fadd v21.4s, v21.4s, v1.4s
  fadd v22.4s, v22.4s, v0.4s
  fadd v23.4s, v23.4s, v1.4s
  fadd v24.4s, v24.4s, v0.4s
  fadd v25.4s, v25.4s, v1.4s
  fadd v26.4s, v26.4s, v0.4s
  fadd v27.4s, v27.4s, v1.4s
  fadd v28.4s, v28.4s, v0.4s
  fadd v29.4s, v29.4s, v1.4s
  fadd v30.4s, v30.4s, v0.4s
  fadd v31.4s, v31.4s, v1.4s

Activation:
  cmp w4, #2
  beq Relu6
  cmp w4, #1
  beq Relu
  b TransToOut
Relu6:
  mov w8, #6
  dup v15.4s, w8
  scvtf v15.4s, v15.4s
  fmin v16.4s, v16.4s, v15.4s
  fmin v17.4s, v17.4s, v15.4s
  fmin v18.4s, v18.4s, v15.4s
  fmin v19.4s, v19.4s, v15.4s
  fmin v20.4s, v20.4s, v15.4s
  fmin v21.4s, v21.4s, v15.4s
  fmin v22.4s, v22.4s, v15.4s
  fmin v23.4s, v23.4s, v15.4s
  fmin v24.4s, v24.4s, v15.4s
  fmin v25.4s, v25.4s, v15.4s
  fmin v26.4s, v26.4s, v15.4s
  fmin v27.4s, v27.4s, v15.4s
  fmin v28.4s, v28.4s, v15.4s
  fmin v29.4s, v29.4s, v15.4s
  fmin v30.4s, v30.4s, v15.4s
  fmin v31.4s, v31.4s, v15.4s
Relu:
  dup v14.4s, wzr
  fmax v16.4s, v16.4s, v14.4s
  fmax v17.4s, v17.4s, v14.4s
  fmax v18.4s, v18.4s, v14.4s
  fmax v19.4s, v19.4s, v14.4s
  fmax v20.4s, v20.4s, v14.4s
  fmax v21.4s, v21.4s, v14.4s
  fmax v22.4s, v22.4s, v14.4s
  fmax v23.4s, v23.4s, v14.4s
  fmax v24.4s, v24.4s, v14.4s
  fmax v25.4s, v25.4s, v14.4s
  fmax v26.4s, v26.4s, v14.4s
  fmax v27.4s, v27.4s, v14.4s
  fmax v28.4s, v28.4s, v14.4s
  fmax v29.4s, v29.4s, v14.4s
  fmax v30.4s, v30.4s, v14.4s
  fmax v31.4s, v31.4s, v14.4s

TransToOut:
  st1 {v16.4s}, [x2], #16
  st1 {v17.4s}, [x2], #16
  st1 {v18.4s}, [x2], #16
  st1 {v19.4s}, [x2], #16
  st1 {v20.4s}, [x2], #16
  st1 {v21.4s}, [x2], #16
  st1 {v22.4s}, [x2], #16
  st1 {v23.4s}, [x2], #16
  st1 {v24.4s}, [x2], #16
  st1 {v25.4s}, [x2], #16
  st1 {v26.4s}, [x2], #16
  st1 {v27.4s}, [x2], #16
  st1 {v28.4s}, [x2], #16
  st1 {v29.4s}, [x2], #16
  st1 {v30.4s}, [x2], #16
  st1 {v31.4s}, [x2], #16

  add w10, w10, #8    // lm row offset + 8
  b L2

End2:
  add w9, w9, #8      // rm col offset + 8
  add x1, x1, x15     // rm ptr + stride
  add x3, x3, x18     // bias ptr + stride
  b L1

End1:
  sub sp, sp, #128
  ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
  ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
  ret
#endif