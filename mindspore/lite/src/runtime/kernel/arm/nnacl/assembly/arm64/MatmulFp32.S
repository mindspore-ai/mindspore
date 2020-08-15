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
//OptLoopMul4                                          RM 4x8 block
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
// void MatmulFloatNeon64(const float *a, const float *b, float *c, const float *bias, int act_type, int depth
//                        int row, int col, int stride, bool write_nhwc)
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

MatmulFloatNeon64:
  sub sp, sp, #128
  st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
  st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64

  mov w18, #32 // sizeof(float) * 8
  mul w15, w5, w18 // block stride of lhs/rhs: sizeof(float) * 8 * depth
  mov x11, x3 // bias flag
  mov x18, #4
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

  cmp w13, #4
  blt CommLoopMul

OptLoopMul4:
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

  sub w13, w13, #4
  cmp w13, #0
  ble Bias
  cmp w13, #4
  bge OptLoopMul4

CommLoopMul:
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
  bgt CommLoopMul

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
  b Write

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
  str s16, [x18]
  cmp w10, #1
  beq WriteEnd
  add x18, x18, x17
  str s18, [x18]
  cmp w10, #2
  beq WriteEnd
  add x18, x18, x17
  str s20, [x18]
  cmp w10, #3
  beq WriteEnd
  add x18, x18, x17
  str s22, [x18]
  cmp w10, #4
  beq WriteEnd
  add x18, x18, x17
  str s24, [x18]
  cmp w10, #5
  beq WriteEnd
  add x18, x18, x17
  str s26, [x18]
  cmp w10, #6
  beq WriteEnd
  add x18, x18, x17
  str s28, [x18]
  cmp w10, #7
  beq WriteEnd
  add x18, x18, x17
  str s30, [x18]
  add x18, x18, x17
  b WriteEnd
Write2:
  dup s17, v16.s[1]
  stp s16, s17, [x18]
  cmp w10, #1
  beq WriteEnd
  add x18, x18, x17
  dup s19, v18.s[1]
  stp s18, s19, [x18]
  cmp w10, #2
  beq WriteEnd
  add x18, x18, x17
  dup s21, v20.s[1]
  stp s20, s21, [x18]
  cmp w10, #3
  beq WriteEnd
  add x18, x18, x17
  dup s23, v22.s[1]
  stp s22, s23, [x18]
  cmp w10, #4
  beq WriteEnd
  add x18, x18, x17
  dup s25, v24.s[1]
  stp s24, s25, [x18]
  cmp w10, #5
  beq WriteEnd
  add x18, x18, x17
  dup s27, v26.s[1]
  stp s26, s27, [x18]
  cmp w10, #6
  beq WriteEnd
  add x18, x18, x17
  dup s29, v28.s[1]
  stp s28, s29, [x18]
  cmp w10, #7
  beq WriteEnd
  add x18, x18, x17
  dup s31, v30.s[1]
  stp s30, s31, [x18]
  add x18, x18, x17
  b WriteEnd
Write3:
  add x13, x18, #8
  dup s17, v16.s[1]
  stp s16, s17, [x18]
  add x18, x18, x17
  st1 {v16.s}[2], [x13], x17
  cmp w10, #1
  beq WriteEnd
  dup s19, v18.s[1]
  stp s18, s19, [x18]
  add x18, x18, x17
  st1 {v18.s}[2], [x13], x17
  cmp w10, #2
  beq WriteEnd
  dup s21, v20.s[1]
  stp s20, s21, [x18]
  add x18, x18, x17
  st1 {v20.s}[2], [x13], x17
  cmp w10, #3
  beq WriteEnd
  dup s23, v22.s[1]
  stp s22, s23, [x18]
  add x18, x18, x17
  st1 {v22.s}[2], [x13], x17
  cmp w10, #4
  beq WriteEnd
  dup s25, v24.s[1]
  stp s24, s25, [x18]
  add x18, x18, x17
  st1 {v24.s}[2], [x13], x17
  cmp w10, #5
  beq WriteEnd
  dup s27, v26.s[1]
  stp s26, s27, [x18]
  add x18, x18, x17
  st1 {v26.s}[2], [x13], x17
  cmp w10, #6
  beq WriteEnd
  dup s29, v28.s[1]
  stp s28, s29, [x18]
  add x18, x18, x17
  st1 {v28.s}[2], [x13], x17
  cmp w10, #7
  beq WriteEnd
  dup s31, v30.s[1]
  stp s30, s31, [x18]
  add x18, x18, x17
  st1 {v30.s}[2], [x13]
  b WriteEnd
Write4:
  st1 {v16.4s}, [x18], x17
  cmp w10, #1
  beq WriteEnd
  st1 {v18.4s}, [x18], x17
  cmp w10, #2
  beq WriteEnd
  st1 {v20.4s}, [x18], x17
  cmp w10, #3
  beq WriteEnd
  st1 {v22.4s}, [x18], x17
  cmp w10, #4
  beq WriteEnd
  st1 {v24.4s}, [x18], x17
  cmp w10, #5
  beq WriteEnd
  st1 {v26.4s}, [x18], x17
  cmp w10, #6
  beq WriteEnd
  st1 {v28.4s}, [x18], x17
  cmp w10, #7
  beq WriteEnd
  st1 {v30.4s}, [x18], x17
  b WriteEnd
Write5:
  add x13, x18, #16
  st1 {v16.4s}, [x18], x17
  str s17, [x13]
  cmp w10, #1
  beq WriteEnd
  add x13, x13, x17
  st1 {v18.4s}, [x18], x17
  str s19, [x13]
  cmp w10, #2
  beq WriteEnd
  add x13, x13, x17
  st1 {v20.4s}, [x18], x17
  str s21, [x13]
  cmp w10, #3
  beq WriteEnd
  add x13, x13, x17
  st1 {v22.4s}, [x18], x17
  str s23, [x13]
  cmp w10, #4
  beq WriteEnd
  add x13, x13, x17
  st1 {v24.4s}, [x18], x17
  str s25, [x13]
  cmp w10, #5
  beq WriteEnd
  add x13, x13, x17
  st1 {v26.4s}, [x18], x17
  str s27, [x13]
  cmp w10, #6
  beq WriteEnd
  add x13, x13, x17
  st1 {v28.4s}, [x18], x17
  str s29, [x13]
  cmp w10, #7
  beq WriteEnd
  add x13, x13, x17
  st1 {v30.4s}, [x18], x17
  str s31, [x13]
  b WriteEnd
Write6:
  add x13, x18, #16
  st1 {v16.4s}, [x18], x17
  dup s16, v17.s[1]
  stp s17, s16, [x13]
  cmp w10, #1
  beq WriteEnd
  add x13, x13, x17
  st1 {v18.4s}, [x18], x17
  dup s18, v19.s[1]
  stp s19, s18, [x13]
  cmp w10, #2
  beq WriteEnd
  add x13, x13, x17
  st1 {v20.4s}, [x18], x17
  dup s20, v21.s[1]
  stp s21, s20, [x13]
  cmp w10, #3
  beq WriteEnd
  add x13, x13, x17
  st1 {v22.4s}, [x18], x17
  dup s22, v23.s[1]
  stp s23, s22, [x13]
  cmp w10, #4
  beq WriteEnd
  add x13, x13, x17
  st1 {v24.4s}, [x18], x17
  dup s24, v25.s[1]
  stp s25, s24, [x13]
  cmp w10, #5
  beq WriteEnd
  add x13, x13, x17
  st1 {v26.4s}, [x18], x17
  dup s26, v27.s[1]
  stp s27, s26, [x13]
  cmp w10, #6
  beq WriteEnd
  add x13, x13, x17
  st1 {v28.4s}, [x18], x17
  dup s28, v29.s[1]
  stp s29, s28, [x13]
  cmp w10, #7
  beq WriteEnd
  add x13, x13, x17
  st1 {v30.4s}, [x18], x17
  dup s30, v31.s[1]
  stp s31, s30, [x13]
  b WriteEnd
Write7:
  add x13, x18, #16
  add x16, x18, #24
  st1 {v16.4s}, [x18], x17
  dup s16, v17.s[1]
  stp s17, s16, [x13]
  add x13, x13, x17
  st1 {v17.s}[2], [x16], x17
  cmp w10, #1
  beq WriteEnd
  st1 {v18.4s}, [x18], x17
  dup s18, v19.s[1]
  stp s19, s18, [x13]
  add x13, x13, x17
  st1 {v19.s}[2], [x16], x17
  cmp w10, #2
  beq WriteEnd
  st1 {v20.4s}, [x18], x17
  dup s20, v21.s[1]
  stp s21, s20, [x13]
  add x13, x13, x17
  st1 {v21.s}[2], [x16], x17
  cmp w10, #3
  beq WriteEnd
  st1 {v22.4s}, [x18], x17
  dup s22, v23.s[1]
  stp s23, s22, [x13]
  add x13, x13, x17
  st1 {v23.s}[2], [x16], x17
  cmp w10, #4
  beq WriteEnd
  st1 {v24.4s}, [x18], x17
  dup s24, v25.s[1]
  stp s25, s24, [x13]
  add x13, x13, x17
  st1 {v25.s}[2], [x16], x17
  cmp w10, #5
  beq WriteEnd
  st1 {v26.4s}, [x18], x17
  dup s26, v27.s[1]
  stp s27, s26, [x13]
  add x13, x13, x17
  st1 {v27.s}[2], [x16], x17
  cmp w10, #6
  beq WriteEnd
  st1 {v28.4s}, [x18], x17
  dup s28, v29.s[1]
  stp s29, s28, [x13]
  add x13, x13, x17
  st1 {v29.s}[2], [x16], x17
  cmp w10, #7
  beq WriteEnd
  st1 {v30.4s}, [x18], x17
  dup s30, v31.s[1]
  stp s31, s30, [x13]
  add x13, x13, x17
  st1 {v31.s}[2], [x16], x17
  b WriteEnd
WriteC8:
  st1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x2], #64
  st1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x2], #64
  st1 {v24.8h, v25.8h, v26.8h, v27.8h}, [x2], #64
  st1 {v28.8h, v29.8h, v30.8h, v31.8h}, [x2], #64
  b WriteEnd
Write8:
  st1 {v16.4s, v17.4s}, [x18], x17
  cmp w10, #1
  beq WriteEnd
  st1 {v18.4s, v19.4s}, [x18], x17
  cmp w10, #2
  beq WriteEnd
  st1 {v20.4s, v21.4s}, [x18], x17
  cmp w10, #3
  beq WriteEnd
  st1 {v22.4s, v23.4s}, [x18], x17
  cmp w10, #4
  beq WriteEnd
  st1 {v24.4s, v25.4s}, [x18], x17
  cmp w10, #5
  beq WriteEnd
  st1 {v26.4s, v27.4s}, [x18], x17
  cmp w10, #6
  beq WriteEnd
  st1 {v28.4s, v29.4s}, [x18], x17
  cmp w10, #7
  beq WriteEnd
  st1 {v30.4s, v31.4s}, [x18], x17

WriteEnd:
  subs w10, w10, #8 // lhs row - 8
  bgt L2

End2:
  subs w7, w7, #8 // rhs col - 8
  add x1, x1, x15 // rhs ptr + stride
  add x3, x3, #32 // bias ptr + stride
  ldrb w13, [sp, #8]
  cbz w13, NoDstStep
  add x2, x2, #32 // dst ptr + stride
NoDstStep:
  bgt L1

End1:
  sub sp, sp, #128
  ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
  ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
  ret
#endif
