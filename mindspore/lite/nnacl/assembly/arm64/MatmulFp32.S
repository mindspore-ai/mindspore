#ifdef __aarch64__
#include "nnacl/assembly_global.h"
    .text
    .align 5
    .global MatmulFloatNeon64
#ifndef __APPLE__
    .type MatmulFloatNeon64, %function
#endif

// void MatmulFloatNeon64(const float *a, const float *b, float *c, const float *bias, int act_type, int depth
//                        int row, int col, size_t stride, size_t writeNhwc, size_t WriteWino)
// x0: a
// x1: b
// x2: c
// x3: bias
// w4: act_type
// w5: depth
// w6: row
// w7: col
// w17: stride
// w13: c8_nhwc_c4

asm_function MatmulFloatNeon64
  sub sp, sp, #128
  st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
  st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64

  ldr x9, [sp, #8]
  ldr x14, [sp, #16]

  mov w18, #32 // sizeof(float) * 8
  mul w15, w5, w18 // block stride of lhs/rhs: sizeof(float) * 8 * depth
  mov x18, #4
  ldr x17, [sp]
  cbz x14, NoWinoSteps
  mul x8, x7, x17
  mov x11, #8
  mul x11, x11, x17
  mul x8, x8, x18
  mul x11, x11, x18
NoWinoSteps:
  mul x17, x17, x18

L1:
  mov w10, w6 // reload lhs row
  mov x12, x0 // reload lhs ptr
  mov x18, x2 // reload dst ptr

L2:
  mov x16, x1 // reload rhs ptr
  mov w13, w5 // reload depth
  dup v8.4s, wzr
  dup v9.4s, wzr
  dup v10.4s, wzr
  dup v11.4s, wzr
  dup v12.4s, wzr
  dup v13.4s, wzr
  dup v14.4s, wzr
  dup v15.4s, wzr
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

LoopStart:
  ld1 {v0.4s, v1.4s, v2.4s}, [x12], #48
  ld1 {v3.4s, v4.4s}, [x16], #32
  fmla v8.4s, v3.4s, v0.s[0]
  fmla v10.4s, v3.4s, v0.s[1]
  fmla v12.4s, v3.4s, v0.s[2]
  fmla v14.4s, v3.4s, v0.s[3]
  fmla v9.4s, v4.4s, v0.s[0]
  fmla v11.4s, v4.4s, v0.s[1]
  fmla v13.4s, v4.4s, v0.s[2]
  fmla v15.4s, v4.4s, v0.s[3]

  subs w13, w13, #1
  beq LoopEnd

Loop:
  ld1 {v0.4s}, [x12], #16
  fmla v16.4s, v3.4s, v1.s[0]
  fmla v18.4s, v3.4s, v1.s[1]
  fmla v20.4s, v3.4s, v1.s[2]
  fmla v22.4s, v3.4s, v1.s[3]
  fmla v17.4s, v4.4s, v1.s[0]
  fmla v19.4s, v4.4s, v1.s[1]
  fmla v21.4s, v4.4s, v1.s[2]
  fmla v23.4s, v4.4s, v1.s[3]
  ld1 {v1.4s}, [x12], #16
  fmla v24.4s, v3.4s, v2.s[0]
  fmla v26.4s, v3.4s, v2.s[1]
  fmla v28.4s, v3.4s, v2.s[2]
  fmla v30.4s, v3.4s, v2.s[3]
  ld1 {v3.4s}, [x16], #16
  fmla v25.4s, v4.4s, v2.s[0]
  fmla v27.4s, v4.4s, v2.s[1]
  fmla v29.4s, v4.4s, v2.s[2]
  fmla v31.4s, v4.4s, v2.s[3]
  ld1 {v4.4s}, [x16], #16
  fmla v8.4s, v3.4s, v0.s[0]
  fmla v10.4s, v3.4s, v0.s[1]
  fmla v12.4s, v3.4s, v0.s[2]
  fmla v14.4s, v3.4s, v0.s[3]
  ld1 {v2.4s}, [x12], #16
  fmla v9.4s, v4.4s, v0.s[0]
  fmla v11.4s, v4.4s, v0.s[1]
  fmla v13.4s, v4.4s, v0.s[2]
  fmla v15.4s, v4.4s, v0.s[3]

  subs w13, w13, #1
  bgt Loop

LoopEnd:
  fmla v16.4s, v3.4s, v1.s[0]
  fmla v18.4s, v3.4s, v1.s[1]
  fmla v20.4s, v3.4s, v1.s[2]
  fmla v22.4s, v3.4s, v1.s[3]
  fmla v17.4s, v4.4s, v1.s[0]
  fmla v19.4s, v4.4s, v1.s[1]
  fmla v21.4s, v4.4s, v1.s[2]
  fmla v23.4s, v4.4s, v1.s[3]
  fmla v24.4s, v3.4s, v2.s[0]
  fmla v26.4s, v3.4s, v2.s[1]
  fmla v28.4s, v3.4s, v2.s[2]
  fmla v30.4s, v3.4s, v2.s[3]
  fmla v25.4s, v4.4s, v2.s[0]
  fmla v27.4s, v4.4s, v2.s[1]
  fmla v29.4s, v4.4s, v2.s[2]
  fmla v31.4s, v4.4s, v2.s[3]

Bias:
  cbz x3, Activation
  ld1 {v0.4s}, [x3], #16
  ld1 {v1.4s}, [x3]
  sub x3, x3, #16
  fadd v8.4s, v8.4s, v0.4s
  fadd v9.4s, v9.4s, v1.4s
  fadd v10.4s, v10.4s, v0.4s
  fadd v11.4s, v11.4s, v1.4s
  fadd v12.4s, v12.4s, v0.4s
  fadd v13.4s, v13.4s, v1.4s
  fadd v14.4s, v14.4s, v0.4s
  fadd v15.4s, v15.4s, v1.4s
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
  cmp w4, #3
  beq Relu6
  cmp w4, #1
  beq Relu
  b Write

Relu6:
  mov w13, #6
  dup v2.4s, w13
  scvtf v2.4s, v2.4s
  fmin v8.4s, v8.4s, v2.4s
  fmin v9.4s, v9.4s, v2.4s
  fmin v10.4s, v10.4s, v2.4s
  fmin v11.4s, v11.4s, v2.4s
  fmin v12.4s, v12.4s, v2.4s
  fmin v13.4s, v13.4s, v2.4s
  fmin v14.4s, v14.4s, v2.4s
  fmin v15.4s, v15.4s, v2.4s
  fmin v16.4s, v16.4s, v2.4s
  fmin v17.4s, v17.4s, v2.4s
  fmin v18.4s, v18.4s, v2.4s
  fmin v19.4s, v19.4s, v2.4s
  fmin v20.4s, v20.4s, v2.4s
  fmin v21.4s, v21.4s, v2.4s
  fmin v22.4s, v22.4s, v2.4s
  fmin v23.4s, v23.4s, v2.4s
  fmin v24.4s, v24.4s, v2.4s
  fmin v25.4s, v25.4s, v2.4s
  fmin v26.4s, v26.4s, v2.4s
  fmin v27.4s, v27.4s, v2.4s
  fmin v28.4s, v28.4s, v2.4s
  fmin v29.4s, v29.4s, v2.4s
  fmin v30.4s, v30.4s, v2.4s
  fmin v31.4s, v31.4s, v2.4s

Relu:
  dup v3.4s, wzr
  fmax v8.4s, v8.4s, v3.4s
  fmax v9.4s, v9.4s, v3.4s
  fmax v10.4s, v10.4s, v3.4s
  fmax v11.4s, v11.4s, v3.4s
  fmax v12.4s, v12.4s, v3.4s
  fmax v13.4s, v13.4s, v3.4s
  fmax v14.4s, v14.4s, v3.4s
  fmax v15.4s, v15.4s, v3.4s
  fmax v16.4s, v16.4s, v3.4s
  fmax v17.4s, v17.4s, v3.4s
  fmax v18.4s, v18.4s, v3.4s
  fmax v19.4s, v19.4s, v3.4s
  fmax v20.4s, v20.4s, v3.4s
  fmax v21.4s, v21.4s, v3.4s
  fmax v22.4s, v22.4s, v3.4s
  fmax v23.4s, v23.4s, v3.4s
  fmax v24.4s, v24.4s, v3.4s
  fmax v25.4s, v25.4s, v3.4s
  fmax v26.4s, v26.4s, v3.4s
  fmax v27.4s, v27.4s, v3.4s
  fmax v28.4s, v28.4s, v3.4s
  fmax v29.4s, v29.4s, v3.4s
  fmax v30.4s, v30.4s, v3.4s
  fmax v31.4s, v31.4s, v3.4s

Write:
  cbnz x14, WriteWino
  cbz x9, WriteC8
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
  str s8, [x18]
  cmp w10, #1
  beq WriteEnd
  add x18, x18, x17
  str s10, [x18]
  cmp w10, #2
  beq WriteEnd
  add x18, x18, x17
  str s12, [x18]
  cmp w10, #3
  beq WriteEnd
  add x18, x18, x17
  str s14, [x18]
  cmp w10, #4
  beq WriteEnd
  add x18, x18, x17
  str s16, [x18]
  cmp w10, #5
  beq WriteEnd
  add x18, x18, x17
  str s18, [x18]
  cmp w10, #6
  beq WriteEnd
  add x18, x18, x17
  str s20, [x18]
  cmp w10, #7
  beq WriteEnd
  add x18, x18, x17
  str s22, [x18]
  cmp w10, #8
  beq WriteEnd
  add x18, x18, x17
  str s24, [x18]
  cmp w10, #9
  beq WriteEnd
  add x18, x18, x17
  str s26, [x18]
  cmp w10, #10
  beq WriteEnd
  add x18, x18, x17
  str s28, [x18]
  cmp w10, #11
  beq WriteEnd
  add x18, x18, x17
  str s30, [x18]
  add x18, x18, x17
  b WriteEnd
Write2:
  dup s9, v8.s[1]
  stp s8, s9, [x18]
  cmp w10, #1
  beq WriteEnd
  add x18, x18, x17
  dup s11, v10.s[1]
  stp s10, s11, [x18]
  cmp w10, #2
  beq WriteEnd
  add x18, x18, x17
  dup s13, v12.s[1]
  stp s12, s13, [x18]
  cmp w10, #3
  beq WriteEnd
  add x18, x18, x17
  dup s15, v14.s[1]
  stp s14, s15, [x18]
  cmp w10, #4
  beq WriteEnd
  add x18, x18, x17
  dup s17, v16.s[1]
  stp s16, s17, [x18]
  cmp w10, #5
  beq WriteEnd
  add x18, x18, x17
  dup s19, v18.s[1]
  stp s18, s19, [x18]
  cmp w10, #6
  beq WriteEnd
  add x18, x18, x17
  dup s21, v20.s[1]
  stp s20, s21, [x18]
  cmp w10, #7
  beq WriteEnd
  add x18, x18, x17
  dup s23, v22.s[1]
  stp s22, s23, [x18]
  cmp w10, #8
  beq WriteEnd
  add x18, x18, x17
  dup s25, v24.s[1]
  stp s24, s25, [x18]
  cmp w10, #9
  beq WriteEnd
  add x18, x18, x17
  dup s27, v26.s[1]
  stp s26, s27, [x18]
  cmp w10, #10
  beq WriteEnd
  add x18, x18, x17
  dup s29, v28.s[1]
  stp s28, s29, [x18]
  cmp w10, #11
  beq WriteEnd
  add x18, x18, x17
  dup s31, v30.s[1]
  stp s30, s31, [x18]
  add x18, x18, x17
  b WriteEnd
Write3:
  add x13, x18, #8
  dup s9, v8.s[1]
  stp s8, s9, [x18]
  add x18, x18, x17
  st1 {v8.s}[2], [x13], x17
  cmp w10, #1
  beq WriteEnd
  dup s11, v10.s[1]
  stp s10, s11, [x18]
  add x18, x18, x17
  st1 {v10.s}[2], [x13], x17
  cmp w10, #2
  beq WriteEnd
  dup s13, v12.s[1]
  stp s12, s13, [x18]
  add x18, x18, x17
  st1 {v12.s}[2], [x13], x17
  cmp w10, #3
  beq WriteEnd
  dup s15, v14.s[1]
  stp s14, s15, [x18]
  add x18, x18, x17
  st1 {v14.s}[2], [x13], x17
  cmp w10, #4
  beq WriteEnd
  dup s17, v16.s[1]
  stp s16, s17, [x18]
  add x18, x18, x17
  st1 {v16.s}[2], [x13], x17
  cmp w10, #5
  beq WriteEnd
  dup s19, v18.s[1]
  stp s18, s19, [x18]
  add x18, x18, x17
  st1 {v18.s}[2], [x13], x17
  cmp w10, #6
  beq WriteEnd
  dup s21, v20.s[1]
  stp s20, s21, [x18]
  add x18, x18, x17
  st1 {v20.s}[2], [x13], x17
  cmp w10, #7
  beq WriteEnd
  dup s23, v22.s[1]
  stp s22, s23, [x18]
  add x18, x18, x17
  st1 {v22.s}[2], [x13], x17
  cmp w10, #8
  beq WriteEnd
  dup s25, v24.s[1]
  stp s24, s25, [x18]
  add x18, x18, x17
  st1 {v24.s}[2], [x13], x17
  cmp w10, #9
  beq WriteEnd
  dup s27, v26.s[1]
  stp s26, s27, [x18]
  add x18, x18, x17
  st1 {v26.s}[2], [x13], x17
  cmp w10, #10
  beq WriteEnd
  dup s29, v28.s[1]
  stp s28, s29, [x18]
  add x18, x18, x17
  st1 {v28.s}[2], [x13], x17
  cmp w10, #11
  beq WriteEnd
  dup s31, v30.s[1]
  stp s30, s31, [x18]
  add x18, x18, x17
  st1 {v30.s}[2], [x13]
  b WriteEnd
Write4:
  st1 {v8.4s}, [x18], x17
  cmp w10, #1
  beq WriteEnd
  st1 {v10.4s}, [x18], x17
  cmp w10, #2
  beq WriteEnd
  st1 {v12.4s}, [x18], x17
  cmp w10, #3
  beq WriteEnd
  st1 {v14.4s}, [x18], x17
  cmp w10, #4
  beq WriteEnd
  st1 {v16.4s}, [x18], x17
  cmp w10, #5
  beq WriteEnd
  st1 {v18.4s}, [x18], x17
  cmp w10, #6
  beq WriteEnd
  st1 {v20.4s}, [x18], x17
  cmp w10, #7
  beq WriteEnd
  st1 {v22.4s}, [x18], x17
  cmp w10, #8
  beq WriteEnd
  st1 {v24.4s}, [x18], x17
  cmp w10, #9
  beq WriteEnd
  st1 {v26.4s}, [x18], x17
  cmp w10, #10
  beq WriteEnd
  st1 {v28.4s}, [x18], x17
  cmp w10, #11
  beq WriteEnd
  st1 {v30.4s}, [x18], x17
  b WriteEnd
Write5:
  add x13, x18, #16
  st1 {v8.4s}, [x18], x17
  str s9, [x13]
  cmp w10, #1
  beq WriteEnd
  add x13, x13, x17
  st1 {v10.4s}, [x18], x17
  str s11, [x13]
  cmp w10, #2
  beq WriteEnd
  add x13, x13, x17
  st1 {v12.4s}, [x18], x17
  str s13, [x13]
  cmp w10, #3
  beq WriteEnd
  add x13, x13, x17
  st1 {v14.4s}, [x18], x17
  str s15, [x13]
  cmp w10, #4
  beq WriteEnd
  add x13, x13, x17
  st1 {v16.4s}, [x18], x17
  str s17, [x13]
  cmp w10, #5
  beq WriteEnd
  add x13, x13, x17
  st1 {v18.4s}, [x18], x17
  str s19, [x13]
  cmp w10, #6
  beq WriteEnd
  add x13, x13, x17
  st1 {v20.4s}, [x18], x17
  str s21, [x13]
  cmp w10, #7
  beq WriteEnd
  add x13, x13, x17
  st1 {v22.4s}, [x18], x17
  str s23, [x13]
  cmp w10, #8
  beq WriteEnd
  add x13, x13, x17
  st1 {v24.4s}, [x18], x17
  str s25, [x13]
  cmp w10, #9
  beq WriteEnd
  add x13, x13, x17
  st1 {v26.4s}, [x18], x17
  str s27, [x13]
  cmp w10, #10
  beq WriteEnd
  add x13, x13, x17
  st1 {v28.4s}, [x18], x17
  str s29, [x13]
  cmp w10, #11
  beq WriteEnd
  add x13, x13, x17
  st1 {v30.4s}, [x18], x17
  str s31, [x13]
  b WriteEnd
Write6:
  add x13, x18, #16
  st1 {v8.4s}, [x18], x17
  dup s8, v9.s[1]
  stp s9, s8, [x13]
  cmp w10, #1
  beq WriteEnd
  add x13, x13, x17
  st1 {v10.4s}, [x18], x17
  dup s10, v11.s[1]
  stp s11, s10, [x13]
  cmp w10, #2
  beq WriteEnd
  add x13, x13, x17
  st1 {v12.4s}, [x18], x17
  dup s12, v13.s[1]
  stp s13, s12, [x13]
  cmp w10, #3
  beq WriteEnd
  add x13, x13, x17
  st1 {v14.4s}, [x18], x17
  dup s14, v15.s[1]
  stp s15, s14, [x13]
  cmp w10, #4
  beq WriteEnd
  add x13, x13, x17
  st1 {v16.4s}, [x18], x17
  dup s16, v17.s[1]
  stp s17, s16, [x13]
  cmp w10, #5
  beq WriteEnd
  add x13, x13, x17
  st1 {v18.4s}, [x18], x17
  dup s18, v19.s[1]
  stp s19, s18, [x13]
  cmp w10, #6
  beq WriteEnd
  add x13, x13, x17
  st1 {v20.4s}, [x18], x17
  dup s20, v21.s[1]
  stp s21, s20, [x13]
  cmp w10, #7
  beq WriteEnd
  add x13, x13, x17
  st1 {v22.4s}, [x18], x17
  dup s22, v23.s[1]
  stp s23, s22, [x13]
  cmp w10, #8
  beq WriteEnd
  add x13, x13, x17
  st1 {v24.4s}, [x18], x17
  dup s24, v25.s[1]
  stp s25, s24, [x13]
  cmp w10, #9
  beq WriteEnd
  add x13, x13, x17
  st1 {v26.4s}, [x18], x17
  dup s26, v27.s[1]
  stp s27, s26, [x13]
  cmp w10, #10
  beq WriteEnd
  add x13, x13, x17
  st1 {v28.4s}, [x18], x17
  dup s28, v29.s[1]
  stp s29, s28, [x13]
  cmp w10, #11
  beq WriteEnd
  add x13, x13, x17
  st1 {v30.4s}, [x18], x17
  dup s30, v31.s[1]
  stp s31, s30, [x13]
  b WriteEnd
Write7:
  add x13, x18, #16
  add x16, x18, #24
  st1 {v8.4s}, [x18], x17
  dup s8, v9.s[1]
  stp s9, s8, [x13]
  add x13, x13, x17
  st1 {v9.s}[2], [x16], x17
  cmp w10, #1
  beq WriteEnd
  st1 {v10.4s}, [x18], x17
  dup s10, v11.s[1]
  stp s11, s10, [x13]
  add x13, x13, x17
  st1 {v11.s}[2], [x16], x17
  cmp w10, #2
  beq WriteEnd
  st1 {v12.4s}, [x18], x17
  dup s12, v13.s[1]
  stp s13, s12, [x13]
  add x13, x13, x17
  st1 {v13.s}[2], [x16], x17
  cmp w10, #3
  beq WriteEnd
  st1 {v14.4s}, [x18], x17
  dup s14, v15.s[1]
  stp s15, s14, [x13]
  add x13, x13, x17
  st1 {v15.s}[2], [x16], x17
  cmp w10, #4
  beq WriteEnd
  st1 {v16.4s}, [x18], x17
  dup s16, v17.s[1]
  stp s17, s16, [x13]
  add x13, x13, x17
  st1 {v17.s}[2], [x16], x17
  cmp w10, #5
  beq WriteEnd
  st1 {v18.4s}, [x18], x17
  dup s18, v19.s[1]
  stp s19, s18, [x13]
  add x13, x13, x17
  st1 {v19.s}[2], [x16], x17
  cmp w10, #6
  beq WriteEnd
  st1 {v20.4s}, [x18], x17
  dup s20, v21.s[1]
  stp s21, s20, [x13]
  add x13, x13, x17
  st1 {v21.s}[2], [x16], x17
  cmp w10, #7
  beq WriteEnd
  st1 {v22.4s}, [x18], x17
  dup s22, v23.s[1]
  stp s23, s22, [x13]
  add x13, x13, x17
  st1 {v23.s}[2], [x16], x17
  cmp w10, #8
  beq WriteEnd
  st1 {v24.4s}, [x18], x17
  dup s24, v25.s[1]
  stp s25, s24, [x13]
  add x13, x13, x17
  st1 {v25.s}[2], [x16], x17
  cmp w10, #9
  beq WriteEnd
  st1 {v26.4s}, [x18], x17
  dup s26, v27.s[1]
  stp s27, s26, [x13]
  add x13, x13, x17
  st1 {v27.s}[2], [x16], x17
  cmp w10, #10
  beq WriteEnd
  st1 {v28.4s}, [x18], x17
  dup s28, v29.s[1]
  stp s29, s28, [x13]
  add x13, x13, x17
  st1 {v29.s}[2], [x16], x17
  cmp w10, #11
  beq WriteEnd
  st1 {v30.4s}, [x18], x17
  dup s30, v31.s[1]
  stp s31, s30, [x13]
  add x13, x13, x17
  st1 {v31.s}[2], [x16], x17
  b WriteEnd
WriteC8:
  st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x2], #64
  st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x2], #64
  st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
  st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x2], #64
  st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x2], #64
  st1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x2], #64
  b WriteEnd
WriteWino:
  st1 {v8.4s, v9.4s}, [x18], x8
  st1 {v10.4s, v11.4s}, [x18], x8
  st1 {v12.4s, v13.4s}, [x18], x8
  st1 {v14.4s, v15.4s}, [x18], x8
  st1 {v16.4s, v17.4s}, [x18], x8
  st1 {v18.4s, v19.4s}, [x18], x8
  st1 {v20.4s, v21.4s}, [x18], x8
  st1 {v22.4s, v23.4s}, [x18], x8
  st1 {v24.4s, v25.4s}, [x18], x8
  st1 {v26.4s, v27.4s}, [x18], x8
  st1 {v28.4s, v29.4s}, [x18], x8
  st1 {v30.4s, v31.4s}, [x18], x8
  b WriteEnd
Write8:
  st1 {v8.4s, v9.4s}, [x18], x17
  cmp w10, #1
  beq WriteEnd
  st1 {v10.4s, v11.4s}, [x18], x17
  cmp w10, #2
  beq WriteEnd
  st1 {v12.4s, v13.4s}, [x18], x17
  cmp w10, #3
  beq WriteEnd
  st1 {v14.4s, v15.4s}, [x18], x17
  cmp w10, #4
  beq WriteEnd
  st1 {v16.4s, v17.4s}, [x18], x17
  cmp w10, #5
  beq WriteEnd
  st1 {v18.4s, v19.4s}, [x18], x17
  cmp w10, #6
  beq WriteEnd
  st1 {v20.4s, v21.4s}, [x18], x17
  cmp w10, #7
  beq WriteEnd
  st1 {v22.4s, v23.4s}, [x18], x17
  cmp w10, #8
  beq WriteEnd
  st1 {v24.4s, v25.4s}, [x18], x17
  cmp w10, #9
  beq WriteEnd
  st1 {v26.4s, v27.4s}, [x18], x17
  cmp w10, #10
  beq WriteEnd
  st1 {v28.4s, v29.4s}, [x18], x17
  cmp w10, #11
  beq WriteEnd
  st1 {v30.4s, v31.4s}, [x18], x17

WriteEnd:
  subs w10, w10, #12 // lhs row - 12
  bgt L2

End2:
  subs w7, w7, #8 // rhs col - 8
  add x1, x1, x15 // rhs ptr + stride
  cbz x3, NoBiasStep
  add x3, x3, #32 // bias ptr + stride
NoBiasStep:
  cbnz x14, WinoDstStep
  cbz x9, NoDstStep
  add x2, x2, #32 // dst ptr + stride
  b NoDstStep
WinoDstStep:
  add x2, x2, x11
NoDstStep:
  bgt L1

End1:
  sub sp, sp, #128
  ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
  ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
  ret
#endif
