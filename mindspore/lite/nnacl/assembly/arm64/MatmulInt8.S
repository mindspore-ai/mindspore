#ifdef __aarch64__
#include "nnacl/assembly_global.h"
    .text
    .align 5
    .global MatmulInt8Neon64
#ifndef __APPLE__
    .type MatmulInt8Neon64, %function
#endif

//void MatmulInt8Neon64(const int8_t *a, const int8_t *b, int8_t *dst, int row4, int col4, int deep16, const int *a_sums,
//                      const int *bias, int act_min, int act_max, int out_zp, int32_t *multiplier, int32_t *left_shift,
//                      int32_t *right_shift, int row, int col, int stride, int filter_peroc)

// x0: a(left matrix ptr)
// x1: b(right matrix ptr)
// x2: out ptr
// w3: row4
// w4: col4
// w5: deep16
// x6: a_sums
// x7: bias
// w8: act_min
// w9: act_max
// w10: out_zp
// x11: multiplier
// x12: left_shift
// x13: right_shift
// w14: row
// w15: col
// w24: stride
// w27: filter_peroc

asm_function MatmulInt8Neon64
  sub sp, sp, #208
  st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
  st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
  stp x19, x20, [sp], #16
  stp x21, x22, [sp], #16
  stp x23, x24, [sp], #16
  stp x25, x26, [sp], #16
  stp x27, x28, [sp], #16

  ldr w8, [sp]
  ldr w9, [sp, #8]
  ldr w10, [sp, #16]
  ldr x11, [sp, #24]
  ldr x12, [sp, #32]
  ldr x13, [sp, #40]
  ldr w14, [sp, #48]
  ldr w15, [sp, #56]
  ldr w24, [sp, #64]
  ldr w27, [sp, #72]

  mov w17, #4       // sizeof(int8)*4
  mul w21, w5, w17  // the stride of a/b: sizeof(int8)*4*deep16
  mov w17, #1
  mov x25, x2
L1:
  cmp w4, #0      // if at the end of col4
  beq End1

  mov w16, w3     // reset a row4 counter
  mov w23, w14    // reset a row counter
  mov x17, x0     // reload a ptr
  mov x22, x6     // reload a_sums ptr
L2:
  cmp w16, #0
  beq End2

  mov x18, x1     // reload b ptr
  mov x19, x7     // reload bias ptr
  mov w20, w5     // reload depth
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
L3:
  cmp w20, #0
  beq End3

  ld1 {v0.16b}, [x17], #16
  ld1 {v1.16b}, [x17], #16
  ld1 {v2.16b}, [x17], #16
  ld1 {v3.16b}, [x17], #16
  ld1 {v4.16b}, [x18], #16
  ld1 {v5.16b}, [x18], #16
  ld1 {v6.16b}, [x18], #16
  ld1 {v7.16b}, [x18], #16

  smull v8.8h, v4.8b, v0.8b
  smull v9.8h, v5.8b, v0.8b
  smull v10.8h, v6.8b, v0.8b
  smull v11.8h, v7.8b, v0.8b
  smull v12.8h, v4.8b, v1.8b
  smull v13.8h, v5.8b, v1.8b
  smull v14.8h, v6.8b, v1.8b
  smull v15.8h, v7.8b, v1.8b

  smlal2 v8.8h, v4.16b, v0.16b
  smlal2 v9.8h, v5.16b, v0.16b
  smlal2 v10.8h, v6.16b, v0.16b
  smlal2 v11.8h, v7.16b, v0.16b
  smlal2 v12.8h, v4.16b, v1.16b
  smlal2 v13.8h, v5.16b, v1.16b
  smlal2 v14.8h, v6.16b, v1.16b
  smlal2 v15.8h, v7.16b, v1.16b

  sadalp v16.4s, v8.8h
  sadalp v17.4s, v9.8h
  sadalp v18.4s, v10.8h
  sadalp v19.4s, v11.8h
  sadalp v20.4s, v12.8h
  sadalp v21.4s, v13.8h
  sadalp v22.4s, v14.8h
  sadalp v23.4s, v15.8h

  smull v8.8h, v4.8b, v2.8b
  smull v9.8h, v5.8b, v2.8b
  smull v10.8h, v6.8b, v2.8b
  smull v11.8h, v7.8b, v2.8b
  smull v12.8h, v4.8b, v3.8b
  smull v13.8h, v5.8b, v3.8b
  smull v14.8h, v6.8b, v3.8b
  smull v15.8h, v7.8b, v3.8b

  smlal2 v8.8h, v4.16b, v2.16b
  smlal2 v9.8h, v5.16b, v2.16b
  smlal2 v10.8h, v6.16b, v2.16b
  smlal2 v11.8h, v7.16b, v2.16b
  smlal2 v12.8h, v4.16b, v3.16b
  smlal2 v13.8h, v5.16b, v3.16b
  smlal2 v14.8h, v6.16b, v3.16b
  smlal2 v15.8h, v7.16b, v3.16b

  sadalp v24.4s, v8.8h
  sadalp v25.4s, v9.8h
  sadalp v26.4s, v10.8h
  sadalp v27.4s, v11.8h
  sadalp v28.4s, v12.8h
  sadalp v29.4s, v13.8h
  sadalp v30.4s, v14.8h
  sadalp v31.4s, v15.8h
  subs w20, w20, #16  // depth + 16
  b L3

End3:
  addp v16.4s, v16.4s, v17.4s
  addp v18.4s, v18.4s, v19.4s
  addp v20.4s, v20.4s, v21.4s
  addp v22.4s, v22.4s, v23.4s
  addp v24.4s, v24.4s, v25.4s
  addp v26.4s, v26.4s, v27.4s
  addp v28.4s, v28.4s, v29.4s
  addp v30.4s, v30.4s, v31.4s

  addp v16.4s, v16.4s, v18.4s
  addp v17.4s, v20.4s, v22.4s
  addp v18.4s, v24.4s, v26.4s
  addp v19.4s, v28.4s, v30.4s

  // Add (Bias+Depth*Za*Zb-Za*Bsums)
  ld1 {v15.4s}, [x19], #16
  add v16.4s, v16.4s, v15.4s
  add v17.4s, v17.4s, v15.4s
  add v18.4s, v18.4s, v15.4s
  add v19.4s, v19.4s, v15.4s

  cmp w27, #0
  beq PerTLoad
PerCLoad:
  ld1 {v20.4s}, [x6], #16
  ld1 {v21.4s}, [x6], #16
  ld1 {v22.4s}, [x6], #16
  ld1 {v23.4s}, [x6], #16

  ld1 {v13.4s}, [x12]
  ld1 {v12.4s}, [x11]
  ld1 {v11.4s}, [x13]
  b Apply

PerTLoad:
  ld1 {v14.4s}, [x22], #16
  dup v20.4s, v14.s[0]
  dup v21.4s, v14.s[1]
  dup v22.4s, v14.s[2]
  dup v23.4s, v14.s[3]

  ld1 {v14.s}[0], [x12]
  dup v13.4s, v14.s[0]
  ld1 {v14.s}[0], [x11]
  dup v12.4s, v14.s[0]
  ld1 {v14.s}[0], [x13]
  dup v11.4s, v14.s[0]
  b Apply

Apply:
  // Subtract (Asums*Zb)
  sub v16.4s, v16.4s, v20.4s
  sub v17.4s, v17.4s, v21.4s
  sub v18.4s, v18.4s, v22.4s
  sub v19.4s, v19.4s, v23.4s

  // Apply left shift
  sqshl v16.4s, v16.4s, v13.4s
  sqshl v17.4s, v17.4s, v13.4s
  sqshl v18.4s, v18.4s, v13.4s
  sqshl v19.4s, v19.4s, v13.4s

  // Apply the fixed-point part of the multiplier.
  sqrdmulh v16.4s, v16.4s, v12.4s
  sqrdmulh v17.4s, v17.4s, v12.4s
  sqrdmulh v18.4s, v18.4s, v12.4s
  sqrdmulh v19.4s, v19.4s, v12.4s

  // Apply right shift
  and v20.16b, v11.16b, v16.16b
  sshr v20.4s, v20.4s, #31
  sqadd v16.4s, v16.4s, v20.4s
  srshl v16.4s, v16.4s, v11.4s
  and v21.16b, v11.16b, v17.16b
  sshr v21.4s, v21.4s, #31
  sqadd v17.4s, v17.4s, v21.4s
  srshl v17.4s, v17.4s, v11.4s
  and v22.16b, v11.16b, v18.16b
  sshr v22.4s, v22.4s, #31
  sqadd v18.4s, v18.4s, v22.4s
  srshl v18.4s, v18.4s, v11.4s
  and v23.16b, v11.16b, v19.16b
  sshr v23.4s, v23.4s, #31
  sqadd v19.4s, v19.4s, v23.4s
  srshl v19.4s, v19.4s, v11.4s

  // Add the destination zero point
  dup v10.4s, w10
  add v16.4s, v16.4s, v10.4s
  add v17.4s, v17.4s, v10.4s
  add v18.4s, v18.4s, v10.4s
  add v19.4s, v19.4s, v10.4s

  // Apply the act_min bound
  dup v9.4s, w8
  smax v16.4s, v16.4s, v9.4s
  smax v17.4s, v17.4s, v9.4s
  smax v18.4s, v18.4s, v9.4s
  smax v19.4s, v19.4s, v9.4s

  // Apply the act_min bound
  dup v8.4s, w9
  smin v16.4s, v16.4s, v8.4s
  smin v17.4s, v17.4s, v8.4s
  smin v18.4s, v18.4s, v8.4s
  smin v19.4s, v19.4s, v8.4s

  // int32 -> int16
  sqxtn v13.4h, v16.4s
  sqxtn2 v13.8h, v17.4s
  sqxtn v14.4h, v18.4s
  sqxtn2 v14.8h, v19.4s

  // int16 -> int8
  sqxtn v15.8b, v13.8h
  sqxtn2 v15.16b, v14.8h

  cmp w23, #4
  blt Write     // if rows < 4
  cmp w15, #4
  blt Write     // if cols < 4

  st1 {v15.s}[0], [x2], x24
  st1 {v15.s}[1], [x2], x24
  st1 {v15.s}[2], [x2], x24
  st1 {v15.s}[3], [x2], x24
  b Endwrite

Write:
  cmp w15, #4
  beq WriteCol4
  cmp w15, #3
  beq WriteCol3
  cmp w15, #2
  beq WriteCol2
  cmp w15, #1
  beq WriteCol1

WriteCol4:
  st1 {v15.s}[0], [x2], x24
  cmp w23, #1
  beq Endwrite
  st1 {v15.s}[1], [x2], x24
  cmp w23, #2
  beq Endwrite
  st1 {v15.s}[2], [x2], x24
  cmp w23, #3
  beq Endwrite
  st1 {v15.s}[3], [x2], x24
  b Endwrite

WriteCol3:
  mov x26, x2
  st1 {v15.b}[0], [x26], #1
  st1 {v15.b}[1], [x26], #1
  st1 {v15.b}[2], [x26], #1
  add x2, x2, x24
  cmp w23, #1
  beq Endwrite
  mov x26, x2
  st1 {v15.b}[4], [x26], #1
  st1 {v15.b}[5], [x26], #1
  st1 {v15.b}[6], [x26], #1
  add x2, x2, x24
  cmp w23, #2
  beq Endwrite
  mov x26, x2
  st1 {v15.b}[8], [x26], #1
  st1 {v15.b}[9], [x26], #1
  st1 {v15.b}[10], [x26], #1
  add x2, x2, x24
  cmp w23, #3
  beq Endwrite
  mov x26, x2
  st1 {v15.b}[12], [x26], #1
  st1 {v15.b}[13], [x26], #1
  st1 {v15.b}[14], [x26], #1
  add x2, x2, x24
  b Endwrite

WriteCol2:
  mov x26, x2
  st1 {v15.b}[0], [x26], #1
  st1 {v15.b}[1], [x26], #1
  add x2, x2, x24
  cmp w23, #1
  beq Endwrite
  mov x26, x2
  st1 {v15.b}[4], [x26], #1
  st1 {v15.b}[5], [x26], #1
  add x2, x2, x24
  cmp w23, #2
  beq Endwrite
  mov x26, x2
  st1 {v15.b}[8], [x26], #1
  st1 {v15.b}[9], [x26], #1
  add x2, x2, x24
  cmp w23, #3
  beq Endwrite
  mov x26, x2
  st1 {v15.b}[12], [x26], #1
  st1 {v15.b}[13], [x26], #1
  add x2, x2, x24
  b Endwrite

WriteCol1:
  st1 {v15.b}[0], [x2], x24
  cmp w23, #1
  beq Endwrite
  st1 {v15.b}[4], [x2], x24
  cmp w23, #2
  beq Endwrite
  st1 {v15.b}[8], [x2], x24
  cmp w23, #3
  beq Endwrite
  st1 {v15.b}[12], [x2], x24
  b Endwrite

Endwrite:
  sub w16, w16, #4      // a row4 counter - 4
  sub w23, w23, #4      // a row counter - 4
  b L2

End2:
  sub w4, w4, #4        // b col4 counter - 4
  sub w15, w15, #4      // b col counter - 4
  add x1, x1, x21       // b ptr + stride
  add x7, x7, #16       // bias ptr + stride
  add x25, x25, #4      // output + stride(4 * sizeof(int8))
  mov x2, x25

  cmp w27, #0
  beq PerTEnd2
  add x12, x12, #16
  add x11, x11, #16
  add x13, x13, #16
PerTEnd2:
  b L1

End1:
  sub sp, sp, #208
  ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
  ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
  ldp x19, x20, [sp], #16
  ldp x21, x22, [sp], #16
  ldp x23, x24, [sp], #16
  ldp x25, x26, [sp], #16
  ldp x27, x28, [sp], #16
  ret
#endif
