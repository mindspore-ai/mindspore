#ifdef __aarch64__
#include "nnacl/assembly_global.h"
    .text
    .align 5
    .global MatmulInt8DpNeon64
#ifndef __APPLE__
    .type MatmulInt8DpNeon64, %function
#endif

//void MatmulInt8DpNeon64(const int8_t *a, const int8_t *b, int8_t *dst, int row8, int col8, int deep4, 
//                      const int *a_sums, const int *bias, int act_min, int act_max, int out_zp,
//                      int *multiplier, int *left_shift, int *right_shift, int row, int col, int stride, int peroc);

// x0: a(left matrix ptr)
// x1: b(right matrix ptr)
// x2: out ptr
// w3: row8
// w4: col8
// w5: deep4
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

asm_function MatmulInt8DpNeon64
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

  mov w17, #8       // sizeof(int8)*8
  mul w21, w5, w17  // the stride of a/b: sizeof(int8)*8*deep4
  mov x25, x2
L1:
  cmp w4, #0      // if at the end of col8
  beq End1

  mov w16, w3     // reset a row8 counter
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
  cmp w20, #16
  blt LoopD4

LoopD16:
  ld1 {v0.16b, v1.16b}, [x17], #32
  ld1 {v2.16b, v3.16b}, [x18], #32

  sdot v16.4s, v2.16b, v0.4b[0]
  sdot v18.4s, v2.16b, v0.4b[1]
  sdot v20.4s, v2.16b, v0.4b[2]
  sdot v22.4s, v2.16b, v0.4b[3]

  ld1 {v4.16b, v5.16b}, [x17], #32
  sdot v24.4s, v2.16b, v1.4b[0]
  sdot v26.4s, v2.16b, v1.4b[1]
  sdot v28.4s, v2.16b, v1.4b[2]
  sdot v30.4s, v2.16b, v1.4b[3]

  ld1 {v6.16b, v7.16b}, [x18], #32
  sdot v17.4s, v3.16b, v0.4b[0]
  sdot v19.4s, v3.16b, v0.4b[1]
  sdot v21.4s, v3.16b, v0.4b[2]
  sdot v23.4s, v3.16b, v0.4b[3]

  sdot v25.4s, v3.16b, v1.4b[0]
  sdot v27.4s, v3.16b, v1.4b[1]
  sdot v29.4s, v3.16b, v1.4b[2]
  sdot v31.4s, v3.16b, v1.4b[3]

  ld1 {v8.16b, v9.16b}, [x17], #32
  sdot v16.4s, v6.16b, v4.4b[0]
  sdot v18.4s, v6.16b, v4.4b[1]
  sdot v20.4s, v6.16b, v4.4b[2]
  sdot v22.4s, v6.16b, v4.4b[3]

  sdot v24.4s, v6.16b, v5.4b[0]
  sdot v26.4s, v6.16b, v5.4b[1]
  sdot v28.4s, v6.16b, v5.4b[2]
  sdot v30.4s, v6.16b, v5.4b[3]

  ld1 {v10.16b, v11.16b}, [x18], #32
  sdot v17.4s, v7.16b, v4.4b[0]
  sdot v19.4s, v7.16b, v4.4b[1]
  sdot v21.4s, v7.16b, v4.4b[2]
  sdot v23.4s, v7.16b, v4.4b[3]

  sdot v25.4s, v7.16b, v5.4b[0]
  sdot v27.4s, v7.16b, v5.4b[1]
  sdot v29.4s, v7.16b, v5.4b[2]
  sdot v31.4s, v7.16b, v5.4b[3]

  ld1 {v12.16b, v13.16b}, [x17], #32
  sdot v16.4s, v10.16b, v8.4b[0]
  sdot v18.4s, v10.16b, v8.4b[1]
  sdot v20.4s, v10.16b, v8.4b[2]
  sdot v22.4s, v10.16b, v8.4b[3]

  sdot v24.4s, v10.16b, v9.4b[0]
  sdot v26.4s, v10.16b, v9.4b[1]
  sdot v28.4s, v10.16b, v9.4b[2]
  sdot v30.4s, v10.16b, v9.4b[3]

  ld1 {v14.16b, v15.16b}, [x18], #32
  sdot v17.4s, v11.16b, v8.4b[0]
  sdot v19.4s, v11.16b, v8.4b[1]
  sdot v21.4s, v11.16b, v8.4b[2]
  sdot v23.4s, v11.16b, v8.4b[3]

  sdot v25.4s, v11.16b, v9.4b[0]
  sdot v27.4s, v11.16b, v9.4b[1]
  sdot v29.4s, v11.16b, v9.4b[2]
  sdot v31.4s, v11.16b, v9.4b[3]

  sdot v16.4s, v14.16b, v12.4b[0]
  sdot v18.4s, v14.16b, v12.4b[1]
  sdot v20.4s, v14.16b, v12.4b[2]
  sdot v22.4s, v14.16b, v12.4b[3]

  sdot v24.4s, v14.16b, v13.4b[0]
  sdot v26.4s, v14.16b, v13.4b[1]
  sdot v28.4s, v14.16b, v13.4b[2]
  sdot v30.4s, v14.16b, v13.4b[3]

  sdot v17.4s, v15.16b, v12.4b[0]
  sdot v19.4s, v15.16b, v12.4b[1]
  sdot v21.4s, v15.16b, v12.4b[2]
  sdot v23.4s, v15.16b, v12.4b[3]

  sdot v25.4s, v15.16b, v13.4b[0]
  sdot v27.4s, v15.16b, v13.4b[1]
  sdot v29.4s, v15.16b, v13.4b[2]
  sdot v31.4s, v15.16b, v13.4b[3]

  subs w20, w20, #16  // depth - 16
  b L3

LoopD4:
  cmp w20, #0
  beq End3

  ld1 {v0.16b, v1.16b}, [x17], #32
  ld1 {v2.16b, v3.16b}, [x18], #32

  sdot v16.4s, v2.16b, v0.4b[0]
  sdot v18.4s, v2.16b, v0.4b[1]
  sdot v20.4s, v2.16b, v0.4b[2]
  sdot v22.4s, v2.16b, v0.4b[3]
  sdot v24.4s, v2.16b, v1.4b[0]
  sdot v26.4s, v2.16b, v1.4b[1]
  sdot v28.4s, v2.16b, v1.4b[2]
  sdot v30.4s, v2.16b, v1.4b[3]
  sdot v17.4s, v3.16b, v0.4b[0]
  sdot v19.4s, v3.16b, v0.4b[1]
  sdot v21.4s, v3.16b, v0.4b[2]
  sdot v23.4s, v3.16b, v0.4b[3]
  sdot v25.4s, v3.16b, v1.4b[0]
  sdot v27.4s, v3.16b, v1.4b[1]
  sdot v29.4s, v3.16b, v1.4b[2]
  sdot v31.4s, v3.16b, v1.4b[3]
  
  subs w20, w20, #4  // depth - 4
  b LoopD4

End3:
  // Add (Bias+Depth*Za*Zb-Za*Bsums)
  ld1 {v15.4s}, [x19], #16  
  ld1 {v14.4s}, [x19], #16  
  add v16.4s, v16.4s, v15.4s
  add v18.4s, v18.4s, v15.4s
  add v20.4s, v20.4s, v15.4s
  add v22.4s, v22.4s, v15.4s
  add v24.4s, v24.4s, v15.4s
  add v26.4s, v26.4s, v15.4s
  add v28.4s, v28.4s, v15.4s
  add v30.4s, v30.4s, v15.4s
  add v17.4s, v17.4s, v14.4s
  add v19.4s, v19.4s, v14.4s
  add v21.4s, v21.4s, v14.4s
  add v23.4s, v23.4s, v14.4s
  add v25.4s, v25.4s, v14.4s
  add v27.4s, v27.4s, v14.4s
  add v29.4s, v29.4s, v14.4s
  add v31.4s, v31.4s, v14.4s

  cmp w27, #0
  beq PerTSumLoad
PerCSumLoad:
  ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x6], #64
  ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x6], #64
  ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x6], #64
  ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x6], #64
  b ApplySum
PerTSumLoad:
  ld1 {v14.4s}, [x22], #16
  ld1 {v15.4s}, [x22], #16
  dup v0.4s, v14.s[0]
  dup v1.4s, v14.s[0]
  dup v2.4s, v14.s[1]
  dup v3.4s, v14.s[1]
  dup v4.4s, v14.s[2]
  dup v5.4s, v14.s[2]
  dup v6.4s, v14.s[3]
  dup v7.4s, v14.s[3]
  dup v8.4s, v15.s[0]
  dup v9.4s, v15.s[0]
  dup v10.4s, v15.s[1]
  dup v11.4s, v15.s[1]
  dup v12.4s, v15.s[2]
  dup v13.4s, v15.s[2]
  dup v14.4s, v15.s[3]
  dup v15.4s, v14.s[0]
ApplySum:
  // Subtract (Asums*Zb)
  sub v16.4s, v16.4s, v0.4s
  sub v17.4s, v17.4s, v1.4s
  sub v18.4s, v18.4s, v2.4s
  sub v19.4s, v19.4s, v3.4s
  sub v20.4s, v20.4s, v4.4s
  sub v21.4s, v21.4s, v5.4s
  sub v22.4s, v22.4s, v6.4s
  sub v23.4s, v23.4s, v7.4s
  sub v24.4s, v24.4s, v8.4s
  sub v25.4s, v25.4s, v9.4s
  sub v26.4s, v26.4s, v10.4s
  sub v27.4s, v27.4s, v11.4s
  sub v28.4s, v28.4s, v12.4s
  sub v29.4s, v29.4s, v13.4s
  sub v30.4s, v30.4s, v14.4s
  sub v31.4s, v31.4s, v15.4s

  cmp w27, #0
  beq PerTRoundLoad
PerCRoundLoad:
  ld1 {v8.4s, v9.4s}, [x12]
  ld1 {v10.4s, v11.4s}, [x11]
  ld1 {v12.4s, v13.4s}, [x13]
  b ApplyRound
PerTRoundLoad:
  ld1 {v14.s}[0], [x12]
  dup v8.4s, v14.s[0]
  dup v9.4s, v14.s[0]
  ld1 {v14.s}[0], [x11]
  dup v10.4s, v14.s[0]
  dup v11.4s, v14.s[0]
  ld1 {v14.s}[0], [x13]
  dup v12.4s, v14.s[0]
  dup v13.4s, v14.s[0]
ApplyRound:
  // Apply left shift
  sqshl v16.4s, v16.4s, v8.4s
  sqshl v17.4s, v17.4s, v9.4s
  sqshl v18.4s, v18.4s, v8.4s
  sqshl v19.4s, v19.4s, v9.4s
  sqshl v20.4s, v20.4s, v8.4s
  sqshl v21.4s, v21.4s, v9.4s
  sqshl v22.4s, v22.4s, v8.4s
  sqshl v23.4s, v23.4s, v9.4s
  sqshl v24.4s, v24.4s, v8.4s
  sqshl v25.4s, v25.4s, v9.4s
  sqshl v26.4s, v26.4s, v8.4s
  sqshl v27.4s, v27.4s, v9.4s
  sqshl v28.4s, v28.4s, v8.4s
  sqshl v29.4s, v29.4s, v9.4s
  sqshl v30.4s, v30.4s, v8.4s
  sqshl v31.4s, v31.4s, v9.4s

  // Apply the fixed-point part of the multiplier.
  sqrdmulh v16.4s, v16.4s, v10.4s
  sqrdmulh v17.4s, v17.4s, v11.4s
  sqrdmulh v18.4s, v18.4s, v10.4s
  sqrdmulh v19.4s, v19.4s, v11.4s
  sqrdmulh v20.4s, v20.4s, v10.4s
  sqrdmulh v21.4s, v21.4s, v11.4s
  sqrdmulh v22.4s, v22.4s, v10.4s
  sqrdmulh v23.4s, v23.4s, v11.4s
  sqrdmulh v24.4s, v24.4s, v10.4s
  sqrdmulh v25.4s, v25.4s, v11.4s
  sqrdmulh v26.4s, v26.4s, v10.4s
  sqrdmulh v27.4s, v27.4s, v11.4s
  sqrdmulh v28.4s, v28.4s, v10.4s
  sqrdmulh v29.4s, v29.4s, v11.4s
  sqrdmulh v30.4s, v30.4s, v10.4s
  sqrdmulh v31.4s, v31.4s, v11.4s

  // Apply right shift
  and v0.16b, v12.16b, v16.16b
  sshr v0.4s, v0.4s, #31
  sqadd v16.4s, v16.4s, v0.4s
  srshl v16.4s, v16.4s, v12.4s
  and v1.16b, v13.16b, v17.16b
  sshr v1.4s, v1.4s, #31
  sqadd v17.4s, v17.4s, v1.4s
  srshl v17.4s, v17.4s, v13.4s
  and v2.16b, v12.16b, v18.16b
  sshr v2.4s, v2.4s, #31
  sqadd v18.4s, v18.4s, v2.4s
  srshl v18.4s, v18.4s, v12.4s
  and v3.16b, v13.16b, v19.16b
  sshr v3.4s, v3.4s, #31
  sqadd v19.4s, v19.4s, v3.4s
  srshl v19.4s, v19.4s, v13.4s
  and v0.16b, v12.16b, v20.16b
  sshr v0.4s, v0.4s, #31
  sqadd v20.4s, v20.4s, v0.4s
  srshl v20.4s, v20.4s, v12.4s
  and v1.16b, v13.16b, v21.16b
  sshr v1.4s, v1.4s, #31
  sqadd v21.4s, v21.4s, v1.4s
  srshl v21.4s, v21.4s, v13.4s
  and v2.16b, v12.16b, v22.16b
  sshr v2.4s, v2.4s, #31
  sqadd v22.4s, v22.4s, v2.4s
  srshl v22.4s, v22.4s, v12.4s
  and v3.16b, v13.16b, v23.16b
  sshr v3.4s, v3.4s, #31
  sqadd v23.4s, v23.4s, v3.4s
  srshl v23.4s, v23.4s, v13.4s
  and v0.16b, v12.16b, v24.16b
  sshr v0.4s, v0.4s, #31
  sqadd v24.4s, v24.4s, v0.4s
  srshl v24.4s, v24.4s, v12.4s
  and v1.16b, v13.16b, v25.16b
  sshr v1.4s, v1.4s, #31
  sqadd v25.4s, v25.4s, v1.4s
  srshl v25.4s, v25.4s, v13.4s
  and v2.16b, v12.16b, v26.16b
  sshr v2.4s, v2.4s, #31
  sqadd v26.4s, v26.4s, v2.4s
  srshl v26.4s, v26.4s, v12.4s
  and v3.16b, v13.16b, v27.16b
  sshr v3.4s, v3.4s, #31
  sqadd v27.4s, v27.4s, v3.4s
  srshl v27.4s, v27.4s, v13.4s
  and v0.16b, v12.16b, v28.16b
  sshr v0.4s, v0.4s, #31
  sqadd v28.4s, v28.4s, v0.4s
  srshl v28.4s, v28.4s, v12.4s
  and v1.16b, v13.16b, v29.16b
  sshr v1.4s, v1.4s, #31
  sqadd v29.4s, v29.4s, v1.4s
  srshl v29.4s, v29.4s, v13.4s
  and v2.16b, v12.16b, v30.16b
  sshr v2.4s, v2.4s, #31
  sqadd v30.4s, v30.4s, v2.4s
  srshl v30.4s, v30.4s, v12.4s
  and v3.16b, v13.16b, v31.16b
  sshr v3.4s, v3.4s, #31
  sqadd v31.4s, v31.4s, v3.4s
  srshl v31.4s, v31.4s, v13.4s

  // Add the destination zero point
  dup v8.4s, w10
  add v16.4s, v16.4s, v8.4s
  add v17.4s, v17.4s, v8.4s
  add v18.4s, v18.4s, v8.4s
  add v19.4s, v19.4s, v8.4s
  add v20.4s, v20.4s, v8.4s
  add v21.4s, v21.4s, v8.4s
  add v22.4s, v22.4s, v8.4s
  add v23.4s, v23.4s, v8.4s
  add v24.4s, v24.4s, v8.4s
  add v25.4s, v25.4s, v8.4s
  add v26.4s, v26.4s, v8.4s
  add v27.4s, v27.4s, v8.4s
  add v28.4s, v28.4s, v8.4s
  add v29.4s, v29.4s, v8.4s
  add v30.4s, v30.4s, v8.4s
  add v31.4s, v31.4s, v8.4s

  // Apply the act_min bound
  dup v7.4s, w8
  smax v16.4s, v16.4s, v7.4s
  smax v17.4s, v17.4s, v7.4s
  smax v18.4s, v18.4s, v7.4s
  smax v19.4s, v19.4s, v7.4s
  smax v20.4s, v20.4s, v7.4s
  smax v21.4s, v21.4s, v7.4s
  smax v22.4s, v22.4s, v7.4s
  smax v23.4s, v23.4s, v7.4s
  smax v24.4s, v24.4s, v7.4s
  smax v25.4s, v25.4s, v7.4s
  smax v26.4s, v26.4s, v7.4s
  smax v27.4s, v27.4s, v7.4s
  smax v28.4s, v28.4s, v7.4s
  smax v29.4s, v29.4s, v7.4s
  smax v30.4s, v30.4s, v7.4s
  smax v31.4s, v31.4s, v7.4s

  // Apply the act_max bound
  dup v6.4s, w9
  smin v16.4s, v16.4s, v6.4s
  smin v17.4s, v17.4s, v6.4s
  smin v18.4s, v18.4s, v6.4s
  smin v19.4s, v19.4s, v6.4s
  smin v20.4s, v20.4s, v6.4s
  smin v21.4s, v21.4s, v6.4s
  smin v22.4s, v22.4s, v6.4s
  smin v23.4s, v23.4s, v6.4s
  smin v24.4s, v24.4s, v6.4s
  smin v25.4s, v25.4s, v6.4s
  smin v26.4s, v26.4s, v6.4s
  smin v27.4s, v27.4s, v6.4s
  smin v28.4s, v28.4s, v6.4s
  smin v29.4s, v29.4s, v6.4s
  smin v30.4s, v30.4s, v6.4s
  smin v31.4s, v31.4s, v6.4s

  // int32 -> int16
  sqxtn v0.4h, v16.4s
  sqxtn2 v0.8h, v17.4s
  sqxtn v1.4h, v18.4s
  sqxtn2 v1.8h, v19.4s
  sqxtn v2.4h, v20.4s
  sqxtn2 v2.8h, v21.4s
  sqxtn v3.4h, v22.4s
  sqxtn2 v3.8h, v23.4s
  sqxtn v4.4h, v24.4s
  sqxtn2 v4.8h, v25.4s
  sqxtn v5.4h, v26.4s
  sqxtn2 v5.8h, v27.4s
  sqxtn v6.4h, v28.4s
  sqxtn2 v6.8h, v29.4s
  sqxtn v7.4h, v30.4s
  sqxtn2 v7.8h, v31.4s

  // int16 -> int8
  sqxtn v8.8b, v0.8h
  sqxtn2 v8.16b, v1.8h
  sqxtn v9.8b, v2.8h
  sqxtn2 v9.16b, v3.8h
  sqxtn v10.8b, v4.8h
  sqxtn2 v10.16b, v5.8h
  sqxtn v11.8b, v6.8h
  sqxtn2 v11.16b, v7.8h

  cmp w23, #8
  blt Write     // if rows < 8
  cmp w15, #8
  blt Write     // if cols < 8

  st1 {v8.d}[0], [x2], x24
  st1 {v8.d}[1], [x2], x24
  st1 {v9.d}[0], [x2], x24
  st1 {v9.d}[1], [x2], x24
  st1 {v10.d}[0], [x2], x24
  st1 {v10.d}[1], [x2], x24
  st1 {v11.d}[0], [x2], x24
  st1 {v11.d}[1], [x2], x24
  b Endwrite

Write:
  cmp w15, #8
  bge WriteCol8
  cmp w15, #7
  beq WriteCol7
  cmp w15, #6
  beq WriteCol6
  cmp w15, #5
  beq WriteCol5
  cmp w15, #4
  beq WriteCol4
  cmp w15, #3
  beq WriteCol3
  cmp w15, #2
  beq WriteCol2
  cmp w15, #1
  beq WriteCol1

WriteCol8:
  st1 {v8.d}[0], [x2], x24
  cmp w23, #1
  beq Endwrite
  st1 {v8.d}[1], [x2], x24
  cmp w23, #2
  beq Endwrite
  st1 {v9.d}[0], [x2], x24
  cmp w23, #3
  beq Endwrite
  st1 {v9.d}[1], [x2], x24
  cmp w23, #4
  beq Endwrite
  st1 {v10.d}[0], [x2], x24
  cmp w23, #5
  beq Endwrite
  st1 {v10.d}[1], [x2], x24
  cmp w23, #6
  beq Endwrite
  st1 {v11.d}[0], [x2], x24
  cmp w23, #7
  beq Endwrite
  st1 {v11.d}[1], [x2], x24
  b Endwrite

WriteCol7:
  mov x26, x2
  st1 {v8.s}[0], [x26], #4
  st1 {v8.h}[2], [x26], #2
  st1 {v8.b}[6], [x26], #1
  add x2, x2, x24
  cmp w23, #1
  beq Endwrite
  mov x26, x2
  st1 {v8.s}[2], [x26], #4
  st1 {v8.h}[6], [x26], #2
  st1 {v8.b}[14], [x26], #1
  add x2, x2, x24
  cmp w23, #2
  beq Endwrite
  mov x26, x2
  st1 {v9.s}[0], [x26], #4
  st1 {v9.h}[2], [x26], #2
  st1 {v9.b}[6], [x26], #1
  add x2, x2, x24
  cmp w23, #3
  beq Endwrite
  mov x26, x2
  st1 {v9.s}[2], [x26], #4
  st1 {v9.h}[6], [x26], #2
  st1 {v9.b}[14], [x26], #1
  add x2, x2, x24
  cmp w23, #4
  beq Endwrite
  mov x26, x2
  st1 {v10.s}[0], [x26], #4
  st1 {v10.h}[2], [x26], #2
  st1 {v10.b}[6], [x26], #1
  add x2, x2, x24
  cmp w23, #5
  beq Endwrite
  mov x26, x2
  st1 {v10.s}[2], [x26], #4
  st1 {v10.h}[6], [x26], #2
  st1 {v10.b}[14], [x26], #1
  add x2, x2, x24
  cmp w23, #6
  beq Endwrite
  mov x26, x2
  st1 {v11.s}[0], [x26], #4
  st1 {v11.h}[2], [x26], #2
  st1 {v11.b}[6], [x26], #1
  add x2, x2, x24
  cmp w23, #7
  beq Endwrite
  mov x26, x2
  st1 {v11.s}[2], [x26], #4
  st1 {v11.h}[6], [x26], #2
  st1 {v11.b}[14], [x26], #1
  add x2, x2, x24
  b Endwrite

WriteCol6:
  mov x26, x2
  st1 {v8.s}[0], [x26], #4
  st1 {v8.h}[2], [x26], #2
  add x2, x2, x24
  cmp w23, #1
  beq Endwrite
  mov x26, x2
  st1 {v8.s}[2], [x26], #4
  st1 {v8.h}[6], [x26], #2
  add x2, x2, x24
  cmp w23, #2
  beq Endwrite
  mov x26, x2
  st1 {v9.s}[0], [x26], #4
  st1 {v9.h}[2], [x26], #2
  add x2, x2, x24
  cmp w23, #3
  beq Endwrite
  mov x26, x2
  st1 {v9.s}[2], [x26], #4
  st1 {v9.h}[6], [x26], #2
  add x2, x2, x24
  cmp w23, #4
  beq Endwrite
  mov x26, x2
  st1 {v10.s}[0], [x26], #4
  st1 {v10.h}[2], [x26], #2
  add x2, x2, x24
  cmp w23, #5
  beq Endwrite
  mov x26, x2
  st1 {v10.s}[2], [x26], #4
  st1 {v10.h}[6], [x26], #2
  add x2, x2, x24
  cmp w23, #6
  beq Endwrite
  mov x26, x2
  st1 {v11.s}[0], [x26], #4
  st1 {v11.h}[2], [x26], #2
  add x2, x2, x24
  cmp w23, #7
  beq Endwrite
  mov x26, x2
  st1 {v11.s}[2], [x26], #4
  st1 {v11.h}[6], [x26], #2
  add x2, x2, x24
  b Endwrite

WriteCol5:
  mov x26, x2
  st1 {v8.s}[0], [x26], #4
  st1 {v8.b}[4], [x26], #1
  add x2, x2, x24
  cmp w23, #1
  beq Endwrite
  mov x26, x2
  st1 {v8.s}[2], [x26], #4
  st1 {v8.b}[12], [x26], #1
  add x2, x2, x24
  cmp w23, #2
  beq Endwrite
  mov x26, x2
  st1 {v9.s}[0], [x26], #4
  st1 {v9.b}[4], [x26], #1
  add x2, x2, x24
  cmp w23, #3
  beq Endwrite
  mov x26, x2
  st1 {v9.s}[2], [x26], #4
  st1 {v9.b}[12], [x26], #1
  add x2, x2, x24
  cmp w23, #4
  beq Endwrite
  mov x26, x2
  st1 {v10.s}[0], [x26], #4
  st1 {v10.b}[4], [x26], #1
  add x2, x2, x24
  cmp w23, #5
  beq Endwrite
  mov x26, x2
  st1 {v10.s}[2], [x26], #4
  st1 {v10.b}[12], [x26], #1
  add x2, x2, x24
  cmp w23, #6
  beq Endwrite
  mov x26, x2
  st1 {v11.s}[0], [x26], #4
  st1 {v11.b}[4], [x26], #1
  add x2, x2, x24
  cmp w23, #7
  beq Endwrite
  mov x26, x2
  st1 {v11.s}[2], [x26], #4
  st1 {v11.b}[12], [x26], #1
  add x2, x2, x24
  b Endwrite

WriteCol4:
  st1 {v8.s}[0], [x2], x24
  cmp w23, #1
  beq Endwrite
  st1 {v8.s}[2], [x2], x24
  cmp w23, #2
  beq Endwrite
  st1 {v9.s}[0], [x2], x24
  cmp w23, #3
  beq Endwrite
  st1 {v9.s}[2], [x2], x24
  cmp w23, #4
  beq Endwrite
  st1 {v10.s}[0], [x2], x24
  cmp w23, #5
  beq Endwrite
  st1 {v10.s}[2], [x2], x24
  cmp w23, #6
  beq Endwrite
  st1 {v11.s}[0], [x2], x24
  cmp w23, #7
  beq Endwrite
  st1 {v11.s}[2], [x2], x24
  b Endwrite

WriteCol3:
  mov x26, x2
  st1 {v8.h}[0], [x26], #2
  st1 {v8.b}[2], [x26], #1
  add x2, x2, x24
  cmp w23, #1
  beq Endwrite
  mov x26, x2
  st1 {v8.h}[4], [x26], #2
  st1 {v8.b}[10], [x26], #1
  add x2, x2, x24
  cmp w23, #2
  beq Endwrite
  mov x26, x2
  st1 {v9.h}[0], [x26], #2
  st1 {v9.b}[2], [x26], #1
  add x2, x2, x24
  cmp w23, #3
  beq Endwrite
  mov x26, x2
  st1 {v9.h}[4], [x26], #2
  st1 {v9.b}[10], [x26], #1
  add x2, x2, x24
  cmp w23, #4
  beq Endwrite
  mov x26, x2
  st1 {v10.h}[0], [x26], #2
  st1 {v10.b}[2], [x26], #1
  add x2, x2, x24
  cmp w23, #5
  beq Endwrite
  mov x26, x2
  st1 {v10.h}[4], [x26], #2
  st1 {v10.b}[10], [x26], #1
  add x2, x2, x24
  cmp w23, #6
  beq Endwrite
  mov x26, x2
  st1 {v11.h}[0], [x26], #2
  st1 {v11.b}[2], [x26], #1
  add x2, x2, x24
  cmp w23, #7
  beq Endwrite
  mov x26, x2
  st1 {v11.h}[4], [x26], #2
  st1 {v11.b}[10], [x26], #1
  add x2, x2, x24
  b Endwrite

WriteCol2:
  st1 {v8.h}[0], [x2], x24
  cmp w23, #1
  beq Endwrite
  st1 {v8.h}[4], [x2], x24
  cmp w23, #2
  beq Endwrite
  st1 {v9.h}[0], [x2], x24
  cmp w23, #3
  beq Endwrite
  st1 {v9.h}[4], [x2], x24
  cmp w23, #4
  beq Endwrite
  st1 {v10.h}[0], [x2], x24
  cmp w23, #5
  beq Endwrite
  st1 {v10.h}[4], [x2], x24
  cmp w23, #6
  beq Endwrite
  st1 {v11.h}[0], [x2], x24
  cmp w23, #7
  beq Endwrite
  st1 {v11.h}[4], [x2], x24
  b Endwrite

WriteCol1:
  st1 {v8.b}[0], [x2], x24
  cmp w23, #1
  beq Endwrite
  st1 {v8.b}[8], [x2], x24
  cmp w23, #2
  beq Endwrite
  st1 {v9.b}[0], [x2], x24
  cmp w23, #3
  beq Endwrite
  st1 {v9.b}[8], [x2], x24
  cmp w23, #4
  beq Endwrite
  st1 {v10.b}[0], [x2], x24
  cmp w23, #5
  beq Endwrite
  st1 {v10.b}[8], [x2], x24
  cmp w23, #6
  beq Endwrite
  st1 {v11.b}[0], [x2], x24
  cmp w23, #7
  beq Endwrite
  st1 {v11.b}[8], [x2], x24
  b Endwrite

Endwrite:  
  sub w16, w16, #8      // a row8 counter - 8
  sub w23, w23, #8      // a row counter - 8
  b L2

End2:
  sub w4, w4, #8        // b col8 counter - 8
  sub w15, w15, #8      // b col counter - 8
  add x1, x1, x21       // b ptr + stride
  add x7, x7, #32       // bias ptr + stride
  add x25, x25, #8      // output + stride(8 * sizeof(int8))
  mov x2, x25

  cmp w27, #0
  beq PerTEnd2
  add x12, x12, #32
  add x11, x11, #32
  add x13, x13, #32
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
