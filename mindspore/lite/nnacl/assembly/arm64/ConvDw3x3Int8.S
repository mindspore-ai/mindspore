#ifdef __aarch64__
#include "nnacl/assembly_global.h"

.text
.align 5
.global ConvDw3x3Int8Neon64
#ifndef __APPLE__
.type ConvDw3x3Int8Neon64, %function
#endif


// void ConvDw3x3Int8Neon64(int8_t *output, const int8_t *input, const int16_t *weight, const int32_t *bias, int input_col_size,
//                          int input_row_size, int channel, int output_h, int output_w, int8_t in_zp, int32_t out_zp,
//                          int *out_multiplier, int *left_shift, int *right_shift, int32_t acc_min, int32_t acc_max,
//                          size_t per_channel)
//
// x0: output
// x1: input
// x2: weight
// x3: bias
// w4: col_size
// w5: row_size
// w6: channel
// w7: output_h
// w8: output_w
// w9: in_zp
// w10: out_zp
// w11: out_multiplier
// w12: left_shift
// w13: right_shift
// w14: acc_min
// w15: acc_max
// w16: per_channel

asm_function ConvDw3x3Int8Neon64
  sub sp, sp, #176
  st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
  st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
  stp x19, x20, [sp], #16
  stp x21, x22, [sp], #16
  stp x23, x24, [sp], #16

  ldr x8, [sp]
  ldr x9, [sp, #8]
  ldr x10, [sp, #16]
  ldr x11, [sp, #24]
  ldr x12, [sp, #32]
  ldr x13, [sp, #40]
  ldr x14, [sp, #48]
  ldr x15, [sp, #56]
  ldr x23, [sp, #64]  // per_channel

  add x19, x3, #16
  add w20, w6, w6   // channel * 2
  add w21, w4, w4   // col_size * 2
  dup v25.8b, w9

  cbnz w23, PER_CHANNEL_DUMP
  PER_LAYER_DUMP:
      ld1r {v27.4s}, [x11]   // out_multiplier
      ld1r {v26.4s}, [x12]   // left_shift
      ld1r {v28.4s}, [x13]   // right_shift
      b MAIN_FUC
  PER_CHANNEL_DUMP:
      ld1 {v27.4s}, [x11]
      ld1 {v26.4s}, [x12]
      ld1 {v28.4s}, [x13]
  MAIN_FUC:
  dup v29.4s, w10
  dup v30.4s, w14
  dup v31.4s, w15
  ldr w24, [x12]

  // Load weights
  ld1 {v0.8h}, [x2], x20
  ld1 {v1.8h}, [x2], x20
  ld1 {v2.8h}, [x2], x20
  ld1 {v3.8h}, [x2], x20
  ld1 {v4.8h}, [x2], x20
  ld1 {v5.8h}, [x2], x20
  ld1 {v6.8h}, [x2], x20
  ld1 {v7.8h}, [x2], x20
  ld1 {v8.8h}, [x2], x20

  mov x16, x1
  add x17, x16, x5
  add x18, x17, x5
  ld1 {v9.8b}, [x16], x4
  ld1 {v10.8b}, [x16], x4
  ld1 {v11.8b}, [x16], x4
  ld1 {v13.8b}, [x17], x4
  ld1 {v14.8b}, [x17], x4
  ld1 {v15.8b}, [x17], x4
  ld1 {v17.8b}, [x18], x4
  ld1 {v18.8b}, [x18], x4
  ld1 {v19.8b}, [x18], x4

  ld1 {v21.4s}, [x3]
  ld1 {v22.4s}, [x19]
  ld1 {v23.4s}, [x3]
  ld1 {v24.4s}, [x19]

  // subtract input zp
  ssubl v9.8h, v9.8b, v25.8b
  ssubl v10.8h, v10.8b, v25.8b
  ssubl v11.8h, v11.8b, v25.8b
  ssubl v13.8h, v13.8b, v25.8b
  ssubl v14.8h, v14.8b, v25.8b
  ssubl v15.8h, v15.8b, v25.8b
  ssubl v17.8h, v17.8b, v25.8b
  ssubl v18.8h, v18.8b, v25.8b
  ssubl v19.8h, v19.8b, v25.8b

  cmp w8, #2
  beq WIDTH2_LEFT
  cmp w8, #1
  beq WIDTH1_LEFT

HEIGHT1_LOOP:
  smlal v21.4s, v0.4h, v9.4h
  ld1 {v12.8b}, [x16]
  smlal2 v22.4s, v0.8h, v9.8h
  ld1 {v16.8b}, [x17]
  smlal v23.4s, v0.4h, v10.4h
  smlal2 v24.4s, v0.8h, v10.8h
  ld1 {v20.8b}, [x18]
  add x1, x1, x21  
  ssubl v12.8h, v12.8b, v25.8b
  smlal v21.4s, v1.4h, v10.4h
  mov x16, x1
  add x17, x16, x5
  add x18, x17, x5
  smlal2 v22.4s, v1.8h, v10.8h
  ld1 {v9.8b}, [x16], x4
  ssubl v16.8h, v16.8b, v25.8b
  smlal v23.4s, v1.4h, v11.4h
  ld1 {v10.8b}, [x16], x4
  ssubl v20.8h, v20.8b, v25.8b
  smlal2 v24.4s, v1.8h, v11.8h
  smlal v21.4s, v2.4h, v11.4h
  smlal2 v22.4s, v2.8h, v11.8h
  ld1 {v11.8b}, [x16], x4
  smlal v23.4s, v2.4h, v12.4h
  smlal2 v24.4s, v2.8h, v12.8h
  smlal v21.4s, v3.4h, v13.4h
  smlal2 v22.4s, v3.8h, v13.8h
  ld1 {v13.8b}, [x17], x4
  smlal v23.4s, v3.4h, v14.4h
  smlal2 v24.4s, v3.8h, v14.8h
  smlal v21.4s, v4.4h, v14.4h
  smlal2 v22.4s, v4.8h, v14.8h
  ld1 {v14.8b}, [x17], x4
  smlal v23.4s, v4.4h, v15.4h
  smlal2 v24.4s, v4.8h, v15.8h
  smlal v21.4s, v5.4h, v15.4h
  smlal2 v22.4s, v5.8h, v15.8h
  ld1 {v15.8b}, [x17], x4
  smlal v23.4s, v5.4h, v16.4h
  smlal2 v24.4s, v5.8h, v16.8h
  smlal v21.4s, v6.4h, v17.4h
  smlal2 v22.4s, v6.8h, v17.8h
  ld1 {v17.8b}, [x18], x4
  smlal v23.4s, v6.4h, v18.4h
  smlal2 v24.4s, v6.8h, v18.8h
  smlal v21.4s, v7.4h, v18.4h
  smlal2 v22.4s, v7.8h, v18.8h
  ld1 {v18.8b}, [x18], x4
  smlal v23.4s, v7.4h, v19.4h
  smlal2 v24.4s, v7.8h, v19.8h
  smlal v21.4s, v8.4h, v19.4h
  smlal2 v22.4s, v8.8h, v19.8h
  ld1 {v19.8b}, [x18], x4
  smlal v23.4s, v8.4h, v20.4h
  smlal2 v24.4s, v8.8h, v20.8h

  cbnz w23, PER_CHANNEL_POST1
  cbz w24, SKIP_LEFTSHIFT1
  sqshl v21.4s, v21.4s, v26.4s
  sqshl v22.4s, v22.4s, v26.4s
  sqshl v23.4s, v23.4s, v26.4s
  sqshl v24.4s, v24.4s, v26.4s
  sqrdmulh v21.4s, v21.4s, v27.4s
  sqrdmulh v22.4s, v22.4s, v27.4s
  sqrdmulh v23.4s, v23.4s, v27.4s
  sqrdmulh v24.4s, v24.4s, v27.4s
  b OUTZP1

SKIP_LEFTSHIFT1:
  sqrdmulh v21.4s, v21.4s, v27.4s
  sqrdmulh v22.4s, v22.4s, v27.4s
  sqrdmulh v23.4s, v23.4s, v27.4s
  sqrdmulh v24.4s, v24.4s, v27.4s
  sqrshl v21.4s, v21.4s, v28.4s
  sqrshl v22.4s, v22.4s, v28.4s
  sqrshl v23.4s, v23.4s, v28.4s
  sqrshl v24.4s, v24.4s, v28.4s
  b OUTZP1

PER_CHANNEL_POST1:
  sqshl v21.4s, v21.4s, v26.4s
  sqshl v23.4s, v23.4s, v26.4s
  sqrdmulh v21.4s, v21.4s, v27.4s
  sqrdmulh v23.4s, v23.4s, v27.4s
  ldr q26, [x12, #16]
  sqrshl v21.4s, v21.4s, v28.4s
  sqrshl v23.4s, v23.4s, v28.4s
  ldr q27, [x11, #16]
  sqshl v22.4s, v22.4s, v26.4s
  sqshl v24.4s, v24.4s, v26.4s
  ldr q28, [x13, #16]
  sqrdmulh v22.4s, v22.4s, v27.4s
  sqrdmulh v24.4s, v24.4s, v27.4s
  ld1 {v26.4s}, [x12]
  sqrshl v22.4s, v22.4s, v28.4s
  sqrshl v24.4s, v24.4s, v28.4s
  ld1 {v27.4s}, [x11]
  ld1 {v28.4s}, [x13]

OUTZP1:
  // Add output zero point
  sqadd v21.4s, v21.4s, v29.4s
  sqadd v22.4s, v22.4s, v29.4s
  sqadd v23.4s, v23.4s, v29.4s
  sqadd v24.4s, v24.4s, v29.4s

  // Apply min bound
  smax v21.4s, v21.4s, v30.4s
  smax v22.4s, v22.4s, v30.4s
  smax v23.4s, v23.4s, v30.4s
  smax v24.4s, v24.4s, v30.4s

  // Apply max bound
  smin v21.4s, v21.4s, v31.4s
  smin v22.4s, v22.4s, v31.4s
  smin v23.4s, v23.4s, v31.4s
  smin v24.4s, v24.4s, v31.4s

  sqxtn v21.4h, v21.4s
  sqxtn2 v21.8h, v22.4s
  ld1 {v22.4s}, [x19]
  ssubl v9.8h, v9.8b, v25.8b
  ssubl v10.8h, v10.8b, v25.8b
  sqxtn v23.4h, v23.4s
  sqxtn2 v23.8h, v24.4s
  ld1 {v24.4s}, [x19]
  sqxtn v21.8b, v21.8h
  sqxtn2 v21.16b, v23.8h
  st1 {v21.8b}, [x0], x6
  mov v23.d[0], v21.d[1]
  ld1 {v21.4s}, [x3]
  st1 {v23.8b}, [x0], x6
  ssubl v11.8h, v11.8b, v25.8b
  ssubl v13.8h, v13.8b, v25.8b
  ld1 {v23.4s}, [x3]
  ssubl v14.8h, v14.8b, v25.8b
  ssubl v15.8h, v15.8b, v25.8b
  ssubl v17.8h, v17.8b, v25.8b
  ssubl v18.8h, v18.8b, v25.8b
  ssubl v19.8h, v19.8b, v25.8b
  sub w8, w8, #2
  cmp w8, #2
  bgt HEIGHT1_LOOP

  cmp w8, #2
  blt WIDTH1_LEFT

WIDTH2_LEFT:
  smlal v21.4s, v0.4h, v9.4h
  smlal2 v22.4s, v0.8h, v9.8h
  ld1 {v12.8b}, [x16]
  ssubl v12.8h, v12.8b, v25.8b
  smlal v23.4s, v0.4h, v10.4h
  smlal2 v24.4s, v0.8h, v10.8h
  smlal v21.4s, v1.4h, v10.4h
  smlal2 v22.4s, v1.8h, v10.8h
  ld1 {v16.8b}, [x17]
  smlal v23.4s, v1.4h, v11.4h
  smlal2 v24.4s, v1.8h, v11.8h
  smlal v21.4s, v2.4h, v11.4h
  smlal2 v22.4s, v2.8h, v11.8h
  ld1 {v20.8b}, [x18]
  smlal v23.4s, v2.4h, v12.4h
  smlal2 v24.4s, v2.8h, v12.8h
  smlal v21.4s, v3.4h, v13.4h
  smlal2 v22.4s, v3.8h, v13.8h
  smlal v23.4s, v3.4h, v14.4h
  smlal2 v24.4s, v3.8h, v14.8h
  smlal v21.4s, v4.4h, v14.4h
  smlal2 v22.4s, v4.8h, v14.8h
  ssubl v16.8h, v16.8b, v25.8b
  smlal v23.4s, v4.4h, v15.4h
  smlal2 v24.4s, v4.8h, v15.8h
  smlal v21.4s, v5.4h, v15.4h
  smlal2 v22.4s, v5.8h, v15.8h
  ssubl v20.8h, v20.8b, v25.8b
  smlal v23.4s, v5.4h, v16.4h
  smlal2 v24.4s, v5.8h, v16.8h
  smlal v21.4s, v6.4h, v17.4h
  smlal2 v22.4s, v6.8h, v17.8h
  smlal v23.4s, v6.4h, v18.4h
  smlal2 v24.4s, v6.8h, v18.8h
  smlal v21.4s, v7.4h, v18.4h
  smlal2 v22.4s, v7.8h, v18.8h
  smlal v23.4s, v7.4h, v19.4h
  smlal2 v24.4s, v7.8h, v19.8h
  smlal v21.4s, v8.4h, v19.4h
  smlal2 v22.4s, v8.8h, v19.8h
  smlal v23.4s, v8.4h, v20.4h
  smlal2 v24.4s, v8.8h, v20.8h

  cbnz w23, PER_CHANNEL_POST2
  cbz w24, SKIP_LEFTSHIFT2
  sqshl v21.4s, v21.4s, v26.4s
  sqshl v22.4s, v22.4s, v26.4s
  sqshl v23.4s, v23.4s, v26.4s
  sqshl v24.4s, v24.4s, v26.4s
  sqrdmulh v21.4s, v21.4s, v27.4s
  sqrdmulh v22.4s, v22.4s, v27.4s
  sqrdmulh v23.4s, v23.4s, v27.4s
  sqrdmulh v24.4s, v24.4s, v27.4s
  b OUTZP2

SKIP_LEFTSHIFT2:
  sqrdmulh v21.4s, v21.4s, v27.4s
  sqrdmulh v22.4s, v22.4s, v27.4s
  sqrdmulh v23.4s, v23.4s, v27.4s
  sqrdmulh v24.4s, v24.4s, v27.4s
  sqrshl v21.4s, v21.4s, v28.4s
  sqrshl v22.4s, v22.4s, v28.4s
  sqrshl v23.4s, v23.4s, v28.4s
  sqrshl v24.4s, v24.4s, v28.4s
  b OUTZP2

PER_CHANNEL_POST2:
  sqshl v21.4s, v21.4s, v26.4s
  sqshl v23.4s, v23.4s, v26.4s
  sqrdmulh v21.4s, v21.4s, v27.4s
  sqrdmulh v23.4s, v23.4s, v27.4s
  ldr q26, [x12, #16]
  sqrshl v21.4s, v21.4s, v28.4s
  sqrshl v23.4s, v23.4s, v28.4s
  ldr q27, [x11, #16]
  sqshl v22.4s, v22.4s, v26.4s
  sqshl v24.4s, v24.4s, v26.4s
  ldr q28, [x13, #16]
  sqrdmulh v22.4s, v22.4s, v27.4s
  sqrdmulh v24.4s, v24.4s, v27.4s
  sqrshl v22.4s, v22.4s, v28.4s
  sqrshl v24.4s, v24.4s, v28.4s

OUTZP2:
  // Add output zero point
  sqadd v21.4s, v21.4s, v29.4s
  sqadd v22.4s, v22.4s, v29.4s
  sqadd v23.4s, v23.4s, v29.4s
  sqadd v24.4s, v24.4s, v29.4s

  // Apply min bound
  smax v21.4s, v21.4s, v30.4s
  smax v22.4s, v22.4s, v30.4s
  smax v23.4s, v23.4s, v30.4s
  smax v24.4s, v24.4s, v30.4s

  // Apply max bound
  smin v21.4s, v21.4s, v31.4s
  smin v22.4s, v22.4s, v31.4s
  smin v23.4s, v23.4s, v31.4s
  smin v24.4s, v24.4s, v31.4s

  sqxtn v21.4h, v21.4s
  sqxtn2 v21.8h, v22.4s
  sqxtn v23.4h, v23.4s
  sqxtn2 v23.8h, v24.4s
  sqxtn v21.8b, v21.8h
  sqxtn2 v21.16b, v23.8h
  st1 {v21.8b}, [x0], x6
  mov v23.d[0], v21.d[1]
  st1 {v23.8b}, [x0], x6
  b End

WIDTH1_LEFT:
  smlal v21.4s, v0.4h, v9.4h
  smlal2 v22.4s, v0.8h, v9.8h
  smlal v21.4s, v1.4h, v10.4h
  smlal2 v22.4s, v1.8h, v10.8h
  smlal v21.4s, v2.4h, v11.4h
  smlal2 v22.4s, v2.8h, v11.8h
  smlal v21.4s, v3.4h, v13.4h
  smlal2 v22.4s, v3.8h, v13.8h
  smlal v21.4s, v4.4h, v14.4h
  smlal2 v22.4s, v4.8h, v14.8h
  smlal v21.4s, v5.4h, v15.4h
  smlal2 v22.4s, v5.8h, v15.8h
  smlal v21.4s, v6.4h, v17.4h
  smlal2 v22.4s, v6.8h, v17.8h
  smlal v21.4s, v7.4h, v18.4h
  smlal2 v22.4s, v7.8h, v18.8h
  smlal v21.4s, v8.4h, v19.4h
  smlal2 v22.4s, v8.8h, v19.8h

  cbnz w23, PER_CHANNEL_POST3
  cbz w24, SKIP_LEFTSHIFT3
  sqshl v21.4s, v21.4s, v26.4s
  sqshl v22.4s, v22.4s, v26.4s
  sqrdmulh v21.4s, v21.4s, v27.4s
  sqrdmulh v22.4s, v22.4s, v27.4s
  b OUTZP3

SKIP_LEFTSHIFT3:
  sqrdmulh v21.4s, v21.4s, v27.4s
  sqrdmulh v22.4s, v22.4s, v27.4s
  sqrshl v21.4s, v21.4s, v28.4s
  sqrshl v22.4s, v22.4s, v28.4s
  b OUTZP3

PER_CHANNEL_POST3:
  sqshl v21.4s, v21.4s, v26.4s
  sqrdmulh v21.4s, v21.4s, v27.4s
  ldr q26, [x12, #16]
  sqrshl v21.4s, v21.4s, v28.4s
  ldr q27, [x11, #16]
  sqshl v22.4s, v22.4s, v26.4s
  ldr q28, [x13, #16]
  sqrdmulh v22.4s, v22.4s, v27.4s
  sqrshl v22.4s, v22.4s, v28.4s

OUTZP3:
  // Add output zero point
  sqadd v21.4s, v21.4s, v29.4s
  sqadd v22.4s, v22.4s, v29.4s

  // Apply min bound
  smax v21.4s, v21.4s, v30.4s
  smax v22.4s, v22.4s, v30.4s

  // Apply max bound
  smin v21.4s, v21.4s, v31.4s
  smin v22.4s, v22.4s, v31.4s

  sqxtn v21.4h, v21.4s
  sqxtn2 v21.8h, v22.4s
  sqxtn v21.8b, v21.8h
  st1 {v21.8b}, [x0], x6

End:
  sub sp, sp, #176
  ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
  ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
  ldp x19, x20, [sp], #16
  ldp x21, x22, [sp], #16
  ldp x23, x24, [sp], #16
  ret

#endif
