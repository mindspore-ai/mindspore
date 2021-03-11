#ifdef __aarch64__
#include "nnacl/assembly_global.h"

.text
.align 5
.global ConvDw3x3Int8Stride2
#ifndef __APPLE__
.type ConvDw3x3Int8Stride2, %function
#endif


// void ConvDw3x3Int8Stride2(int8_t *output, const int8_t *input, const int16_t *weight, const int32_t *bias, int input_col_size,
//                           int input_row_size, int channel, int output_h, int output_w, int8_t in_zp, int32_t out_zp,
//                           int *out_multiplier, int *left_shift, int *right_shift, int32_t acc_min, int32_t acc_max
//                           size_t per_channel)
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

asm_function ConvDw3x3Int8Stride2
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
    dup v28.8b, w9    // in_zp
    ldr w24, [x12]
    
    dup v29.4s, w10
    dup v30.4s, w14
    dup v31.4s, w15
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
    ssubl v9.8h, v9.8b, v28.8b
    ld1 {v11.8b}, [x16], x4
    ssubl v10.8h, v10.8b, v28.8b
    ld1 {v14.8b}, [x17], x4
    ssubl v11.8h, v11.8b, v28.8b
    ld1 {v15.8b}, [x17], x4
    ssubl v14.8h, v14.8b, v28.8b
    ld1 {v16.8b}, [x17], x4
    ssubl v15.8h, v15.8b, v28.8b
    ld1 {v19.8b}, [x18], x4
    ssubl v16.8h, v16.8b, v28.8b
    ld1 {v20.8b}, [x18], x4
    ssubl v19.8h, v19.8b, v28.8b
    ld1 {v21.8b}, [x18], x4
    ssubl v20.8h, v20.8b, v28.8b
    ssubl v21.8h, v21.8b, v28.8b

    ld1 {v24.4s}, [x3]
    ld1 {v25.4s}, [x19]
    ld1 {v26.4s}, [x3]
    ld1 {v27.4s}, [x19]

    cmp w8, #2
    beq WIDTH2_LEFT
    cmp w8, #1
    beq WIDTH1_LEFT

HEIGHT1_LOOP:
    smlal v24.4s, v0.4h, v9.4h
    ld1 {v12.8b}, [x16], x4
    smlal2 v25.4s, v0.8h, v9.8h
    ld1 {v17.8b}, [x17], x4
    ssubl v12.8h, v12.8b, v28.8b
    smlal v26.4s, v0.4h, v11.4h
    ld1 {v22.8b}, [x18], x4
    ssubl v17.8h, v17.8b, v28.8b
    smlal2 v27.4s, v0.8h, v11.8h
    ld1 {v13.8b}, [x16], x4
    ssubl v22.8h, v22.8b, v28.8b
    smlal v24.4s, v1.4h, v10.4h
    ld1 {v18.8b}, [x17], x4
    ssubl v13.8h, v13.8b, v28.8b
    smlal2 v25.4s, v1.8h, v10.8h
    ld1 {v23.8b}, [x18], x4
    ssubl v18.8h, v18.8b, v28.8b
    smlal v26.4s, v1.4h, v12.4h
    mov v9.16b, v13.16b
    ssubl v23.8h, v23.8b, v28.8b
    smlal2 v27.4s, v1.8h, v12.8h
    ld1 {v10.8b}, [x16], x4
    smlal v24.4s, v2.4h, v11.4h
    smlal2 v25.4s, v2.8h, v11.8h
    ld1 {v11.8b}, [x16], x4
    ssubl v10.8h, v10.8b, v28.8b
    smlal v26.4s, v2.4h, v13.4h
    ssubl v11.8h, v11.8b, v28.8b
    smlal2 v27.4s, v2.8h, v13.8h

    smlal v24.4s, v3.4h, v14.4h
    smlal2 v25.4s, v3.8h, v14.8h
    mov v14.16b, v18.16b
    smlal v26.4s, v3.4h, v16.4h
    smlal2 v27.4s, v3.8h, v16.8h
    smlal v24.4s, v4.4h, v15.4h
    smlal2 v25.4s, v4.8h, v15.8h
    ld1 {v15.8b}, [x17], x4
    smlal v26.4s, v4.4h, v17.4h
    smlal2 v27.4s, v4.8h, v17.8h
    smlal v24.4s, v5.4h, v16.4h
    smlal2 v25.4s, v5.8h, v16.8h
    ld1 {v16.8b}, [x17], x4
    ssubl v15.8h, v15.8b, v28.8b
    smlal v26.4s, v5.4h, v18.4h
    ssubl v16.8h, v16.8b, v28.8b
    smlal2 v27.4s, v5.8h, v18.8h

    smlal v24.4s, v6.4h, v19.4h
    smlal2 v25.4s, v6.8h, v19.8h
    mov v19.16b, v23.16b
    smlal v26.4s, v6.4h, v21.4h
    smlal2 v27.4s, v6.8h, v21.8h
    smlal v24.4s, v7.4h, v20.4h
    smlal2 v25.4s, v7.8h, v20.8h
    ld1 {v20.8b}, [x18], x4
    smlal v26.4s, v7.4h, v22.4h
    smlal2 v27.4s, v7.8h, v22.8h
    smlal v24.4s, v8.4h, v21.4h
    smlal2 v25.4s, v8.8h, v21.8h
    ld1 {v21.8b}, [x18], x4
    ssubl v20.8h, v20.8b, v28.8b
    smlal v26.4s, v8.4h, v23.4h
    ssubl v21.8h, v21.8b, v28.8b
    smlal2 v27.4s, v8.8h, v23.8h

    cbnz w23, PER_CHANNEL_POST1
    ld1r {v17.4s}, [x11]
    cbz w24, SKIP_LEFTSHIFT1
    ld1r {v12.4s}, [x12]
    sqshl v24.4s, v24.4s, v12.4s
    sqshl v25.4s, v25.4s, v12.4s
    sqshl v26.4s, v26.4s, v12.4s
    sqshl v27.4s, v27.4s, v12.4s
    sqrdmulh v24.4s, v24.4s, v17.4s
    sqrdmulh v25.4s, v25.4s, v17.4s
    sqrdmulh v26.4s, v26.4s, v17.4s
    sqrdmulh v27.4s, v27.4s, v17.4s
    b OUTZP1

SKIP_LEFTSHIFT1:
    ld1r {v22.4s}, [x13]
    sqrdmulh v24.4s, v24.4s, v17.4s
    sqrdmulh v25.4s, v25.4s, v17.4s
    sqrdmulh v26.4s, v26.4s, v17.4s
    sqrdmulh v27.4s, v27.4s, v17.4s
    sqrshl v24.4s, v24.4s, v22.4s
    sqrshl v25.4s, v25.4s, v22.4s
    sqrshl v26.4s, v26.4s, v22.4s
    sqrshl v27.4s, v27.4s, v22.4s
    b OUTZP1

PER_CHANNEL_POST1:
    ld1 {v12.4s}, [x12]
    sqshl v24.4s, v24.4s, v12.4s
    ldr q13, [x12, #16]
    sqshl v25.4s, v25.4s, v13.4s
    ld1 {v17.4s}, [x11]
    sqshl v26.4s, v26.4s, v12.4s
    sqshl v27.4s, v27.4s, v13.4s
    ldr q18, [x11, #16]
    sqrdmulh v24.4s, v24.4s, v17.4s
    sqrdmulh v25.4s, v25.4s, v18.4s
    ld1 {v22.4s}, [x13]
    sqrdmulh v26.4s, v26.4s, v17.4s
    sqrdmulh v27.4s, v27.4s, v18.4s
    ldr q23, [x13, #16]
    sqrshl v24.4s, v24.4s, v22.4s
    sqrshl v25.4s, v25.4s, v23.4s
    sqrshl v26.4s, v26.4s, v22.4s
    sqrshl v27.4s, v27.4s, v23.4s

OUTZP1:
    // Add output zero point
    sqadd v24.4s, v24.4s, v29.4s
    sqadd v25.4s, v25.4s, v29.4s
    sqadd v26.4s, v26.4s, v29.4s
    sqadd v27.4s, v27.4s, v29.4s

    // Apply min bound
    smax v24.4s, v24.4s, v30.4s
    smax v25.4s, v25.4s, v30.4s
    smax v26.4s, v26.4s, v30.4s
    smax v27.4s, v27.4s, v30.4s

    // Apply max bound
    smin v24.4s, v24.4s, v31.4s
    smin v25.4s, v25.4s, v31.4s
    smin v26.4s, v26.4s, v31.4s
    smin v27.4s, v27.4s, v31.4s

    sqxtn v24.4h, v24.4s
    sqxtn2 v24.8h, v25.4s
    ld1 {v25.4s}, [x19]
    sqxtn v26.4h, v26.4s
    sqxtn2 v26.8h, v27.4s
    ld1 {v27.4s}, [x19]
    sqxtn v24.8b, v24.8h
    sqxtn2 v24.16b, v26.8h
    st1 {v24.8b}, [x0], x6
    mov v26.d[0], v24.d[1]
    ld1 {v24.4s}, [x3]
    st1 {v26.8b}, [x0], x6
    ld1 {v26.4s}, [x3]
    sub w8, w8, #2
    cmp w8, #2
    bgt HEIGHT1_LOOP

    cmp w8, #2
    blt WIDTH1_LEFT

WIDTH2_LEFT:
    smlal v24.4s, v0.4h, v9.4h
    ld1 {v12.8b}, [x16], x4
    smlal2 v25.4s, v0.8h, v9.8h
    ld1 {v17.8b}, [x17], x4
    ssubl v12.8h, v12.8b, v28.8b
    smlal v26.4s, v0.4h, v11.4h
    ld1 {v22.8b}, [x18], x4
    ssubl v17.8h, v17.8b, v28.8b
    smlal2 v27.4s, v0.8h, v11.8h
    ld1 {v13.8b}, [x16], x4
    ssubl v22.8h, v22.8b, v28.8b
    smlal v24.4s, v1.4h, v10.4h
    ld1 {v18.8b}, [x17], x4
    ssubl v13.8h, v13.8b, v28.8b
    smlal2 v25.4s, v1.8h, v10.8h
    ld1 {v23.8b}, [x18], x4
    ssubl v18.8h, v18.8b, v28.8b
    smlal v26.4s, v1.4h, v12.4h
    ssubl v23.8h, v23.8b, v28.8b
    smlal2 v27.4s, v1.8h, v12.8h
    smlal v24.4s, v2.4h, v11.4h
    smlal2 v25.4s, v2.8h, v11.8h
    smlal v26.4s, v2.4h, v13.4h
    smlal2 v27.4s, v2.8h, v13.8h

    smlal v24.4s, v3.4h, v14.4h
    smlal2 v25.4s, v3.8h, v14.8h
    smlal v26.4s, v3.4h, v16.4h
    smlal2 v27.4s, v3.8h, v16.8h
    smlal v24.4s, v4.4h, v15.4h
    smlal2 v25.4s, v4.8h, v15.8h
    smlal v26.4s, v4.4h, v17.4h
    smlal2 v27.4s, v4.8h, v17.8h
    smlal v24.4s, v5.4h, v16.4h
    smlal2 v25.4s, v5.8h, v16.8h
    smlal v26.4s, v5.4h, v18.4h
    smlal2 v27.4s, v5.8h, v18.8h

    smlal v24.4s, v6.4h, v19.4h
    smlal2 v25.4s, v6.8h, v19.8h
    smlal v26.4s, v6.4h, v21.4h
    smlal2 v27.4s, v6.8h, v21.8h
    smlal v24.4s, v7.4h, v20.4h
    smlal2 v25.4s, v7.8h, v20.8h
    smlal v26.4s, v7.4h, v22.4h
    smlal2 v27.4s, v7.8h, v22.8h
    smlal v24.4s, v8.4h, v21.4h
    smlal2 v25.4s, v8.8h, v21.8h
    smlal v26.4s, v8.4h, v23.4h
    smlal2 v27.4s, v8.8h, v23.8h

    cbnz w23, PER_CHANNEL_POST2
    ld1r {v17.4s}, [x11]
    cbz w24, SKIP_LEFTSHIFT2
    ld1r {v12.4s}, [x12]
    sqshl v24.4s, v24.4s, v12.4s
    sqshl v25.4s, v25.4s, v12.4s
    sqshl v26.4s, v26.4s, v12.4s
    sqshl v27.4s, v27.4s, v12.4s
    sqrdmulh v24.4s, v24.4s, v17.4s
    sqrdmulh v25.4s, v25.4s, v17.4s
    sqrdmulh v26.4s, v26.4s, v17.4s
    sqrdmulh v27.4s, v27.4s, v17.4s
    b OUTZP2

SKIP_LEFTSHIFT2:
    ld1r {v22.4s}, [x13]
    sqrdmulh v24.4s, v24.4s, v17.4s
    sqrdmulh v25.4s, v25.4s, v17.4s
    sqrdmulh v26.4s, v26.4s, v17.4s
    sqrdmulh v27.4s, v27.4s, v17.4s
    sqrshl v24.4s, v24.4s, v22.4s
    sqrshl v25.4s, v25.4s, v22.4s
    sqrshl v26.4s, v26.4s, v22.4s
    sqrshl v27.4s, v27.4s, v22.4s
    b OUTZP2

PER_CHANNEL_POST2:
    ld1 {v12.4s}, [x12]
    sqshl v24.4s, v24.4s, v12.4s
    ldr q13, [x12, #16]
    sqshl v25.4s, v25.4s, v13.4s
    ld1 {v17.4s}, [x11]
    sqshl v26.4s, v26.4s, v12.4s
    sqshl v27.4s, v27.4s, v13.4s
    ldr q18, [x11, #16]
    sqrdmulh v24.4s, v24.4s, v17.4s
    sqrdmulh v25.4s, v25.4s, v18.4s
    ld1 {v22.4s}, [x13]
    sqrdmulh v26.4s, v26.4s, v17.4s
    sqrdmulh v27.4s, v27.4s, v18.4s
    ldr q23, [x13, #16]
    sqrshl v24.4s, v24.4s, v22.4s
    sqrshl v25.4s, v25.4s, v23.4s
    sqrshl v26.4s, v26.4s, v22.4s
    sqrshl v27.4s, v27.4s, v23.4s

OUTZP2:
    // Add output zero point
    sqadd v24.4s, v24.4s, v29.4s
    sqadd v25.4s, v25.4s, v29.4s
    sqadd v26.4s, v26.4s, v29.4s
    sqadd v27.4s, v27.4s, v29.4s

    // Apply min bound
    smax v24.4s, v24.4s, v30.4s
    smax v25.4s, v25.4s, v30.4s
    smax v26.4s, v26.4s, v30.4s
    smax v27.4s, v27.4s, v30.4s

    // Apply max bound
    smin v24.4s, v24.4s, v31.4s
    smin v25.4s, v25.4s, v31.4s
    smin v26.4s, v26.4s, v31.4s
    smin v27.4s, v27.4s, v31.4s

    sqxtn v24.4h, v24.4s
    sqxtn2 v24.8h, v25.4s
    sqxtn v26.4h, v26.4s
    sqxtn2 v26.8h, v27.4s
    sqxtn v24.8b, v24.8h
    sqxtn2 v24.16b, v26.8h
    st1 {v24.8b}, [x0], x6
    mov v26.d[0], v24.d[1]
    st1 {v26.8b}, [x0], x6
    b End

WIDTH1_LEFT:
    smlal v24.4s, v0.4h, v9.4h
    smlal2 v25.4s, v0.8h, v9.8h
    smlal v24.4s, v1.4h, v10.4h
    smlal2 v25.4s, v1.8h, v10.8h
    smlal v24.4s, v2.4h, v11.4h
    smlal2 v25.4s, v2.8h, v11.8h
    smlal v24.4s, v3.4h, v14.4h
    smlal2 v25.4s, v3.8h, v14.8h
    smlal v24.4s, v4.4h, v15.4h
    smlal2 v25.4s, v4.8h, v15.8h
    smlal v24.4s, v5.4h, v16.4h
    smlal2 v25.4s, v5.8h, v16.8h
    smlal v24.4s, v6.4h, v19.4h
    smlal2 v25.4s, v6.8h, v19.8h
    smlal v24.4s, v7.4h, v20.4h
    smlal2 v25.4s, v7.8h, v20.8h
    smlal v24.4s, v8.4h, v21.4h
    smlal2 v25.4s, v8.8h, v21.8h

    cbnz w23, PER_CHANNEL_POST3
    ld1r {v17.4s}, [x11]
    cbz w24, SKIP_LEFTSHIFT3
    ld1r {v12.4s}, [x12]
    sqshl v24.4s, v24.4s, v12.4s
    sqshl v25.4s, v25.4s, v12.4s
    sqrdmulh v24.4s, v24.4s, v17.4s
    sqrdmulh v25.4s, v25.4s, v17.4s
    b OUTZP3

SKIP_LEFTSHIFT3:
    ld1r {v22.4s}, [x13]
    sqrdmulh v24.4s, v24.4s, v17.4s
    sqrdmulh v25.4s, v25.4s, v17.4s
    sqrshl v24.4s, v24.4s, v22.4s
    sqrshl v25.4s, v25.4s, v22.4s
    b OUTZP3

PER_CHANNEL_POST3:
    ld1 {v12.4s}, [x12]
    sqshl v24.4s, v24.4s, v12.4s
    ldr q13, [x12, #16]
    sqshl v25.4s, v25.4s, v13.4s
    ld1 {v17.4s}, [x11]
    ldr q18, [x11, #16]
    sqrdmulh v24.4s, v24.4s, v17.4s
    sqrdmulh v25.4s, v25.4s, v18.4s
    ld1 {v22.4s}, [x13]
    ldr q23, [x13, #16]
    sqrshl v24.4s, v24.4s, v22.4s
    sqrshl v25.4s, v25.4s, v23.4s

OUTZP3:
    // Add output zero point
    sqadd v24.4s, v24.4s, v29.4s
    sqadd v25.4s, v25.4s, v29.4s

    // Apply min bound
    smax v24.4s, v24.4s, v30.4s
    smax v25.4s, v25.4s, v30.4s

    // Apply max bound
    smin v24.4s, v24.4s, v31.4s
    smin v25.4s, v25.4s, v31.4s

    sqxtn v24.4h, v24.4s
    sqxtn2 v24.8h, v25.4s
    sqxtn v24.8b, v24.8h
    st1 {v24.8b}, [x0], x6

End:
    sub sp, sp, #176
    ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
    ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
    ldp x19, x20, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x23, x24, [sp], #16
    ret
#endif
