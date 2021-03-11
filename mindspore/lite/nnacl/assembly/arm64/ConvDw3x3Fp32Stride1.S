#ifdef __aarch64__
#include "nnacl/assembly_global.h"

.text
.align 5
.global ConvDw3x3Stride1
#ifndef __APPLE__
.type ConvDw3x3Stride1, %function
#endif


// void ConvDw3x3Stride1(float *output, const float *buffer, const float *weight, const float *bias, int col_size,
//                       int row_size, int channel, int output_h, int output_w, size_t relu, size_t relu6)
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
// w9: relu
// w10: relu6

asm_function ConvDw3x3Stride1
    sub sp, sp, #128
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64

    ldr w8, [sp]
    ldr w9, [sp, #8]
    ldr w10, [sp, #16]

    mov w11, #4
    mul w15, w4, w11   // col_size * 4
    mul w16, w6, w11   // channel * 4
    mul w17, w5, w11   // row_size * 4
    mov w11, #2
    mul w14, w11, w15  // col_size * 2 * 4

    movi v23.4s, #6
    scvtf v23.4s, v23.4s
    dup v24.4s, wzr

    // Load weights
    ld1 {v0.4s}, [x2], x16
    ld1 {v1.4s}, [x2], x16
    ld1 {v2.4s}, [x2], x16
    ld1 {v3.4s}, [x2], x16
    ld1 {v4.4s}, [x2], x16
    ld1 {v5.4s}, [x2], x16
    ld1 {v6.4s}, [x2], x16
    ld1 {v7.4s}, [x2], x16
    ld1 {v8.4s}, [x2], x16

    mov x11, x1
    add x12, x11, x17
    add x13, x12, x17
    ld1 {v9.4s}, [x11], x15
    ld1 {v10.4s}, [x11], x15
    ld1 {v11.4s}, [x11], x15
    ld1 {v13.4s}, [x12], x15
    ld1 {v14.4s}, [x12], x15
    ld1 {v15.4s}, [x12], x15
    ld1 {v17.4s}, [x13], x15
    ld1 {v18.4s}, [x13], x15
    ld1 {v19.4s}, [x13], x15

    ld1 {v21.4s}, [x3]
    ld1 {v22.4s}, [x3]

    cmp w8, #2
    beq WIDTH2_LEFT
    cmp w8, #1
    beq WIDTH1_LEFT

WIDTH2_LOOP:
    fmla v21.4s, v0.4s, v9.4s
    ld1 {v12.4s}, [x11]
    ld1 {v16.4s}, [x12]
    fmla v22.4s, v0.4s, v10.4s
    ld1 {v20.4s}, [x13]
    add x1, x1, x14
    fmla v21.4s, v1.4s, v10.4s
    mov x11, x1
    add x12, x11, x17
    add x13, x12, x17
    ld1 {v9.4s}, [x11], x15
    fmla v22.4s, v1.4s, v11.4s
    ld1 {v10.4s}, [x11], x15
    fmla v21.4s, v2.4s, v11.4s
    fmla v22.4s, v2.4s, v12.4s
    fmla v21.4s, v3.4s, v13.4s
    ld1 {v11.4s}, [x11], x15
    fmla v22.4s, v3.4s, v14.4s
    fmla v21.4s, v4.4s, v14.4s
    ld1 {v13.4s}, [x12], x15
    fmla v22.4s, v4.4s, v15.4s
    fmla v21.4s, v5.4s, v15.4s
    ld1 {v14.4s}, [x12], x15
    fmla v22.4s, v5.4s, v16.4s
    fmla v21.4s, v6.4s, v17.4s
    ld1 {v15.4s}, [x12], x15
    fmla v22.4s, v6.4s, v18.4s
    fmla v21.4s, v7.4s, v18.4s
    ld1 {v17.4s}, [x13], x15
    fmla v22.4s, v7.4s, v19.4s
    fmla v21.4s, v8.4s, v19.4s
    ld1 {v18.4s}, [x13], x15
    fmla v22.4s, v8.4s, v20.4s
    ld1 {v19.4s}, [x13], x15

    cbnz x10, WIDTH2_RELU6
    cbnz x9, WIDTH2_RELU
    b WIDTH2_WRITE
    WIDTH2_RELU6:
        fmin v21.4s, v21.4s, v23.4s
        fmin v22.4s, v22.4s, v23.4s
    WIDTH2_RELU:
        fmax v21.4s, v21.4s, v24.4s
        fmax v22.4s, v22.4s, v24.4s
    WIDTH2_WRITE:
        st1 {v21.4s}, [x0], x16
        ld1 {v21.4s}, [x3]
        st1 {v22.4s}, [x0], x16
        ld1 {v22.4s}, [x3]

        sub w8, w8, #2
        cmp w8, #2
        bgt WIDTH2_LOOP

        cmp w8, #2
        blt WIDTH1_LEFT

WIDTH2_LEFT:
    fmla v21.4s, v0.4s, v9.4s
    ld1 {v12.4s}, [x11]
    fmla v22.4s, v0.4s, v10.4s
    fmla v21.4s, v1.4s, v10.4s
    ld1 {v16.4s}, [x12]
    fmla v22.4s, v1.4s, v11.4s
    fmla v21.4s, v2.4s, v11.4s
    ld1 {v20.4s}, [x13]
    fmla v22.4s, v2.4s, v12.4s
    fmla v21.4s, v3.4s, v13.4s
    fmla v22.4s, v3.4s, v14.4s
    fmla v21.4s, v4.4s, v14.4s
    fmla v22.4s, v4.4s, v15.4s
    fmla v21.4s, v5.4s, v15.4s
    fmla v22.4s, v5.4s, v16.4s
    fmla v21.4s, v6.4s, v17.4s
    fmla v22.4s, v6.4s, v18.4s
    fmla v21.4s, v7.4s, v18.4s
    fmla v22.4s, v7.4s, v19.4s
    fmla v21.4s, v8.4s, v19.4s
    fmla v22.4s, v8.4s, v20.4s

    cbnz x10, WIDTH2_LEFT_RELU6
    cbnz x9, WIDTH2_LEFT_RELU
    b WIDTH2_LEFT_WRITE
    WIDTH2_LEFT_RELU6:
        fmin v21.4s, v21.4s, v23.4s
        fmin v22.4s, v22.4s, v23.4s
    WIDTH2_LEFT_RELU:
        fmax v21.4s, v21.4s, v24.4s
        fmax v22.4s, v22.4s, v24.4s
    WIDTH2_LEFT_WRITE:
        st1 {v21.4s}, [x0], x16
        st1 {v22.4s}, [x0], x16
        b End

WIDTH1_LEFT:
    fmla v21.4s, v0.4s, v9.4s
    fmla v21.4s, v1.4s, v10.4s
    fmla v21.4s, v2.4s, v11.4s
    fmla v21.4s, v3.4s, v13.4s
    fmla v21.4s, v4.4s, v14.4s
    fmla v21.4s, v5.4s, v15.4s
    fmla v21.4s, v6.4s, v17.4s
    fmla v21.4s, v7.4s, v18.4s
    fmla v21.4s, v8.4s, v19.4s

    cbnz x10, WIDTH1_RELU6
    cbnz x9, WIDTH1_RELU
    b WIDTH1_WRITE
    WIDTH1_RELU6:
        fmin v21.4s, v21.4s, v23.4s
    WIDTH1_RELU:
        fmax v21.4s, v21.4s, v24.4s
    WIDTH1_WRITE:
        st1 {v21.4s}, [x0]

End:
    sub sp, sp, #128
    ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
    ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
    ret
#endif
