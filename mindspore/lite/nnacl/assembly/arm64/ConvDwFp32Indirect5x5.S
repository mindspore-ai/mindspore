#ifdef __aarch64__
#include "nnacl/assembly_global.h"

.text
.align 5
.global ConvDwFp32Indirect5x5
#ifndef __APPLE__
.type ConvDwFp32Indirect5x5, %function
#endif

// void ConvDwFp32Indirect5x5(float *output, float **input, const float *weights, const float *bias, int channels, int output_width,
//                            size_t input_stride, size_t relu, size_t relu6)
// x0: output, x1: input, x2: weights, x3: bias, x4: channels, x5: output_width, x6: input_stride, x7: relu, x8: relu6

asm_function ConvDwFp32Indirect5x5
    sub sp, sp, #160
    stp x19, x20, [sp, #64]
    stp x21, x22, [sp, #80]
    stp x23, x24, [sp, #96]
    stp x25, x26, [sp, #112]
    stp x27, x28, [sp, #128]
    stp x29, x30, [sp, #144]
    ldrb w8, [sp, #160]
    stp x2, x3, [sp]
    stp x4, x6, [sp, #16]
    stp x7, x8, [sp, #32]

    movi v31.4s, #6
    scvtf v31.4s, v31.4s
    dup v30.4s, wzr

    mov x3, x5
    cmp x3, #0
    beq End

    LoopPixel:
        ldp x5, x4, [sp]          // weight, bias
        ld1 {v29.4s}, [x4], #16
        ldr x2, [sp, #16]         // channel

        ldp x6, x7, [x1]
        ldp x8, x9, [x1, #16]
        ldp x10, x11, [x1, #32]
        ldp x12, x13, [x1, #48]
        ldp x14, x15, [x1, #64]
        ldp x16, x17, [x1, #80]
        ldp x18, x19, [x1, #96]
        ldp x20, x21, [x1, #112]
        ldp x22, x23, [x1, #128]
        ldp x24, x25, [x1, #144]
        ldp x26, x27, [x1, #160]
        ldp x28, x29, [x1, #176]
        ldr x30, [x1, #192]

        ld1 {v0.4s}, [x6], #16
        ld1 {v1.4s}, [x7], #16
        ld1 {v2.4s}, [x8], #16
        ld1 {v3.4s}, [x9], #16
        ld1 {v4.4s}, [x10], #16

        ld1 {v18.4s}, [x5], #16
        ld1 {v19.4s}, [x5], #16
        ld1 {v20.4s}, [x5], #16
        ld1 {v21.4s}, [x5], #16
        ld1 {v22.4s}, [x5], #16
        stp x5, x4, [sp, #48]

        cmp x2, #4
        ble LeftLoop
        LoopC4:
            ldr x5, [sp, #48]
            // column 0
            fmla v29.4s, v0.4s, v18.4s
            ld1 {v5.4s}, [x11], #16
            ld1 {v23.4s}, [x5], #16
            fmla v29.4s, v1.4s, v19.4s
            ld1 {v6.4s}, [x12], #16
            ld1 {v24.4s}, [x5], #16
            fmla v29.4s, v2.4s, v20.4s
            ld1 {v7.4s}, [x13], #16
            ld1 {v25.4s}, [x5], #16
            fmla v29.4s, v3.4s, v21.4s
            ld1 {v16.4s}, [x14], #16
            ld1 {v26.4s}, [x5], #16
            fmla v29.4s, v4.4s, v22.4s
            ld1 {v17.4s}, [x15], #16
            ld1 {v27.4s}, [x5], #16
            // column 1
            fmla v29.4s, v5.4s, v23.4s
            ld1 {v0.4s}, [x16], #16
            ld1 {v18.4s}, [x5], #16
            fmla v29.4s, v6.4s, v24.4s
            ld1 {v1.4s}, [x17], #16
            ld1 {v19.4s}, [x5], #16
            fmla v29.4s, v7.4s, v25.4s
            ld1 {v2.4s}, [x18], #16
            ld1 {v20.4s}, [x5], #16
            fmla v29.4s, v16.4s, v26.4s
            ld1 {v3.4s}, [x19], #16
            ld1 {v21.4s}, [x5], #16
            fmla v29.4s, v17.4s, v27.4s
            ld1 {v4.4s}, [x20], #16
            ld1 {v22.4s}, [x5], #16
            // column 2
            fmla v29.4s, v0.4s, v18.4s
            ld1 {v5.4s}, [x21], #16
            ld1 {v23.4s}, [x5], #16
            fmla v29.4s, v1.4s, v19.4s
            ld1 {v6.4s}, [x22], #16
            ld1 {v24.4s}, [x5], #16
            fmla v29.4s, v2.4s, v20.4s
            ld1 {v7.4s}, [x23], #16
            ld1 {v25.4s}, [x5], #16
            fmla v29.4s, v3.4s, v21.4s
            ld1 {v16.4s}, [x24], #16
            ld1 {v26.4s}, [x5], #16
            fmla v29.4s, v4.4s, v22.4s
            ld1 {v17.4s}, [x25], #16
            ld1 {v27.4s}, [x5], #16
            // column 3
            fmla v29.4s, v5.4s, v23.4s
            ld1 {v0.4s}, [x26], #16
            ld1 {v18.4s}, [x5], #16
            fmla v29.4s, v6.4s, v24.4s
            ld1 {v1.4s}, [x27], #16
            ld1 {v19.4s}, [x5], #16
            fmla v29.4s, v7.4s, v25.4s
            ld1 {v2.4s}, [x28], #16
            ld1 {v20.4s}, [x5], #16
            fmla v29.4s, v16.4s, v26.4s
            ld1 {v3.4s}, [x29], #16
            ld1 {v21.4s}, [x5], #16
            fmla v29.4s, v17.4s, v27.4s
            ld1 {v4.4s}, [x30], #16
            ld1 {v22.4s}, [x5], #16
            // column 4
            fmla v29.4s, v0.4s, v18.4s
            fmla v29.4s, v1.4s, v19.4s
            ld1 {v0.4s}, [x6], #16
            ld1 {v18.4s}, [x5], #16
            fmla v29.4s, v2.4s, v20.4s
            ld1 {v1.4s}, [x7], #16
            ld1 {v19.4s}, [x5], #16
            fmla v29.4s, v3.4s, v21.4s
            ld1 {v2.4s}, [x8], #16
            ld1 {v20.4s}, [x5], #16
            fmla v29.4s, v4.4s, v22.4s
            ld1 {v3.4s}, [x9], #16
            ld1 {v21.4s}, [x5], #16
            ld1 {v4.4s}, [x10], #16
            ld1 {v22.4s}, [x5], #16
            str x5, [sp, #48]

            ldp x4, x5, [sp, #32]
            cbnz x5, RELU6
            cbnz x4, RELU
            b WRITE
            RELU6:
                fmin v29.4s, v29.4s, v31.4s
            RELU:
                fmax v29.4s, v29.4s, v30.4s
            WRITE:
                st1 {v29.4s}, [x0], #16

            ldr x4, [sp, #56]
            ld1 {v29.4s}, [x4], #16
            str x4, [sp, #56]
            sub x2, x2, #4
            cmp x2, #4
            bgt LoopC4

        LeftLoop:
            // column 0
            ldr x5, [sp, #48]
            fmla v29.4s, v0.4s, v18.4s
            ld1 {v5.4s}, [x11], #16
            ld1 {v23.4s}, [x5], #16
            fmla v29.4s, v1.4s, v19.4s
            ld1 {v6.4s}, [x12], #16
            ld1 {v24.4s}, [x5], #16
            fmla v29.4s, v2.4s, v20.4s
            ld1 {v7.4s}, [x13], #16
            ld1 {v25.4s}, [x5], #16
            fmla v29.4s, v3.4s, v21.4s
            ld1 {v16.4s}, [x14], #16
            ld1 {v26.4s}, [x5], #16
            fmla v29.4s, v4.4s, v22.4s
            ld1 {v17.4s}, [x15], #16
            ld1 {v27.4s}, [x5], #16
            // column 1
            fmla v29.4s, v5.4s, v23.4s
            ld1 {v0.4s}, [x16], #16
            ld1 {v18.4s}, [x5], #16
            fmla v29.4s, v6.4s, v24.4s
            ld1 {v1.4s}, [x17], #16
            ld1 {v19.4s}, [x5], #16
            fmla v29.4s, v7.4s, v25.4s
            ld1 {v2.4s}, [x18], #16
            ld1 {v20.4s}, [x5], #16
            fmla v29.4s, v16.4s, v26.4s
            ld1 {v3.4s}, [x19], #16
            ld1 {v21.4s}, [x5], #16
            fmla v29.4s, v17.4s, v27.4s
            ld1 {v4.4s}, [x20], #16
            ld1 {v22.4s}, [x5], #16
            // column 2
            fmla v29.4s, v0.4s, v18.4s
            ld1 {v5.4s}, [x21], #16
            ld1 {v23.4s}, [x5], #16
            fmla v29.4s, v1.4s, v19.4s
            ld1 {v6.4s}, [x22], #16
            ld1 {v24.4s}, [x5], #16
            fmla v29.4s, v2.4s, v20.4s
            ld1 {v7.4s}, [x23], #16
            ld1 {v25.4s}, [x5], #16
            fmla v29.4s, v3.4s, v21.4s
            ld1 {v16.4s}, [x24], #16
            ld1 {v26.4s}, [x5], #16
            fmla v29.4s, v4.4s, v22.4s
            ld1 {v17.4s}, [x25], #16
            ld1 {v27.4s}, [x5], #16
            // column 3
            fmla v29.4s, v5.4s, v23.4s
            ld1 {v0.4s}, [x26], #16
            ld1 {v18.4s}, [x5], #16
            fmla v29.4s, v6.4s, v24.4s
            ld1 {v1.4s}, [x27], #16
            ld1 {v19.4s}, [x5], #16
            fmla v29.4s, v7.4s, v25.4s
            ld1 {v2.4s}, [x28], #16
            ld1 {v20.4s}, [x5], #16
            fmla v29.4s, v16.4s, v26.4s
            ld1 {v3.4s}, [x29], #16
            ld1 {v21.4s}, [x5], #16
            fmla v29.4s, v17.4s, v27.4s
            ld1 {v4.4s}, [x30], #16
            ld1 {v22.4s}, [x5], #16
            // column 4
            fmla v29.4s, v0.4s, v18.4s
            fmla v29.4s, v1.4s, v19.4s
            fmla v29.4s, v2.4s, v20.4s
            fmla v29.4s, v3.4s, v21.4s
            fmla v29.4s, v4.4s, v22.4s

            ldp x4, x5, [sp, #32]
            cbnz x5, LeftRelu6
            cbnz x4, LeftRelu
            b LeftWrite
            LeftRelu6:
                fmin v29.4s, v29.4s, v31.4s
            LeftRelu:
                fmax v29.4s, v29.4s, v30.4s
            LeftWrite:
                cmp x2, #4
                bne Write3
                st1 {v29.4s}, [x0], #16
                b NextPixel
            Write3:
                sxtw x2, w2
                tbnz w2, #1, Write2
                tbnz w2, #0, Write1
            Write2:
                st1 {v29.2s}, [x0], #8
                ext v29.16b, v29.16b, v29.16b, #8
                tbz w2, #0, NextPixel
            Write1:
                str s29, [x0], #4

    NextPixel:
        ldr x2, [sp, #24]
        add x1, x1, x2
        sub x3, x3, #1
        cmp x3, #0
        bgt LoopPixel
End:
    ldp x19, x20, [sp, #64]
    ldp x21, x22, [sp, #80]
    ldp x23, x24, [sp, #96]
    ldp x25, x26, [sp, #112]
    ldp x27, x28, [sp, #128]
    ldp x29, x30, [sp, #144]
    add sp, sp, #160
ret
#endif
