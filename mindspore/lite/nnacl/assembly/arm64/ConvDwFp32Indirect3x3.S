#ifdef __aarch64__
#include "nnacl/assembly_global.h"

.text
.align 5
.global ConvDwFp32Indirect3x3
#ifndef __APPLE__
.type ConvDwFp32Indirect3x3, %function
#endif

// void ConvDwFp32Indirect3x3(float *output, float **input, const float *weights, const float *bias, int channels, int output_width,
//                            size_t input_stride, size_t relu, size_t relu6)
// x0: output, x1: input, x2: weights, x3: bias, x4: channels, x5: output_width, x6: input_stride, x7: relu, x8: relu6

asm_function ConvDwFp32Indirect3x3
    sub sp, sp, #16
    stp x19, x20, [sp], #16

    movi v31.4s, #6
    scvtf v31.4s, v31.4s
    dup v30.4s, wzr

    ldr x8, [sp]
    cmp x5, #0
    beq End

    LoopPixel:
        ldp x12, x13, [x1]
        ldp x14, x15, [x1, #16]
        ldp x16, x17, [x1, #32]
        ldp x18, x19, [x1, #48]
        ldr x20, [x1, #64]
        mov x9, x2
        mov x10, x3
        mov x11, x4

        ld1 {v0.4s}, [x12], #16
        ld1 {v1.4s}, [x13], #16
        ld1 {v2.4s}, [x14], #16

        ld1 {v17.4s}, [x9], #16
        ld1 {v18.4s}, [x9], #16
        ld1 {v19.4s}, [x9], #16

        ld1 {v29.4s}, [x10], #16
        cmp x11, #4
        ble LeftLoop
        LoopC4:
            fmla v29.4s, v0.4s, v17.4s
            ld1 {v3.4s}, [x15], #16
            ld1 {v20.4s}, [x9], #16
            fmla v29.4s, v1.4s, v18.4s
            ld1 {v4.4s}, [x16], #16
            ld1 {v21.4s}, [x9], #16
            fmla v29.4s, v2.4s, v19.4s
            ld1 {v5.4s}, [x17], #16
            ld1 {v22.4s}, [x9], #16
            fmla v29.4s, v3.4s, v20.4s
            ld1 {v6.4s}, [x18], #16
            ld1 {v23.4s}, [x9], #16
            fmla v29.4s, v4.4s, v21.4s
            ld1 {v7.4s}, [x19], #16
            ld1 {v24.4s}, [x9], #16
            fmla v29.4s, v5.4s, v22.4s
            ld1 {v16.4s}, [x20], #16
            ld1 {v25.4s}, [x9], #16
            fmla v29.4s, v6.4s, v23.4s
            ld1 {v0.4s}, [x12], #16
            ld1 {v17.4s}, [x9], #16
            fmla v29.4s, v7.4s, v24.4s
            ld1 {v1.4s}, [x13], #16
            ld1 {v18.4s}, [x9], #16
            fmla v29.4s, v16.4s, v25.4s
            ld1 {v2.4s}, [x14], #16
            ld1 {v19.4s}, [x9], #16

            cbnz x8, Relu6
            cbnz x7, Relu
            b Write
            Relu6:
                fmin v29.4s, v29.4s, v31.4s
            Relu:
                fmax v29.4s, v29.4s, v30.4s
            Write:
                st1 {v29.4s}, [x0], #16

            ld1 {v29.4s}, [x10], #16
            sub x11, x11, #4
            cmp x11, #4
            bgt LoopC4

        LeftLoop:
            fmla v29.4s, v0.4s, v17.4s
            ld1 {v3.4s}, [x15], #16
            ld1 {v20.4s}, [x9], #16
            fmla v29.4s, v1.4s, v18.4s
            ld1 {v4.4s}, [x16], #16
            ld1 {v21.4s}, [x9], #16
            fmla v29.4s, v2.4s, v19.4s
            ld1 {v5.4s}, [x17], #16
            ld1 {v22.4s}, [x9], #16
            fmla v29.4s, v3.4s, v20.4s
            ld1 {v6.4s}, [x18], #16
            ld1 {v23.4s}, [x9], #16
            fmla v29.4s, v4.4s, v21.4s
            ld1 {v7.4s}, [x19], #16
            ld1 {v24.4s}, [x9], #16
            fmla v29.4s, v5.4s, v22.4s
            ld1 {v16.4s}, [x20], #16
            ld1 {v25.4s}, [x9], #16
            fmla v29.4s, v6.4s, v23.4s
            fmla v29.4s, v7.4s, v24.4s
            fmla v29.4s, v16.4s, v25.4s

            cbnz x8, LeftRelu6
            cbnz x7, LeftRelu
            b LeftWrite
            LeftRelu6:
                fmin v29.4s, v29.4s, v31.4s
            LeftRelu:
                fmax v29.4s, v29.4s, v30.4s
            LeftWrite:
                cmp x11, #4
                bne Write3
                st1 {v29.4s}, [x0], #16
                b NextPixel
            Write3:
                sxtw x11, w11
                tbnz w11, #1, Write2
                tbnz w11, #0, Write1
            Write2:
                st1 {v29.2s}, [x0], #8
                ext v29.16b, v29.16b, v29.16b, #8
                tbz w11, #0, NextPixel
            Write1:
                str s29, [x0], #4

    NextPixel:
        add x1, x1, x6
        sub x5, x5, #1
        cmp x5, #0
        bgt LoopPixel
End:
    sub sp, sp, #16
    ldp x19, x20, [sp], #16
ret
#endif
