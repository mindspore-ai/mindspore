#ifdef __aarch64__
#include "nnacl/assembly_global.h"

.text
.align 5
.global ConvDw3x3Vertical
#ifndef __APPLE__
.type ConvDw3x3Vertical, %function
#endif

// void ConvDw3x3Vertical(float *dst, const float *src, const float *weight, const float *bias, int in_kh_step,
//                        int in_kw_step, int channel, size_t relu, size_t relu6)
                     
// x0: dst, x1: src, x2: weight, x3: bias, x4: in_kh_step, x5: in_kw_step, x6: channel, x7: relu,  x8: relu6

asm_function ConvDw3x3Vertical
    // registers v8 ~ v15 must be preserved by a callee across subroutine calls, according to
    // https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst#simd-and-floating-point-registers
    // x19 ~ x29 should be also preserved
    // whereas our coding style do not permit such amount of parameters
    ldr x8, [sp]

    mov x9, #4
    mul x13, x6, x9        // x6 * 4
    mul x4, x4, x9
    mul x5, x5, x9
    mov x9, #3
    mul x14, x13, x9       // x6 * 3 * 4

    movi v26.4s, #6
    scvtf v26.4s, v26.4s
    dup v27.4s, wzr

    ld1 {v23.4s}, [x3], #16
    mov x9, x1
    mov x10, x2
    
    ld1 {v0.4s}, [x9], x5
    add x11, x1, x4
    ld1 {v4.4s}, [x10], x13
    ld1 {v1.4s}, [x9], x5
    add x12, x2, x14
    ld1 {v5.4s}, [x10], x13
    ld1 {v2.4s}, [x11], x5
    ld1 {v6.4s}, [x12], x13
    ld1 {v3.4s}, [x11], x5
    ld1 {v7.4s}, [x12], x13
    ld1 {v16.4s}, [x9], x5
    ld1 {v18.4s}, [x10], x13
    ld1 {v17.4s}, [x11], x5
    ld1 {v19.4s}, [x12], x13

    cmp x6, #4
    ble LoopC4Post

    LoopC4:
        add x1, x1, #16
        add x2, x2, #16
        fmla v23.4s, v0.4s, v4.4s
        mov x9, x1
        mov x10, x2
        ld1 {v0.4s}, [x9], x5
        ld1 {v4.4s}, [x10], x13
        add x11, x1, x4
        fmla v23.4s, v1.4s, v5.4s
        add x12, x2, x14
        ld1 {v1.4s}, [x9], x5
        fmla v23.4s, v2.4s, v6.4s
        ld1 {v5.4s}, [x10], x13
        ld1 {v2.4s}, [x11], x5
        fmla v23.4s, v3.4s, v7.4s
        ld1 {v6.4s}, [x12], x13
        ld1 {v3.4s}, [x11], x5
        fmla v23.4s, v16.4s, v18.4s
        ld1 {v7.4s}, [x12], x13
        ld1 {v16.4s}, [x9], x5
        fmla v23.4s, v17.4s, v19.4s
        ld1 {v18.4s}, [x10], x13
        ld1 {v17.4s}, [x11], x5
        ld1 {v19.4s}, [x12], x13

        cbnz x8, C4_RELU6
        cbnz x7, C4_RELU
        b C4_WRITE
        C4_RELU6:
            fmin v23.4s, v23.4s, v26.4s
        C4_RELU:
            fmax v23.4s, v23.4s, v27.4s
        C4_WRITE:
            st1 {v23.4s}, [x0], #16
            ld1 {v23.4s}, [x3], #16

        sub x6, x6, #4
        cmp x6, #4
        bgt LoopC4

    LoopC4Post:
        fmla v23.4s, v0.4s, v4.4s
        fmla v23.4s, v1.4s, v5.4s
        fmla v23.4s, v2.4s, v6.4s
        fmla v23.4s, v3.4s, v7.4s
        fmla v23.4s, v16.4s, v18.4s
        fmla v23.4s, v17.4s, v19.4s

        cbnz x8, RELU6
        cbnz x7, RELU
        b WRITE
        RELU6:
            fmin v23.4s, v23.4s, v26.4s
        RELU:
            fmax v23.4s, v23.4s, v27.4s
        WRITE:
            st1 {v23.4s}, [x0], #16
    ret
#endif
