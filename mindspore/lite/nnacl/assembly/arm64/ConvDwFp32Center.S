#ifdef __aarch64__
#include "nnacl/assembly_global.h"

.text
.align 5
.global ConvDwFp32Center
#ifndef __APPLE__
.type ConvDwFp32Center, %function
#endif

// void ConvDwFp32Center(float *dst, const float *src, const float *weight, const float *bias, size_t height, size_t width,
//                      size_t kernel_h, size_t kernel_w, size_t out_h_step, size_t block_channel, size_t in_sh_step, size_t in_sw_step,
//                      size_t in_kh_step, size_t in_kw_step, size_t relu, size_t relu6);
// x0: dst, x1: src, x2: weight, x3: bias, x4: height, x5: width, x6: kernel_h, x7: kernel_w,
// x8: out_h_step, x9: block_channel, x10: in_sh_step, x11: in_sw_step, x12: in_kh_step, x13: in_kw_step
// x14: relu, x15: relu6
asm_function ConvDwFp32Center
    // registers v8 ~ v15 must be preserved by a callee across subroutine calls, according to
    // https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst#simd-and-floating-point-registers
    // x19 ~ x29 should be also preserved
    // whereas our coding style do not permit such amount of parameters
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

    ld1 {v24.4s}, [x3]
    movi v26.4s, #6
    scvtf v26.4s, v26.4s
    dup v27.4s, wzr

    LoopH:
        mov x23, x1
        mov x24, x5
        mov x3, x0
        cmp x24, #8
        blt LoopW
        cmp x24, #16
        blt LoopW8

        LoopW16:
            mov x19, #16
            mul x19, x19, x11
            mov x16, x23
            mov x17, x2
            mov x20, x6
            mov v0.16b, v24.16b
            mov v1.16b, v24.16b
            mov v2.16b, v24.16b
            mov v3.16b, v24.16b
            mov v4.16b, v24.16b
            mov v5.16b, v24.16b
            mov v6.16b, v24.16b
            mov v7.16b, v24.16b
            mov v8.16b, v24.16b
            mov v9.16b, v24.16b
            mov v10.16b, v24.16b
            mov v11.16b, v24.16b
            mov v12.16b, v24.16b
            mov v13.16b, v24.16b
            mov v14.16b, v24.16b
            mov v15.16b, v24.16b
            LoopKh16:
                mov x18, x7
                mov x21, x16
                LoopKw16:
                    mov x22, x21
                    ld1 {v25.4s}, [x17], #16
                    ld1 {v16.4s}, [x22], x11
                    ld1 {v17.4s}, [x22], x11
                    fmla v0.4s, v16.4s, v25.4s
                    fmla v1.4s, v17.4s, v25.4s
                    ld1 {v18.4s}, [x22], x11
                    ld1 {v19.4s}, [x22], x11
                    fmla v2.4s, v18.4s, v25.4s
                    fmla v3.4s, v19.4s, v25.4s
                    ld1 {v20.4s}, [x22], x11
                    ld1 {v21.4s}, [x22], x11
                    fmla v4.4s, v20.4s, v25.4s
                    fmla v5.4s, v21.4s, v25.4s
                    ld1 {v22.4s}, [x22], x11
                    ld1 {v23.4s}, [x22], x11
                    fmla v6.4s, v22.4s, v25.4s
                    fmla v7.4s, v23.4s, v25.4s
                    ld1 {v16.4s}, [x22], x11
                    ld1 {v17.4s}, [x22], x11
                    fmla v8.4s, v16.4s, v25.4s
                    fmla v9.4s, v17.4s, v25.4s
                    ld1 {v18.4s}, [x22], x11
                    ld1 {v19.4s}, [x22], x11
                    fmla v10.4s, v18.4s, v25.4s
                    fmla v11.4s, v19.4s, v25.4s
                    ld1 {v20.4s}, [x22], x11
                    ld1 {v21.4s}, [x22], x11
                    fmla v12.4s, v20.4s, v25.4s
                    fmla v13.4s, v21.4s, v25.4s
                    ld1 {v22.4s}, [x22], x11
                    ld1 {v23.4s}, [x22], x11
                    fmla v14.4s, v22.4s, v25.4s
                    fmla v15.4s, v23.4s, v25.4s
                    subs x18, x18, #1
                    add x21, x21, x13
                    bne LoopKw16
                add x16, x16, x12
                subs x20, x20, #1
                bne LoopKh16
            cbnz x15, Relu616
            cbnz x14, Relu16
            b Write16
        Relu616:
            fmin v0.4s, v0.4s, v26.4s
            fmin v1.4s, v1.4s, v26.4s
            fmin v2.4s, v2.4s, v26.4s
            fmin v3.4s, v3.4s, v26.4s
            fmin v4.4s, v4.4s, v26.4s
            fmin v5.4s, v5.4s, v26.4s
            fmin v6.4s, v6.4s, v26.4s
            fmin v7.4s, v7.4s, v26.4s
            fmin v8.4s, v8.4s, v26.4s
            fmin v9.4s, v9.4s, v26.4s
            fmin v10.4s, v10.4s, v26.4s
            fmin v11.4s, v11.4s, v26.4s
            fmin v12.4s, v12.4s, v26.4s
            fmin v13.4s, v13.4s, v26.4s
            fmin v14.4s, v14.4s, v26.4s
            fmin v15.4s, v15.4s, v26.4s
        Relu16:
            fmax v0.4s, v0.4s, v27.4s
            fmax v1.4s, v1.4s, v27.4s
            fmax v2.4s, v2.4s, v27.4s
            fmax v3.4s, v3.4s, v27.4s
            fmax v4.4s, v4.4s, v27.4s
            fmax v5.4s, v5.4s, v27.4s
            fmax v6.4s, v6.4s, v27.4s
            fmax v7.4s, v7.4s, v27.4s
            fmax v8.4s, v8.4s, v27.4s
            fmax v9.4s, v9.4s, v27.4s
            fmax v10.4s, v10.4s, v27.4s
            fmax v11.4s, v11.4s, v27.4s
            fmax v12.4s, v12.4s, v27.4s
            fmax v13.4s, v13.4s, v27.4s
            fmax v14.4s, v14.4s, v27.4s
            fmax v15.4s, v15.4s, v27.4s
        Write16:
            st1 {v0.4s}, [x3], x9
            st1 {v1.4s}, [x3], x9
            st1 {v2.4s}, [x3], x9
            st1 {v3.4s}, [x3], x9
            st1 {v4.4s}, [x3], x9
            st1 {v5.4s}, [x3], x9
            st1 {v6.4s}, [x3], x9
            st1 {v7.4s}, [x3], x9
            st1 {v8.4s}, [x3], x9
            st1 {v9.4s}, [x3], x9
            st1 {v10.4s}, [x3], x9
            st1 {v11.4s}, [x3], x9
            st1 {v12.4s}, [x3], x9
            st1 {v13.4s}, [x3], x9
            st1 {v14.4s}, [x3], x9
            st1 {v15.4s}, [x3], x9
            add x23, x23, x19
            sub x24, x24, #16
            cmp x24, #0
            ble LoopWEnd
            cmp x24, #8
            blt LoopW
            cmp x24, #16
            bge LoopW16
        LoopW8:
            mov x19, #8
            mul x19, x19, x11
            mov x16, x23
            mov x17, x2
            mov x20, x6
            mov v0.16b, v24.16b
            mov v1.16b, v24.16b
            mov v2.16b, v24.16b
            mov v3.16b, v24.16b
            mov v4.16b, v24.16b
            mov v5.16b, v24.16b
            mov v6.16b, v24.16b
            mov v7.16b, v24.16b
            LoopKh8:
                mov x18, x7
                mov x21, x16
                LoopKw8:
                    mov x22, x21
                    ld1 {v25.4s}, [x17], #16
                    ld1 {v16.4s}, [x22], x11
                    ld1 {v17.4s}, [x22], x11
                    fmla v0.4s, v16.4s, v25.4s
                    fmla v1.4s, v17.4s, v25.4s
                    ld1 {v18.4s}, [x22], x11
                    ld1 {v19.4s}, [x22], x11
                    fmla v2.4s, v18.4s, v25.4s
                    fmla v3.4s, v19.4s, v25.4s
                    ld1 {v20.4s}, [x22], x11
                    ld1 {v21.4s}, [x22], x11
                    fmla v4.4s, v20.4s, v25.4s
                    fmla v5.4s, v21.4s, v25.4s
                    ld1 {v22.4s}, [x22], x11
                    ld1 {v23.4s}, [x22], x11
                    fmla v6.4s, v22.4s, v25.4s
                    fmla v7.4s, v23.4s, v25.4s
                    subs x18, x18, #1
                    add x21, x21, x13
                    bne LoopKw8
                add x16, x16, x12
                subs x20, x20, #1
                bne LoopKh8
            cbnz x15, Relu68
            cbnz x14, Relu8
            b Write8
        Relu68:
            fmin v0.4s, v0.4s, v26.4s
            fmin v1.4s, v1.4s, v26.4s
            fmin v2.4s, v2.4s, v26.4s
            fmin v3.4s, v3.4s, v26.4s
            fmin v4.4s, v4.4s, v26.4s
            fmin v5.4s, v5.4s, v26.4s
            fmin v6.4s, v6.4s, v26.4s
            fmin v7.4s, v7.4s, v26.4s
        Relu8:
            fmax v0.4s, v0.4s, v27.4s
            fmax v1.4s, v1.4s, v27.4s
            fmax v2.4s, v2.4s, v27.4s
            fmax v3.4s, v3.4s, v27.4s
            fmax v4.4s, v4.4s, v27.4s
            fmax v5.4s, v5.4s, v27.4s
            fmax v6.4s, v6.4s, v27.4s
            fmax v7.4s, v7.4s, v27.4s
        Write8:
            st1 {v0.4s}, [x3], x9
            st1 {v1.4s}, [x3], x9
            st1 {v2.4s}, [x3], x9
            st1 {v3.4s}, [x3], x9
            st1 {v4.4s}, [x3], x9
            st1 {v5.4s}, [x3], x9
            st1 {v6.4s}, [x3], x9
            st1 {v7.4s}, [x3], x9
            add x23, x23, x19
            sub x24, x24, #8
            cmp x24, #0
            ble LoopWEnd
            cmp x24, #8
            bge LoopW8
        LoopW:
            mov x16, x23
            mov x17, x2
            mov x20, x6
            mov v0.16b, v24.16b
            LoopKh:
                mov x18, x7
                mov x22, x16
                LoopKw:
                    ld1 {v16.4s}, [x22], x13
                    ld1 {v25.4s}, [x17], #16
                    fmla v0.4s, v16.4s, v25.4s
                    subs x18, x18, #1
                    bne LoopKw
                add x16, x16, x12
                subs x20, x20, #1
                bne LoopKh
            cbnz x15, Relu6
            cbnz x14, Relu
            b Write
        Relu6:
            fmin v0.4s, v0.4s, v26.4s
        Relu:
            fmax v0.4s, v0.4s, v27.4s
        Write:
            st1 {v0.4s}, [x3], x9
            add x23, x23, x11
            subs x24, x24, #1
            bne LoopW
    LoopWEnd:
        add x0, x0, x8
        add x1, x1, x10
        subs x4, x4, #1
        bne LoopH

    sub sp, sp, #176
    ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
    ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
    ldp x19, x20, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x23, x24, [sp], #16
    ret
#endif
