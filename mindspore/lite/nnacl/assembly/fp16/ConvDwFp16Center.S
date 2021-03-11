#ifdef __aarch64__
#include "nnacl/assembly_global.h"

.text
.align 5
.global ConvDwFp16Center
#ifndef __APPLE__
.type ConvDwFp16Center, %function
#endif

// void ConvDwFp16Center(float16_t *dst, const float16_t *src, const float16_t *weight, const float16_t *bias, size_t height, size_t width,
//                      size_t kernel_h, size_t kernel_w, size_t out_h_step, size_t block_channel, size_t in_sh_step, size_t in_sw_step,
//                      size_t in_kh_step, size_t in_kw_step, size_t relu, size_t relu6);
// x0: dst, x1: src, x2: weight, x3: bias, x4: height, x5: weight, x6: kernel_h, x7: kernel_w, 
// x8: out_h_step, x9: block_channel, x10: in_sh_step, x11: in_sw_step, x12: in_kh_step, x13: in_kw_step
// x14: relu, x15: relu6
asm_function ConvDwFp16Center
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

    ld1 {v24.8h}, [x3]
    movi v26.8h, #0x46, lsl #8
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
                    ld1 {v25.8h}, [x17], #16
                    ld1 {v16.8h}, [x22], x11
                    ld1 {v17.8h}, [x22], x11
                    fmla v0.8h, v16.8h, v25.8h
                    fmla v1.8h, v17.8h, v25.8h
                    ld1 {v18.8h}, [x22], x11
                    ld1 {v19.8h}, [x22], x11
                    fmla v2.8h, v18.8h, v25.8h
                    fmla v3.8h, v19.8h, v25.8h
                    ld1 {v20.8h}, [x22], x11
                    ld1 {v21.8h}, [x22], x11
                    fmla v4.8h, v20.8h, v25.8h
                    fmla v5.8h, v21.8h, v25.8h
                    ld1 {v22.8h}, [x22], x11
                    ld1 {v23.8h}, [x22], x11
                    fmla v6.8h, v22.8h, v25.8h
                    fmla v7.8h, v23.8h, v25.8h
                    ld1 {v16.8h}, [x22], x11
                    ld1 {v17.8h}, [x22], x11
                    fmla v8.8h, v16.8h, v25.8h
                    fmla v9.8h, v17.8h, v25.8h
                    ld1 {v18.8h}, [x22], x11
                    ld1 {v19.8h}, [x22], x11
                    fmla v10.8h, v18.8h, v25.8h
                    fmla v11.8h, v19.8h, v25.8h
                    ld1 {v20.8h}, [x22], x11
                    ld1 {v21.8h}, [x22], x11
                    fmla v12.8h, v20.8h, v25.8h
                    fmla v13.8h, v21.8h, v25.8h
                    ld1 {v22.8h}, [x22], x11
                    ld1 {v23.8h}, [x22], x11
                    fmla v14.8h, v22.8h, v25.8h
                    fmla v15.8h, v23.8h, v25.8h
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
            fmin v0.8h, v0.8h, v26.8h
            fmin v1.8h, v1.8h, v26.8h
            fmin v2.8h, v2.8h, v26.8h
            fmin v3.8h, v3.8h, v26.8h
            fmin v4.8h, v4.8h, v26.8h
            fmin v5.8h, v5.8h, v26.8h
            fmin v6.8h, v6.8h, v26.8h
            fmin v7.8h, v7.8h, v26.8h
            fmin v8.8h, v8.8h, v26.8h
            fmin v9.8h, v9.8h, v26.8h
            fmin v10.8h, v10.8h, v26.8h
            fmin v11.8h, v11.8h, v26.8h
            fmin v12.8h, v12.8h, v26.8h
            fmin v13.8h, v13.8h, v26.8h
            fmin v14.8h, v14.8h, v26.8h
            fmin v15.8h, v15.8h, v26.8h
        Relu16:
            fmax v0.8h, v0.8h, v27.8h
            fmax v1.8h, v1.8h, v27.8h
            fmax v2.8h, v2.8h, v27.8h
            fmax v3.8h, v3.8h, v27.8h
            fmax v4.8h, v4.8h, v27.8h
            fmax v5.8h, v5.8h, v27.8h
            fmax v6.8h, v6.8h, v27.8h
            fmax v7.8h, v7.8h, v27.8h
            fmax v8.8h, v8.8h, v27.8h
            fmax v9.8h, v9.8h, v27.8h
            fmax v10.8h, v10.8h, v27.8h
            fmax v11.8h, v11.8h, v27.8h
            fmax v12.8h, v12.8h, v27.8h
            fmax v13.8h, v13.8h, v27.8h
            fmax v14.8h, v14.8h, v27.8h
            fmax v15.8h, v15.8h, v27.8h
        Write16:
            st1 {v0.8h}, [x3], x9
            st1 {v1.8h}, [x3], x9
            st1 {v2.8h}, [x3], x9
            st1 {v3.8h}, [x3], x9
            st1 {v4.8h}, [x3], x9
            st1 {v5.8h}, [x3], x9
            st1 {v6.8h}, [x3], x9
            st1 {v7.8h}, [x3], x9
            st1 {v8.8h}, [x3], x9
            st1 {v9.8h}, [x3], x9
            st1 {v10.8h}, [x3], x9
            st1 {v11.8h}, [x3], x9
            st1 {v12.8h}, [x3], x9
            st1 {v13.8h}, [x3], x9
            st1 {v14.8h}, [x3], x9
            st1 {v15.8h}, [x3], x9
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
                    ld1 {v25.8h}, [x17], #16
                    ld1 {v16.8h}, [x22], x11
                    ld1 {v17.8h}, [x22], x11
                    fmla v0.8h, v16.8h, v25.8h
                    fmla v1.8h, v17.8h, v25.8h
                    ld1 {v18.8h}, [x22], x11
                    ld1 {v19.8h}, [x22], x11
                    fmla v2.8h, v18.8h, v25.8h
                    fmla v3.8h, v19.8h, v25.8h
                    ld1 {v20.8h}, [x22], x11
                    ld1 {v21.8h}, [x22], x11
                    fmla v4.8h, v20.8h, v25.8h
                    fmla v5.8h, v21.8h, v25.8h
                    ld1 {v22.8h}, [x22], x11
                    ld1 {v23.8h}, [x22], x11
                    fmla v6.8h, v22.8h, v25.8h
                    fmla v7.8h, v23.8h, v25.8h
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
            fmin v0.8h, v0.8h, v26.8h
            fmin v1.8h, v1.8h, v26.8h
            fmin v2.8h, v2.8h, v26.8h
            fmin v3.8h, v3.8h, v26.8h
            fmin v4.8h, v4.8h, v26.8h
            fmin v5.8h, v5.8h, v26.8h
            fmin v6.8h, v6.8h, v26.8h
            fmin v7.8h, v7.8h, v26.8h
        Relu8:
            fmax v0.8h, v0.8h, v27.8h
            fmax v1.8h, v1.8h, v27.8h
            fmax v2.8h, v2.8h, v27.8h
            fmax v3.8h, v3.8h, v27.8h
            fmax v4.8h, v4.8h, v27.8h
            fmax v5.8h, v5.8h, v27.8h
            fmax v6.8h, v6.8h, v27.8h
            fmax v7.8h, v7.8h, v27.8h
        Write8:
            st1 {v0.8h}, [x3], x9
            st1 {v1.8h}, [x3], x9
            st1 {v2.8h}, [x3], x9
            st1 {v3.8h}, [x3], x9
            st1 {v4.8h}, [x3], x9
            st1 {v5.8h}, [x3], x9
            st1 {v6.8h}, [x3], x9
            st1 {v7.8h}, [x3], x9
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
                    ld1 {v16.8h}, [x22], x13
                    ld1 {v25.8h}, [x17], #16
                    fmla v0.8h, v16.8h, v25.8h
                    subs x18, x18, #1
                    bne LoopKw
                add x16, x16, x12
                subs x20, x20, #1
                bne LoopKh
            cbnz x15, Relu6
            cbnz x14, Relu
            b Write
        Relu6:
            fmin v0.8h, v0.8h, v26.8h
        Relu:
            fmax v0.8h, v0.8h, v27.8h
        Write:
            st1 {v0.8h}, [x3], x9
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
