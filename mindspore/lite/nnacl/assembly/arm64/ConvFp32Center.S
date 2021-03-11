#ifdef __aarch64__
#include "nnacl/assembly_global.h"

.text
.align 5
.global ConvSwFp32Center
#ifndef __APPLE__
.type ConvSwFp32Center, %function
#endif

// void ConvSwFp32Center(float *dst, const float *src, const float *weight, const float *bias, size_t height, size_t width,
//                      size_t kernel_h, size_t kernel_w, size_t out_h_step, size_t block_channel, size_t ic4, size_t in_sh_step,
//                      size_t in_sw_step, size_t in_kh_step, size_t in_kw_step, size_t relu, size_t relu6);
// x0: dst, x1: src, x2: weight, x3: bias, x4: height, x5: width, x6: kernel_h, x7: kernel_w,
// x8: out_h_step, x9: block_channel, x10: ic4, x11: in_sh_step, x12: in_sw_step, x13: in_kh_step, x14: in_kw_step
// x26: relu, x16: relu6
asm_function ConvSwFp32Center
    // registers v8 ~ v15 must be preserved by a callee across subroutine calls, according to
    // https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst#simd-and-floating-point-registers
    // x19 ~ x29 should be also preserved
    // whereas our coding style do not permit such amount of parameters
    sub sp, sp, #208
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
    stp x19, x20, [sp], #16
    stp x21, x22, [sp], #16
    stp x23, x24, [sp], #16
    stp x25, x26, [sp], #16
    stp x27, x28, [sp], #16

    ldr x8, [sp]
    ldr x9, [sp, #8]
    ldr x10, [sp, #16]
    ldr x11, [sp, #24]
    ldr x12, [sp, #32]
    ldr x13, [sp, #40]
    ldr x14, [sp, #48]
    mul x15, x6, x7
    mul x15, x10, x15
    mov x16, #16
    mul x15, x15, x16

    ld1 {v25.4s}, [x3]
    movi v26.4s, #6
    scvtf v26.4s, v26.4s
    dup v27.4s, wzr

    LoopH:
        mov x17, x1
        mov x18, x5
        mov x3, x0
        cmp x18, #8
        blt LoopW
        cmp x18, #16
        blt LoopW8

        LoopW16:
            mov x19, #16
            mul x19, x19, x12
            mov x20, x17
            mov x21, x2
            mov x22, x6
            mov v0.16b, v25.16b
            mov v1.16b, v25.16b
            mov v2.16b, v25.16b
            mov v3.16b, v25.16b
            mov v4.16b, v25.16b
            mov v5.16b, v25.16b
            mov v6.16b, v25.16b
            mov v7.16b, v25.16b
            mov v8.16b, v25.16b
            mov v9.16b, v25.16b
            mov v10.16b, v25.16b
            mov v11.16b, v25.16b
            mov v12.16b, v25.16b
            mov v13.16b, v25.16b
            mov v14.16b, v25.16b
            mov v15.16b, v25.16b
            LoopKh16:
                mov x23, x7
                mov x24, x20
                LoopKw16:
                    mov x25, x24
                    mov x27, x10
                    LoopIc16:
                        mov x26, x25
                        mov x16, x21
                        ld1 {v28.4s}, [x16], x15
                        ld1 {v29.4s}, [x16], x15
                        ld1 {v30.4s}, [x16], x15
                        ld1 {v31.4s}, [x16], x15
                        zip1 v20.4s, v28.4s, v29.4s
                        zip2 v21.4s, v28.4s, v29.4s
                        zip1 v22.4s, v30.4s, v31.4s
                        zip2 v23.4s, v30.4s, v31.4s
                        ld1 {v16.4s}, [x26], x12
                        ld1 {v17.4s}, [x26], x12
                        trn1 v28.2d, v20.2d, v22.2d
                        trn2 v29.2d, v20.2d, v22.2d
                        trn1 v30.2d, v21.2d, v23.2d
                        trn2 v31.2d, v21.2d, v23.2d
                        ld1 {v18.4s}, [x26], x12
                        ld1 {v19.4s}, [x26], x12
                        fmla v0.4s, v28.4s, v16.s[0]
                        fmla v1.4s, v28.4s, v17.s[0]
                        fmla v0.4s, v29.4s, v16.s[1]
                        fmla v1.4s, v29.4s, v17.s[1]
                        fmla v0.4s, v30.4s, v16.s[2]
                        fmla v1.4s, v30.4s, v17.s[2]
                        fmla v0.4s, v31.4s, v16.s[3]
                        fmla v1.4s, v31.4s, v17.s[3]
                        ld1 {v20.4s}, [x26], x12
                        ld1 {v21.4s}, [x26], x12
                        fmla v2.4s, v28.4s, v18.s[0]
                        fmla v3.4s, v28.4s, v19.s[0]
                        fmla v2.4s, v29.4s, v18.s[1]
                        fmla v3.4s, v29.4s, v19.s[1]
                        fmla v2.4s, v30.4s, v18.s[2]
                        fmla v3.4s, v30.4s, v19.s[2]
                        fmla v2.4s, v31.4s, v18.s[3]
                        fmla v3.4s, v31.4s, v19.s[3]
                        ld1 {v22.4s}, [x26], x12
                        ld1 {v23.4s}, [x26], x12
                        fmla v4.4s, v28.4s, v20.s[0]
                        fmla v5.4s, v28.4s, v21.s[0]
                        fmla v4.4s, v29.4s, v20.s[1]
                        fmla v5.4s, v29.4s, v21.s[1]
                        fmla v4.4s, v30.4s, v20.s[2]
                        fmla v5.4s, v30.4s, v21.s[2]
                        fmla v4.4s, v31.4s, v20.s[3]
                        fmla v5.4s, v31.4s, v21.s[3]
                        ld1 {v16.4s}, [x26], x12
                        ld1 {v17.4s}, [x26], x12
                        fmla v6.4s, v28.4s, v22.s[0]
                        fmla v7.4s, v28.4s, v23.s[0]
                        fmla v6.4s, v29.4s, v22.s[1]
                        fmla v7.4s, v29.4s, v23.s[1]
                        fmla v6.4s, v30.4s, v22.s[2]
                        fmla v7.4s, v30.4s, v23.s[2]
                        fmla v6.4s, v31.4s, v22.s[3]
                        fmla v7.4s, v31.4s, v23.s[3]
                        ld1 {v18.4s}, [x26], x12
                        ld1 {v19.4s}, [x26], x12
                        fmla v8.4s, v28.4s, v16.s[0]
                        fmla v9.4s, v28.4s, v17.s[0]
                        fmla v8.4s, v29.4s, v16.s[1]
                        fmla v9.4s, v29.4s, v17.s[1]
                        fmla v8.4s, v30.4s, v16.s[2]
                        fmla v9.4s, v30.4s, v17.s[2]
                        fmla v8.4s, v31.4s, v16.s[3]
                        fmla v9.4s, v31.4s, v17.s[3]
                        ld1 {v20.4s}, [x26], x12
                        ld1 {v21.4s}, [x26], x12
                        fmla v10.4s, v28.4s, v18.s[0]
                        fmla v11.4s, v28.4s, v19.s[0]
                        fmla v10.4s, v29.4s, v18.s[1]
                        fmla v11.4s, v29.4s, v19.s[1]
                        fmla v10.4s, v30.4s, v18.s[2]
                        fmla v11.4s, v30.4s, v19.s[2]
                        fmla v10.4s, v31.4s, v18.s[3]
                        fmla v11.4s, v31.4s, v19.s[3]
                        ld1 {v22.4s}, [x26], x12
                        ld1 {v23.4s}, [x26], x12
                        fmla v12.4s, v28.4s, v20.s[0]
                        fmla v13.4s, v28.4s, v21.s[0]
                        fmla v12.4s, v29.4s, v20.s[1]
                        fmla v13.4s, v29.4s, v21.s[1]
                        fmla v12.4s, v30.4s, v20.s[2]
                        fmla v13.4s, v30.4s, v21.s[2]
                        fmla v12.4s, v31.4s, v20.s[3]
                        fmla v13.4s, v31.4s, v21.s[3]
                        fmla v14.4s, v28.4s, v22.s[0]
                        fmla v15.4s, v28.4s, v23.s[0]
                        fmla v14.4s, v29.4s, v22.s[1]
                        fmla v15.4s, v29.4s, v23.s[1]
                        fmla v14.4s, v30.4s, v22.s[2]
                        fmla v15.4s, v30.4s, v23.s[2]
                        fmla v14.4s, v31.4s, v22.s[3]
                        fmla v15.4s, v31.4s, v23.s[3]
                        add x21, x21, #16
                        add x25, x25, #16
                        subs x27, x27, #1
                        bgt LoopIc16
                    subs x23, x23, #1
                    add x24, x24, x14
                    bne LoopKw16
                add x20, x20, x13
                subs x22, x22, #1
                bne LoopKh16
            ldr x16, [sp, #64]
            cbnz x16, Relu616
            ldr x26, [sp, #56]
            cbnz x26, Relu16
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
            add x17, x17, x19
            sub x18, x18, #16
            cmp x18, #0
            ble LoopWEnd
            cmp x18, #8
            blt LoopW
            cmp x18, #16
            bge LoopW16
        LoopW8:
            mov x19, #8
            mul x19, x19, x12
            mov x20, x17
            mov x21, x2
            mov x22, x6
            mov v0.16b, v25.16b
            mov v1.16b, v25.16b
            mov v2.16b, v25.16b
            mov v3.16b, v25.16b
            mov v4.16b, v25.16b
            mov v5.16b, v25.16b
            mov v6.16b, v25.16b
            mov v7.16b, v25.16b
            LoopKh8:
                mov x23, x7
                mov x24, x20
                LoopKw8:
                    mov x25, x24
                    mov x27, x10
                    LoopIc8:
                        mov x26, x25
                        mov x16, x21
                        ld1 {v28.4s}, [x16], x15
                        ld1 {v29.4s}, [x16], x15
                        ld1 {v30.4s}, [x16], x15
                        ld1 {v31.4s}, [x16], x15
                        zip1 v20.4s, v28.4s, v29.4s
                        zip2 v21.4s, v28.4s, v29.4s
                        zip1 v22.4s, v30.4s, v31.4s
                        zip2 v23.4s, v30.4s, v31.4s
                        ld1 {v16.4s}, [x26], x12
                        ld1 {v17.4s}, [x26], x12
                        trn1 v28.2d, v20.2d, v22.2d
                        trn2 v29.2d, v20.2d, v22.2d
                        trn1 v30.2d, v21.2d, v23.2d
                        trn2 v31.2d, v21.2d, v23.2d
                        ld1 {v18.4s}, [x26], x12
                        ld1 {v19.4s}, [x26], x12
                        fmla v0.4s, v28.4s, v16.s[0]
                        fmla v1.4s, v28.4s, v17.s[0]
                        fmla v0.4s, v29.4s, v16.s[1]
                        fmla v1.4s, v29.4s, v17.s[1]
                        fmla v0.4s, v30.4s, v16.s[2]
                        fmla v1.4s, v30.4s, v17.s[2]
                        fmla v0.4s, v31.4s, v16.s[3]
                        fmla v1.4s, v31.4s, v17.s[3]
                        ld1 {v20.4s}, [x26], x12
                        ld1 {v21.4s}, [x26], x12
                        fmla v2.4s, v28.4s, v18.s[0]
                        fmla v3.4s, v28.4s, v19.s[0]
                        fmla v2.4s, v29.4s, v18.s[1]
                        fmla v3.4s, v29.4s, v19.s[1]
                        fmla v2.4s, v30.4s, v18.s[2]
                        fmla v3.4s, v30.4s, v19.s[2]
                        fmla v2.4s, v31.4s, v18.s[3]
                        fmla v3.4s, v31.4s, v19.s[3]
                        ld1 {v22.4s}, [x26], x12
                        ld1 {v23.4s}, [x26], x12
                        fmla v4.4s, v28.4s, v20.s[0]
                        fmla v5.4s, v28.4s, v21.s[0]
                        fmla v4.4s, v29.4s, v20.s[1]
                        fmla v5.4s, v29.4s, v21.s[1]
                        fmla v4.4s, v30.4s, v20.s[2]
                        fmla v5.4s, v30.4s, v21.s[2]
                        fmla v4.4s, v31.4s, v20.s[3]
                        fmla v5.4s, v31.4s, v21.s[3]
                        fmla v6.4s, v28.4s, v22.s[0]
                        fmla v7.4s, v28.4s, v23.s[0]
                        fmla v6.4s, v29.4s, v22.s[1]
                        fmla v7.4s, v29.4s, v23.s[1]
                        fmla v6.4s, v30.4s, v22.s[2]
                        fmla v7.4s, v30.4s, v23.s[2]
                        fmla v6.4s, v31.4s, v22.s[3]
                        fmla v7.4s, v31.4s, v23.s[3]
                        add x21, x21, #16
                        add x25, x25, #16
                        subs x27, x27, #1
                        bgt LoopIc8
                    subs x23, x23, #1
                    add x24, x24, x14
                    bne LoopKw8
                add x20, x20, x13
                subs x22, x22, #1
                bne LoopKh8
            ldr x16, [sp, #64]
            cbnz x16, Relu68
            ldr x26, [sp, #56]
            cbnz x26, Relu8
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
            add x17, x17, x19
            sub x18, x18, #8
            cmp x18, #0
            ble LoopWEnd
            cmp x18, #8
            bge LoopW8
        LoopW:
            mov x20, x17
            mov x21, x2
            mov x22, x6
            mov v0.16b, v25.16b
            LoopKh:
                mov x23, x7
                mov x24, x20
                LoopKw:
                    mov x25, x24
                    mov x27, x10
                    LoopIc:
                        mov x26, x25
                        mov x16, x21
                        ld1 {v28.4s}, [x16], x15
                        ld1 {v29.4s}, [x16], x15
                        ld1 {v30.4s}, [x16], x15
                        ld1 {v31.4s}, [x16], x15
                        zip1 v20.4s, v28.4s, v29.4s
                        zip2 v21.4s, v28.4s, v29.4s
                        zip1 v22.4s, v30.4s, v31.4s
                        zip2 v23.4s, v30.4s, v31.4s
                        ld1 {v16.4s}, [x26], x12
                        trn1 v28.2d, v20.2d, v22.2d
                        trn2 v29.2d, v20.2d, v22.2d
                        trn1 v30.2d, v21.2d, v23.2d
                        trn2 v31.2d, v21.2d, v23.2d
                        fmla v0.4s, v28.4s, v16.s[0]
                        fmla v0.4s, v29.4s, v16.s[1]
                        fmla v0.4s, v30.4s, v16.s[2]
                        fmla v0.4s, v31.4s, v16.s[3]
                        add x21, x21, #16
                        add x25, x25, #16
                        subs x27, x27, #1
                        bgt LoopIc
                    subs x23, x23, #1
                    add x24, x24, x14
                    bne LoopKw
                add x20, x20, x13
                subs x22, x22, #1
                bne LoopKh
            ldr x16, [sp, #64]
            cbnz x16, Relu6
            ldr x26, [sp, #56]
            cbnz x26, Relu
            b Write
        Relu6:
            fmin v0.4s, v0.4s, v26.4s
        Relu:
            fmax v0.4s, v0.4s, v27.4s
        Write:
            st1 {v0.4s}, [x3], x9
            add x17, x17, x12
            subs x18, x18, #1
            bne LoopW
    LoopWEnd:
        add x0, x0, x8
        add x1, x1, x11
        subs x4, x4, #1
        bne LoopH

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
