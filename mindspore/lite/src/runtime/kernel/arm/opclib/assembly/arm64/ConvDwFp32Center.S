#ifdef __aarch64__

.text
.align 5
.global ConvDwFp32Center
#ifndef __APPLE__
.type ConvDwFp32Center, %function
#endif

// void ConvDwFp32Center(float *dst, const float *src, const float *weight, const float *bias, size_t height, size_t width,
//                      size_t kernel_h, size_t kernel_w, size_t out_h_step, size_t block_channel, size_t in_sh_step, size_t in_sw_step,
//                      size_t in_kh_step, size_t in_kw_step, size_t relu, size_t relu6);
// x0: dst, x1: src, x2: weight, x3: bias, x4: height, x5: weight, x6: kernel_h, x7: kernel_w, 
// x8: out_h_step, x9: block_channel, x10: in_sh_step, x11: in_sw_step, x12: in_kh_step, x13: in_kw_step
// x14: relu, x15: relu6
ConvDwFp32Center:
    // registers v8 ~ v15 must be preserved by a callee across subroutine calls, according to
    // https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst#simd-and-floating-point-registers
    // x19 ~ x29 should be also preserved
    // whereas our coding style do not permit such amount of parameters
    sub sp, sp, #48
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

    mov x16, #4
    mul x8, x8, x16
    mul x9, x9, x16
    mul x10, x10, x16
    mul x11, x11, x16
    mul x12, x12, x16
    mul x13, x13, x16
    mov x16, #16
    mul x19, x7, x16

    ld1 {v5.4s}, [x3]

    LoopH:
        mov x23, x1
        mov x24, x5
        mov x3, x0
        LoopW:
            mov x16, x23
            mov x17, x2
            mov x20, x6
            ld1 {v0.4s}, [x3]
            fadd v0.4s, v0.4s, v5.4s
            LoopKh:
                mov x18, x7
                mov x21, x17
                mov x22, x16
                LoopKw:
                    ld1 {v1.4s}, [x22], x13
                    ld1 {v2.4s}, [x21], #16
                    fmla v0.4s, v1.4s, v2.4s
                    subs x18, x18, #1
                    bne LoopKw
                add x16, x16, x12
                add x17, x17, x19
                subs x20, x20, #1
                bne LoopKh
            cbnz x15, Relu6
            cbnz x14, Relu
            b Write
        Relu6:
            movi v4.4s, #6
            scvtf v4.4s, v4.4s
            fmin v0.4s, v0.4s, v4.4s
        Relu:
            dup v3.4s, wzr
            fmax v0.4s, v0.4s, v3.4s
        Write:
            st1 {v0.4s}, [x3], x9
            add x23, x23, x11
            subs x24, x24, #1
            bne LoopW
        add x0, x0, x8
        add x1, x1, x10
        subs x4, x4, #1
        bne LoopH

    sub sp, sp, #48
    ldp x19, x20, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x23, x24, [sp], #16
    ret
#endif
