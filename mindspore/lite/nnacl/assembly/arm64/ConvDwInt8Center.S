#ifdef __aarch64__
#include "nnacl/assembly_global.h"

.text
.align 5
.global ConvDwInt8Center
#ifndef __APPLE__
.type ConvDwInt8Center, %function
#endif

// void ConvDwInt8Center(int8_t *dst, const int8_t *src, const int16_t *weight, const int32_t *bias, size_t height,
//                       size_t width, size_t kernel_h, size_t kernel_w, size_t out_h_step, size_t block_channel,
//                       size_t in_sh_step, size_t in_sw_step, size_t in_kh_step, size_t in_kw_step, int8_t *in_zp,
//                       int32_t *out_zp, int32_t *out_multiplier, int32_t *left_shift, int32_t *right_shift,
//                       int32_t *acc_min, int32_t *acc_max)

// x0: dst, x1: src, x2: weight, x3: bias, x4: height, x5: weight, x6: kernel_h, x7: kernel_w, 
// x8: out_h_step, x9: block_channel, x10: in_sh_step, x11: in_sw_step, x12: in_kh_step, x13: in_kw_step
// x14: in_zp, #56: out_zp, #64: out_multiplier, #72:left_shift, #80: right_shift, #88: acc_min, #96: acc_max
asm_function ConvDwInt8Center
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

    ldr x14, [sp, #48] // input_zp
    ld1 {v19.8b}, [x14], #8

    ldr x15, [sp, #56] // output_zp
    ld1 {v20.4s}, [x15], #16
    ld1 {v21.4s}, [x15], #16

    ldr x16, [sp, #64] // out_multiplier
    ld1 {v22.4s}, [x16], #16
    ld1 {v23.4s}, [x16], #16

    ldr x17, [sp, #72] // left_shift
    ld1 {v24.4s}, [x17], #16
    ld1 {v25.4s}, [x17], #16

    ldr x18, [sp, #80] // right shift
    ld1 {v26.4s}, [x18], #16
    ld1 {v27.4s}, [x18], #16

    ldr x19, [sp, #88] // acc_min
    ld1 {v28.4s}, [x19], #16
    ld1 {v29.4s}, [x19], #16

    ldr x20, [sp, #96] // acc_max
    ld1 {v30.4s}, [x20], #16
    ld1 {v31.4s}, [x20], #16

    ld1 {v17.4s}, [x3], #16
    ld1 {v18.4s}, [x3], #16

    LoopH:
        mov x23, x1
        mov x24, x5
        mov x3, x0

        LoopW4:
            mov x19, #4
            mul x19, x19, x11
            mov x25, #4
            mul x25, x25, x9

            mov x16, x23
            mov x17, x2
            mov x20, x6

            mov v0.16b, v17.16b
            mov v1.16b, v18.16b
            mov v2.16b, v17.16b
            mov v3.16b, v18.16b
            mov v4.16b, v17.16b
            mov v5.16b, v18.16b
            mov v6.16b, v17.16b
            mov v7.16b, v18.16b
            LoopKh4:
                mov x18, x7
                mov x21, x16
                LoopKw4:
                    mov x22, x21
                    ld1 {v16.8h}, [x17], #16

                    ld1 {v15.8b}, [x22], x11
                    ssubl v14.8h, v15.8b, v19.8b
                    smlal v0.4s, v14.4h, v16.4h
                    smlal2 v1.4s, v14.8h, v16.8h

                    ld1 {v13.8b}, [x22], x11
                    ssubl v12.8h, v13.8b, v19.8b
                    smlal v2.4s, v12.4h, v16.4h
                    smlal2 v3.4s, v12.8h, v16.8h

                    ld1 {v11.8b}, [x22], x11
                    ssubl v10.8h, v11.8b, v19.8b
                    smlal v4.4s, v10.4h, v16.4h
                    smlal2 v5.4s, v10.8h, v16.8h

                    ld1 {v9.8b}, [x22], x11
                    ssubl v8.8h, v9.8b, v19.8b
                    smlal v6.4s, v8.4h, v16.4h
                    smlal2 v7.4s, v8.8h, v16.8h

                    subs x18, x18, #1
                    add x21, x21, x13
                    bne LoopKw4
                add x16, x16, x12
                subs x20, x20, #1
                bne LoopKh4

            sqshl v0.4s, v0.4s, v24.4s
            sqshl v1.4s, v1.4s, v25.4s
            sqshl v2.4s, v2.4s, v24.4s
            sqshl v3.4s, v3.4s, v25.4s
            sqshl v4.4s, v4.4s, v24.4s
            sqshl v5.4s, v5.4s, v25.4s
            sqshl v6.4s, v6.4s, v24.4s
            sqshl v7.4s, v7.4s, v25.4s

            sqrdmulh v0.4s, v0.4s, v22.4s
            sqrdmulh v1.4s, v1.4s, v23.4s
            sqrdmulh v2.4s, v2.4s, v22.4s
            sqrdmulh v3.4s, v3.4s, v23.4s
            sqrdmulh v4.4s, v4.4s, v22.4s
            sqrdmulh v5.4s, v5.4s, v23.4s
            sqrdmulh v6.4s, v6.4s, v22.4s
            sqrdmulh v7.4s, v7.4s, v23.4s

            sqrshl v0.4s, v0.4s, v26.4s
            sqrshl v1.4s, v1.4s, v27.4s
            sqrshl v2.4s, v2.4s, v26.4s
            sqrshl v3.4s, v3.4s, v27.4s
            sqrshl v4.4s, v4.4s, v26.4s
            sqrshl v5.4s, v5.4s, v27.4s
            sqrshl v6.4s, v6.4s, v26.4s
            sqrshl v7.4s, v7.4s, v27.4s

            add v0.4s, v0.4s, v20.4s
            add v1.4s, v1.4s, v21.4s
            add v2.4s, v2.4s, v20.4s
            add v3.4s, v3.4s, v21.4s
            add v4.4s, v4.4s, v20.4s
            add v5.4s, v5.4s, v21.4s
            add v6.4s, v6.4s, v20.4s
            add v7.4s, v7.4s, v21.4s
            smax v0.4s, v0.4s, v28.4s
            smax v1.4s, v1.4s, v29.4s
            smax v2.4s, v2.4s, v28.4s
            smax v3.4s, v3.4s, v29.4s
            smax v4.4s, v4.4s, v28.4s
            smax v5.4s, v5.4s, v29.4s
            smax v6.4s, v6.4s, v28.4s
            smax v7.4s, v7.4s, v29.4s
            smin v0.4s, v0.4s, v30.4s
            smin v1.4s, v1.4s, v31.4s
            smin v2.4s, v2.4s, v30.4s
            smin v3.4s, v3.4s, v31.4s
            smin v4.4s, v4.4s, v30.4s
            smin v5.4s, v5.4s, v31.4s
            smin v6.4s, v6.4s, v30.4s
            smin v7.4s, v7.4s, v31.4s

            sqxtn v0.4h, v0.4s
            sqxtn v1.4h, v1.4s
            sqxtn v2.4h, v2.4s
            sqxtn v3.4h, v3.4s
            sqxtn v4.4h, v4.4s
            sqxtn v5.4h, v5.4s
            sqxtn v6.4h, v6.4s
            sqxtn v7.4h, v7.4s
            sqxtn v0.8b, v0.8h
            sqxtn v1.8b, v1.8h
            sqxtn v2.8b, v2.8h
            sqxtn v3.8b, v3.8h
            sqxtn v4.8b, v4.8h
            sqxtn v5.8b, v5.8h
            sqxtn v6.8b, v6.8h
            sqxtn v7.8b, v7.8h

            mov x16, x3
            add x17, x16, x9
            add x18, x17, x9
            add x21, x18, x9

            st1 {v0.s}[0], [x16], #4
            st1 {v1.s}[0], [x16], #4
            st1 {v2.s}[0], [x17], #4
            st1 {v3.s}[0], [x17], #4
            st1 {v4.s}[0], [x18], #4
            st1 {v5.s}[0], [x18], #4
            st1 {v6.s}[0], [x21], #4
            st1 {v7.s}[0], [x21], #4

            add x3, x3, x25
            add x23, x23, x19
            sub x24, x24, #4
            cmp x24, #0
            ble LoopWEnd
            cmp x24, #4
            bge LoopW4

        LoopW:
            mov x16, x23
            mov x17, x2
            mov x20, x6
            mov v0.16b, v17.16b
            mov v1.16b, v18.16b
            LoopKh:
                mov x18, x7
                mov x22, x16
                LoopKw:
                    ld1 {v15.8b}, [x22], x13
                    ssubl v14.8h, v15.8b, v19.8b
                    ld1 {v16.8h}, [x17], #16
                    smlal v0.4s, v14.4h, v16.4h
                    smlal2 v1.4s, v14.8h, v16.8h
                    subs x18, x18, #1
                    bne LoopKw
                add x16, x16, x12
                subs x20, x20, #1
                bne LoopKh

            sqshl v0.4s, v0.4s, v24.4s
            sqrdmulh v0.4s, v0.4s, v22.4s
            sqshl v1.4s, v1.4s, v25.4s
            sqrdmulh v1.4s, v1.4s, v23.4s

            sqrshl v0.4s, v0.4s, v26.4s
            sqrshl v1.4s, v1.4s, v27.4s

            add v0.4s, v0.4s, v20.4s
            smax v0.4s, v0.4s, v28.4s
            smin v0.4s, v0.4s, v30.4s

            sqxtn v0.4h, v0.4s
            sqxtn v0.8b, v0.8h

            add v1.4s, v1.4s, v21.4s
            smax v1.4s, v1.4s, v29.4s
            smin v1.4s, v1.4s, v31.4s

            sqxtn v1.4h, v1.4s
            sqxtn v1.8b, v1.8h

            mov x17, x3
            st1 {v0.s}[0], [x17], #4
            st1 {v1.s}[0], [x17], #4
            add x3, x3, x9

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
