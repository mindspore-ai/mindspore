#ifdef __aarch64__

.text
.align 5
.global ConvDw3x3BorderPixelInt8
#ifndef __APPLE__
.type ConvDw3x3BorderPixelInt8, %function
#endif

// void ConvDw3x3BorderPixelInt8(int8_t *dst, const int8_t *src, const int16_t *weight, const int32_t *bias, size_t height,
//                               size_t width, size_t in_kh_step, size_t in_kw_step, size_t channel, size_t in_zp, size_t out_zp,
//                               size_t out_multiplier, size_t left_shift, size_t right_shift, size_t acc_min, size_t acc_max) {

// x0: dst, x1: src, x2: weight, x3: bias, x4: height, x5: width, x6: in_kh_step, x7: in_kw_step,
// x8: channel, x9: in_zp,  x10: out_zp, x11: out_multiplier, x12: left_shift, x13: right_shift
// x14: acc_min, x15: acc_max
ConvDw3x3BorderPixelInt8:
    // registers v8 ~ v15 must be preserved by a callee across subroutine calls, according to
    // https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst#simd-and-floating-point-registers
    // x19 ~ x29 should be also preserved
    // whereas our coding style do not permit such amount of parameters
    ldr x8, [sp]
    ldrb w9, [sp, #8]
    dup v25.8b, w9         // in_zp
    ldr x9, [sp, #16]
    dup v26.4s, w9         // out_zp
    ldr x9, [sp, #24]
    dup v27.4s, w9         // out_multiplier
    ldr x9, [sp, #32]
    dup v28.4s, w9         // left_shift
    ldr x9, [sp, #40]
    dup v29.4s, w9         // right_shift
    ldr x9, [sp, #48]
    dup v30.4s, w9         // acc_min
    ldr x9, [sp, #56]
    dup v31.4s, w9         // acc_max

    mov x9, #2
    mul x13, x8, x9        // x8 * 2
    mov x9, #3
    mul x14, x13, x9       // x8 * 3 * 2

    LoopC:
        mov x9, x1
        mov x10, x2
        mov x17, x4  // height

        ld1 {v5.4s}, [x3], #16
        mov v3.16b, v5.16b
        ld1 {v6.4s}, [x3], #16
        mov v4.16b, v6.16b
        LoopH:
            mov x11, x9
            mov x12, x10
            mov x18, x5  // width
            LoopW:
                ld1 {v0.8b}, [x11], x7
                ssubl v1.8h, v0.8b, v25.8b

                ld1 {v2.8h}, [x12], x13   // weight
                smlal v3.4s, v1.4h, v2.4h
                smlal2 v4.4s, v1.8h, v2.8h

                subs x18, x18, #1
                bne LoopW
            subs x17, x17, #1
            add x9, x9, x6
            add x10, x10, x14
            bne LoopH

        sqshl v3.4s, v3.4s, v28.4s
        sqshl v4.4s, v4.4s, v28.4s
        sqrdmulh v3.4s, v3.4s, v27.4s
        sqrdmulh v4.4s, v4.4s, v27.4s

        and v12.16b, v29.16b, v3.16b
        sshr v12.4s, v12.4s, #31
        sqadd v3.4s, v3.4s, v12.4s
        srshl v3.4s, v3.4s, v29.4s

        and v11.16b, v29.16b, v4.16b
        sshr v11.4s, v11.4s, #31
        sqadd v4.4s, v4.4s, v11.4s
        srshl v4.4s, v4.4s, v29.4s

        add v3.4s, v3.4s, v26.4s
        add v4.4s, v4.4s, v26.4s
        smax v3.4s, v3.4s, v30.4s
        smax v4.4s, v4.4s, v30.4s
        smin v3.4s, v3.4s, v31.4s
        smin v4.4s, v4.4s, v31.4s

        sqxtn v3.4h, v3.4s
        sqxtn v4.4h, v4.4s
        sqxtn v3.8b, v3.8h
        sqxtn v4.8b, v4.8h

        st1 {v3.s}[0], [x0], #4
        st1 {v4.s}[0], [x0], #4
        add x1, x1, #8
        add x2, x2, #16
        sub x8, x8, #8
        cmp x8, #8
        bge LoopC
    ret
#endif
