#ifdef __aarch64__

.text
.align 5
.global ConvDw3x3Int8Corner
#ifndef __APPLE__
.type ConvDw3x3Int8Corner, %function
#endif

// void ConvDw3x3Int8Corner(int8_t *dst, const int8_t *src, const int16_t *weight, const int32_t *bias, size_t in_kh_step,
//                          size_t in_kw_step, size_t channel, size_t in_zp, size_t out_zp, size_t out_multiplier,
//                          size_t left_shift, size_t right_shift, size_t acc_min, size_t acc_max)

// x0: dst, x1: src, x2: weight, x3: bias, x4: in_kh_step, x5: in_kw_step,
// x6: channel, x7: in_zp,  x8: out_zp, x9: out_multiplier, x10: left_shift, x11: right_shift
// x11: acc_min, x13: acc_max
ConvDw3x3Int8Corner:
    // registers v8 ~ v15 must be preserved by a callee across subroutine calls, according to
    // https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst#simd-and-floating-point-registers
    // x19 ~ x29 should be also preserved
    // whereas our coding style do not permit such amount of parameters
    dup v25.8b, w7         // in_zp
    ldr x9, [sp]
    dup v26.4s, w9        // out_zp
    ldr x9, [sp, #8]
    dup v27.4s, w9         // out_multiplier
    ldr x9, [sp, #16]
    dup v28.4s, w9         // left_shift
    ldr x9, [sp, #24]
    dup v29.4s, w9         // right_shift
    ldr x9, [sp, #32]
    dup v30.4s, w9         // acc_min
    ldr x9, [sp, #40]
    dup v31.4s, w9         // acc_max

    mov x9, #2
    mul x13, x6, x9        // x6 * 2
    mov x9, #3
    mul x14, x13, x9       // x6 * 3 * 2

    ld1 {v23.4s}, [x3], #16
    ld1 {v24.4s}, [x3], #16
    mov x9, x1
    mov x10, x2

    ld1 {v0.8b}, [x9], x5
    ssubl v0.8h, v0.8b, v25.8b
    add x11, x1, x4
    ld1 {v4.8h}, [x10], x13   // weight
    add x12, x2, x14
    ld1 {v1.8b}, [x9], x5
    ssubl v1.8h, v1.8b, v25.8b
    ld1 {v5.8h}, [x10], x13
    ld1 {v2.8b}, [x11], x5
    ssubl v2.8h, v2.8b, v25.8b
    ld1 {v6.8h}, [x12], x13
    ld1 {v3.8b}, [x11], x5
    ssubl v3.8h, v3.8b, v25.8b
    ld1 {v7.8h}, [x12], x13

    cmp x6, #8
    ble LoopC8Post

    LoopC8:
        add x1, x1, #8
        add x2, x2, #16
        smlal v23.4s, v0.4h, v4.4h
        smlal2 v24.4s, v0.8h, v4.8h
        mov x9, x1
        mov x10, x2
        ld1 {v0.8b}, [x9], x5
        ssubl v0.8h, v0.8b, v25.8b
        ld1 {v4.8h}, [x10], x13   // weight
        add x11, x1, x4
        smlal v23.4s, v1.4h, v5.4h
        smlal2 v24.4s, v1.8h, v5.8h
        add x12, x2, x14
        ld1 {v1.8b}, [x9], x5
        ssubl v1.8h, v1.8b, v25.8b
        smlal v23.4s, v2.4h, v6.4h
        ld1 {v5.8h}, [x10], x13
        smlal2 v24.4s, v2.8h, v6.8h
        ld1 {v2.8b}, [x11], x5
        ssubl v2.8h, v2.8b, v25.8b
        smlal v23.4s, v3.4h, v7.4h
        ld1 {v6.8h}, [x12], x13
        smlal2 v24.4s, v3.8h, v7.8h

        sqshl v23.4s, v23.4s, v28.4s
        sqshl v24.4s, v24.4s, v28.4s
        sqrdmulh v23.4s, v23.4s, v27.4s
        sqrdmulh v24.4s, v24.4s, v27.4s

        and v21.16b, v29.16b, v23.16b
        sshr v21.4s, v21.4s, #31
        sqadd v23.4s, v23.4s, v21.4s
        srshl v23.4s, v23.4s, v29.4s

        and v22.16b, v29.16b, v24.16b
        sshr v22.4s, v22.4s, #31
        sqadd v24.4s, v24.4s, v22.4s
        srshl v24.4s, v24.4s, v29.4s

        ld1 {v3.8b}, [x11], x5
        ssubl v3.8h, v3.8b, v25.8b
        ld1 {v7.8h}, [x12], x13

        add v23.4s, v23.4s, v26.4s
        add v24.4s, v24.4s, v26.4s
        smax v23.4s, v23.4s, v30.4s
        smax v24.4s, v24.4s, v30.4s
        smin v23.4s, v23.4s, v31.4s
        smin v24.4s, v24.4s, v31.4s

        sqxtn v23.4h, v23.4s
        sqxtn v24.4h, v24.4s
        sqxtn v23.8b, v23.8h
        sqxtn v24.8b, v24.8h

        st1 {v23.s}[0], [x0], #4
        st1 {v24.s}[0], [x0], #4
        ld1 {v23.4s}, [x3], #16
        ld1 {v24.4s}, [x3], #16
        sub x6, x6, #8
        cmp x6, #8
        bgt LoopC8

    LoopC8Post:
        smlal v23.4s, v0.4h, v4.4h
        smlal2 v24.4s, v0.8h, v4.8h
        smlal v23.4s, v1.4h, v5.4h
        smlal2 v24.4s, v1.8h, v5.8h
        smlal v23.4s, v2.4h, v6.4h
        smlal2 v24.4s, v2.8h, v6.8h
        smlal v23.4s, v3.4h, v7.4h
        smlal2 v24.4s, v3.8h, v7.8h

        sqshl v23.4s, v23.4s, v28.4s
        sqshl v24.4s, v24.4s, v28.4s
        sqrdmulh v23.4s, v23.4s, v27.4s
        sqrdmulh v24.4s, v24.4s, v27.4s

        and v21.16b, v29.16b, v23.16b
        sshr v21.4s, v21.4s, #31
        sqadd v23.4s, v23.4s, v21.4s
        srshl v23.4s, v23.4s, v29.4s

        and v22.16b, v29.16b, v24.16b
        sshr v22.4s, v22.4s, #31
        sqadd v24.4s, v24.4s, v22.4s
        srshl v24.4s, v24.4s, v29.4s

        add v23.4s, v23.4s, v26.4s
        add v24.4s, v24.4s, v26.4s
        smax v23.4s, v23.4s, v30.4s
        smax v24.4s, v24.4s, v30.4s
        smin v23.4s, v23.4s, v31.4s
        smin v24.4s, v24.4s, v31.4s

        sqxtn v23.4h, v23.4s
        sqxtn v24.4h, v24.4s
        sqxtn v23.8b, v23.8h
        sqxtn v24.8b, v24.8h

        st1 {v23.s}[0], [x0], #4
        st1 {v24.s}[0], [x0], #4
    ret
#endif
