#ifdef __aarch64__
#include "nnacl/assembly_global.h"

.text
.align 5
.global ConvDw3x3Int8Vertical
#ifndef __APPLE__
.type ConvDw3x3Int8Vertical, %function
#endif

// void ConvDw3x3Int8Vertical(int8_t *dst, const int8_t *src, const int16_t *weight, const int32_t *bias, size_t in_kh_step,
//                            size_t in_kw_step, size_t channel, size_t in_zp, size_t out_zp, int32_t *out_multiplier,
//                            int32_t *left_shift, int32_t *right_shift, size_t acc_min, size_t acc_max, size_t per_channel)

// x0: dst, x1: src, x2: weight, x3: bias, x4: in_kh_step, x5: in_kw_step,
// x6: channel, x7: in_zp,  x8: out_zp, x9: out_multiplier, x10: left_shift, x11: right_shift
// x12: acc_min, x13: acc_max, x14: per_channel
asm_function ConvDw3x3Int8Vertical
    // registers v8 ~ v15 must be preserved by a callee across subroutine calls, according to
    // https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst#simd-and-floating-point-registers
    // x19 ~ x29 should be also preserved
    // whereas our coding style do not permit such amount of parameters
    sub sp, sp, #32
    stp x19, x20, [sp], #16
    stp x21, x22, [sp], #16

    dup v25.8b, w7                      // in_zp
    ldr x8, [sp]
    dup v26.4s, w8                      // out_zp
    ldr x9, [sp, #8]                    // out_multiplier
    ldr x10, [sp, #16]                  // left_shift
    ldr x11, [sp, #24]                  // right_shift
    ldr x12, [sp, #32]
    dup v30.4s, w12                     // acc_min
    ldr x13, [sp, #40]
    dup v31.4s, w13                     // acc_max
    ldr x14, [sp, #48]                  // per_channel
    cbnz x14, PerChannelDump
    PerLayerDump:
        ld1r {v27.4s}, [x9]
        ld1r {v28.4s}, [x10]
        ld1r {v29.4s}, [x11]
        b ContinueFunc
    PerChannelDump:
        ld1 {v27.4s}, [x9], #16
        ld1 {v28.4s}, [x10], #16
        ld1 {v29.4s}, [x11], #16
    ContinueFunc:

    mov x12, #2
    mul x21, x6, x12                    // x6 * 2
    mov x12, #3
    mul x22, x21, x12                   // x6 * 3 * 2

    ld1 {v23.4s}, [x3], #16
    ld1 {v24.4s}, [x3], #16
    mov x12, x1
    mov x13, x2

    ld1 {v0.8b}, [x12], x5
    ssubl v0.8h, v0.8b, v25.8b
    add x19, x1, x4
    ld1 {v4.8h}, [x13], x21   // weight
    add x20, x2, x22
    ld1 {v1.8b}, [x12], x5
    ssubl v1.8h, v1.8b, v25.8b
    ld1 {v5.8h}, [x13], x21
    ld1 {v2.8b}, [x19], x5
    ssubl v2.8h, v2.8b, v25.8b
    ld1 {v6.8h}, [x20], x21
    ld1 {v3.8b}, [x19], x5
    ssubl v3.8h, v3.8b, v25.8b
    ld1 {v7.8h}, [x20], x21
    ld1 {v16.8b}, [x12], x5
    ssubl v16.8h, v16.8b, v25.8b
    ld1 {v18.8h}, [x13], x21
    ld1 {v17.8b}, [x19], x5
    ssubl v17.8h, v17.8b, v25.8b
    ld1 {v19.8h}, [x20], x21

    cmp x6, #8
    ble LoopC8Post

    LoopC8:
        add x1, x1, #8
        add x2, x2, #16
        smlal v23.4s, v0.4h, v4.4h
        smlal2 v24.4s, v0.8h, v4.8h
        mov x12, x1
        mov x13, x2
        ld1 {v0.8b}, [x12], x5
        ssubl v0.8h, v0.8b, v25.8b
        ld1 {v4.8h}, [x13], x21   // weight
        add x19, x1, x4
        smlal v23.4s, v1.4h, v5.4h
        smlal2 v24.4s, v1.8h, v5.8h

        add x20, x2, x22
        ld1 {v1.8b}, [x12], x5
        ssubl v1.8h, v1.8b, v25.8b
        smlal v23.4s, v2.4h, v6.4h
        ld1 {v5.8h}, [x13], x21
        smlal2 v24.4s, v2.8h, v6.8h

        ld1 {v2.8b}, [x19], x5
        ssubl v2.8h, v2.8b, v25.8b
        smlal v23.4s, v3.4h, v7.4h
        ld1 {v6.8h}, [x20], x21
        smlal2 v24.4s, v3.8h, v7.8h

        ld1 {v3.8b}, [x19], x5
        ssubl v3.8h, v3.8b, v25.8b
        smlal v23.4s, v16.4h, v18.4h
        ld1 {v7.8h}, [x20], x21
        smlal2 v24.4s, v16.8h, v18.8h

        ld1 {v16.8b}, [x12], x5
        ssubl v16.8h, v16.8b, v25.8b
        smlal v23.4s, v17.4h, v19.4h
        ld1 {v18.8h}, [x13], x21
        smlal2 v24.4s, v17.8h, v19.8h
        ld1 {v17.8b}, [x19], x5
        ssubl v17.8h, v17.8b, v25.8b
        ld1 {v19.8h}, [x20], x21

        cbnz x14, PerChannelPostLoop
            ldr w8, [x10]
            cbz w8, RightShiftLoop
            sqshl v23.4s, v23.4s, v28.4s
            sqshl v24.4s, v24.4s, v28.4s
            sqrdmulh v23.4s, v23.4s, v27.4s
            sqrdmulh v24.4s, v24.4s, v27.4s
            b AddZpLoop

            RightShiftLoop:
            sqrdmulh v23.4s, v23.4s, v27.4s
            sqrdmulh v24.4s, v24.4s, v27.4s
            sqrshl v23.4s, v23.4s, v29.4s
            sqrshl v24.4s, v24.4s, v29.4s
            b AddZpLoop
        PerChannelPostLoop:
            sqshl v23.4s, v23.4s, v28.4s
            ld1 {v28.4s}, [x10], #16
            sqrdmulh v23.4s, v23.4s, v27.4s
            ld1 {v27.4s}, [x9], #16
            sqrshl v23.4s, v23.4s, v29.4s
            ld1 {v29.4s}, [x11], #16
            sqshl v24.4s, v24.4s, v28.4s
            ld1 {v28.4s}, [x10], #16
            sqrdmulh v24.4s, v24.4s, v27.4s
            ld1 {v27.4s}, [x9], #16
            sqrshl v24.4s, v24.4s, v29.4s
            ld1 {v29.4s}, [x11], #16

        AddZpLoop:
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
        smlal v23.4s, v16.4h, v18.4h
        smlal2 v24.4s, v16.8h, v18.8h
        smlal v23.4s, v17.4h, v19.4h
        smlal2 v24.4s, v17.8h, v19.8h

        cbnz x14, PerChannelPost
            ldr w8, [x10]
            cbz w8, RightShift
            sqshl v23.4s, v23.4s, v28.4s
            sqshl v24.4s, v24.4s, v28.4s
            sqrdmulh v23.4s, v23.4s, v27.4s
            sqrdmulh v24.4s, v24.4s, v27.4s
            b AddZp

            RightShift:
            sqrdmulh v23.4s, v23.4s, v27.4s
            sqrdmulh v24.4s, v24.4s, v27.4s
            sqrshl v23.4s, v23.4s, v29.4s
            sqrshl v24.4s, v24.4s, v29.4s
            b AddZp
        PerChannelPost:
            sqshl v23.4s, v23.4s, v28.4s
            ld1 {v28.4s}, [x10], #16
            sqrdmulh v23.4s, v23.4s, v27.4s
            ld1 {v27.4s}, [x9], #16
            sqrshl v23.4s, v23.4s, v29.4s
            ld1 {v29.4s}, [x11], #16
            sqshl v24.4s, v24.4s, v28.4s
            sqrdmulh v24.4s, v24.4s, v27.4s
            sqrshl v24.4s, v24.4s, v29.4s

        AddZp:
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
    sub sp, sp, #32
    ldp x19, x20, [sp], #16
    ldp x21, x22, [sp], #16
    ret
#endif
