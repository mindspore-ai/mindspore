#ifdef __aarch64__
#include "nnacl/assembly_global.h"

.text
.align 5
.global ConvDwInt8PostAlign4
#ifndef __APPLE__
.type ConvDwInt8PostAlign4, %function
#endif

// void ConvDwInt8PostAlign4(int8_t *dst, int32_t *buffer, int num_pixels, int32_t output_zp, int32_t out_multiplier,
//                           int32_t left_shift, int32_t right_shift, int32_t acc_min, int32_t acc_max);
// x0: dst, x1: buffer, x2: num_pixels, x3: output_zp, x4: out_multiplier,
// x5: left_shift, x6: right_shift, x7: acc_min, x8: acc_max

asm_function ConvDwInt8PostAlign4
    // registers v8 ~ v15 must be preserved by a callee across subroutine calls, according to
    // https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst#simd-and-floating-point-registers
    // x19 ~ x29 should be also preserved
    // whereas our coding style do not permit such amount of parameters
    ldr x8, [sp]

    dup v26.4s, w5
    dup v27.4s, w4
    dup v28.4s, w6

    dup v29.4s, w3
    dup v30.4s, w7
    dup v31.4s, w8

    cmp x2, #16
    blt LoopDepth8

    LoopDepth16:
        ld1 {v0.4s}, [x1], #16
        ld1 {v1.4s}, [x1], #16
        ld1 {v2.4s}, [x1], #16
        ld1 {v3.4s}, [x1], #16

        cbz w5, RightShiftDepth16
        sqshl v0.4s, v0.4s, v26.4s
        sqshl v1.4s, v1.4s, v26.4s
        sqshl v2.4s, v2.4s, v26.4s
        sqshl v3.4s, v3.4s, v26.4s
        sqrdmulh v0.4s, v0.4s, v27.4s
        sqrdmulh v1.4s, v1.4s, v27.4s
        sqrdmulh v2.4s, v2.4s, v27.4s
        sqrdmulh v3.4s, v3.4s, v27.4s
        b AddZpDepth16

        RightShiftDepth16:
        sqrdmulh v0.4s, v0.4s, v27.4s
        sqrdmulh v1.4s, v1.4s, v27.4s
        sqrdmulh v2.4s, v2.4s, v27.4s
        sqrdmulh v3.4s, v3.4s, v27.4s

        and v4.16b, v0.16b, v28.16b
        sshr v4.4s, v4.4s, #31
        sqadd v0.4s, v0.4s, v4.4s
        srshl v0.4s, v0.4s, v28.4s
        and v5.16b, v1.16b, v28.16b
        sshr v5.4s, v5.4s, #31
        sqadd v1.4s, v1.4s, v5.4s
        srshl v1.4s, v1.4s, v28.4s
        and v6.16b, v2.16b, v28.16b
        sshr v6.4s, v6.4s, #31
        sqadd v2.4s, v2.4s, v6.4s
        srshl v2.4s, v2.4s, v28.4s
        and v7.16b, v3.16b, v28.16b
        sshr v7.4s, v7.4s, #31
        sqadd v3.4s, v3.4s, v7.4s
        srshl v3.4s, v3.4s, v28.4s

        AddZpDepth16:
        add v0.4s, v0.4s, v29.4s
        add v1.4s, v1.4s, v29.4s
        add v2.4s, v2.4s, v29.4s
        add v3.4s, v3.4s, v29.4s

        smax v0.4s, v0.4s, v30.4s
        smax v1.4s, v1.4s, v30.4s
        smax v2.4s, v2.4s, v30.4s
        smax v3.4s, v3.4s, v30.4s

        smin v0.4s, v0.4s, v31.4s
        smin v1.4s, v1.4s, v31.4s
        smin v2.4s, v2.4s, v31.4s
        smin v3.4s, v3.4s, v31.4s

        sqxtn v0.4h, v0.4s
        sqxtn v1.4h, v1.4s
        sqxtn v2.4h, v2.4s
        sqxtn v3.4h, v3.4s

        sqxtn v0.8b, v0.8h
        sqxtn v1.8b, v1.8h
        sqxtn v2.8b, v2.8h
        sqxtn v3.8b, v3.8h

        st1 {v0.s}[0], [x0], #4
        st1 {v1.s}[0], [x0], #4
        st1 {v2.s}[0], [x0], #4
        st1 {v3.s}[0], [x0], #4

        sub x2, x2, #16
        cmp x2, #16
        bge LoopDepth16

    LoopDepth8:
        cmp x2, #8
        blt LoopDepth4
        ld1 {v0.4s}, [x1], #16
        ld1 {v1.4s}, [x1], #16

        cbz w5, RightShiftDepth8
        sqshl v0.4s, v0.4s, v26.4s
        sqshl v1.4s, v1.4s, v26.4s
        sqrdmulh v0.4s, v0.4s, v27.4s
        sqrdmulh v1.4s, v1.4s, v27.4s
        b AddZpDepth8

        RightShiftDepth8:
        sqrdmulh v0.4s, v0.4s, v27.4s
        sqrdmulh v1.4s, v1.4s, v27.4s
        and v4.16b, v0.16b, v28.16b
        sshr v4.4s, v4.4s, #31
        sqadd v0.4s, v0.4s, v4.4s
        srshl v0.4s, v0.4s, v28.4s
        and v5.16b, v1.16b, v28.16b
        sshr v5.4s, v5.4s, #31
        sqadd v1.4s, v1.4s, v5.4s
        srshl v1.4s, v1.4s, v28.4s

        AddZpDepth8:
        add v0.4s, v0.4s, v29.4s
        add v1.4s, v1.4s, v29.4s
        smax v0.4s, v0.4s, v30.4s
        smax v1.4s, v1.4s, v30.4s
        smin v0.4s, v0.4s, v31.4s
        smin v1.4s, v1.4s, v31.4s

        sqxtn v0.4h, v0.4s
        sqxtn v1.4h, v1.4s

        sqxtn v0.8b, v0.8h
        sqxtn v1.8b, v1.8h

        st1 {v0.s}[0], [x0], #4
        st1 {v1.s}[0], [x0], #4

        sub x2, x2, #8
        cmp x2, #8
        bge LoopDepth8

    LoopDepth4:
        cmp x2, #4
        blt End
        ld1 {v0.4s}, [x1], #16

        sqshl v0.4s, v0.4s, v26.4s
        sqrdmulh v0.4s, v0.4s, v27.4s
        and v4.16b, v0.16b, v28.16b
        sshr v4.4s, v4.4s, #31
        sqadd v0.4s, v0.4s, v4.4s
        srshl v0.4s, v0.4s, v28.4s

        add v0.4s, v0.4s, v29.4s
        smax v0.4s, v0.4s, v30.4s
        smin v0.4s, v0.4s, v31.4s

        sqxtn v0.4h, v0.4s
        sqxtn v0.8b, v0.8h

        st1 {v0.s}[0], [x0], #4

        sub x2, x2, #4
        bge LoopDepth4
    End:
    ret
#endif
