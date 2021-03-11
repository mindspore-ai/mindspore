#ifdef __aarch64__
#include "nnacl/assembly_global.h"

.text
.align 5
.global DeconvDwInt8Post
#ifndef __APPLE__
.type DeconvDwInt8Post, %function
#endif

// void DeconvDwInt8Post(int8_t *dst, int32_t *output_buffer, const int32_t *bias, int block_channel, int pixel_nums,
//                       int out_multiplier, int left_shift, int right_shift, int32_t out_zp, int32_t acc_min,
//                       int32_t acc_max)
// x0: dst, x1: output_buffer, x2: bias, x3: block_channel, x4: pixel_nums, x5: out_multiplier
// x6: left_shift, x7: right_shift, x8: out_zp, x9: acc_min, x10: acc_max

asm_function DeconvDwInt8Post
    // registers v8 ~ v15 must be preserved by a callee across subroutine calls, according to
    // https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst#simd-and-floating-point-registers
    // x19 ~ x29 should be also preserved
    // whereas our coding style do not permit such amount of parameters
    ld1 {v25.4s}, [x2]

    dup v26.4s, w6    // left_shift
    dup v27.4s, w5    // out_multiplier
    dup v28.4s, w7    // right_shift

    ldr w8, [sp]
    dup v29.4s, w8    // out_zp
    ldr w9, [sp, #8]
    dup v30.4s, w9   // acc_min
    ldr w10, [sp, #16]
    dup v31.4s, w10   // acc_max

    LoopCount:
        ld1 {v0.4s}, [x1], #16
        add v0.4s, v0.4s, v25.4s
        sqshl v0.4s, v0.4s, v26.4s
        sqrdmulh v0.4s, v0.4s, v27.4s
        sqrshl v0.4s, v0.4s, v28.4s

        add v0.4s, v0.4s, v29.4s
        smax v0.4s, v0.4s, v30.4s
        smin v0.4s, v0.4s, v31.4s

        sqxtn v0.4h, v0.4s
        sqxtn v0.8b, v0.8h

        st1 {v0.s}[0], [x0], x3

        sub x4, x4, #1
        cmp x4, #1
        bge LoopCount
    ret
#endif
