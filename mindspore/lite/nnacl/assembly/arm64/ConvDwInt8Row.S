#ifdef __aarch64__
#include "nnacl/assembly_global.h"

.text
.align 5
.global ConvDwInt8Row
#ifndef __APPLE__
.type ConvDwInt8Row, %function
#endif

// void ConvDwInt8Row(int32_t *output_ptr, const int8_t *input_ptr, const int16_t *weight_ptr, int num_pixels,
//                    int output_channel, int input_step, int8_t input_zp)
// x0: output_ptr, x1: input_ptr, x2: weight_ptr, x3: num_pixels,
// x4: output_channel, x5: input_step, x6: input_zp
//
asm_function ConvDwInt8Row
    // registers v8 ~ v15 must be preserved by a callee across subroutine calls, according to
    // https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst#simd-and-floating-point-registers
    // x19 ~ x29 should be also preserved
    // whereas our coding style do not permit such amount of parameters
cmp x3, #0
beq End

mov x10, x0

dup v31.8b, w6

LoopOutPixel:
mov x7, x1
mov x8, x2
mov x9, x4

    LoopDepth16In:
    cmp x9, #16
    blt L8
    sub x9, x9, #16

    ld1 {v0.8b, v1.8b}, [x7], #16
    ld1 {v2.8h, v3.8h}, [x8], #32
    ld1 {v16.4s, v17.4s}, [x0], #32

    ssubl v20.8h, v0.8b, v31.8b
    smlal v16.4s, v20.4h, v2.4h
    smlal2 v17.4s, v20.8h, v2.8h


    cmp x9, #16
    blt LoopDepth16Out
    LoopDepth16:

    st1 {v16.4s, v17.4s}, [x10], #32
    ld1 {v18.4s, v19.4s}, [x0], #32
    ssubl v21.8h, v1.8b, v31.8b
    smlal v18.4s, v21.4h, v3.4h
    smlal2 v19.4s, v21.8h, v3.8h
    st1 {v18.4s, v19.4s}, [x10], #32

    ld1 {v0.8b, v1.8b}, [x7], #16
    ld1 {v2.8h, v3.8h}, [x8], #32
    ld1 {v16.4s, v17.4s}, [x0], #32

    ssubl v20.8h, v0.8b, v31.8b
    smlal v16.4s, v20.4h, v2.4h
    smlal2 v17.4s, v20.8h, v2.8h

    sub x9, x9, #16
    cmp x9, #16
    bge LoopDepth16

    LoopDepth16Out:

    st1 {v16.4s, v17.4s}, [x10], #32
    ld1 {v18.4s, v19.4s}, [x0], #32
    ssubl v21.8h, v1.8b, v31.8b
    smlal v18.4s, v21.4h, v3.4h
    smlal2 v19.4s, v21.8h, v3.8h
    st1 {v18.4s, v19.4s}, [x10], #32

    L8:
    cmp x9, #8
    blt L0

    LoopDepth8:
    ld1 {v0.8b}, [x7], #8
    ld1 {v2.8h}, [x8], #16
    ld1 {v16.4s, v17.4s}, [x0], #32

    ssubl v20.8h, v0.8b, v31.8b
    smlal v16.4s, v20.4h, v2.4h
    smlal2 v17.4s, v20.8h, v2.8h
    st1 {v16.4s, v17.4s}, [x10], #32

    sub x9, x9, #8
    cmp x9, #8
    bge LoopDepth8

    L0:
    cmp x9, #0
    beq Loop16LineEnd

    LoopDepth0:
    ldrsb w14, [x7], #1
    ldrsh w15, [x8], #2
    ldr w16, [x0], #4
    sub w14, w14, w6

    sxth w14, w14
    madd w14, w14, w15, w16
    str w14, [x10], #4

    subs x9, x9, #1
    bne LoopDepth0

    Loop16LineEnd:

subs x3, x3, #1
add x1, x1, x5
bne LoopOutPixel

End:
ret

#endif
