#ifdef __aarch64__
#include "nnacl/assembly_global.h"

.text
.align 5
.global ConvDwFp16Row
#ifndef __APPLE__
.type ConvDwFp16Row, %function
#endif

// void ConvDwFp16Row(float16_t* output_ptr, const float16_t* input_ptr,const float16_t* filter_ptr,
//                    size_t num_pixels, size_t input_channel, size_t input_step)
// x0: output_ptr, x1: input_ptr, x2: filter_ptr, x3: num_pixels,
// x4: input_channel, x5: input_step
//
asm_function ConvDwFp16Row
    // registers v8 ~ v15 must be preserved by a callee across subroutine calls, according to
    // https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst#simd-and-floating-point-registers
    // x19 ~ x29 should be also preserved
    // whereas our coding style do not permit such amount of parameters
cmp x3, #0
beq End

mov x9, x0
mov x12, #2 // sizeof(float16_t)
mul x5, x5, x12

LoopOutPixel:
mov x6, x1
mov x7, x2
mov x8, x4

LoopInputDepth32In:
    cmp x8, #32
    blt Loop8
    sub x8, x8, #32

    ld1 {v0.8h, v1.8h}, [x6], #32
    ld1 {v2.8h, v3.8h}, [x7], #32
    ld1 {v16.8h, v17.8h}, [x0], #32

    cmp x8, #32
    blt LoopInputDepth32Out
    LoopInputDepth32:
    fmla v16.8h, v0.8h, v2.8h
    fmla v17.8h, v1.8h, v3.8h

    st1 {v16.8h, v17.8h}, [x9], #32

    ld1 {v4.8h, v5.8h}, [x6], #32
    ld1 {v6.8h, v7.8h}, [x7], #32
    ld1 {v18.8h, v19.8h}, [x0], #32

    fmla v18.8h, v4.8h, v6.8h
    fmla v19.8h, v5.8h, v7.8h

    st1 {v18.8h, v19.8h}, [x9], #32

    ld1 {v0.8h, v1.8h}, [x6], #32
    ld1 {v2.8h, v3.8h}, [x7], #32
    ld1 {v16.8h, v17.8h}, [x0], #32

    sub x8, x8, #32
    cmp x8, #32
    bge LoopInputDepth32

    LoopInputDepth32Out:
    fmla v16.8h, v0.8h, v2.8h
    fmla v17.8h, v1.8h, v3.8h
    st1 {v16.8h, v17.8h}, [x9], #32

    ld1 {v4.8h, v5.8h}, [x6], #32
    ld1 {v6.8h, v7.8h}, [x7], #32
    ld1 {v18.8h, v19.8h}, [x0], #32

    fmla v18.8h, v4.8h, v6.8h
    fmla v19.8h, v5.8h, v7.8h

    st1 {v18.8h, v19.8h}, [x9], #32

    Loop8:
    cmp x8, #8
    blt L0

    LoopInputDepth8:
    ld1 {v0.8h}, [x6], #16
    ld1 {v2.8h}, [x7], #16
    ld1 {v16.8h}, [x0], #16
    fmla v16.8h, v0.8h, v2.8h
    st1 {v16.8h}, [x9], #16
    sub x8, x8, #8
    cmp x8, #8
    bge LoopInputDepth8

    L0:
    cmp x8, #0
    beq Loop8LineEnd

    LoopInputDepth0:
    ldr h0, [x6], #2
    ldr h1, [x7], #2
    ldr h2, [x0], #2
    fmul h0, h0, h1
    fadd h2, h2, h0
    str h2, [x9], #2
    subs x8, x8, #1
    bne LoopInputDepth0

    Loop8LineEnd:

subs x3, x3, #1
add x1, x1, x5
bne LoopOutPixel

End:
ret

#endif
