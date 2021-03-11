#ifdef __aarch64__
#include "nnacl/assembly_global.h"

.text
.align 5
.global ConvDwFp32Row
#ifndef __APPLE__
.type ConvDwFp32Row, %function
#endif

// void ConvDwFp32Row(float* output_ptr, const float* input_ptr,const float* filter_ptr,
//                    size_t num_pixels, size_t input_channel, size_t input_step)
// x0: output_ptr, x1: input_ptr, x2: filter_ptr, x3: num_pixels,
// x4: input_channel, x5: input_step
//
asm_function ConvDwFp32Row
    // registers v8 ~ v15 must be preserved by a callee across subroutine calls, according to
    // https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst#simd-and-floating-point-registers
    // x19 ~ x29 should be also preserved
    // whereas our coding style do not permit such amount of parameters
cmp x3, #0
beq End

mov x9, x0
mov x12, #4
mul x5, x5, x12

LoopOutPixel:
mov x6, x1
mov x7, x2
mov x8, x4

    LoopDepth16In:
    cmp x8, #16
    blt L4
    sub x8, x8, #16

    ld1 {v0.4s, v1.4s}, [x6], #32
    ld1 {v2.4s, v3.4s}, [x7], #32
    ld1 {v16.4s, v17.4s}, [x0], #32

    cmp x8, #16
    blt LoopDepth16Out
    LoopDepth16:
    fmla v16.4s, v0.4s, v2.4s
    fmla v17.4s, v1.4s, v3.4s

    st1 {v16.4s, v17.4s}, [x9], #32

    ld1 {v4.4s, v5.4s}, [x6], #32
    ld1 {v6.4s, v7.4s}, [x7], #32
    ld1 {v18.4s, v19.4s}, [x0], #32

    fmla v18.4s, v4.4s, v6.4s
    fmla v19.4s, v5.4s, v7.4s

    st1 {v18.4s, v19.4s}, [x9], #32

    ld1 {v0.4s, v1.4s}, [x6], #32
    ld1 {v2.4s, v3.4s}, [x7], #32
    ld1 {v16.4s, v17.4s}, [x0], #32

    sub x8, x8, #16
    cmp x8, #16
    bge LoopDepth16

    LoopDepth16Out:
    fmla v16.4s, v0.4s, v2.4s
    fmla v17.4s, v1.4s, v3.4s
    st1 {v16.4s, v17.4s}, [x9], #32
    
    ld1 {v4.4s, v5.4s}, [x6], #32
    ld1 {v6.4s, v7.4s}, [x7], #32
    ld1 {v18.4s, v19.4s}, [x0], #32
    
    fmla v18.4s, v4.4s, v6.4s
    fmla v19.4s, v5.4s, v7.4s
    
    st1 {v18.4s, v19.4s}, [x9], #32

    L4:
    cmp x8, #4
    blt L0

    LoopDepth4:
    ld1 {v0.4s}, [x6], #16
    ld1 {v2.4s}, [x7], #16
    ld1 {v16.4s}, [x0], #16
    fmla v16.4s, v0.4s, v2.4s
    st1 {v16.4s}, [x9], #16
    sub x8, x8, #4
    cmp x8, #4
    bge LoopDepth4

    L0:
    cmp x8, #0
    beq Loop16LineEnd

    LoopDepth0:
    ldr s0, [x6], #4
    ldr s1, [x7], #4
    ldr s2, [x0], #4
    fmul s0, s0, s1
    fadd s2, s2, s0
    str s2, [x9], #4
    subs x8, x8, #1
    bne LoopDepth0

    Loop16LineEnd:

subs x3, x3, #1
add x1, x1, x5
bne LoopOutPixel

End:
ret

#endif
