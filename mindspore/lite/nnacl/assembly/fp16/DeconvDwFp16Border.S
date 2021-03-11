#ifdef __aarch64__
#include "nnacl/assembly_global.h"

.text
.align 5
.global DeconvDwFp16Border
#ifndef __APPLE__
.type DeconvDwFp16Border, %function
#endif

// void DeconvDwFp16Border(float *dst, const float *src, const float *weight, size_t height, size_t width,
//                         size_t in_kh_step, size_t in_kw_step, size_t kernel_w)

// x0: dst, x1: src, x2: weight, x3: height, x4: width, x5: in_kh_step, x6: in_kw_step, x7: kernel_w
asm_function DeconvDwFp16Border
    // registers v8 ~ v15 must be preserved by a callee across subroutine calls, according to
    // https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst#simd-and-floating-point-registers
    // x19 ~ x29 should be also preserved
    // whereas our coding style do not permit such amount of parameters
    ld1 {v1.8h}, [x1]

    mov x13, x0
    mov x14, x2
    LoopH:
        mov x15, x13
        mov x16, x14
        mov x17, x4
        LoopW:
            ld1 {v0.8h}, [x15]
            ld1 {v2.8h}, [x16], #16
            fmla v0.8h, v1.8h, v2.8h
            st1 {v0.8h}, [x15], x6
            subs x17, x17, #1
            bne LoopW
        subs x3, x3, #1
        add x13, x13, x5
        add x14, x14, x7
        bne LoopH
        ret
#endif
