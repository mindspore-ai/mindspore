#ifdef __aarch64__
#include "nnacl/assembly_global.h"

.text
.align 5
.global Float32ToFloat16
#ifndef __APPLE__
.type Float32ToFloat16, %function
#endif

// void Float32ToFloat16(const float *input, float16_t output, int number);
// x0: input, x1: output, x2: number
asm_function Float32ToFloat16
    // registers v8 ~ v15 must be preserved by a callee across subroutine calls, according to
    // https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst#simd-and-floating-point-registers
    // x19 ~ x29 should be also preserved
    // whereas our coding style do not permit such amount of parameters
    cmp x2, #0
    beq LoopEnd
    cmp x2, #64
    blt Loop
    Loop64:
        ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], #64
        ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x0], #64
        ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x0], #64
        ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x0], #64
        fcvtn v0.4h, v16.4s
        fcvtn2 v0.8h, v17.4s
        fcvtn v1.4h, v18.4s
        fcvtn2 v1.8h, v19.4s
        fcvtn v2.4h, v20.4s
        fcvtn2 v2.8h, v21.4s
        fcvtn v3.4h, v22.4s
        fcvtn2 v3.8h, v23.4s
        fcvtn v4.4h, v24.4s
        fcvtn2 v4.8h, v25.4s
        fcvtn v5.4h, v26.4s
        fcvtn2 v5.8h, v27.4s
        fcvtn v6.4h, v28.4s
        fcvtn2 v6.8h, v29.4s
        fcvtn v7.4h, v30.4s
        fcvtn2 v7.8h, v31.4s
        st1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x1], #64
        st1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x1], #64
        subs x2, x2, #64
        ble LoopEnd
        cmp x2, #64
        bge Loop64
    Loop:
        ldr s0, [x0], #4
        fcvt h0, s0
        str h0, [x1], #2
        subs x2, x2, #1
        bgt Loop
    LoopEnd:
        ret
#endif
