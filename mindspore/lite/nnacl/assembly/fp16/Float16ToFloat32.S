#ifdef __aarch64__
#include "nnacl/assembly_global.h"

.text
.align 5
.global Float16ToFloat32
#ifndef __APPLE__
.type Float16ToFloat32, %function
#endif

// void Float16ToFloat32(const float16_t *input, float *output, int number);
// x0: input, x1: output, x2: number
asm_function Float16ToFloat32
    // registers v8 ~ v15 must be preserved by a callee across subroutine calls, according to
    // https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst#simd-and-floating-point-registers
    // x19 ~ x29 should be also preserved
    // whereas our coding style do not permit such amount of parameters
    cmp x2, #0
    beq LoopEnd
    cmp x2, #64
    blt Loop
    Loop64:
        ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x0], #64
        ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x0], #64
        fcvtl v16.4s, v0.4h
        fcvtl2 v17.4s, v0.8h
        fcvtl v18.4s, v1.4h
        fcvtl2 v19.4s, v1.8h
        fcvtl v20.4s, v2.4h
        fcvtl2 v21.4s, v2.8h
        fcvtl v22.4s, v3.4h
        fcvtl2 v23.4s, v3.8h
        fcvtl v24.4s, v4.4h
        fcvtl2 v25.4s, v4.8h
        fcvtl v26.4s, v5.4h
        fcvtl2 v27.4s, v5.8h
        fcvtl v28.4s, v6.4h
        fcvtl2 v29.4s, v6.8h
        fcvtl v30.4s, v7.4h
        fcvtl2 v31.4s, v7.8h
        st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x1], #64
        st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x1], #64
        st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x1], #64
        st1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x1], #64
        subs x2, x2, #64
        ble LoopEnd
        cmp x2, #64
        bge Loop64
    Loop:
        ldr h0, [x0], #2
        fcvt s0, h0
        str s0, [x1], #4
        subs x2, x2, #1
        bgt Loop
    LoopEnd:
        ret
#endif
