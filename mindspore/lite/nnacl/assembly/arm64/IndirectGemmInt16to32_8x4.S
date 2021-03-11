#ifdef __aarch64__
#include "nnacl/assembly_global.h"

.text
.align 5
.global IndirectGemmInt16to32_8x4
#ifndef __APPLE__
.type IndirectGemmInt16to32_8x4, %function
#endif

// void IndirectGemmInt16to32_8x4(int *output, short *input, short *weight, size_t ksize, size_t ic8, size_t oc4, size_t offset);
// x0: output, x1: input, x2: weight, x3: ksize, x4: ic8, x5: oc4, x6: offset
asm_function IndirectGemmInt16to32_8x4

    .macro INIT_ZERO
        dup v28.4s, wzr
        mov v29.16b, v28.16b
        mov v30.16b, v28.16b
        mov v31.16b, v28.16b
    .endm

    // registers v8 ~ v15 must be preserved by a callee across subroutine calls, according to
    // https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst#simd-and-floating-point-registers
    // x19 ~ x29 should be also preserved
    // whereas our coding style do not permit such amount of parameters
    LoopOc:
        mov x7, x3
        mov x8, x1

        LoopKsize:
            mov x9, x0
            INIT_ZERO

            // load input
            ld1 {v0.8h, v1.8h}, [x8], #32
            // load weight
            ld1 {v16.8h}, [x2], #16
            smull v24.4s, v16.4h, v0.h[0]
            smull v25.4s, v16.4h, v1.h[0]
            // load weight
            ld1 {v17.8h}, [x2], #16
            smlal2 v24.4s, v16.8h, v0.h[1]
            smlal2 v25.4s, v16.8h, v1.h[1]
            // load input
            ld1 {v2.8h, v3.8h}, [x8], #32
            smlal v24.4s, v17.4h, v0.h[2]
            smlal v25.4s, v17.4h, v1.h[2]
            smlal2 v24.4s, v17.8h, v0.h[3]
            smlal2 v25.4s, v17.8h, v1.h[3]
            // load weight
            ld1 {v18.8h, v19.8h}, [x2], #32
            smull v26.4s, v16.4h, v2.h[0]
            smull v27.4s, v16.4h, v3.h[0]

            subs x10, x4, #1
            beq LoopIcEnd

            LoopIc:

                smlal2 v26.4s, v16.8h, v2.h[1]
                smlal2 v27.4s, v16.8h, v3.h[1]
                smlal v26.4s, v17.4h, v2.h[2]
                smlal v27.4s, v17.4h, v3.h[2]
                smlal2 v26.4s, v17.8h, v2.h[3]
                smlal2 v27.4s, v17.8h, v3.h[3]

                smlal v24.4s, v18.4h, v0.h[4]
                smlal v25.4s, v18.4h, v1.h[4]
                smlal2 v24.4s, v18.8h, v0.h[5]
                smlal2 v25.4s, v18.8h, v1.h[5]
                smlal v24.4s, v19.4h, v0.h[6]
                smlal v25.4s, v19.4h, v1.h[6]
                smlal2 v24.4s, v19.8h, v0.h[7]
                smlal2 v25.4s, v19.8h, v1.h[7]
                // load input
                ld1 {v4.8h, v5.8h}, [x8], #32
                smlal v26.4s, v18.4h, v2.h[4]
                smlal v27.4s, v18.4h, v3.h[4]
                smlal2 v26.4s, v18.8h, v2.h[5]
                smlal2 v27.4s, v18.8h, v3.h[5]
                smlal v26.4s, v19.4h, v2.h[6]
                smlal v27.4s, v19.4h, v3.h[6]
                smlal2 v26.4s, v19.8h, v2.h[7]
                smlal2 v27.4s, v19.8h, v3.h[7]

                // load input
                ld1 {v6.8h, v7.8h}, [x8], #32
                smlal v28.4s, v16.4h, v4.h[0]
                smlal v29.4s, v16.4h, v5.h[0]
                smlal2 v28.4s, v16.8h, v4.h[1]
                smlal2 v29.4s, v16.8h, v5.h[1]
                smlal v28.4s, v17.4h, v4.h[2]
                smlal v29.4s, v17.4h, v5.h[2]
                smlal2 v28.4s, v17.8h, v4.h[3]
                smlal2 v29.4s, v17.8h, v5.h[3]

                smlal v30.4s, v16.4h, v6.h[0]
                smlal v31.4s, v16.4h, v7.h[0]
                smlal2 v30.4s, v16.8h, v6.h[1]
                smlal2 v31.4s, v16.8h, v7.h[1]
                smlal v30.4s, v17.4h, v6.h[2]
                smlal v31.4s, v17.4h, v7.h[2]
                smlal2 v30.4s, v17.8h, v6.h[3]
                smlal2 v31.4s, v17.8h, v7.h[3]
                // load weight
                ld1 {v16.8h, v17.8h}, [x2], #32
                smlal v28.4s, v18.4h, v4.h[4]
                smlal v29.4s, v18.4h, v5.h[4]
                smlal2 v28.4s, v18.8h, v4.h[5]
                smlal2 v29.4s, v18.8h, v5.h[5]
                smlal v28.4s, v19.4h, v4.h[6]
                smlal v29.4s, v19.4h, v5.h[6]
                smlal2 v28.4s, v19.8h, v4.h[7]
                smlal2 v29.4s, v19.8h, v5.h[7]
                // load input
                ld1 {v0.8h, v1.8h}, [x8], #32
                smlal v30.4s, v18.4h, v6.h[4]
                smlal v31.4s, v18.4h, v7.h[4]
                smlal2 v30.4s, v18.8h, v6.h[5]
                smlal2 v31.4s, v18.8h, v7.h[5]
                smlal v30.4s, v19.4h, v6.h[6]
                smlal v31.4s, v19.4h, v7.h[6]
                smlal2 v30.4s, v19.8h, v6.h[7]
                smlal2 v31.4s, v19.8h, v7.h[7]
                // load input
                ld1 {v2.8h, v3.8h}, [x8], #32
                smlal v24.4s, v16.4h, v0.h[0]
                smlal v25.4s, v16.4h, v1.h[0]
                smlal2 v24.4s, v16.8h, v0.h[1]
                smlal2 v25.4s, v16.8h, v1.h[1]
                // load weight
                ld1 {v18.8h, v19.8h}, [x2], #32
                smlal v24.4s, v17.4h, v0.h[2]
                smlal v25.4s, v17.4h, v1.h[2]
                smlal2 v24.4s, v17.8h, v0.h[3]
                smlal2 v25.4s, v17.8h, v1.h[3]
                smlal v26.4s, v16.4h, v2.h[0]
                smlal v27.4s, v16.4h, v3.h[0]

                subs x10, x10, #1
                bne LoopIc

            LoopIcEnd:
                smlal2 v26.4s, v16.8h, v2.h[1]
                smlal2 v27.4s, v16.8h, v3.h[1]
                smlal v26.4s, v17.4h, v2.h[2]
                smlal v27.4s, v17.4h, v3.h[2]
                smlal2 v26.4s, v17.8h, v2.h[3]
                smlal2 v27.4s, v17.8h, v3.h[3]

                smlal v24.4s, v18.4h, v0.h[4]
                smlal v25.4s, v18.4h, v1.h[4]
                smlal2 v24.4s, v18.8h, v0.h[5]
                smlal2 v25.4s, v18.8h, v1.h[5]
                smlal v24.4s, v19.4h, v0.h[6]
                smlal v25.4s, v19.4h, v1.h[6]
                smlal2 v24.4s, v19.8h, v0.h[7]
                smlal2 v25.4s, v19.8h, v1.h[7]
                // load input
                ld1 {v4.8h, v5.8h}, [x8], #32
                smlal v26.4s, v18.4h, v2.h[4]
                smlal v27.4s, v18.4h, v3.h[4]
                smlal2 v26.4s, v18.8h, v2.h[5]
                st1 {v24.4s}, [x9], x6
                smlal2 v27.4s, v18.8h, v3.h[5]
                smlal v26.4s, v19.4h, v2.h[6]
                st1 {v25.4s}, [x9], x6
                smlal v27.4s, v19.4h, v3.h[6]
                smlal2 v26.4s, v19.8h, v2.h[7]
                smlal2 v27.4s, v19.8h, v3.h[7]

                // load input
                ld1 {v6.8h, v7.8h}, [x8], #32
                smlal v28.4s, v16.4h, v4.h[0]
                smlal v29.4s, v16.4h, v5.h[0]
                smlal2 v28.4s, v16.8h, v4.h[1]
                smlal2 v29.4s, v16.8h, v5.h[1]
                smlal v28.4s, v17.4h, v4.h[2]
                st1 {v26.4s}, [x9], x6
                smlal v29.4s, v17.4h, v5.h[2]
                smlal2 v28.4s, v17.8h, v4.h[3]
                smlal2 v29.4s, v17.8h, v5.h[3]
                st1 {v27.4s}, [x9], x6
                smlal v30.4s, v16.4h, v6.h[0]
                smlal v31.4s, v16.4h, v7.h[0]
                smlal2 v30.4s, v16.8h, v6.h[1]
                smlal2 v31.4s, v16.8h, v7.h[1]
                smlal v30.4s, v17.4h, v6.h[2]
                smlal v31.4s, v17.4h, v7.h[2]
                smlal2 v30.4s, v17.8h, v6.h[3]
                smlal2 v31.4s, v17.8h, v7.h[3]
                smlal v28.4s, v18.4h, v4.h[4]
                smlal v29.4s, v18.4h, v5.h[4]
                smlal2 v28.4s, v18.8h, v4.h[5]
                smlal2 v29.4s, v18.8h, v5.h[5]
                smlal v28.4s, v19.4h, v4.h[6]
                smlal v29.4s, v19.4h, v5.h[6]
                smlal2 v28.4s, v19.8h, v4.h[7]
                smlal2 v29.4s, v19.8h, v5.h[7]
                smlal v30.4s, v18.4h, v6.h[4]
                smlal v31.4s, v18.4h, v7.h[4]
                st1 {v28.4s}, [x9], x6
                smlal2 v30.4s, v18.8h, v6.h[5]
                smlal2 v31.4s, v18.8h, v7.h[5]
                smlal v30.4s, v19.4h, v6.h[6]
                st1 {v29.4s}, [x9], x6
                smlal v31.4s, v19.4h, v7.h[6]
                smlal2 v30.4s, v19.8h, v6.h[7]
                smlal2 v31.4s, v19.8h, v7.h[7]

                st1 {v30.4s}, [x9], x6
                st1 {v31.4s}, [x9]

            subs x7, x7, #1
            add x0, x0, #16
            bne LoopKsize

        subs x5, x5, #1
        bne LoopOc

    ret
#endif
