#ifdef __aarch64__
#include "nnacl/assembly_global.h"

.text
.align 5
.global MatrixMultiplyWinograd
#ifndef __APPLE__
.type MatrixMultiplyWinograd, %function
#endif

// MatrixMultiplyWinograd(float *matix_a, float *matrix_b, float *matrix_c, int m, int k, int n, int in_channel, int c4_channel)
               // x0: matrix_a, x1: matrix_b, x2: matrix_c, x3: m, x4: k, x5: n, x6: in_channel, x7: c4_channel
asm_function MatrixMultiplyWinograd
    // registers v8 ~ v15 must be preserved by a callee across subroutine calls, according to
    // https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst#simd-and-floating-point-registers
    // x19 ~ x29 should be also preserved
    // whereas our coding style do not permit such amount of parameters
    sub sp, sp, #48
    st1 {v8.4s}, [sp], #16
    stp x19, x20, [sp], #16
    stp x21, x22, [sp], #16
    mov x8, #4
    mul x10, x5, x8
    mov x17, x3  // m
    mul x13, x6, x8   // in_channel * 4
    mul x21, x13, x4  // in_channel * k

    LoopM:
        mov x15, x5 // n
        mov x14, x1  // mat_b
        LoopN:
            mov x16, x0  // mat_a_m
            sub x18, x5, x15   // ni
            sub x19, x17, x3   // mi
            mul x18, x18, x17  // ni * m
            mov x11, x6 // in_channel
            add x18, x18, x19  // (ni * m) + mi
            mul x18, x18, x7   // x18 * c4_channel
            add x20, x2, x18   // dst + offset
            cmp x11, #16
            bge LoopC16
            cmp x11, #8
            bge LoopC8
            cmp x11, #4
            bge LoopC4
            cmp x11, #1
            bge LoopC
            b EndLoopC
            LoopC16:
                mov x12, x14
                mov x9, x4  // new_k
                dup v5.4s, wzr
                dup v6.4s, wzr
                dup v7.4s, wzr
                dup v8.4s, wzr
                LoopK16:
                    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x16], x13
                    ldr s4, [x12]
                    add x12, x12, x10
                    fmla v5.4s, v0.4s, v4.s[0]
                    fmla v6.4s, v1.4s, v4.s[0]
                    fmla v7.4s, v2.4s, v4.s[0]
                    fmla v8.4s, v3.4s, v4.s[0]
                    subs x9, x9, #1
                    bne LoopK16
                Write16:
                    st1 {v5.4s}, [x20], #16
                    st1 {v6.4s}, [x20], #16
                    st1 {v7.4s}, [x20], #16
                    st1 {v8.4s}, [x20], #16

                sub x16, x16, x21  // back x13 * k
                add x16, x16, #64 // add 64B
                subs x11, x11, #16
                beq EndLoopC
                cmp x11, #16
                bge LoopC16
                cmp x11, #8
                bge LoopC8
                cmp x11, #4
                bge LoopC4
                cmp x11, #1
                bge LoopC

            LoopC8:
                dup v5.4s, wzr
                dup v6.4s, wzr
                mov x9, x4  // new_k
                mov x12, x14
                LoopK8:
                    ld1 {v0.4s, v1.4s}, [x16], x13
                    ldr s4, [x12]
                    add x12, x12, x10
                    fmla v5.4s, v0.4s, v4.s[0]
                    fmla v6.4s, v1.4s, v4.s[0]
                    subs x9, x9, #1
                    bne LoopK8
                Write8:
                    st1 {v5.4s}, [x20], #16
                    st1 {v6.4s}, [x20], #16

                sub x16, x16, x21  // back x13 * k
                add x16, x16, #32 // add 64B
                subs x11, x11, #8
                beq EndLoopC
                cmp x11, #8
                bge LoopC8
                cmp x11, #4
                bge LoopC4
                cmp x11, #1
                bge LoopC

            LoopC4:
                dup v5.4s, wzr
                mov x9, x4  // new_k
                mov x12, x14
                LoopK4:
                    ld1 {v0.4s}, [x16], x13
                    ldr s4, [x12]
                    add x12, x12, x10
                    fmla v5.4s, v0.4s, v4.s[0]
                    subs x9, x9, #1
                    bne LoopK4
                Write4:
                    st1 {v5.4s}, [x20], #16

                sub x16, x16, x21  // ptr back x13 * k
                add x16, x16, #16 // add 16B
                subs x11, x11, #4
                beq EndLoopC
                cmp x11, #4
                bge LoopC4
                cmp x11, #1
                bge LoopC

            LoopC:
                dup v5.4s, wzr
                mov x9, x4  // new_k
                mov x12, x14
                LoopK:
                    ldr s0, [x16]
                    add x16, x16, x13
                    ldr s4, [x12]
                    add x12, x12, x10
                    fmla v5.4s, v0.4s, v4.s[0]
                    subs x9, x9, #1
                    bne LoopK
                Write1:
                    str s5, [x20], #4

                sub x16, x16, x21  // ptr back x13 * k
                add x16, x16, #4 // ptr add 4B
                subs x11, x11, #1
                beq EndLoopC
                b LoopC

            EndLoopC:
                add x14, x14, #4
                subs x15, x15, #1
                beq EndLoopN
                b LoopN
        EndLoopN:
            subs x3, x3, #1
            beq EndLoopM
            add x0, x0, x21
            b LoopM
    EndLoopM:
        sub sp, sp, #48
        ld1 {v8.4s}, [sp], #16
        ldp x19, x20, [sp], #16
        ldp x21, x22, [sp], #16
    ret
#endif
