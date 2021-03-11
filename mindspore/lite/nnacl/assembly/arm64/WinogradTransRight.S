#ifdef __aarch64__
#include "nnacl/assembly_global.h"

    .text
    .align 5
    .global WinogradTransRight
#ifndef __APPLE__
    .type WinogradTransRight, %function
#endif

asm_function WinogradTransRight
//void WinogradTransRight(const float* S, const float* B, float* M, size_t w, size_t h, size_t k, size_t length);
//x0: S
//x1: B
//x2: M
//x3: w
//x4: h
//x5: k
//x6: length

mov x8, #16 // 4 * sizeof(float)
mul x8, x6, x8
mul x9, x5, x8 // step for S
mov x10, #4
mul x10, x4, x10 // step for B

LoopH:
    mov x7, x1
    mov x15, x3
    LoopW:
        mov x17, x0
        mov x13, x7
        dup v30.4s, wzr
        mov x11, x6
        InitZero:
            st1 {v30.4s}, [x2], #16
            subs x11, x11, #1
            bne InitZero
        sub x2, x2, x8
        mov x12, x5

        LoopKStart4:
            cmp x12, #4
            blt LoopKStart3
            mov x16, x15
            mov x18, x4
            LoopK4:
                ld1 {v0.s}[0], [x13], x10
                ld1 {v0.s}[1], [x13], x10
                ld1 {v0.s}[2], [x13], x10
                ld1 {v0.s}[3], [x13], x10
                mov x11, x6
                mov x14, x13

                add x14, x17, x8
                add x16, x14, x8
                add x18, x16, x8

                LoopLength4:
                    ld1 {v16.4s}, [x2]
                    ld1 {v20.4s}, [x17], #16
                    fmla v16.4s, v20.4s, v0.s[0]
                    ld1 {v21.4s}, [x14], #16
                    fmul v17.4s, v21.4s, v0.s[1]
                    ld1 {v20.4s}, [x16], #16
                    fmla v16.4s, v20.4s, v0.s[2]
                    ld1 {v21.4s}, [x18], #16
                    fmla v17.4s, v21.4s, v0.s[3]

                    fadd v17.4s, v16.4s, v17.4s
                    st1 {v17.4s}, [x2], #16
                    subs x11, x11, #1
                    bne LoopLength4
                sub x2, x2, x8
                sub x12, x12, #4
                mov x17, x18

                cmp x12, #4
                bge LoopK4

        LoopKStart3:
            cmp x12, #3
            blt LoopKStart
            mov x16, x15
            LoopK3:
                ld1 {v0.s}[0], [x13], x10
                ld1 {v0.s}[1], [x13], x10
                ld1 {v0.s}[2], [x13], x10
                mov x11, x6
                mov x14, x13

                add x14, x17, x8
                add x16, x14, x8

                LoopLength3:
                    ld1 {v16.4s}, [x2]
                    ld1 {v20.4s}, [x17], #16
                    fmla v16.4s, v20.4s, v0.s[0]
                    ld1 {v21.4s}, [x14], #16
                    fmul v17.4s, v21.4s, v0.s[1]
                    ld1 {v20.4s}, [x16], #16
                    fmla v16.4s, v20.4s, v0.s[2]

                    fadd v17.4s, v16.4s, v17.4s
                    st1 {v17.4s}, [x2], #16
                    subs x11, x11, #1
                    bne LoopLength3
                sub x2, x2, x8
                sub x12, x12, #3
                mov x17, x18
                cmp x12, #3
                bge LoopK3

        LoopKStart:
            cmp x12, #0
            beq LoopKEnd

            LoopK:
                ld1r {v31.4s}, [x13], x10

                mov x11, x6
                LoopLength:
                    ld1 {v0.4s}, [x2]
                    ld1 {v1.4s}, [x17], #16
                    fmla v0.4s, v1.4s, v31.4s

                    st1 {v0.4s}, [x2], #16
                    subs x11, x11, #1
                    bne LoopLength
                    subs x12, x12, #1

                sub x2, x2, x8
                bne LoopK
        LoopKEnd:
            subs x15, x15, #1
            add x2, x2, x8
            add x7, x7, #4 //sizeof(float)
            bne LoopW

    add x0, x0, x9
    subs x4, x4, #1
    bne LoopH

    ret
#endif
