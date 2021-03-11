#ifdef __aarch64__
#include "nnacl/assembly_global.h"

    .text
    .align 5
    .global WinogradTransLeft
#ifndef __APPLE__
    .type WinogradTransLeft, %function
#endif

asm_function WinogradTransLeft
//void WinogradTransLeft(const float* S, const float* B, float* M, size_t w, size_t h, size_t k, size_t length);
//x0: S
//x1: B
//x2: M
//x3: w
//x4: h
//x5: k
//x6:length

sub sp, sp, #32
stp x19, x20, [sp], #32

mov x8, #16 // 4 * sizeof(float)
mul x8, x6, x8
mul x9, x3, x8
sub x9, x9, x8
add x7, x9, x8 // step for S
mov x10, #4
mul x10, x4, x10 // step for B

LoopH:
    mov x13, x0
    mov x15, x3
    LoopW:
        mov x14, x13
        mov x17, x1
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
            mov x19, x4
        LoopK4:
            ld1 {v0.s}[0], [x17], x10
            ld1 {v0.s}[1], [x17], x10
            ld1 {v0.s}[2], [x17], x10
            ld1 {v0.s}[3], [x17], x10
            mov x11, x6
            mov x18, x17
            add x18, x14, x7
            add x16, x18, x7
            add x19, x16, x7

            LoopLength4:
                ld1 {v16.4s}, [x2]
                ld1 {v20.4s}, [x14], #16
                fmla v16.4s, v20.4s, v0.s[0]
                ld1 {v21.4s}, [x18], #16
                fmul v17.4s, v21.4s, v0.s[1]
                ld1 {v20.4s}, [x16], #16
                fmla v16.4s, v20.4s, v0.s[2]
                ld1 {v21.4s}, [x19], #16
                fmla v17.4s, v21.4s, v0.s[3]
                fadd v17.4s, v16.4s, v17.4s
                st1 {v17.4s}, [x2], #16
                subs x11, x11, #1
                bne LoopLength4

            sub x2, x2, x8
            sub x12, x12, #4
            add x14, x19, x9
            cmp x12, #4
            bge LoopK4

        LoopKStart3:
            cmp x12, #3
            blt LoopKStart
            mov x16, x15
        LoopK3:
            ld1 {v0.s}[0], [x17], x10
            ld1 {v0.s}[1], [x17], x10
            ld1 {v0.s}[2], [x17], x10
            mov x11, x6
            mov x18, x17
            add x18, x14, x7
            add x16, x18, x7
            LoopLength3:
                ld1 {v16.4s}, [x2]
                ld1 {v20.4s}, [x14], #16
                fmla v16.4s, v20.4s, v0.s[0]
                ld1 {v21.4s}, [x18], #16
                fmul v17.4s, v21.4s, v0.s[1]
                ld1 {v20.4s}, [x16], #16
                fmla v16.4s, v20.4s, v0.s[2]
                fadd v17.4s, v16.4s, v17.4s
                st1 {v17.4s}, [x2], #16
                subs x11, x11, #1
                bne LoopLength3

            sub x2, x2, x8
            sub x12, x12, #3
            add x14, x16, x9
            cmp x12, #3
            bge LoopK3

        LoopKStart:
            cmp x12, #0
            beq LKEnd
        LoopK:
            ld1r {v31.4s}, [x17], x10
            mov x11, x6
            LoopLength:
                ld1 {v0.4s}, [x2]
                ld1 {v1.4s}, [x14], #16
                fmla v0.4s, v1.4s, v31.4s
                st1 {v0.4s}, [x2], #16
                subs x11, x11, #1
                bne LoopLength

            subs x12, x12, #1
            sub x2, x2, x8
            add x14, x14, x9
            bne LoopK

        LKEnd:
        subs x15, x15, #1
        add x13, x13, x8
        add x2, x2, x8
        bne LoopW

    add x1, x1, #4 //sizeof(float)
    subs x4, x4, #1
    bne LoopH

    sub sp, sp, #32
    ldp x19, x20, [sp], #32
    ret

#endif
