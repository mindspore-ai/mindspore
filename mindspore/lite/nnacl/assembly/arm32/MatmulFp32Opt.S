#ifdef ENABLE_ARM32
#include "nnacl/assembly_global.h"
    .text
    .align 5
    .global MatmulFloatNeon32Opt
#ifndef __APPLE__
    .type MatmulFloatNeon32Opt, %function
#endif

// void MatmulFloatNeon32Opt(const float *a, const float *b, float *c, const float *bias, int act_type, int depth
//                        int row, int col, size_t stride, size_t writeMode)
// r0: a
// r1: b
// r2: c
// r3: bias
// r4: act_type
// r5: depth
// r6: row
// r7: col
// r8: stride
// lr: writeNhwc/writeWino

asm_function MatmulFloatNeon32Opt
    // r4-r8 and q4-q7 must be saved according to https://static.docs.arm.com/ihi0042/i/aapcs32.pdf
    push {r0-r8, r10, r11, lr}
    add sp, sp, #48

    ldr r5, [sp, #4]
    ldr r6, [sp, #8]
    ldr r7, [sp, #12]
    ldr r8, [sp, #16]

    mov lr, #16 // sizeof(float) * 4
    mul r12, r5, lr // block stride of lhs/rhs: sizeof(float) * 4 * depth
    ldr lr, [sp, #20]
    cmp lr, #0
    bne NoC8Steps
    mov lr, #32
    mul r10, r6, lr
NoC8Steps:
    cmp lr, #2
    bne NoWinoSteps
    mov lr, #4
    mul r11, r7, r8 // stride * col * sizeof(float)
    mul r11, r11, lr
    mov lr, #32
    mul r10, r8, lr // stride * 8 * sizeof(float)
NoWinoSteps:
    mov lr, #4
    mul r8, r8, lr // stride * sizeof(float)

LoopRow:
    ldr r1, [sp, #-44] // reload rhs ptr
    ldr r7, [sp, #12] // reload rhs col
    ldr r3, [sp, #-36] // reload bias ptr

    LoopCol:
        ldr lr, [sp, #20]
        cmp lr, #0
        beq NoReloadDst
        ldr r2, [sp, #-40] // reload dst ptr
    NoReloadDst:
        ldr r0, [sp, #-48] // reload lhs ptr
        ldr r5, [sp, #4] // reload depth
        vld1.32 {q0}, [r0]!
        vld1.32 {q1, q2}, [r1]!
        vmul.f32 q8, q1, d0[0]
        vmul.f32 q9, q2, d0[0]
        vmul.f32 q10, q1, d0[1]
        vmul.f32 q11, q2, d0[1]
        vmul.f32 q12, q1, d1[0]
        vmul.f32 q13, q2, d1[0]
        vmul.f32 q14, q1, d1[1]
        vmul.f32 q15, q2, d1[1]

        subs r5, r5, #1
        beq Bias

        LoopDepth:
            vld1.32 {q0}, [r0]!
            vld1.32 {q1, q2}, [r1]!
            vmla.f32 q8, q1, d0[0]
            vmla.f32 q9, q2, d0[0]
            vmla.f32 q10, q1, d0[1]
            vmla.f32 q11, q2, d0[1]
            vmla.f32 q12, q1, d1[0]
            vmla.f32 q13, q2, d1[0]
            vmla.f32 q14, q1, d1[1]
            vmla.f32 q15, q2, d1[1]

            subs r5, r5, #1
            bne LoopDepth

        Bias:
            cmp r3, #0
            beq Activation
            vld1.32 {q0}, [r3]!
            vld1.32 {q1}, [r3]!
            vadd.f32 q8, q8, q0
            vadd.f32 q9, q9, q1
            vadd.f32 q10, q10, q0
            vadd.f32 q11, q11, q1
            vadd.f32 q12, q12, q0
            vadd.f32 q13, q13, q1
            vadd.f32 q14, q14, q0
            vadd.f32 q15, q15, q1

        Activation:
            ldr lr, [sp]
            cmp lr, #3
            beq Relu6
            cmp lr, #1
            beq Relu
            b Write

        Relu6:
            vmov.i32 q2, #6
            vcvt.f32.s32 q2, q2
            vmin.f32 q8, q8, q2
            vmin.f32 q9, q9, q2
            vmin.f32 q10, q10, q2
            vmin.f32 q11, q11, q2
            vmin.f32 q12, q12, q2
            vmin.f32 q13, q13, q2
            vmin.f32 q14, q14, q2
            vmin.f32 q15, q15, q2

        Relu:
            veor q3, q3, q3
            vmax.f32 q8, q8, q3
            vmax.f32 q9, q9, q3
            vmax.f32 q10, q10, q3
            vmax.f32 q11, q11, q3
            vmax.f32 q12, q12, q3
            vmax.f32 q13, q13, q3
            vmax.f32 q14, q14, q3
            vmax.f32 q15, q15, q3

        Write:
            ldr lr, [sp, #20]
            cmp lr, #2
            beq WriteWino
            cmp lr, #0
            beq WriteC8
            cmp r7, #1
            beq Write1
            cmp r7, #2
            beq Write2
            cmp r7, #3
            beq Write3
            cmp r7, #4
            beq Write4
            cmp r7, #5
            beq Write5
            cmp r7, #6
            beq Write6
            cmp r7, #7
            beq Write7
            b Write8

        Write1:
            add lr, r2, #4
            str lr, [sp, #-40]
            vst1.32 d16[0], [r2]
            cmp r6, #1
            beq WriteEnd
            add r2, r2, r8
            vst1.32 d20[0], [r2]
            cmp r6, #2
            beq WriteEnd
            add r2, r2, r8
            vst1.32 d24[0], [r2]
            cmp r6, #3
            beq WriteEnd
            add r2, r2, r8
            vst1.32 d28[0], [r2]
            add r2, r2, r8
            add r2, r2, #4
            b WriteEnd
        Write2:
            add lr, r2, #8
            str lr, [sp, #-40]
            vst1.32 d16, [r2]
            cmp r6, #1
            beq WriteEnd
            add r2, r2, r8
            vst1.32 d20, [r2]
            cmp r6, #2
            beq WriteEnd
            add r2, r2, r8
            vst1.32 d24, [r2]
            cmp r6, #3
            beq WriteEnd
            add r2, r2, r8
            vst1.32 d28, [r2]
            add r2, r2, r8
            add r2, r2, #8
            b WriteEnd
        Write3:
            add lr, r2, #12
            str lr, [sp, #-40]
            add r4, r2, #8
            vst1.32 d16, [r2]
            vst1.32 d17[0], [r4]
            cmp r6, #1
            beq WriteEnd
            add r2, r2, r8
            add r4, r4, r8
            vst1.32 d20, [r2]
            vst1.32 d21[0], [r4]
            cmp r6, #2
            beq WriteEnd
            add r2, r2, r8
            add r4, r4, r8
            vst1.32 d24, [r2]
            vst1.32 d25[0], [r4]
            cmp r6, #3
            beq WriteEnd
            add r2, r2, r8
            add r4, r4, r8
            vst1.32 d28, [r2]
            vst1.32 d29[0], [r4]
            add r2, r2, r8
            add r2, r2, #12
            b WriteEnd
        Write4:
            add lr, r2, #16
            str lr, [sp, #-40]
            vst1.32 q8, [r2]
            cmp r6, #1
            beq WriteEnd
            add r2, r2, r8
            vst1.32 q10, [r2]
            cmp r6, #2
            beq WriteEnd
            add r2, r2, r8
            vst1.32 q12, [r2]
            cmp r6, #3
            beq WriteEnd
            add r2, r2, r8
            vst1.32 q14, [r2]
            add r2, r2, r8
            add r2, r2, #16
            b WriteEnd
        Write5:
            add lr, r2, #20
            str lr, [sp, #-40]
            add r4, r2, #16
            vst1.32 q8, [r2]
            vst1.32 d18[0], [r4]
            cmp r6, #1
            beq WriteEnd
            add r2, r2, r8
            add r4, r4, r8
            vst1.32 q10, [r2]
            vst1.32 d22[0], [r4]
            cmp r6, #2
            beq WriteEnd
            add r2, r2, r8
            add r4, r4, r8
            vst1.32 q12, [r2]
            vst1.32 d26[0], [r4]
            cmp r6, #3
            beq WriteEnd
            add r2, r2, r8
            add r4, r4, r8
            vst1.32 q14, [r2]
            vst1.32 d30[0], [r4]
            add r2, r2, r8
            add r2, r2, #20
            b WriteEnd
        Write6:
            add lr, r2, #24
            str lr, [sp, #-40]
            add r4, r2, #16
            vst1.32 q8, [r2]
            vst1.32 d18, [r4]
            cmp r6, #1
            beq WriteEnd
            add r2, r2, r8
            add r4, r4, r8
            vst1.32 q10, [r2]
            vst1.32 d22, [r4]
            cmp r6, #2
            beq WriteEnd
            add r2, r2, r8
            add r4, r4, r8
            vst1.32 q12, [r2]
            vst1.32 d26, [r4]
            cmp r6, #3
            beq WriteEnd
            add r2, r2, r8
            add r4, r4, r8
            vst1.32 q14, [r2]
            vst1.32 d30, [r4]
            add r2, r2, r8
            add r2, r2, #24
            b WriteEnd
        Write7:
            add lr, r2, #28
            str lr, [sp, #-40]
            add lr, r2, #24
            add r4, r2, #16
            vst1.32 q8, [r2]
            vst1.32 d18, [r4]
            vst1.32 d19[0], [lr]
            cmp r6, #1
            beq WriteEnd
            add r2, r2, r8
            add r4, r4, r8
            add lr, lr, r8
            vst1.32 q10, [r2]
            vst1.32 d22, [r4]
            vst1.32 d23[0], [lr]
            cmp r6, #2
            beq WriteEnd
            add r2, r2, r8
            add r4, r4, r8
            add lr, lr, r8
            vst1.32 q12, [r2]
            vst1.32 d26, [r4]
            vst1.32 d27[0], [lr]
            cmp r6, #3
            beq WriteEnd
            add r2, r2, r8
            add r4, r4, r8
            add lr, lr, r8
            vst1.32 q14, [r2]
            vst1.32 d30, [r4]
            vst1.32 d31[0], [lr]
            add r2, r2, r8
            add r2, r2, #28
            b WriteEnd
        WriteC8:
            mov lr, r2
            vst1.32 {q8, q9}, [lr]!
            vst1.32 {q10, q11}, [lr]!
            vst1.32 {q12, q13}, [lr]!
            vst1.32 {q14, q15}, [lr]!
            add r2, r2, r10
            b WriteEnd
        WriteWino:
            add lr, r2, r10
            vst1.32 {q8, q9}, [r2]
            add r2, r2, r11
            vst1.32 {q10, q11}, [r2]
            add r2, r2, r11
            vst1.32 {q12, q13}, [r2]
            add r2, r2, r11
            vst1.32 {q14, q15}, [r2]
            str lr, [sp, #-40]
            b WriteEnd
        Write8:
            add lr, r2, #32
            str lr, [sp, #-40]
            vst1.32 {q8, q9}, [r2]
            cmp r6, #1
            beq WriteEnd
            add r2, r2, r8
            vst1.32 {q10, q11}, [r2]
            cmp r6, #2
            beq WriteEnd
            add r2, r2, r8
            vst1.32 {q12, q13}, [r2]
            cmp r6, #3
            beq WriteEnd
            add r2, r2, r8
            vst1.32 {q14, q15}, [r2]
            add r2, r2, r8
            add r2, r2, #32

        WriteEnd:
            cmp r7, #8
            ble LoopColEnd
            sub r7, r7, #8 // rhs col - 8
            b LoopCol

    LoopColEnd:
        ldr r0, [sp, #-48]
        add r0, r0, r12 // rhs ptr + stride
        str r0, [sp, #-48]
        ldr lr, [sp, #20]
        cmp lr, #0
        beq C8DstStep
        cmp lr, #2
        beq WinoDstStep
        mov lr, #4
        ldr r7, [sp, #12] // reload rhs col
        mul lr, lr, r7
        sub r2, r2, lr
        str r2, [sp, #-40]
        b NoDstStep
    C8DstStep:
        ldr lr, [sp, #-40]
        add r2, lr, #128
        str r2, [sp, #-40]
        b NoDstStep
    WinoDstStep:
        add r2, r2, r10
        str r2, [sp, #-40]
    NoDstStep:
        cmp r6, #4
        ble LoopRowEnd
        sub r6, r6, #4 // lhs row - 4
        b LoopRow

LoopRowEnd:
    sub sp, sp, #48
    pop {r0-r8, r10, r11, pc}
#endif
