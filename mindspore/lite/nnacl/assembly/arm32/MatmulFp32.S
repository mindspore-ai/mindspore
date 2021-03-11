#ifdef ENABLE_ARM32
#include "nnacl/assembly_global.h"
    .text
    .align 5
    .global MatmulFloatNeon32
#ifndef __APPLE__
    .type MatmulFloatNeon32, %function
#endif

// void MatmulFloatNeon32(const float *a, const float *b, float *c, const float *bias, int act_type, int depth
//                        int row, int col, size_t stride, size_t writeNhwc, size_t WriteWino)
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

asm_function MatmulFloatNeon32
    // r4-r8 and q4-q7 must be saved according to https://static.docs.arm.com/ihi0042/i/aapcs32.pdf
    push {r0-r8, r10, r11, lr}
    add sp, sp, #48

    ldr r5, [sp, #4]
    ldr r7, [sp, #12]
    ldr r8, [sp, #16]

    mov lr, #32 // sizeof(float) * 8
    mul r12, r5, lr // block stride of lhs/rhs: sizeof(float) * 8 * depth
    ldr lr, [sp, #24]
    cmp lr, #0
    beq NoWinoSteps
    mov lr, #4
    mul r11, r7, r8 // stride * col * sizeof(float)
    mul r11, r11, lr
    mov lr, #32
    mul r10, r8, lr // stride * 8 * sizeof(float)
NoWinoSteps:
    mov lr, #4
    mul r8, r8, lr // stride * sizeof(float)

LoopCol:
    ldr r6, [sp, #8] // reload lhs row
    ldr r0, [sp, #-48] // reload lhs ptr
    ldr r2, [sp, #-40] // reload dst ptr

    LoopRow:
        ldr r1, [sp, #-44] // reload rhs ptr
        ldr r5, [sp, #4] // reload depth
        veor q8, q8, q8
        veor q9, q9, q9
        veor q10, q10, q10
        veor q11, q11, q11
        veor q12, q12, q12
        veor q13, q13, q13
        veor q14, q14, q14
        veor q15, q15, q15

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
            vld1.32 {q1}, [r3]
            sub r3, r3, #16
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
            ldr lr, [sp, #24]
            cmp lr, #0
            bne WriteWino
            ldr lr, [sp, #20]
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
            b WriteEnd
        Write2:
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
            b WriteEnd
        Write3:
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
            b WriteEnd
        Write4:
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
            b WriteEnd
        Write5:
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
            b WriteEnd
        Write6:
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
            b WriteEnd
        Write7:
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
            b WriteEnd
        WriteC8:
            vst1.32 {q8, q9}, [r2]!
            vst1.32 {q10, q11}, [r2]!
            vst1.32 {q12, q13}, [r2]!
            vst1.32 {q14, q15}, [r2]!
            str r2, [sp, #-40]
            b WriteEnd
        WriteWino:
            vst1.32 {q8, q9}, [r2]
            add r2, r2, r11
            vst1.32 {q10, q11}, [r2]
            add r2, r2, r11
            vst1.32 {q12, q13}, [r2]
            add r2, r2, r11
            vst1.32 {q14, q15}, [r2]
            add r2, r2, r11
            b WriteEnd
        Write8:
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

        WriteEnd:
            cmp r6, #4
            ble LoopRowEnd
            sub r6, r6, #4 // lhs row - 4
            b LoopRow

    LoopRowEnd:
        ldr r1, [sp, #-44]
        add r1, r1, r12 // rhs ptr + stride
        str r1, [sp, #-44]
        cmp r3, #0
        beq NoBiasStep
        add r3, r3, #32 // bias ptr + stride
    NoBiasStep:
        ldr lr, [sp, #24]
        cmp lr, #0
        bne WinoDstStep
        ldr lr, [sp, #20]
        cmp lr, #0
        beq NoDstStep
        ldr r2, [sp, #-40]
        add r2, r2, #32 // dst ptr + stride
        str r2, [sp, #-40]
        b NoDstStep
    WinoDstStep:
        ldr r2, [sp, #-40]
        add r2, r2, r10
        str r2, [sp, #-40]
    NoDstStep:
        cmp r7, #8
        ble LoopColEnd
        sub r7, r7, #8 // rhs col - 8
        b LoopCol

LoopColEnd:
    sub sp, sp, #48
    pop {r0-r8, r10, r11, pc}
#endif
