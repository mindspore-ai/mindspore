#ifdef ENABLE_ARM32
#include "nnacl/assembly_global.h"
    .text
    .align 5
    .global MatmulFloatNeon32Opt12x4
#ifndef __APPLE__
    .type MatmulFloatNeon32Opt12x4, %function
#endif

// void MatmulFloatNeon32Opt12x4(const float *a, const float *b, float *c, const float *bias, int act_type, int depth
//                               int row, int col, size_t stride, size_t writeMode)
// r0: a
// r1: b
// r2: c
// r3: bias
// r4: act_type
// r5: depth
// r6: row
// r7: col
// r8: stride
// lr: OutType_C8 = 0, OutType_Nhwc = 1, OutType_TileC8 = 2

asm_function MatmulFloatNeon32Opt12x4
    // r4-r8 and q4-q7 must be saved according to https://static.docs.arm.com/ihi0042/i/aapcs32.pdf
    push {r0-r8, r10, r11, lr}
    vpush {q4-q7}
    add sp, sp, #112

    ldr r5, [sp, #4]
    ldr r6, [sp, #8]
    ldr r7, [sp, #12]
    ldr r8, [sp, #16]

    mov lr, #48 // sizeof(float) * 12
    mul r12, r5, lr // block stride of lhs: sizeof(float) * 12 * depth
    mov lr, #4
    mul r8, r8, lr // stride * sizeof(float)

LoopRowStart:
    cmp r6, #4
    ble LoopRow4
    cmp r6, #8
    ble LoopRow8

LoopRow:
    ldr r1, [sp, #-44] // reload rhs ptr
    ldr r7, [sp, #12] // reload rhs col
    ldr r3, [sp, #-36] // reload bias ptr

    LoopCol:
        ldr r2, [sp, #-40] // reload dst ptr
        ldr r0, [sp, #-48] // reload lhs ptr
        ldr r5, [sp, #4] // reload depth
        vld1.32 {q3}, [r1]!
        vld1.32 {q0, q1}, [r0]!
        vmul.f32 q4, q3, d0[0]
        vmul.f32 q5, q3, d0[1]
        vmul.f32 q6, q3, d1[0]
        vld1.32 {q2}, [r0]!
        vmul.f32 q7, q3, d1[1]

        vmul.f32 q8, q3, d2[0]
        vmul.f32 q9, q3, d2[1]
        vmul.f32 q10, q3, d3[0]
        vmul.f32 q11, q3, d3[1]

        vmul.f32 q12, q3, d4[0]
        vmul.f32 q13, q3, d4[1]
        vmul.f32 q14, q3, d5[0]
        vmul.f32 q15, q3, d5[1]

        subs r5, r5, #1
        beq Bias

        LoopDepth:
            vld1.32 {q3}, [r1]!
            vld1.32 {q0, q1}, [r0]!
            vmla.f32 q4, q3, d0[0]
            vmla.f32 q5, q3, d0[1]
            vmla.f32 q6, q3, d1[0]
            vld1.32 {q2}, [r0]!
            vmla.f32 q7, q3, d1[1]

            vmla.f32 q8, q3, d2[0]
            vmla.f32 q9, q3, d2[1]
            vmla.f32 q10, q3, d3[0]
            vmla.f32 q11, q3, d3[1]

            vmla.f32 q12, q3, d4[0]
            vmla.f32 q13, q3, d4[1]
            vmla.f32 q14, q3, d5[0]
            vmla.f32 q15, q3, d5[1]

            subs r5, r5, #1
            bne LoopDepth

        Bias:
            cmp r3, #0
            beq Activation
            vld1.32 {q0}, [r3]!
            vadd.f32 q4, q4, q0
            vadd.f32 q5, q5, q0
            vadd.f32 q6, q6, q0
            vadd.f32 q7, q7, q0
            vadd.f32 q8, q8, q0
            vadd.f32 q9, q9, q0
            vadd.f32 q10, q10, q0
            vadd.f32 q11, q11, q0
            vadd.f32 q12, q12, q0
            vadd.f32 q13, q13, q0
            vadd.f32 q14, q14, q0
            vadd.f32 q15, q15, q0

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
            vmin.f32 q4, q4, q2
            vmin.f32 q5, q5, q2
            vmin.f32 q6, q6, q2
            vmin.f32 q7, q7, q2
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
            vmax.f32 q4, q4, q3
            vmax.f32 q5, q5, q3
            vmax.f32 q6, q6, q3
            vmax.f32 q7, q7, q3
            vmax.f32 q8, q8, q3
            vmax.f32 q9, q9, q3
            vmax.f32 q10, q10, q3
            vmax.f32 q11, q11, q3
            vmax.f32 q12, q12, q3
            vmax.f32 q13, q13, q3
            vmax.f32 q14, q14, q3
            vmax.f32 q15, q15, q3
            b Write

LoopRow8:
    ldr r1, [sp, #-44] // reload rhs ptr
    ldr r7, [sp, #12] // reload rhs col
    ldr r3, [sp, #-36] // reload bias ptr

    LoopCol_R8:
        ldr r2, [sp, #-40] // reload dst ptr
        ldr r0, [sp, #-48] // reload lhs ptr
        ldr r5, [sp, #4] // reload depth
        vld1.32 {q3}, [r1]!
        vld1.32 {q0, q1}, [r0]!
        vmul.f32 q4, q3, d0[0]
        vmul.f32 q5, q3, d0[1]
        vmul.f32 q6, q3, d1[0]
        vld1.32 {q2}, [r0]!
        vmul.f32 q7, q3, d1[1]

        vmul.f32 q8, q3, d2[0]
        vmul.f32 q9, q3, d2[1]
        vmul.f32 q10, q3, d3[0]
        vmul.f32 q11, q3, d3[1]

        subs r5, r5, #1
        beq Bias_R8

        LoopDepth_R8:
            vld1.32 {q3}, [r1]!
            vld1.32 {q0, q1}, [r0]!
            vmla.f32 q4, q3, d0[0]
            vmla.f32 q5, q3, d0[1]
            vmla.f32 q6, q3, d1[0]
            vld1.32 {q2}, [r0]!
            vmla.f32 q7, q3, d1[1]

            vmla.f32 q8, q3, d2[0]
            vmla.f32 q9, q3, d2[1]
            vmla.f32 q10, q3, d3[0]
            vmla.f32 q11, q3, d3[1]

            subs r5, r5, #1
            bne LoopDepth_R8

        Bias_R8:
            cmp r3, #0
            beq Activation_R8
            vld1.32 {q0}, [r3]!
            vadd.f32 q4, q4, q0
            vadd.f32 q5, q5, q0
            vadd.f32 q6, q6, q0
            vadd.f32 q7, q7, q0
            vadd.f32 q8, q8, q0
            vadd.f32 q9, q9, q0
            vadd.f32 q10, q10, q0
            vadd.f32 q11, q11, q0

        Activation_R8:
            ldr lr, [sp]
            cmp lr, #3
            beq Relu6_R8
            cmp lr, #1
            beq Relu_R8
            b Write

        Relu6_R8:
            vmov.i32 q2, #6
            vcvt.f32.s32 q2, q2
            vmin.f32 q4, q4, q2
            vmin.f32 q5, q5, q2
            vmin.f32 q6, q6, q2
            vmin.f32 q7, q7, q2
            vmin.f32 q8, q8, q2
            vmin.f32 q9, q9, q2
            vmin.f32 q10, q10, q2
            vmin.f32 q11, q11, q2

        Relu_R8:
            veor q3, q3, q3
            vmax.f32 q4, q4, q3
            vmax.f32 q5, q5, q3
            vmax.f32 q6, q6, q3
            vmax.f32 q7, q7, q3
            vmax.f32 q8, q8, q3
            vmax.f32 q9, q9, q3
            vmax.f32 q10, q10, q3
            vmax.f32 q11, q11, q3
            b Write

LoopRow4:
    ldr r1, [sp, #-44] // reload rhs ptr
    ldr r7, [sp, #12] // reload rhs col
    ldr r3, [sp, #-36] // reload bias ptr

    LoopCol_R4:
        ldr r2, [sp, #-40] // reload dst ptr
        ldr r0, [sp, #-48] // reload lhs ptr
        ldr r5, [sp, #4] // reload depth
        vld1.32 {q3}, [r1]!
        vld1.32 {q0, q1}, [r0]!
        vmul.f32 q4, q3, d0[0]
        vmul.f32 q5, q3, d0[1]
        vmul.f32 q6, q3, d1[0]
        vld1.32 {q2}, [r0]!
        vmul.f32 q7, q3, d1[1]

        subs r5, r5, #1
        beq Bias_R4

        LoopDepth_R4:
            vld1.32 {q3}, [r1]!
            vld1.32 {q0, q1}, [r0]!
            vmla.f32 q4, q3, d0[0]
            vmla.f32 q5, q3, d0[1]
            vmla.f32 q6, q3, d1[0]
            vld1.32 {q2}, [r0]!
            vmla.f32 q7, q3, d1[1]

            subs r5, r5, #1
            bne LoopDepth_R4

        Bias_R4:
            cmp r3, #0
            beq Activation_R4
            vld1.32 {q0}, [r3]!
            vadd.f32 q4, q4, q0
            vadd.f32 q5, q5, q0
            vadd.f32 q6, q6, q0
            vadd.f32 q7, q7, q0

        Activation_R4:
            ldr lr, [sp]
            cmp lr, #3
            beq Relu6_R4
            cmp lr, #1
            beq Relu_R4
            b Write

        Relu6_R4:
            vmov.i32 q2, #6
            vcvt.f32.s32 q2, q2
            vmin.f32 q4, q4, q2
            vmin.f32 q5, q5, q2
            vmin.f32 q6, q6, q2
            vmin.f32 q7, q7, q2

        Relu_R4:
            veor q3, q3, q3
            vmax.f32 q4, q4, q3
            vmax.f32 q5, q5, q3
            vmax.f32 q6, q6, q3
            vmax.f32 q7, q7, q3

        Write:
            cmp r7, #1
            beq Write1
            cmp r7, #2
            beq Write2
            cmp r7, #3
            beq Write3
            b Write4

        Write1:
            add lr, r2, #4
            str lr, [sp, #-40]
            vst1.32 d8[0], [r2]
            cmp r6, #1
            beq WriteEnd
            add r2, r2, r8
            vst1.32 d10[0], [r2]
            cmp r6, #2
            beq WriteEnd
            add r2, r2, r8
            vst1.32 d12[0], [r2]
            cmp r6, #3
            beq WriteEnd
            add r2, r2, r8
            vst1.32 d14[0], [r2]
            cmp r6, #4
            beq WriteEnd
            add r2, r2, r8
            vst1.32 d16[0], [r2]
            cmp r6, #5
            beq WriteEnd
            add r2, r2, r8
            vst1.32 d18[0], [r2]
            cmp r6, #6
            beq WriteEnd
            add r2, r2, r8
            vst1.32 d20[0], [r2]
            cmp r6, #7
            beq WriteEnd
            add r2, r2, r8
            vst1.32 d22[0], [r2]
            cmp r6, #8
            beq WriteEnd
            add r2, r2, r8
            vst1.32 d24[0], [r2]
            cmp r6, #9
            beq WriteEnd
            add r2, r2, r8
            vst1.32 d26[0], [r2]
            cmp r6, #10
            beq WriteEnd
            add r2, r2, r8
            vst1.32 d28[0], [r2]
            cmp r6, #11
            beq WriteEnd
            add r2, r2, r8
            vst1.32 d30[0], [r2]
            add r2, r2, r8
            add r2, r2, #4
            b WriteEnd
        Write2:
            add lr, r2, #8
            str lr, [sp, #-40]
            vst1.32 d8, [r2]
            cmp r6, #1
            beq WriteEnd
            add r2, r2, r8
            vst1.32 d10, [r2]
            cmp r6, #2
            beq WriteEnd
            add r2, r2, r8
            vst1.32 d12, [r2]
            cmp r6, #3
            beq WriteEnd
            add r2, r2, r8
            vst1.32 d14, [r2]
            cmp r6, #4
            beq WriteEnd
            add r2, r2, r8
            vst1.32 d16, [r2]
            cmp r6, #5
            beq WriteEnd
            add r2, r2, r8
            vst1.32 d18, [r2]      
            cmp r6, #6
            beq WriteEnd
            add r2, r2, r8
            vst1.32 d20, [r2]      
            cmp r6, #7
            beq WriteEnd
            add r2, r2, r8
            vst1.32 d22, [r2]
            cmp r6, #8
            beq WriteEnd
            add r2, r2, r8
            vst1.32 d24, [r2]
            cmp r6, #9
            beq WriteEnd
            add r2, r2, r8
            vst1.32 d26, [r2]
            cmp r6, #10
            beq WriteEnd
            add r2, r2, r8
            vst1.32 d28, [r2]
            cmp r6, #11
            beq WriteEnd
            add r2, r2, r8
            vst1.32 d30, [r2]     
            add r2, r2, r8
            add r2, r2, #8
            b WriteEnd
        Write3:
            add lr, r2, #12
            str lr, [sp, #-40]
            add r4, r2, #8
            vst1.32 d8, [r2]
            vst1.32 d9[0], [r4]
            cmp r6, #1
            beq WriteEnd
            add r2, r2, r8
            add r4, r4, r8
            vst1.32 d10, [r2]
            vst1.32 d11[0], [r4]
            cmp r6, #2
            beq WriteEnd
            add r2, r2, r8
            add r4, r4, r8
            vst1.32 d12, [r2]
            vst1.32 d13[0], [r4]
            cmp r6, #3
            beq WriteEnd
            add r2, r2, r8
            add r4, r4, r8
            vst1.32 d14, [r2]
            vst1.32 d15[0], [r4]
            cmp r6, #4
            beq WriteEnd
            add r2, r2, r8
            add r4, r4, r8
            vst1.32 d16, [r2]
            vst1.32 d17[0], [r4] 
            cmp r6, #5
            beq WriteEnd
            add r2, r2, r8
            add r4, r4, r8
            vst1.32 d18, [r2]
            vst1.32 d19[0], [r4]
            cmp r6, #6
            beq WriteEnd
            add r2, r2, r8
            add r4, r4, r8
            vst1.32 d20, [r2]
            vst1.32 d21[0], [r4]
            cmp r6, #7
            beq WriteEnd
            add r2, r2, r8
            add r4, r4, r8
            vst1.32 d22, [r2]
            vst1.32 d23[0], [r4]
            cmp r6, #8
            beq WriteEnd
            add r2, r2, r8
            add r4, r4, r8
            vst1.32 d24, [r2]
            vst1.32 d25[0], [r4]
            cmp r6, #9
            beq WriteEnd
            add r2, r2, r8
            add r4, r4, r8
            vst1.32 d26, [r2]
            vst1.32 d27[0], [r4]
            cmp r6, #10
            beq WriteEnd
            add r2, r2, r8
            add r4, r4, r8
            vst1.32 d28, [r2]
            vst1.32 d29[0], [r4]
            cmp r6, #11
            beq WriteEnd
            add r2, r2, r8
            add r4, r4, r8
            vst1.32 d30, [r2]
            vst1.32 d31[0], [r4]
            add r2, r2, r8
            add r2, r2, #12
            b WriteEnd
        Write4:
            add lr, r2, #16
            str lr, [sp, #-40]
            vst1.32 q4, [r2]
            cmp r6, #1
            beq WriteEnd
            add r2, r2, r8
            vst1.32 q5, [r2]
            cmp r6, #2
            beq WriteEnd
            add r2, r2, r8
            vst1.32 q6, [r2]
            cmp r6, #3
            beq WriteEnd
            add r2, r2, r8
            vst1.32 q7, [r2]
            cmp r6, #4
            beq WriteEnd
            add r2, r2, r8
            vst1.32 q8, [r2]
            cmp r6, #5
            beq WriteEnd
            add r2, r2, r8
            vst1.32 q9, [r2]
            cmp r6, #6
            beq WriteEnd
            add r2, r2, r8
            vst1.32 q10, [r2]
            cmp r6, #7
            beq WriteEnd
            add r2, r2, r8
            vst1.32 q11, [r2]
            cmp r6, #8
            beq WriteEnd
            add r2, r2, r8
            vst1.32 q12, [r2]
            cmp r6, #9
            beq WriteEnd
            add r2, r2, r8
            vst1.32 q13, [r2]
            cmp r6, #10
            beq WriteEnd
            add r2, r2, r8
            vst1.32 q14, [r2]
            cmp r6, #11
            beq WriteEnd
            add r2, r2, r8
            vst1.32 q15, [r2]       
            add r2, r2, r8
            add r2, r2, #16
            b WriteEnd
        WriteEnd:
            cmp r7, #4
            ble LoopColEnd
            sub r7, r7, #4 // rhs col - 4
            b LoopCol

    LoopColEnd:
        ldr r0, [sp, #-48]
        add r0, r0, r12     // lhs ptr + stride
        str r0, [sp, #-48]
        mov lr, #4
        ldr r7, [sp, #12]   // reload rhs col
        mul lr, lr, r7
        sub r2, r2, lr
        str r2, [sp, #-40]
        cmp r6, #12
        ble LoopRowEnd
        sub r6, r6, #12 // lhs row - 12
        b LoopRowStart

LoopRowEnd:
    sub sp, sp, #112
    vpop {q4-q7}
    pop {r0-r8, r10, r11, pc}
#endif
