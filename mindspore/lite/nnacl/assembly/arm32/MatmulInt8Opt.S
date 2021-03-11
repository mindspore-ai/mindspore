#ifdef __arm__
#ifndef __aarch64__
#include "nnacl/assembly_global.h"

.text
.align 5
.global MatmulInt8Opt
#ifndef __APPLE__
.type MatmulInt8Opt, %function
#endif

//void MatmulInt8Neon32Opt(const int8_t *a, const int8_t *b, int8_t *dst, int row, int col, int deep16, 
//                         const int *input_sums, const int *weight_bias, int act_min, int act_max, int out_zp,
//                         int *multiplier, int *left_shift, int *right_shift, int stride, int per_channel,
//                         int *filter_zp);
// #-48: a, #-44: b, #-40: dst, #-36: row
// #0: col, #4: deep16, #8: input_sums, #12: weight_bias, #16: act_min, #20: act_max, #24: out_zp
// #28: multiplier, #32: left_shift, #36: right_shift, #40: stride, #44: per_channel, #48: filter_zp

asm_function MatmulInt8Opt
    push {r0-r8, r10, r11, lr}
    vpush {q4-q7}
    add sp, sp, #112

    ldr r5, [sp, #4]
    ldr r6, [sp, #8] // reload a_sums ptr
    ldr r8, [sp, #40]
    mov r10, #4
    mul r10, r10, r5 // lhs step
    mov r11, #4
    mul r11, r11, r8 // dst step
LoopRow:
    ldr r1, [sp, #-44] //reload rhs ptr
    ldr r4, [sp] // reload rhs col
    ldr lr, [sp, #-40]
    vmov.32 d4[0], lr // reload dst ptr
    ldr lr, [sp, #32]
    vmov.32 d4[1], lr // reload left shift
    ldr lr, [sp, #28]
    vmov.32 d5[0], lr // reload multiplier
    ldr lr, [sp, #36]
    vmov.32 d5[1], lr // reload right shift
    ldr r7, [sp, #48] // reload filter_zp
    ldr r12, [sp, #12] // reload bias ptr

    LoopCol:
        vmov.32 r2, d4[0] // reload dst ptr
        ldr r0, [sp, #-48] //reload lhs ptr
        ldr r5, [sp, #4] // reaload depth

        vmov.i32 q6, #0
        vmov.i32 q7, #0
        vmov.i32 q8, #0
        vmov.i32 q9, #0
        vmov.i32 q10, #0
        vmov.i32 q11, #0
        vmov.i32 q12, #0
        vmov.i32 q13, #0

        LoopDepth:
            vld1.8 {q0-q1}, [r0]!
            vld1.8 {q4-q5}, [r1]!
            vmull.s8 q14, d0, d8
            vmull.s8 q15, d2, d8
            vmlal.s8 q14, d1, d9
            vmlal.s8 q15, d3, d9
            vpadal.s16 q6, q14
            vpadal.s16 q8, q15
            vmull.s8 q14, d0, d10
            vmull.s8 q15, d2, d10
            vmlal.s8 q14, d1, d11
            vmlal.s8 q15, d3, d11
            vld1.8 {q0-q1}, [r0]!

            vpadal.s16 q7, q14
            vpadal.s16 q9, q15

            vmull.s8 q14, d0, d8
            vmull.s8 q15, d2, d8
            vmlal.s8 q14, d1, d9
            vmlal.s8 q15, d3, d9
            vpadal.s16 q10, q14
            vpadal.s16 q12, q15
            vmull.s8 q14, d0, d10
            vmull.s8 q15, d2, d10
            vmlal.s8 q14, d1, d11
            vmlal.s8 q15, d3, d11

            vpadal.s16 q11, q14
            vpadal.s16 q13, q15

            cmp r5, #16
            ble LoopDepthEnd
            sub r5, r5, #16
            b LoopDepth

    LoopDepthEnd:
        vpadd.i32 d12, d12, d13
        vpadd.i32 d14, d14, d15
        vpadd.i32 d16, d16, d17
        vpadd.i32 d18, d18, d19
        vpadd.i32 d20, d20, d21
        vpadd.i32 d22, d22, d23
        vpadd.i32 d24, d24, d25
        vpadd.i32 d26, d26, d27

        vpadd.i32 d28, d12, d14
        vpadd.i32 d29, d16, d18
        vpadd.i32 d30, d20, d22
        vpadd.i32 d31, d24, d26

    Bias:
        cmp r12, #0
        beq NoBias
        vld1.32 {d26}, [r12]!
        vadd.i32 d28, d28, d26
        vadd.i32 d29, d29, d26
        vadd.i32 d30, d30, d26
        vadd.i32 d31, d31, d26

    NoBias:
        ldr lr, [sp, #44]
        cmp lr, #0
        bne PerChannel

    PerTensor:
        vld1.32 {d24, d25}, [r6]
        vdup.32 d20, d24[0]
        vdup.32 d21, d24[1]
        vdup.32 d22, d25[0]
        vdup.32 d23, d25[1]
        vsub.s32 d28, d28, d20
        vsub.s32 d29, d29, d21
        vsub.s32 d30, d30, d22
        vsub.s32 d31, d31, d23

        vmov.32 lr, d4[1]
        vld1.32 {q9[]}, [lr]
        vshl.s32 q14, q14, q9
        vshl.s32 q15, q15, q9

        vmov.32 lr, d5[0]
        vld1.32 {q8[]}, [lr]
        vqrdmulh.s32 q14, q14, q8
        vqrdmulh.s32 q15, q15, q8

        vmov.32 lr, d5[1]
        vld1.32 {q7[]}, [lr]
        vand q6, q7, q14
        vshr.s32 q6, q6, #31
        vqadd.s32 q14, q14, q6
        vrshl.s32 q14, q14, q7
        vand q5, q7, q15
        vshr.s32 q5, q5, #31
        vqadd.s32 q15, q15, q5
        vrshl.s32 q15, q15, q7
        b Quantize

    PerChannel:
        vld1.32 {d24, d25}, [r6]
        vdup.32 d20, d24[0]
        vdup.32 d21, d24[1]
        vdup.32 d22, d25[0]
        vdup.32 d23, d25[1]
        vld1.32 {d19}, [r7]!
        vmul.s32 d24, d20, d19
        vmul.s32 d25, d21, d19
        vmul.s32 d26, d22, d19
        vmul.s32 d27, d23, d19
        vsub.s32 d28, d28, d24
        vsub.s32 d29, d29, d25
        vsub.s32 d30, d30, d26
        vsub.s32 d31, d31, d27

        vmov.32 lr, d4[1]
        vld1.32 {d23}, [lr]!
        vmov.32 d4[1], lr
        vshl.s32 d28, d28, d23
        vshl.s32 d29, d29, d23
        vshl.s32 d30, d30, d23
        vshl.s32 d31, d31, d23

        vmov.32 lr, d5[0]
        vld1.32 {d22}, [lr]!
        vmov.32 d5[0], lr
        vqrdmulh.s32 d28, d28, d22
        vqrdmulh.s32 d29, d29, d22
        vqrdmulh.s32 d30, d30, d22
        vqrdmulh.s32 d31, d31, d22

        vmov.32 lr, d5[1]
        vld1.32 {d21}, [lr]!
        vmov.32 d5[1], lr
        vand d20, d21, d28
        vshr.s32 d20, d20, #31
        vqadd.s32 d28, d28, d20
        vrshl.s32 d28, d28, d21
        vand d19, d21, d29
        vshr.s32 d19, d19, #31
        vqadd.s32 d29, d29, d19
        vrshl.s32 d29, d29, d21
        vand d18, d21, d30
        vshr.s32 d18, d18, #31
        vqadd.s32 d30, d30, d18
        vrshl.s32 d30, d30, d21
        vand d17, d21, d31
        vshr.s32 d17, d17, #31
        vqadd.s32 d31, d31, d17
        vrshl.s32 d31, d31, d21

    Quantize:
        ldr lr, [sp, #24]
        vdup.32 q0, lr
        vadd.i32 q14, q14, q0
        vadd.i32 q15, q15, q0

        ldr lr, [sp, #16]
        vdup.32 q1, lr
        vmax.s32 q14, q14, q1
        vmax.s32 q15, q15, q1

        ldr lr, [sp, #20]
        vdup.32 q0, lr
        vmin.s32 q14, q14, q0
        vmin.s32 q15, q15, q0

        vqmovn.s32 d28, q14
        vqmovn.s32 d29, q15

        vqmovn.s16 d30, q14

    cmp r4, #1
    beq Write1
    b Write2

    Write1:
        vmov.32 lr, d4[0]
        add lr, lr, #1
        vmov.32 d4[0], lr
        vst1.8 {d30[0]}, [r2], r8
        cmp r3, #1
        beq WriteEnd
        vst1.8 {d30[2]}, [r2], r8
        cmp r3, #2
        beq WriteEnd
        vst1.8 {d30[4]}, [r2], r8
        cmp r3, #3
        beq WriteEnd
        vst1.8 {d30[6]}, [r2], r8
        b WriteEnd

    Write2:
        vmov.32 lr, d4[0]
        add lr, lr, #2
        vmov.32 d4[0], lr
        vst1.16 {d30[0]}, [r2], r8
        cmp r3, #1
        beq WriteEnd
        vst1.16 {d30[1]}, [r2], r8
        cmp r3, #2
        beq WriteEnd
        vst1.16 {d30[2]}, [r2], r8
        cmp r3, #3
        beq WriteEnd
        vst1.16 {d30[3]}, [r2], r8

    WriteEnd:
        cmp r4, #2
        ble LoopColEnd
        sub r4, r4, #2
        b LoopCol

LoopColEnd:
    cmp r3, #4
    ble LoopRowEnd
    ldr lr, [sp, #-48]
    add lr, lr, r10
    str lr, [sp, #-48]
    ldr lr, [sp, #-40]
    add lr, lr, r11
    str lr, [sp, #-40]
    sub r3, r3, #4
    add r6, r6, #16
    b LoopRow

LoopRowEnd:
    sub sp, sp, #112
    vpop {q4-q7}
    pop {r0-r8, r10, r11, pc}
#endif
#endif
