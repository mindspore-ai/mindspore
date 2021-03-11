#ifdef __arm__
#ifndef __aarch64__
#include "nnacl/assembly_global.h"

.text
.align 5
.global IndirectGemmInt8_2x4
#ifndef __APPLE__
.type IndirectGemmInt8_2x4, %function
#endif

// void IndirectGemmInt8_2x4(int8_t *output, int8_t *input, int8_t *weight, int32_t *bias, size_t ksize, size_t ic4,
// size_t oc, size_t offset, int32_t *input_sum, size_t act_min, size_t act_max, size_t out_zp, int32_t *out_multiplier,
// int32_t *shift_before, int32_t *shift_after, size_t asymmetric, size_t per_channel, size_t per_channel_offset);
// r0: output, r1: input, r2: weight, r3: bias, r4: kSize, r5: ic4, r6: oc, r7: offset
// r8: input_sum, r10: act_min, r11: act_max, r10: out_zp, r11: out_multiplier, r10: shift_before, r11: shift_after
asm_function IndirectGemmInt8_2x4

    .macro INIT_BIAS
        veor q10, q10, q10
        veor q11, q11, q11
        veor q12, q12, q12
        veor q13, q13, q13
        veor q14, q14, q14
        veor q15, q15, q15
    .endm

    // at return, clang generates "push {lr}, pop {pc}"" while gcc will generate "bx lr"
    // according to https://stackoverflow.com/questions/53625807
    // even if we jump to link register instead of saving it, we still have to save it in subroutine calls anyway
    // clang's rule seems more simple, though there are no subroutine calls here
    // r4-r8 and q4-q7 must be saved according to https://static.docs.arm.com/ihi0042/i/aapcs32.pdf
    push {r4-r8, r10, r11, lr}
    vpush {q4-q7}
    add sp, sp, #96

    ldr r4, [sp]
    ldr r5, [sp, #4]
    ldr r6, [sp, #8]
    ldr r7, [sp, #12]

    mul r5, r4, r5
    mov r4, #1

    LoopOc:

        mov r8, r4
        mov r12, r1

        LoopKsize:
            INIT_BIAS
            mov r11, r0
            
            // as some processors do not support sdot intrinsic, we use instruction word
            // dp support is stilled judged dymaticly, instruction word is just used to ensure compilation
            // according to https://static.docs.arm.com/ddi0596/g/ISA_A64_xml_v86A-2020-03_OPT.pdf
            // the instruction word of sdot vd.4s, vn.16b, vm.4b[index] is
            // 0100 1111 10Lm mmmm 1110 H0nn nnnd dddd
            // mmmmm/nnnnn/ddddd is the number of neon register, HL is the high/low bit of index

            // load input for output 1-2
            vld1.8 {q0, q1}, [r12]!
            // load weight for oc 1-2
            vld1.8 {q2, q3}, [r2]!
            vmull.s8 q6, d0, d4
            vmull.s8 q7, d0, d6
            vmlal.s8 q6, d1, d5
            vmlal.s8 q7, d1, d7
            vpaddl.s16 q8, q6
            vpaddl.s16 q9, q7
            // load weight for oc 3-4
            vld1.8 {q4, q5}, [r2]!
            vmull.s8 q6, d0, d8
            vmull.s8 q7, d0, d10
            vmlal.s8 q6, d1, d9
            vmlal.s8 q7, d1, d11

            subs r10, r5, #1
            beq LoopIcEnd

            LoopIc:
                // load input for output 1
                vld1.8 {q0}, [r12]!
                vpadal.s16 q10, q6
                vpadal.s16 q11, q7
                vmull.s8 q6, d2, d4
                vmull.s8 q7, d2, d6
                vmlal.s8 q6, d3, d5
                vmlal.s8 q7, d3, d7
                vld1.8 {q2, q3}, [r2]!
                vpadal.s16 q12, q6
                vpadal.s16 q13, q7
                vmull.s8 q6, d2, d8
                vmull.s8 q7, d2, d10
                vmlal.s8 q6, d3, d9
                vmlal.s8 q7, d3, d11
                vld1.8 {q4, q5}, [r2]!
                vpadal.s16 q14, q6
                vpadal.s16 q15, q7
                vmull.s8 q6, d0, d4
                vmull.s8 q7, d0, d6
                vmlal.s8 q6, d1, d5
                vmlal.s8 q7, d1, d7
                vld1.8 {q1}, [r12]!
                vpadal.s16 q8, q6
                vpadal.s16 q9, q7
                vmull.s8 q6, d0, d8
                vmull.s8 q7, d0, d10
                vmlal.s8 q6, d1, d9
                vmlal.s8 q7, d1, d11       

                subs r10, r10, #1
                bne LoopIc

            LoopIcEnd:
                vpadal.s16 q10, q6
                vpadal.s16 q11, q7
                vmull.s8 q6, d2, d4
                vmull.s8 q7, d2, d6
                vmlal.s8 q6, d3, d5
                vmlal.s8 q7, d3, d7
                vpadal.s16 q12, q6
                vpadal.s16 q13, q7
                vmull.s8 q6, d2, d8
                vmull.s8 q7, d2, d10
                vmlal.s8 q6, d3, d9
                vmlal.s8 q7, d3, d11
                vpadal.s16 q14, q6
                vpadal.s16 q15, q7

                // pairwise add
                vpadd.i32 d16, d16, d17
                vpadd.i32 d18, d18, d19
                vpadd.i32 d20, d20, d21
                vpadd.i32 d22, d22, d23
                vpadd.i32 d24, d24, d25
                vpadd.i32 d26, d26, d27
                vpadd.i32 d28, d28, d29
                vpadd.i32 d30, d30, d31

                vpadd.i32 d16, d16, d18
                vpadd.i32 d17, d20, d22
                vpadd.i32 d24, d24, d26
                vpadd.i32 d25, d28, d30

                // load sum
                ldr lr, [sp, #44]
                cmp lr, #0
                beq NoSum
                ldr r10, [sp, #16]
                ldr lr, [sp, #48]
                cmp lr, #0
                beq SymSum
                ldr lr, [sp, #52]
                vld1.32 q0, [r10]
                add r10, r10, lr
                vld1.32 q1, [r10]
                b AddSum
            SymSum:
                vld1.32 q0[], [r10]!
                vld1.32 q1[], [r10]!
            AddSum:
                vsub.i32 q8, q8, q0
                vsub.i32 q12, q12, q1
            NoSum:
                cmp r3, #0
                beq NoBias
                vld1.32 q2, [r3]
                vadd.i32 q8, q8, q2
                vadd.i32 q12, q12, q2

            NoBias:
                ldr lr, [sp, #48]
                cmp lr, #0
                bne PerChannel
                ldr lr, [sp, #36]
                vld1.32 q3[], [lr]
                ldr lr, [sp, #32]
                vld1.32 q4[], [lr]
                ldr lr, [sp, #40]
                vld1.32 q5[], [lr]
                b QuantizeStart
            PerChannel:
                ldr lr, [sp, #36]
                vld1.32 q3, [lr]
                ldr lr, [sp, #32]
                vld1.32 q4, [lr]
                ldr lr, [sp, #40]
                vld1.32 q5, [lr]
            QuantizeStart:
                vshl.s32 q8, q8, q3
                vshl.s32 q12, q12, q3

                vqrdmulh.s32 q8, q8, q4
                vqrdmulh.s32 q12, q12, q4

                vand q3, q5, q8
                vshr.s32 q3, q3, #31
                vqadd.s32 q8, q8, q3
                vrshl.s32 q8, q8, q5
                vand q4, q5, q12
                vshr.s32 q4, q4, #31
                vqadd.s32 q12, q12, q4
                vrshl.s32 q12, q12, q5
                
                ldr r10, [sp, #28]
                vdup.32 q6, r10
                vadd.i32 q8, q8, q6
                vadd.i32 q12, q12, q6

                ldr r10, [sp, #20]
                vdup.32 q0, r10
                vmax.s32 q8, q8, q0
                vmax.s32 q12, q12, q0

                ldr r10, [sp, #24]
                vdup.32 q1, r10
                vmin.s32 q8, q8, q1
                vmin.s32 q12, q12, q1

                vqmovn.s32 d30, q8
                vqmovn.s32 d31, q12
                vqmovn.s16 d0, q15

            // prefetching is not preferred while writing results in spite of cache missing
            // you could try prfm pstl2strm
            WriteStart:
                cmp r6, #1
                beq Write1
                cmp r6, #2
                beq Write2
                cmp r6, #3
                beq Write3
                b Write4
            Write1:
                vst1.8 {d0[0]}, [r11], r7
                vst1.8 {d0[1]}, [r11]
                add r0, r0, #1
                b WriteEnd
            Write2:
                vst1.16 {d0[0]}, [r11], r7
                vst1.16 {d0[1]}, [r11]
                add r0, r0, #2
                b WriteEnd
            Write3:
                add r14, r11, #2
                vst1.16 {d0[0]}, [r11], r7
                vst1.16 {d0[1]}, [r11]
                vst1.8 {d0[0]}, [r14], r7
                vst1.8 {d0[1]}, [r14]
                add r0, r0, #3
                b WriteEnd
            Write4:
                vst1.32 {d0[0]}, [r11], r7
                vst1.32 {d0[1]}, [r11]
                add r0, r0, #4

        WriteEnd:

            subs r8, r8, #1
            bne LoopKsize

        cmp r6, #4
        ble LoopOcEnd
        ldr lr, [sp, #48]
        cmp lr, #0
        beq NoChannelForward
        ldr lr, [sp, #44]
        cmp lr, #0
        beq NoSumForward
        ldr lr, [sp, #16]
        add lr, lr, #16
        str lr, [sp, #16]
    NoSumForward:
        ldr lr, [sp, #36]
        add lr, lr, #16
        str lr, [sp, #36]
        ldr lr, [sp, #32]
        add lr, lr, #16
        str lr, [sp, #32]
        ldr lr, [sp, #40]
        add lr, lr, #16
        str lr, [sp, #40]
    NoChannelForward:
        sub r6, r6, #4
        cmp r3, #0
        beq NoStepFowrard
        add r3, r3, #16
    NoStepFowrard:
        b LoopOc

LoopOcEnd:
    sub sp, sp, #96
    vpop {q4-q7}
    pop {r4-r8, r10, r11, pc}
#endif
#endif
