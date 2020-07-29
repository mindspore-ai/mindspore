#ifdef __aarch64__

.text
.align 5
.global IndirectGemmInt8_2x4
#ifndef __APPLE__
.type IndirectGemmInt8_2x4, %function
#endif

// void IndirectGemmInt8_2x4(int8_t *output, int8_t *input, int8_t *weight, int32_t *bias, size_t ksize, size_t ic4,
// size_t oc, size_t offset, int32_t *input_sum, size_t act_min, size_t act_max, size_t out_zp, size_t out_multiplier,
// size_t shift_before, size_t shift_after);
// x0: output, x1: input, r2: weight, x3: bias, x4: kSize, x5: ic4, x6: oc, x7: offset
// x8: input_sum, x10: act_min, x11: act_max, x10: out_zp, x11: out_multiplier, x10: shift_before, x11: shift_after
IndirectGemmInt8_2x4:

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
    add sp, sp, #160

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

            subs x10, x5, #1
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

                subs x10, x10, #1
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

                // load sum
                ldr r10, [sp, #16]
                vld1.32 q0[], [r10]!
                vld1.32 q1[], [r10]!
                // pairwise add
                vpadd.i32 q8, q8, q9
                vpadd.i32 q10, q10, q11
                vpadd.i32 q12, q12, q13
                vpadd.i32 q14, q14, q15
                vpadd.i32 q8, q8, q10
                vpadd.i32 q12, q12, q14
                vsub.i32 q8, q8, q0
                vsub.i32 q12, q12, q1
                cbz r3, NoBias
                vld1.32 q2, [r3]
                vadd.i32 q8, q8, q2
                vadd.i32 q12, q12, q2

            NoBias:
                ldr r10, [sp, #36]
                vdup.32 q3, r10
                vshl.s32 q8, q8, q3
                vshl.s32 q12, q12, q3

                ldr r10, [sp, #32]
                vdup.32 q4, r10
                vqrdmulh.s32 q8, q8, q4
                vqrdmulh.s32 q12, q12, q4

                ldr r10, [sp, #40]
                vdup.32 q5, r10
                vrshl.s32 q8, q8, q5
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
                vqmovn.s16 d0, q14

            // prefetching is not prefered while writing results in spite of cache missings
            // you could try prfm pstl2strm
            WriteStart:
                cmp x6, #1
                beq Write1
                cmp x6, #2
                beq Write2
                cmp x6, #3
                beq Write3
                b Write4
            Write1:
                vst1.8 {d0[0]}, [x11], x7
                vst1.8 {d0[1]}, [x11]
                add r0, r0, #1
                b WriteEnd
            Write2:
                vst1.16 {d0[0]}, [x11], x7
                vst1.16 {d0[1]}, [x11]
                add r0, r0, #2
                b WriteEnd
            Write3:
                add x14, x11, #2
                vst1.16 {d0[0]}, [x11], x7
                vst1.16 {d0[1]}, [x11]
                vst1.8 {d0[0]}, [x14], x7
                vst1.8 {d0[1]}, [x14]
                add r0, r0, #3
                b WriteEnd
            Write4:
                vst1.32 {d0[0]}, [x11], x7
                vst1.32 {d0[1]}, [x11]
                add r0, r0, #4

        WriteEnd:

            subs r8, r8, #1
            bne LoopKsize

        subs r6, r6, #4
        cbz r3, NoStepFowrard
        add r3, r3, #16
    NoStepFowrard:
        bgt LoopOc

    add sp, sp, #160
    vpop {q4-q7}
    pop {r4-r8, r10, r11, pc}
#endif
