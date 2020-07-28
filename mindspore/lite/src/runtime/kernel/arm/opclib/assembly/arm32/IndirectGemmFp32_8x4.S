#ifdef __aarch64__

.text
.align 5
.global IndirectGemmFp32_8x4
#ifndef __APPLE__
.type IndirectGemmFp32_8x4, %function
#endif

// void IndirectGemmFp32_8x4(float *output, float *input, float *weight, float *bias,
//     size_t kSize, size_t ic4, size_t oc8, size_t offset, size_t mode, size_t writeC4, size_t relu, size_t relu6);
// r0: output, r1: input, r2: weight, r3: bias, r4: kSize, r5: ic4, r6: oc, r7: offset
// r8:mode, r10: writeMode, x10: relu, r10:relu6
// mode = 0 for general convolution, where one conv unit is a row
// mode = 1 for winograd/common gemm, where the total channels of one input is a row
IndirectGemmFp32_8x4:

    .macro INIT_BIAS
        veor q10, q10, q10
        cbz x3, InitBias
        vld1.32 q10, [x3]
    InitBias:
        vmov q11, q10
        vmov q12, q10
        vmov q13, q10
        vmov q14, q10
        vmov q15, q10
    .endm

    // registers v8 ~ v15 must be preserved by a callee across subroutine calls, according to
    // https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst#simd-and-floating-point-registers
    // r19 ~ r29 should be also preserved
    // whereas our coding style do not permit such amount of parameters
    push {r4-r8, r10, r11, lr}
    vpush {q4-q7}
    add sp, sp, #160

    ldr r4, [sp]
    ldr r5, [sp, #4]
    ldr r6, [sp, #8]
    ldr r7, [sp, #12]
    ldr r8, [sp, #16]

    cbnz r8, LoopOc
    // step is one for common convolution, where ic8 should multiply by kernel size 
    // step is (a+b-1) for F(a,b) in winograd
    mul r5, r4, r5
    mov r4, #1

    LoopOc:
        mov r8, r4
        mov r12, r1

        LoopKsize:

            mov r11, r0
            INIT_BIAS

            // load input for output 1-2
            vld1.32 {q0, q1, q2, q3}, [x12]!
            // load weight
            vld1.32 {q4, q5}, [x2]!
            // step for output 1-2
            vmul.f32 q8, q4, d0[0]
            vmul.f32 q9, q4, d2[0]
            vmla.f32 q8, q5, d0[1]
            vmla.f32 q9, q5, d2[1]
            vld1.32 {q6, q7}, [x2]!

            subs x10, x5, #1
            beq LoopIcEnd

            LoopIc:
                vmla.f32 q8, q6, d1[0]
                vmla.f32 q9, q6, d3[0]
                vmla.f32 q8, q7, d1[1]
                vmla.f32 q9, q7, d3[1]
                vmla.f32 q10, q4, d4[0]
                vmla.f32 q11, q4, d6[0]
                vmla.f32 q10, q5, d4[1]
                vmla.f32 q11, q5, d6[1]
                vld1.s32 {q0, q1}, [r12]!
                vmla.f32 q10, q6, d5[0]
                vmla.f32 q11, q6, d7[0]
                vmla.f32 q10, q7, d5[1]
                vmla.f32 q11, q7, d7[1]
                vld1.s32 {q2, q3}, [r12]!
                vmla.f32 q12, q4, d0[0]
                vmla.f32 q13, q4, d2[0]
                vmla.f32 q12, q5, d0[1]
                vmla.f32 q13, q5, d2[1]
                vmla.f32 q14, q4, d4[0]
                vmla.f32 q15, q4, d6[0]
                vmla.f32 q14, q5, d4[1]
                vmla.f32 q15, q5, d6[1]
                vld1.s32 {q4, q5}, [r2]!
                vmla.f32 q12, q6, d1[0]
                vmla.f32 q13, q6, d3[0]
                vmla.f32 q12, q7, d1[1]
                vmla.f32 q13, q7, d3[1]
                vld1.s32 {q0, q1}, [r12]!
                vmla.f32 q14, q6, d5[0]
                vmla.f32 q15, q6, d7[0]
                vmla.f32 q14, q7, d5[1]
                vmla.f32 q15, q7, d7[1]
                vld1.s32 {q6, q7}, [r2]!
                vmla.f32 q8, q4, d0[0]
                vmla.f32 q9, q4, d2[0]
                vmla.f32 q8, q5, d0[1]
                vmla.f32 q9, q5, d2[1]
                vld1.s32 {q2, q3}, [r12]!

                subs r10, r10, #1
                bne LoopIc

            LoopIcEnd:
                vmla.f32 q8, q6, d1[0]
                vmla.f32 q9, q6, d3[0]
                vmla.f32 q8, q7, d1[1]
                vmla.f32 q9, q7, d3[1]
                vmla.f32 q10, q4, d4[0]
                vmla.f32 q11, q4, d6[0]
                vmla.f32 q10, q5, d4[1]
                vmla.f32 q11, q5, d6[1]
                vld1.s32 {q0, q1}, [r12]!
                vmla.f32 q10, q6, d5[0]
                vmla.f32 q11, q6, d7[0]
                vmla.f32 q10, q7, d5[1]
                vmla.f32 q11, q7, d7[1]
                vld1.s32 {q2, q3}, [r12]!
                vmla.f32 q12, q4, d0[0]
                vmla.f32 q13, q4, d2[0]
                vmla.f32 q12, q5, d0[1]
                vmla.f32 q13, q5, d2[1]
                vmla.f32 q14, q4, d4[0]
                vmla.f32 q15, q4, d6[0]
                vmla.f32 q14, q5, d4[1]
                vmla.f32 q15, q5, d6[1]
                vmla.f32 q12, q6, d1[0]
                vmla.f32 q13, q6, d3[0]
                vmla.f32 q12, q7, d1[1]
                vmla.f32 q13, q7, d3[1]
                vmla.f32 q14, q6, d5[0]
                vmla.f32 q15, q6, d7[0]
                vmla.f32 q14, q7, d5[1]
                vmla.f32 q15, q7, d7[1]

                ldr r10, [sp, #28]
                cbnz r10, Relu6
                ldr r10, [sp, #24]
                cbnz x10, Relu
                b WriteStart
            Relu6:
                vmov.i32 q14, #6
                vcvt.f32.s32 q14, q14
                vmin.f32 q0, q0, q14
                vmin.f32 q1, q1, q14
                vmin.f32 q2, q2, q14
                vmin.f32 q3, q3, q14
                vmin.f32 q4, q4, q14
                vmin.f32 q5, q5, q14
                vmin.f32 q6, q6, q14
                vmin.f32 q7, q15, q14
            Relu:
                veor q7, q7, q7
                vmax.f32 q0, q8, q7
                vmax.f32 q1, q9, q7
                vmax.f32 q2, q10, q7
                vmax.f32 q3, q11, q7
                vmax.f32 q4, q12, q7
                vmax.f32 q5, q13, q7
                vmax.f32 q6, q14, q7
                vmax.f32 q15, q15, q7

            WriteStart:
                ldr r10, [sp, #20]
                cbnz x10, WriteC4
                cmp r6, #1
                beq Write1
                cmp r6, #2
                beq Write2
                cmp r6, #3
                beq Write3
                b Write4
            Write1:
                str s0, [r11]
                add r11, r11, x7
                str s4, [r11]
                add r11, r11, x7
                str s8, [r11]
                add r11, r11, x7
                str s12, [r11]
                add r11, r11, x7
                str s16, [r11]
                add r11, r11, x7
                str s20, [r11]
                add r11, r11, x7
                str s24, [r11]
                add r11, r11, x7
                str s28, [r11]
                add r0, r0, #4
                b WriteEnd
            Write2:
                str d0, [r11]
                add r11, r11, x7
                str d2, [r11]
                add r11, r11, x7
                str d4, [r11]
                add r11, r11, x7
                str d6, [r11]
                add r11, r11, x7
                str d8, [r11]
                add r11, r11, x7
                str d10, [r11]
                add r11, r11, x7
                str d12, [r11]
                add r11, r11, x7
                str d14, [r11]
                add r0, r0, #8
                b WriteEnd
            Write3:
                add r12, r11, #8
                str d0, [r11]
                add r11, r11, x7
                str s2, [r12]
                add r12, r12, r7
                str d2, [r11]
                add r11, r11, x7
                str s6, [r12]
                add r12, r12, r7
                str d4, [r11]
                add r11, r11, x7
                str s10, [r12]
                add r12, r12, r7
                str d6, [r11]
                add r11, r11, x7
                str s14, [r12]
                add r12, r12, r7
                str d8, [r11]
                add r11, r11, x7
                str s18, [r12]
                add r12, r12, r7
                str d10, [r11]
                add r11, r11, x7
                str s22, [r12]
                add r12, r12, r7
                str d12, [r11]
                add r11, r11, x7
                str s26, [r12]
                add r12, r12, r7
                str d14, [r11]
                str s30, [r12]
                add r0, r0, #12
                b WriteEnd
            WriteC4:
                vst1.32 q0, [r11], x7
                vst1.32 q1, [r11], x7
                vst1.32 q2, [r11], x7
                vst1.32 q3, [r11], x7
                vst1.32 q4, [r11], x7
                vst1.32 q5, [r11], x7
                vst1.32 q6, [r11], x7
                vst1.32 q7, [r11]
                add r0, r0, #16
                b WriteEnd
            Write4:
                // prefetching is not prefered while writing results in spite of cache missings
                // you could try prfm pstl2strm
                // there are almost no benefits observed though
                vst1.32 q0, [r11], x7
                vst1.32 q1, [r11], x7
                vst1.32 q2, [r11], x7
                vst1.32 q3, [r11], x7
                vst1.32 q4, [r11], x7
                vst1.32 q5, [r11], x7
                vst1.32 q6, [r11], x7
                vst1.32 q7, [r11]
                add r0, r0, #16

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
