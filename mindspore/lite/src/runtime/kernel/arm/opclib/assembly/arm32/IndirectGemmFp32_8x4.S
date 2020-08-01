#ifdef __arm__
#ifndef __aarch64__

.text
.align 5
.global IndirectGemmFp32_8x4
#ifndef __APPLE__
.type IndirectGemmFp32_8x4, %function
#endif

// void IndirectGemmFp32_8x4(float *output, float *input, float *weight, float *bias,
//     size_t kSize, size_t ic4, size_t oc8, size_t offset, size_t mode, size_t writeC4, size_t relu, size_t relu6);
// r0: output, r1: input, r2: weight, r3: bias, r4: kSize, r5: ic4, r6: oc, r7: offset
// r8:mode, r10: writeMode, r10: relu, r10:relu6
// mode = 0 for general convolution, where one conv unit is a row
// mode = 1 for winograd/common gemm, where the total channels of one input is a row
IndirectGemmFp32_8x4:

    .macro INIT_BIAS
        veor q10, q10, q10
        cmp r3, #0
        beq InitBias
        vld1.32 q10, [r3]
    InitBias:
        vmov q11, q10
        vmov q12, q10
        vmov q13, q10
        vmov q14, q10
        vmov q15, q10
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
    ldr r8, [sp, #16]

    cmp r8, #0
    bne LoopOc
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
            vld1.32 {q0, q1}, [r12]!
            vld1.32 {q2, q3}, [r12]!
            // load weight
            vld1.32 {q4, q5}, [r2]!
            // step for output 1-2
            vmul.f32 q8, q4, d0[0]
            vmul.f32 q9, q4, d2[0]
            vmla.f32 q8, q5, d0[1]
            vmla.f32 q9, q5, d2[1]
            vld1.32 {q6, q7}, [r2]!

            subs r10, r5, #1
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
                cmp r10, #0
                bne Relu6
                ldr r10, [sp, #24]
                cmp r10, #0
                bne Relu
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
                cmp r10, #0
                bne WriteC4
                cmp r6, #1
                beq Write1
                cmp r6, #2
                beq Write2
                cmp r6, #3
                beq Write3
                b Write4
            Write1:
                vst1.32 d0[0], [r11]
                add r11, r11, r7
                vst1.32 d2[0], [r11]
                add r11, r11, r7
                vst1.32 d4[0], [r11]
                add r11, r11, r7
                vst1.32 d6[0], [r11]
                add r11, r11, r7
                vst1.32 d8[0], [r11]
                add r11, r11, r7
                vst1.32 d10[0], [r11]
                add r11, r11, r7
                vst1.32 d12[0], [r11]
                add r11, r11, r7
                vst1.32 d14[0], [r11]
                add r0, r0, #4
                b WriteEnd
            Write2:
                vst1.32 d0, [r11]
                add r11, r11, r7
                vst1.32 d2, [r11]
                add r11, r11, r7
                vst1.32 d4, [r11]
                add r11, r11, r7
                vst1.32 d6, [r11]
                add r11, r11, r7
                vst1.32 d8, [r11]
                add r11, r11, r7
                vst1.32 d10, [r11]
                add r11, r11, r7
                vst1.32 d12, [r11]
                add r11, r11, r7
                vst1.32 d14, [r11]
                add r0, r0, #8
                b WriteEnd
            Write3:
                add r12, r11, #8
                vst1.32 d0, [r11]
                add r11, r11, r7
                vst1.32 d1[0], [r12]
                add r12, r12, r7
                vst1.32 d2, [r11]
                add r11, r11, r7
                vst1.32 d3[0], [r12]
                add r12, r12, r7
                vst1.32 d4, [r11]
                add r11, r11, r7
                vst1.32 d5[0], [r12]
                add r12, r12, r7
                vst1.32 d6, [r11]
                add r11, r11, r7
                vst1.32 d7[0], [r12]
                add r12, r12, r7
                vst1.32 d8, [r11]
                add r11, r11, r7
                vst1.32 d9[0], [r12]
                add r12, r12, r7
                vst1.32 d10, [r11]
                add r11, r11, r7
                vst1.32 d11[0], [r12]
                add r12, r12, r7
                vst1.32 d12, [r11]
                add r11, r11, r7
                vst1.32 d13[0], [r12]
                add r12, r12, r7
                vst1.32 d14, [r11]
                vst1.32 d15[0], [r12]
                add r0, r0, #12
                b WriteEnd
            WriteC4:
                vst1.32 q0, [r11], r7
                vst1.32 q1, [r11], r7
                vst1.32 q2, [r11], r7
                vst1.32 q3, [r11], r7
                vst1.32 q4, [r11], r7
                vst1.32 q5, [r11], r7
                vst1.32 q6, [r11], r7
                vst1.32 q7, [r11]
                add r0, r0, #16
                b WriteEnd
            Write4:
                // prefetching is not prefered while writing results in spite of cache missings
                // you could try prfm pstl2vst1.32m
                // there are almost no benefits observed though
                vst1.32 q0, [r11], r7
                vst1.32 q1, [r11], r7
                vst1.32 q2, [r11], r7
                vst1.32 q3, [r11], r7
                vst1.32 q4, [r11], r7
                vst1.32 q5, [r11], r7
                vst1.32 q6, [r11], r7
                vst1.32 q7, [r11]
                add r0, r0, #16

        WriteEnd:

            subs r8, r8, #1
            bne LoopKsize

        subs r6, r6, #4
        cmp r3, #0
        beq NoStepFowrard
        add r3, r3, #16
    NoStepFowrard:
        bgt LoopOc

    add sp, sp, #160
    vpop {q4-q7}
    pop {r4-r8, r10, r11, pc}
#endif
#endif
