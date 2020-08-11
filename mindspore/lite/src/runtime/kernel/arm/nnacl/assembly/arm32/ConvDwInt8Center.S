#ifdef __arm__
#ifndef __aarch64__

.text
.align 5
.global ConvDwInt8Center
#ifndef __APPLE__
.type ConvDwInt8Center, %function
#endif

// void ConvDwInt8Center(int8_t *dst, const int16_t *src, const int16_t *weight, const int32_t *bias, size_t height, size_t width,
//                      size_t kernel_h, size_t kernel_w, size_t out_h_step, size_t block_channel, size_t in_sh_step, size_t in_sw_step,
//                      size_t in_kh_step, size_t in_kw_step, int out_multiplier, int left_shift,
//                      int right_shift, int32_t out_zp, int32_t acc_min, int32_t acc_max);
// r0: dst, r1: src, r2: weight, r3: bias, #48: height, #52: width, #56: kernel_h, #60: kernel_w, 
// #64: out_h_step, #68: block_channel, #72: in_sh_step, #76: in_sw_step, #80: in_kh_step,#84: in_kw_step
// #88: out_multiplier, #92: left_shift, #96: right_shift, #100: out_zp, #104: acc_min, #108: acc_max
ConvDwInt8Center:
    // at return, clang generates "push {lr}, pop {pc}"" while gcc will generate "bx lr"
    // according to https://stackoverflow.com/questions/53625807
    // even if we jump to link register instead of saving it, we still have to save it in subroutine calls anyway
    // clang's rule seems more simple, though there are no subroutine calls here
    // r4-r8 and q4-q7 must be saved according to https://static.docs.arm.com/ihi0042/i/aapcs32.pdf
    push {r0-r8, r10, r11, lr}
    vpush {q4-q7}
    add sp, sp, #112

    ldr r4, [sp, #48]

    ldr r12, [sp, #92]
    vdup.32 q9, r12

    ldr r11, [sp, #88]
    vdup.32 q10, r11

    ldr r10, [sp, #96]
    vdup.32 q11, r10

    ldr r8, [sp, #100]
    vdup.32 q12, r8
 
    ldr r7, [sp, #104]
    vdup.32 q13, r7

    ldr r6, [sp, #108]
    vdup.32 q14, r6

    vld1.32 {q15}, [r3]

    LoopH:
        ldr r1, [sp, #4] // src_w
        ldr r5, [sp, #52] // width
        ldr r0, [sp] // dst_w
        LoopW4:
            ldr r11, [sp, #76] // in_sw_step
            mov r8, r1 // src_kh
            ldr r2, [sp, #8] // weight_kh
            ldr r6, [sp, #56] // kernel_h
            vmov q0, q15
            LoopKh4:
                ldr r12, [sp, #80] //in_kh_step
                ldr r7, [sp, #60] // kernel_w
                mov r10, r8 // src_kw
                LoopKw4:
                    vld1.16 {d24}, [r2]!
                    vld1.16 {d8}, [r10]
                    add r10, r10, r11
                    vmlal.s16 q0, d8, d24
                    vld1.16 {d10}, [r10]
                    add r10, r10, r11
                    vmlal.s16 q1, d10, d24
                    vld1.16 {d12}, [r10]
                    add r10, r10, r11
                    vmlal.s16 q2, d12, d24
                    vld1.16 {d14}, [r10]
                    add r10, r10, r11
                    vmlal.s16 q3, d14, d24
                    subs r7, r7, #1
                    bne LoopKw4
                ldr r12, [sp, #80]
                add r8, r8, r12
                subs r6, r6, #1
                bne LoopKh4

            vshl.s32 q0, q0, q9
            vshl.s32 q1, q1, q9
            vshl.s32 q2, q2, q9
            vshl.s32 q3, q3, q9
            vqrdmulh.s32 q0, q0, q10
            vqrdmulh.s32 q1, q1, q10
            vqrdmulh.s32 q2, q2, q10
            vqrdmulh.s32 q3, q3, q10
            vand q4, q0, q11
            vshr.s32 q4, q4, #31
            vqadd.s32 q0, q0, q4
            vrshl.s32 q0, q0, q11
            vand q5, q1, q11
            vshr.s32 q5, q5, #31
            vqadd.s32 q1, q1, q5
            vrshl.s32 q1, q1, q11
            vand q6, q2, q11
            vshr.s32 q6, q6, #31
            vqadd.s32 q2, q2, q6
            vrshl.s32 q2, q2, q11
            vand q7, q3, q11
            vshr.s32 q7, q7, #31
            vqadd.s32 q3, q3, q7
            vrshl.s32 q3, q3, q11
            vadd.i32 q0, q0, q12
            vadd.i32 q1, q1, q12
            vadd.i32 q2, q2, q12
            vadd.i32 q3, q3, q12
            vmax.s32 q0, q0, q13
            vmax.s32 q1, q1, q13
            vmax.s32 q2, q2, q13
            vmax.s32 q3, q3, q13
            vmin.s32 q0, q0, q14
            vmin.s32 q1, q1, q14
            vmin.s32 q2, q2, q14
            vmin.s32 q3, q3, q14

            vqmovn.s32 d0, q0
            vqmovn.s32 d2, q1
            vqmovn.s32 d4, q2
            vqmovn.s32 d6, q3
            vqmovn.s16 d0, q0
            vqmovn.s16 d2, q1
            vqmovn.s16 d4, q2
            vqmovn.s16 d6, q3

            mov r3, r0
            ldr r12, [sp, #68]
            vst1.8 {d0[0]}, [r3]!
            vst1.8 {d0[1]}, [r3]!
            vst1.8 {d0[2]}, [r3]!
            vst1.8 {d0[3]}, [r3]!
            add r0, r0, r12
            mov r3, r0
            vst1.8 {d2[0]}, [r3]!
            vst1.8 {d2[1]}, [r3]!
            vst1.8 {d2[2]}, [r3]!
            vst1.8 {d2[3]}, [r3]!
            add r0, r0, r12
            mov r3, r0
            vst1.8 {d4[0]}, [r3]!
            vst1.8 {d4[1]}, [r3]!
            vst1.8 {d4[2]}, [r3]!
            vst1.8 {d4[3]}, [r3]!
            add r0, r0, r12
            mov r3, r0
            vst1.8 {d6[0]}, [r3]!
            vst1.8 {d6[1]}, [r3]!
            vst1.8 {d6[2]}, [r3]!
            vst1.8 {d6[3]}, [r3]!
            add r0, r0, r12
            mov r3, r0
            mov r12, #4
            mul r11, r11, r12
            add r1, r1, r11
            sub r5, r5, #4
            cmp r5, #0
            ble LoopWEnd
            cmp r5, #4
            bge LoopW4
        LoopW:
            mov r8, r1 // src_kh
            ldr r2, [sp, #8] // weight_kh
            ldr r6, [sp, #56] // kernel_h
            vmov q0, q15
            LoopKh:
                ldr r12, [sp, #84] //in_kw_step 
                ldr r7, [sp, #60] // kernel_w
                mov r10, r8 // src_kw
                LoopKw:
                    vld1.16 {d2}, [r10]
                    add r10, r10, r12
                    vld1.16 {d24}, [r2]!
                    vmlal.s16 q0, d2, d24
                    subs r7, r7, #1
                    bne LoopKw
                ldr r12, [sp, #80]
                add r8, r8, r12
                subs r6, r6, #1
                bne LoopKh

            vshl.s32 q0, q0, q9
            vqrdmulh.s32 q0, q0, q10
            vand q4, q0, q11
            vshr.s32 q4, q4, #31
            vqadd.s32 q0, q0, q4
            vrshl.s32 q0, q0, q11
            vadd.i32 q0, q0, q12
            vmax.s32 q0, q0, q13
            vmin.s32 q0, q0, q14

            vqmovn.s32 d0, q0
            vqmovn.s16 d0, q0

            mov r3, r0
            ldr r12, [sp, #68]
            vst1.8 {d0[0]}, [r3]!
            vst1.8 {d0[1]}, [r3]!
            vst1.8 {d0[2]}, [r3]!
            vst1.8 {d0[3]}, [r3]!
            add r0, r0, r12
            ldr r12, [sp, #76]
            add r1, r1, r12
            subs r5, r5, #1
            bne LoopW
        ldr r3, [sp, #64]
        ldr r12, [sp]
        add r12, r12, r3
        str r12, [sp]
        ldr r3, [sp, #72]
        ldr r12, [sp, #4]
        add r12, r12, r3
        str r12, [sp, #4]
        subs r4, r4, #1
        bne LoopH
LoopWEnd:
    sub sp, sp, #112
    vpop {q4-q7}
    pop {r0-r8, r10, r11, pc}
#endif
#endif
