#ifdef __arm__
#ifndef __aarch64__
#include "nnacl/assembly_global.h"

.text
.align 5
.global ConvDwInt8Center
#ifndef __APPLE__
.type ConvDwInt8Center, %function
#endif
// void DepthwiseCenterInt8(int8_t *dst, const int8_t *src, const int16_t *weight, const int32_t *bias, int height,
//                          int width, int kernel_h, int kernel_w, int out_h_step, int block_channel, int in_sh_step,
//                          int in_sw_step, int in_kh_step, int in_kw_step, int8_t *in_zp, int32_t *out_zp,
//                          int32_t *out_multiplier, int32_t *left_shift, int32_t *right_shift, int32_t *acc_min,
//                          int32_t *acc_max)
// #-48: dst, #-44: src, #-40: weight, #-36: bias, #0: height, #4: width, #8: kernel_h, #12: kernel_w,
// #16: out_h_step, #20: block_channel, #24: in_sh_step, #28: in_sw_step, #32: in_kh_step, #36: in_kw_step
// #40: in_zp, #44: out_zp, #48: out_multiplier, #52: left_shift, #56: right_shift, #60:acc_min, #64: acc_max
asm_function ConvDwInt8Center
// at return, clang generates "push {lr}, pop {pc}"" while gcc will generate "bx lr"
// according to https://stackoverflow.com/questions/53625807
// even if we jump to link register instead of saving it, we still have to save it in subroutine calls anyway
// clang's rule seems more simple, though there are no subroutine calls here
// r4-r8 and q4-q7 must be saved according to https://static.docs.arm.com/ihi0042/i/aapcs32.pdf
    push {r0-r8, r10, r11, lr}
    vpush {q4-q7}

    ldr lr, [sp, #168]
    vld1.32 {q0, q1}, [lr]
    vpush {q0, q1}
    ldr lr, [sp, #204]
    vld1.32 {q0, q1}, [lr]
    vpush {q0, q1}
    ldr lr, [sp, #240]
    vld1.32 {q0, q1}, [lr]
    vpush {q0, q1}
    add sp, sp, #208

    ldr r1, [sp, #-36]
    vld1.32 {q8, q9}, [r1]
    ldr r1, [sp, #44]
    vld1.32 {q10, q11}, [r1]
    ldr r1, [sp, #48]
    vld1.32 {q12, q13}, [r1]
    ldr r1, [sp, #52]
    vld1.32 {q14, q15}, [r1]

    ldr r11, [sp, #28]
    ldr r4, [sp]
    LoopH:
        ldr r1, [sp, #-44]
        ldr r0, [sp, #-48]
        ldr r5, [sp, #4]
        LoopW2:
            vmov q4, q8
            vmov q5, q9
            vmov q6, q8
            vmov q7, q9
            mov r7, r1
            ldr r3, [sp, #-40]
            ldr r6, [sp, #8]
            LoopKH:
                mov r9, r7
                ldr r10, [sp, #12]
                    LoopKW:
                        mov r8, r9
                        vld1.16 {q0}, [r3]!
                        ldr lr, [sp, #40]
                        vld1.8 {d2}, [lr]

                        vld1.8 {d3}, [r8]
                        add r8, r8, r11
                        vsubl.s8 q2, d3, d2
                        vmlal.s16 q4, d4, d0
                        vmlal.s16 q5, d5, d1

                        vld1.8 {d3}, [r8]
                        add r8, r8, r11
                        vsubl.s8 q2, d3, d2
                        vmlal.s16 q6, d4, d0
                        vmlal.s16 q7, d5, d1

                        ldr r12, [sp, #36]
                        add r9, r9, r12
                        subs r10, r10, #1
                        bne LoopKW
                    ldr r12, [sp, #32]
                    add r7, r7, r12
                    subs r6, r6, #1
                    bne LoopKH

                vshl.s32 q4, q4, q14
                vshl.s32 q5, q5, q15
                vshl.s32 q6, q6, q14
                vshl.s32 q7, q7, q15

                vqrdmulh.s32 q4, q4, q12
                vqrdmulh.s32 q5, q5, q13
                vqrdmulh.s32 q6, q6, q12
                vqrdmulh.s32 q7, q7, q13

                sub lr, sp, #144
                vld1.32 {q0, q1}, [lr]

                vand q2, q4, q0
                vshr.s32 q2, q2, #31
                vqadd.s32 q4, q4, q2
                vrshl.s32 q4, q4, q0

                vand q2, q5, q1
                vshr.s32 q2, q2, #31
                vqadd.s32 q5, q5, q2
                vrshl.s32 q5, q5, q1

                vand q2, q6, q0
                vshr.s32 q2, q2, #31
                vqadd.s32 q6, q6, q2
                vrshl.s32 q6, q6, q0

                vand q2, q7, q1
                vshr.s32 q2, q2, #31
                vqadd.s32 q7, q7, q2
                vrshl.s32 q7, q7, q1

                vadd.i32 q4, q4, q10
                vadd.i32 q5, q5, q11
                vadd.i32 q6, q6, q10
                vadd.i32 q7, q7, q11

                sub lr, sp, #176
                vld1.32 {q0, q1}, [lr]
                vmax.s32 q4, q4, q0
                vmax.s32 q5, q5, q1
                vmax.s32 q6, q6, q0
                vmax.s32 q7, q7, q1

                sub lr, sp, #208
                vld1.32 {q0, q1}, [lr]
                vmin.s32 q4, q4, q0
                vmin.s32 q5, q5, q1
                vmin.s32 q6, q6, q0
                vmin.s32 q7, q7, q1

                vqmovn.s32 d0, q4
                vqmovn.s32 d1, q5
                vqmovn.s32 d2, q6
                vqmovn.s32 d3, q7
                vqmovn.s16 d0, q0
                vqmovn.s16 d1, q1


                ldr r12, [sp, #20]
                mov r8, r0
                vst1.8 {d0[0]}, [r8]!
                vst1.8 {d0[1]}, [r8]!
                vst1.8 {d0[2]}, [r8]!
                vst1.8 {d0[3]}, [r8]!
                vst1.8 {d0[4]}, [r8]!
                vst1.8 {d0[5]}, [r8]!
                vst1.8 {d0[6]}, [r8]!
                vst1.8 {d0[7]}, [r8]!
                add r0, r0, r12

                mov r8, r0
                vst1.8 {d1[0]}, [r8]!
                vst1.8 {d1[1]}, [r8]!
                vst1.8 {d1[2]}, [r8]!
                vst1.8 {d1[3]}, [r8]!
                vst1.8 {d1[4]}, [r8]!
                vst1.8 {d1[5]}, [r8]!
                vst1.8 {d1[6]}, [r8]!
                vst1.8 {d1[7]}, [r8]!
                add r0, r0, r12

            add r1, r1, r11
            add r1, r1, r11
            subs r5, r5, #2
            beq LoopEndW
            cmp r5, #2
            bge LoopW2

        LoopW:
            vmov q4, q8
            vmov q5, q9
            mov r7, r1
            ldr r3, [sp, #-40]
            ldr r6, [sp, #8]
            LoopKH1:
                mov r9, r7
                ldr r10, [sp, #12]
                LoopKW1:
                    vld1.16 {q0}, [r3]!
                    ldr lr, [sp, #40]
                    vld1.8 {d2}, [lr]

                    vld1.8 {d3}, [r9]
                    vsubl.s8 q2, d3, d2
                    vmlal.s16 q4, d4, d0
                    vmlal.s16 q5, d5, d1

                    ldr r12, [sp, #36]
                    add r9, r9, r12
                    subs r10, r10, #1
                    bne LoopKW1
                ldr r12, [sp, #32]
                add r7, r7, r12
                subs r6, r6, #1
                bne LoopKH1

                vshl.s32 q4, q4, q14
                vshl.s32 q5, q5, q15

                vqrdmulh.s32 q4, q4, q12
                vqrdmulh.s32 q5, q5, q13

                sub lr, sp, #144
                vld1.32 {q0, q1}, [lr]
                vand q2, q4, q0
                vshr.s32 q2, q2, #31
                vqadd.s32 q4, q4, q2
                vrshl.s32 q4, q4, q0

                vand q2, q5, q1
                vshr.s32 q2, q2, #31
                vqadd.s32 q5, q5, q2
                vrshl.s32 q5, q5, q1

                vadd.i32 q4, q4, q10
                vadd.i32 q5, q5, q11

                sub lr, sp, #176
                vld1.32 {q0, q1}, [lr]
                vmax.s32 q4, q4, q0
                vmax.s32 q5, q5, q1

                sub lr, sp, #208
                vld1.32 {q0, q1}, [lr]
                vmin.s32 q4, q4, q0
                vmin.s32 q5, q5, q1

                vqmovn.s32 d0, q4
                vqmovn.s32 d1, q5
                vqmovn.s16 d0, q0

                mov r8, r0
                vst1.8 {d0[0]}, [r8]!
                vst1.8 {d0[1]}, [r8]!
                vst1.8 {d0[2]}, [r8]!
                vst1.8 {d0[3]}, [r8]!
                vst1.8 {d0[4]}, [r8]!
                vst1.8 {d0[5]}, [r8]!
                vst1.8 {d0[6]}, [r8]!
                vst1.8 {d0[7]}, [r8]!
            ldr r12, [sp, #20]
            add r0, r0, r12
            add r1, r1, r11
            subs r5, r5, #1
            bne LoopW

        LoopEndW:
            ldr r12, [sp, #16]
            ldr r1, [sp, #-48]
            add r1, r1, r12
            str r1, [sp, #-48]
            ldr r12, [sp, #24]
            ldr r1, [sp, #-44]
            add r1, r1, r12
            str r1, [sp, #-44]
            subs r4, r4, #1
            bne LoopH

    LoopEndH:
        sub sp, sp, #208
        vpop {q0, q1}
        vpop {q0, q1}
        vpop {q0, q1}
        vpop {q4-q7}
        pop {r0-r8, r10, r11, pc}
#endif
#endif
