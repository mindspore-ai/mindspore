#ifdef __arm__
#ifndef __aarch64__
#include "nnacl/assembly_global.h"

.text
.align 5
.global ConvDwFp32Center
#ifndef __APPLE__
.type ConvDwFp32Center, %function
#endif

// void ConvDwFp32Center(float *dst, const float *src, const float *weight, const float *bias, size_t height, size_t width,
//                      size_t kernel_h, size_t kernel_w, size_t out_h_step, size_t block_channel, size_t in_sh_step, size_t in_sw_step,
//                      size_t in_kh_step, size_t in_kw_step, size_t relu, size_t relu6);
// r0: dst, r1: src, r2: weight, r3: bias, #0: height, #4: width, #8: kernel_h, #12: kernel_w,
// #16: out_h_step, #20: block_channel, #24: in_sh_step, #28: in_sw_step, #32: in_kh_step,#36: in_kw_step
// #40: relu, #44: relu6
asm_function ConvDwFp32Center
    // at return, clang generates "push {lr}, pop {pc}"" while gcc will generate "bx lr"
    // according to https://stackoverflow.com/questions/53625807
    // even if we jump to link register instead of saving it, we still have to save it in subroutine calls anyway
    // clang's rule seems more simple, though there are no subroutine calls here
    // r4-r8 and q4-q7 must be saved according to https://static.docs.arm.com/ihi0042/i/aapcs32.pdf
    push {r0-r8, r10, r11, lr}
    vpush {q4-q7}
    add sp, sp, #112

    ldr r4, [sp] // height

    vld1.32 {q13}, [r3]
    vmov.i32 q14, #6
    vcvt.f32.s32 q14, q14
    veor q15, q15, q15

    LoopH:
        ldr r1, [sp, #-44] // src_w, src_h = src
        ldr r5, [sp, #4] // width
        ldr r0, [sp, #-48] // dst_w, dst_h = dst
        cmp r5, #4
        blt LoopW
        LoopW4:
            ldr r11, [sp, #28] // in_sw_step
            mov r8, r1 // src_kh, src_w
            ldr r2, [sp, #-40] // weight_kh, weight
            ldr r6, [sp, #8] // kernel_h
            vmov q0, q13
            vmov q1, q13
            vmov q2, q13
            vmov q3, q13
            LoopKh4:
                ldr r7, [sp, #12] // kernel_w
                mov lr, r8 // src_kw, src_kh
                LoopKw4:
                    ldr r12, [sp, #36] //in_kw_step
                    mov r10, lr
                    vld1.32 {q12}, [r2]!
                    vld1.32 {q4}, [r10]
                    add r10, r10, r11
                    vmla.f32 q0, q4, q12
                    vld1.32 {q5}, [r10]
                    add r10, r10, r11
                    vmla.f32 q1, q5, q12
                    vld1.32 {q6}, [r10]
                    add r10, r10, r11
                    vmla.f32 q2, q6, q12
                    vld1.32 {q7}, [r10]
                    add r10, r10, r11
                    vmla.f32 q3, q7, q12
                    subs r7, r7, #1
                    add lr, lr, r12
                    bne LoopKw4
                ldr r12, [sp, #32]   // in_kh_step
                add r8, r8, r12
                subs r6, r6, #1
                bne LoopKh4
            ldr r12, [sp, #44]
            cmp r12, #0
            bne Relu64
            ldr r12, [sp, #40]
            cmp r12, #0
            bne Relu4
            b Write4
        Relu64:
            vmin.f32 q0, q0, q14
            vmin.f32 q1, q1, q14
            vmin.f32 q2, q2, q14
            vmin.f32 q3, q3, q14
        Relu4:
            vmax.f32 q0, q0, q15
            vmax.f32 q1, q1, q15
            vmax.f32 q2, q2, q15
            vmax.f32 q3, q3, q15
        Write4:
            ldr r12, [sp, #20]  // block_channel
            vst1.32 {q0}, [r0]
            add r0, r0, r12
            vst1.32 {q1}, [r0]
            add r0, r0, r12
            vst1.32 {q2}, [r0]
            add r0, r0, r12
            vst1.32 {q3}, [r0]
            add r0, r0, r12
            mov r12, #4
            mul r11, r11, r12
            add r1, r1, r11  // src_w += in_sw_step
            sub r5, r5, #4
            cmp r5, #0
            ble LoopWEnd
            cmp r5, #4
            bge LoopW
        LoopW:
            mov r8, r1 // src_kh, src_w
            ldr r2, [sp, #-40] // weight_kh, weight
            ldr r6, [sp, #8] // kernel_h
            vmov q0, q13   // bias
            LoopKh:
                ldr r7, [sp, #12] // kernel_w
                mov r10, r8 // src_kw, src_kh
                LoopKw:
                    ldr r12, [sp, #36] //in_kw_step
                    vld1.32 {q1}, [r10]
                    add r10, r10, r12
                    vld1.32 {q12}, [r2]!
                    vmla.f32 q0, q1, q12
                    subs r7, r7, #1
                    bne LoopKw
                ldr r12, [sp, #32]  // in_kh_step
                add r8, r8, r12
                subs r6, r6, #1
                bne LoopKh
            ldr r12, [sp, #44]
            cmp r12, #0
            bne Relu6
            ldr r12, [sp, #40]
            cmp r12, #0
            bne Relu
            b Write
        Relu6:
            vmin.f32 q0, q0, q14
        Relu:
            vmax.f32 q0, q0, q15
        Write:
            ldr r12, [sp, #20]  // block_channel
            vst1.32 {q0}, [r0]   // dst_kw += block_channel
            add r0, r0, r12
            ldr r12, [sp, #28]   // in_sw_step
            add r1, r1, r12  // src_w += in_sw_step
            subs r5, r5, #1
            bne LoopW
        ldr r3, [sp, #16]   // out_h_step
        ldr r12, [sp, #-48]
        add r12, r12, r3
        str r12, [sp, #-48]

        ldr r3, [sp, #24]    // in_sh_step
        ldr r12, [sp, #-44]   // src_h += in_sh_step
        add r12, r12, r3
        str r12, [sp, #-44]

        subs r4, r4, #1   // height
        bne LoopH
LoopWEnd:
    sub sp, sp, #112
    vpop {q4-q7}
    pop {r0-r8, r10, r11, pc}
#endif
#endif
