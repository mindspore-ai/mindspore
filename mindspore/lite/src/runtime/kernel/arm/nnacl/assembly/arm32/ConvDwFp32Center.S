#ifdef __arm__
#ifndef __aarch64__

.text
.align 5
.global ConvDwFp32Center
#ifndef __APPLE__
.type ConvDwFp32Center, %function
#endif

// void ConvDwFp32Center(float *dst, const float *src, const float *weight, const float *bias, size_t height, size_t width,
//                      size_t kernel_h, size_t kernel_w, size_t out_h_step, size_t block_channel, size_t in_sh_step, size_t in_sw_step,
//                      size_t in_kh_step, size_t in_kw_step, size_t relu, size_t relu6);
// r0: dst, r1: src, r2: weight, r3: bias, #48: height, #52: weight, #56: kernel_h, #60: kernel_w, 
// #64: out_h_step, #68: block_channel, #72: in_sh_step, #76: in_sw_step, #80: in_kh_step,#84: in_kw_step
// #88: relu, #92: relu6
ConvDwFp32Center:
    // at return, clang generates "push {lr}, pop {pc}"" while gcc will generate "bx lr"
    // according to https://stackoverflow.com/questions/53625807
    // even if we jump to link register instead of saving it, we still have to save it in subroutine calls anyway
    // clang's rule seems more simple, though there are no subroutine calls here
    // r4-r8 and q4-q7 must be saved according to https://static.docs.arm.com/ihi0042/i/aapcs32.pdf
    push {r0-r8, r10, r11, lr}
    vpush {q4-q7}
    add sp, sp, #112

    ldr r4, [sp, #48]

    vld1.32 {q13}, [r3]
    vmov.i32 q14, #6
    vcvt.f32.s32 q14, q14
    veor q15, q15, q15

    LoopH:
        ldr r1, [sp, #4] // src_w
        ldr r5, [sp, #52] // width
        ldr r0, [sp] // dst_w
        cmp r5, #4
        blt LoopW
        LoopW4:
            ldr r11, [sp, #76] // in_sw_step
            mov r8, r1 // src_kh
            ldr r2, [sp, #8] // weight_kh
            ldr r6, [sp, #56] // kernel_h
            vmov q0, q13
            LoopKh4:
                ldr r12, [sp, #80] //in_kh_step 
                ldr r7, [sp, #60] // kernel_w
                mov lr, r8 // src_kw
                LoopKw4:
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
                ldr r12, [sp, #80]
                add r8, r8, r12
                subs r6, r6, #1
                bne LoopKh4
            ldr r12, [sp, #92]
            cmp r12, #0
            bne Relu64
            ldr r12, [sp, #88]
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
            ldr r12, [sp, #68]
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
            add r1, r1, r11
            sub r5, r5, #4
            cmp r5, #0
            ble LoopWEnd
            cmp r5, #4
            bge LoopW
        LoopW:
            mov r8, r1 // src_kh
            ldr r2, [sp, #8] // weight_kh
            ldr r6, [sp, #56] // kernel_h
            vmov q0, q13
            LoopKh:
                ldr r12, [sp, #84] //in_kw_step 
                ldr r7, [sp, #60] // kernel_w
                mov r10, r8 // src_kw
                LoopKw:
                    vld1.32 {q1}, [r10]
                    add r10, r10, r12
                    vld1.32 {q12}, [r2]!
                    vmla.f32 q0, q1, q12
                    subs r7, r7, #1
                    bne LoopKw
                ldr r12, [sp, #80]
                add r8, r8, r12
                subs r6, r6, #1
                bne LoopKh
            ldr r12, [sp, #92]
            cmp r12, #0
            bne Relu6
            ldr r12, [sp, #88]
            cmp r12, #0
            bne Relu
            b Write
        Relu6:
            vmin.f32 q0, q0, q14
        Relu:
            vmax.f32 q0, q0, q15
        Write:
            ldr r12, [sp, #68]
            vst1.32 {q0}, [r0]
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
