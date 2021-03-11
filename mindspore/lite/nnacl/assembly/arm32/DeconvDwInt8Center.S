#ifdef __arm__
#ifndef __aarch64__
#include "nnacl/assembly_global.h"

.text
.align 5
.global DeconvDwInt8Center
#ifndef __APPLE__
.type DeconvDwInt8Center, %function
#endif

// void DeconvDwInt8Center(int32_t *dst, const int16_t *src, const int16_t *weight, size_t height, size_t width,
//                         size_t kernel_h, size_t kernel_w, size_t out_h_step, size_t block_channel, size_t in_sh_step,
//                         size_t in_sw_step, size_t in_kh_step, size_t in_kw_step);
// r0: dst, r1: src, r2: weight, r3: height, r4: width, #52: kernel_h, #56: kernel_w, #60: out_h_step
// #64: block_channel, #68: in_sh_step, #72: in_sw_step, #76: in_kh_step, #80: in_kw_step
asm_function DeconvDwInt8Center
    // at return, clang generates "push {lr}, pop {pc}"" while gcc will generate "bx lr"
    // according to https://stackoverflow.com/questions/53625807
    // even if we jump to link register instead of saving it, we still have to save it in subroutine calls anyway
    // clang's rule seems more simple, though there are no subroutine calls here
    // r4-r8 and q4-q7 must be saved according to https://static.docs.arm.com/ihi0042/i/aapcs32.pdf
    push {r0-r8, r10, r11, lr}

    ldr r10, [sp, #80] // in_kw_step
    ldr r11, [sp, #76] // in_kh_step

    LoopH:
        ldr r0, [sp] // dst_w
        ldr r1, [sp, #4] // src_w
        ldr r4, [sp, #48] // width
        LoopW:
            mov r6, r0 // dst_kh
            ldr r2, [sp, #8] // weight_kh
            ldr r5, [sp, #52] // kernel_h
            vld1.16 {d2}, [r1]
            LoopKh:
                mov r7, r6 // dst_kw
                ldr r12, [sp, #56] // kernel_w
                LoopKw:
                    vld1.32 {q0}, [r7]
                    vld1.16 {d24}, [r2]!
                    vmlal.s16 q0, d2, d24
                    vst1.32 {q0}, [r7]
                    add r7, r7, r10
                    subs r12, r12, #1
                    bne LoopKw
                add r6, r6, r11
                subs r5, r5, #1
                bne LoopKh
            ldr r12, [sp, #72]
            add r0, r0, r12
            ldr r8, [sp, #64]
            add r1, r1, r8
            subs r4, r4, #1
            bne LoopW
        ldr r8, [sp, #68]
        ldr r12, [sp]
        add r12, r12, r8
        str r12, [sp]
        ldr r8, [sp, #60]
        ldr r12, [sp, #4]
        add r12, r12, r8
        str r12, [sp, #4]
        subs r3, r3, #1
        bne LoopH

    pop {r0-r8, r10, r11, pc}
#endif
#endif
