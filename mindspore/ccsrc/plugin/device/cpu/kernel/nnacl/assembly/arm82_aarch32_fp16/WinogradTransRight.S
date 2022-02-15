/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifdef ENABLE_ARM32
#include "nnacl/assembly_global.h"

.text
.align 5

// void WinogradTransRightFp16(const float16_t *S, const float16_t *B, float16_t *M, size_t w, size_t h, size_t k,
//                            size_t length);
//r0: S
//r1: B
//r2: M
//r3: w
//r4: h
//r5: k
//r6: length
asm_function WinogradTransRightFp16
    push {r1, r3, r4-r11, lr}
    vpush {q4-q7}
    add sp, sp, #108
    ldr r4, [sp]
    ldr r5, [sp, #4]
    ldr r6, [sp, #8]

    mov r8, #8 // 4 * sizeof(float16_t)
    mul r8, r6, r8  // length * 4 * 2
    mul r7, r5, r8  // step for S = k * unitStep * 4
    add r10, r4, r4 // step for B = 2 * h

cmp r4, #1
blt LoopHEnd
cmp r3, #1
blt LoopHEnd
LoopH:
    ldr r3, [sp, #-40]  // w
    ldr r1, [sp, #-44]
    LoopW:
        mov r11, r0  // S
        mov lr, r1 // B_src
        veor q6, q6, q6
        ldr r6, [sp, #8]
        InitZero:
            vst1.16 {d12}, [r2]!
            subs r6, r6, #1
            bne InitZero
        sub r2, r2, r8

        ldr r5, [sp, #4]
        cmp r5, #4
        bge LoopK4
        cmp r5, #3
        bge LoopK3
        cmp r5, #1
        bge LoopK1
        b LoopKEnd

        LoopK4:
            ldr r6, [sp, #8]
            vld1.16 {d1[0]}, [lr], r10
            vld1.16 {d3[0]}, [lr], r10
            vld1.16 {d5[0]}, [lr], r10
            vld1.16 {d7[0]}, [lr], r10

            add r12, r11, r8
            add r14, r12, r8
            add r9, r14, r8
            LoopK4L4:
                vld1.16 {d12}, [r2]
                vld1.16 {d0}, [r11]!
                vld1.16 {d2}, [r12]!
                vmla.f16 d12, d0, d1[0]
                vld1.16 {d4}, [r14]!
                vmla.f16 d12, d2, d3[0]
                vld1.16 {d6}, [r9]!
                vmla.f16 d12, d4, d5[0]
                vmla.f16 d12, d6, d7[0]
                vst1.16 {d12}, [r2]!  // dst
                subs r6, r6, #1  // length
                bne LoopK4L4

            subs r5, r5, #4  // k
            beq LoopKEnd
            sub r2, r2, r8  // dst - step
            mov r11, r9
            cmp r5, #4
            bge LoopK4
            cmp r5, #3
            bge LoopK3
            b LoopK1

        LoopK3:
            ldr r6, [sp, #8]
            vld1.16 {d1[0]}, [lr], r10
            vld1.16 {d3[0]}, [lr], r10
            vld1.16 {d5[0]}, [lr], r10

            add r12, r11, r8
            add r9, r12, r8
            LoopK3L4:
                vld1.16 {d12}, [r2]
                vld1.16 {d0}, [r11]!
                vld1.16 {d2}, [r12]!
                vmla.f16 d12, d0, d1[0]
                vld1.16 {d4}, [r9]!
                vmla.f16 d12, d2, d3[0]
                vmla.f16 d12, d4, d5[0]
                vst1.16 {d12}, [r2]!  // dst
                subs r6, r6, #1  // length
                bne LoopK3L4

            subs r5, r5, #3  // k
            beq LoopKEnd
            sub r2, r2, r8  // dst - step
            mov r11, r9
            cmp r5, #3
            bge LoopK3
            b LoopK1

        LoopK1:
            ldr r6, [sp, #8]
            vld1.16 {d1[0]}, [lr], r10

            LoopK1L4:
                vld1.16 {d12}, [r2]
                vld1.16 {d0}, [r11]!
                vmla.f16 d12, d0, d1[0]
                vst1.16 {d12}, [r2]!  // dst
                subs r6, r6, #1  // length
                bne LoopK1L4

            subs r5, r5, #1  // k
            beq LoopKEnd
            sub r2, r2, r8  // dst - step
            b LoopK1

        LoopKEnd:
            add r1, r1, #2  // B[x]
            subs r3, r3, #1
            bne LoopW
    LoopWEnd:
        add r0, r0, r7
        subs r4, r4, #1
        beq LoopHEnd
        b LoopH
LoopHEnd:
    sub sp, sp, #108
    vpop {q4-q7}
    pop {r1, r3, r4-r11, pc}
#endif
