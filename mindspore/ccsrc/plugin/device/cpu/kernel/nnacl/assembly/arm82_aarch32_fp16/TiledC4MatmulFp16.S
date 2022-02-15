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

asm_function TiledC4MatmulFp16
// void TiledC4MatmulFp16(float16_t *dst, const float16_t *src, const float16_t *weight, size_t cal_num, size_t ic4,
//                        size_t oc4);
// r0: dst
// r1: src
// r2: weight
// r3: cal_num
// r4(sp): ic4
// r5(sp + #4): oc4
push {r4-r11, lr}
vpush {q4-q7}
add sp, sp, #100
ldr r4, [sp]
ldr r5, [sp, #4]  // oc4
add r3, r3, r3
mov r7, r1

cmp r5, #1
blt LoopOCEnd
cmp r4, #1
blt LoopICEnd
LoopOC:
    ldr r4, [sp]
    veor q15, q15, q15
    veor q14, q14, q14
    veor q13, q13, q13
    veor q12, q12, q12
    LoopIC:
        vld1.16 {q4, q5}, [r2]!  // weight
        vld1.16 {q2, q3}, [r1]!  // 16 number src
        vmla.f16 d24, d8, d4[0]
        vmla.f16 d24, d9, d4[1]
        vmla.f16 d24, d10, d4[2]
        vmla.f16 d24, d11, d4[3]
        
        vmla.f16 d25, d8, d5[0]
        vmla.f16 d25, d9, d5[1]
        vmla.f16 d25, d10, d5[2]
        vmla.f16 d25, d11, d5[3]

        vmla.f16 d26, d8, d6[0]
        vmla.f16 d26, d9, d6[1]
        vmla.f16 d26, d10, d6[2]
        vmla.f16 d26, d11, d6[3]

        vmla.f16 d27, d8, d7[0]
        vmla.f16 d27, d9, d7[1]
        vmla.f16 d27, d10, d7[2]
        vmla.f16 d27, d11, d7[3]

        vld1.16 {q0, q1}, [r1]!  // 16 number src
        vmla.f16 d28, d8, d0[0]
        vmla.f16 d28, d9, d0[1]
        vmla.f16 d28, d10, d0[2]
        vmla.f16 d28, d11, d0[3]
        
        vmla.f16 d29, d8, d1[0]
        vmla.f16 d29, d9, d1[1]
        vmla.f16 d29, d10, d1[2]
        vmla.f16 d29, d11, d1[3]

        vmla.f16 d30, d8, d2[0]
        vmla.f16 d30, d9, d2[1]
        vmla.f16 d30, d10, d2[2]
        vmla.f16 d30, d11, d2[3]

        vmla.f16 d31, d8, d3[0]
        vmla.f16 d31, d9, d3[1]
        vmla.f16 d31, d10, d3[2]
        vmla.f16 d31, d11, d3[3]

        subs r4, r4, #1
        bne LoopIC
        b LoopICEnd
    LoopICEnd:
        mov lr, r0
        vst1.16 {q12, q13}, [lr]!
        vst1.16 {q14, q15}, [lr]!
        add r0, r0, r3  // dst += cal_num
        mov r1, r7
        subs r5, r5, #1
        bne LoopOC
LoopOCEnd:
    sub sp, sp, #100
    vpop {q4-q7}
    pop {r4-r11, pc}
#endif
