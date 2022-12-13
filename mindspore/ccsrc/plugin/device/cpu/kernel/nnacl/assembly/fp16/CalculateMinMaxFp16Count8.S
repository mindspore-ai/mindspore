/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifdef ENABLE_ARM64
#include "nnacl/assembly_global.h"

.text
.align 5

// void CalculateMinMaxCount8Fp16(const float16_t *data, int count_8, float16_t *real_min, float16_t *real_max);
// x0: data
// w1: count_8
// x2: real_min
// x3: real_max

asm_function CalculateMinMaxCount8Fp16
    // registers v8 ~ v15 must be preserved by a callee across subroutine calls, according to
    // https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst#simd-and-floating-point-registers
    // x19 ~ x29 should be also preserved
    // whereas our coding style do not permit such amount of parameters

    mov x4, x0 // reload data
    mov w5, w1 // reload count
    ld1 {v31.8h}, [x4]
    ld1 {v30.8h}, [x4], #16
    subs w5, w5, #8
    ble Write

    Loop:
    ld1 {v0.8h}, [x4], #16
    fmin v31.8h, v31.8h, v0.8h
    fmax v30.8h, v30.8h, v0.8h
    subs w5, w5, #8
    bgt Loop

    Write:
    fminv h6, v31.8h
    fmaxv h7, v30.8h

    str h6, [x2]
    str h7, [x3]

    ret
#endif
