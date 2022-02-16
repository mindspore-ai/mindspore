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
#ifdef ENABLE_ARM64
#include "nnacl/assembly_global.h"

.text
.align 5

// void VecMatmulFp16Neon64(const float16_t *a, const float16_t *b, float16_t *c, const float16_t *bias, int act_type,
//                          int depth, int col)
// x0: a
// x1: b
// x2: c
// x3: bias
// w4: act_type
// w5: depth
// w6: col

asm_function VecMatmulFp16Neon64_2
  sub sp, sp, #128
  st1 {v8.8h, v9.8h, v10.8h, v11.8h}, [sp], #64
  st1 {v12.8h, v13.8h, v14.8h, v15.8h}, [sp], #64

LoopCol:
  mov x15, x0   // reload a ptr
  ld1 {v0.8h}, [x3], #16    // acc0
  ld1 {v1.8h}, [x3], #16    // acc1
  mov w9, #0        // tmp depth

Loop2x8Inner:
  sub w18, w5, w9
  cmp w18, #8
  blt DepthRemain

  ld1 {v2.8h}, [x15], #16   // a
  ld1 {v3.8h, v4.8h, v5.8h, v6.8h}, [x1], #64
  ld1 {v7.8h, v8.8h, v9.8h, v10.8h}, [x1], #64
  ld1 {v11.8h, v12.8h, v13.8h, v14.8h}, [x1], #64
  ld1 {v15.8h, v16.8h, v17.8h, v18.8h}, [x1], #64

  fmla v0.8h, v3.8h, v2.h[0]
  fmla v0.8h, v5.8h, v2.h[1]
  fmla v0.8h, v7.8h, v2.h[2]
  fmla v0.8h, v9.8h, v2.h[3]
  fmla v0.8h, v11.8h, v2.h[4]
  fmla v0.8h, v13.8h, v2.h[5]
  fmla v0.8h, v15.8h, v2.h[6]
  fmla v0.8h, v17.8h, v2.h[7]
  fmla v1.8h, v4.8h, v2.h[0]
  fmla v1.8h, v6.8h, v2.h[1]
  fmla v1.8h, v8.8h, v2.h[2]
  fmla v1.8h, v10.8h, v2.h[3]
  fmla v1.8h, v12.8h, v2.h[4]
  fmla v1.8h, v14.8h, v2.h[5]
  fmla v1.8h, v16.8h, v2.h[6]
  fmla v1.8h, v18.8h, v2.h[7]

  add w9, w9, #8
  b Loop2x8Inner

DepthRemain:  // last depth [0, 8)
  cmp w18, #0
  ble Act
  ld1 {v2.h}[0], [x15], #2
  ld1 {v3.8h}, [x1], #16
  ld1 {v4.8h}, [x1], #16
  fmla v0.8h, v3.8h, v2.h[0]
  fmla v1.8h, v4.8h, v2.h[0]
  sub w18, w18, #1
  b DepthRemain

Act:
  cmp w4, #3
  beq Relu6
  cmp w4, #1
  beq Relu
  b Write

Relu6:
  movi v19.8h, #0x46, lsl #8
  fmin v0.8h, v0.8h, v19.8h
  fmin v1.8h, v1.8h, v19.8h

Relu:
  dup v20.8h, wzr
  fmax v0.8h, v0.8h, v20.8h
  fmax v1.8h, v1.8h, v20.8h

Write:
  cmp w6, #8
  blt WriteMod8
  st1 {v0.8h}, [x2], #16
  sub w6, w6, #8
  mov v0.16b, v1.16b
  cmp w6, #8
  blt WriteMod8
  st1 {v1.8h}, [x2], #16
  sub w6, w6, #8
  cbz w6, End
  b LoopCol

WriteMod8:
  cmp w6, #0
  ble End
  cmp w6, #1
  beq Write1
  cmp w6, #2
  beq Write2
  cmp w6, #3
  beq Write3
  cmp w6, #4
  beq Write4
  cmp w6, #5
  beq Write5
  cmp w6, #6
  beq Write6
  cmp w6, #7
  beq Write7

Write1:
  st1 {v0.h}[0], [x2], #2
  b End
Write2:
  st1 {v0.h}[0], [x2], #2
  st1 {v0.h}[1], [x2], #2
  b End
Write3:
  st1 {v0.h}[0], [x2], #2
  st1 {v0.h}[1], [x2], #2
  st1 {v0.h}[2], [x2], #2
  b End
Write4:
  st1 {v0.h}[0], [x2], #2
  st1 {v0.h}[1], [x2], #2
  st1 {v0.h}[2], [x2], #2
  st1 {v0.h}[3], [x2], #2
  b End
Write5:
  st1 {v0.h}[0], [x2], #2
  st1 {v0.h}[1], [x2], #2
  st1 {v0.h}[2], [x2], #2
  st1 {v0.h}[3], [x2], #2
  st1 {v0.h}[4], [x2], #2
  b End
Write6:
  st1 {v0.h}[0], [x2], #2
  st1 {v0.h}[1], [x2], #2
  st1 {v0.h}[2], [x2], #2
  st1 {v0.h}[3], [x2], #2
  st1 {v0.h}[4], [x2], #2
  st1 {v0.h}[5], [x2], #2
  b End
Write7:
  st1 {v0.h}[0], [x2], #2
  st1 {v0.h}[1], [x2], #2
  st1 {v0.h}[2], [x2], #2
  st1 {v0.h}[3], [x2], #2
  st1 {v0.h}[4], [x2], #2
  st1 {v0.h}[5], [x2], #2
  st1 {v0.h}[6], [x2], #2
  b End

End:
  sub sp, sp, #128
  ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [sp], #64
  ld1 {v12.8h, v13.8h, v14.8h, v15.8h}, [sp], #64
  ret
#endif
