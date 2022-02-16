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

// void SPMM8x8Fp32(const float *a, const float *b, const uint32_t *nnz, const size_t *dmap, float *c,
//                  const float *bias, ActType act_type, size_t out_stride);
// x0: a
// x1: b
// x2: nnz
// x3: dmap
// x4: c
// x5: bias
// w6: act_type
// x7: out_stride

// wdata tmp           w8
// loop_oc_count       w9
// loop_nnz_count      w10
// dmap tmp            w11
// a_ptr
// 8 x 1 fp32 A        v0-v1
// fp32 B-value        v2
// uint32 B-NNZ        x9
// uint32 B-INDEX      x10
// 4 MIN               v3
// 4 MAX               v4
// 2 vacc              v5-v6
// 8 x 8 fp32 C        v16-v31

// v16[0] v18[0] v20[0] v22[0] v24[0] v26[0] v28[0] v30[0]
// v16[1] v18[1] v20[1] v22[1] v24[1] v26[1] v28[1] v30[1]
// v16[2] v18[2] v20[2] v22[2] v24[2] v26[2] v28[2] v30[2]
// v16[3] v18[3] v20[3] v22[3] v24[3] v26[3] v28[3] v30[3]
// v17[0] v19[0] v21[0] v23[0] v25[0] v27[0] v29[0] v31[0]
// v17[1] v19[1] v21[1] v23[1] v25[1] v27[1] v29[1] v31[1]
// v17[2] v19[2] v21[2] v23[2] v25[2] v27[2] v29[2] v31[2]
// v17[3] v19[3] v21[3] v23[3] v25[3] v27[3] v29[3] v31[3]

asm_function SPMM8x8Fp32
    sub sp, sp, #144
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
    stp x19, x20, [sp], #16

    // init output with bias
    ldr w8, [x5], #4
    dup v16.4s, w8
    dup v17.4s, w8
    ldr w8, [x5], #4
    dup v18.4s, w8
    dup v19.4s, w8
    ldr w8, [x5], #4
    dup v20.4s, w8
    dup v21.4s, w8
    ldr w8, [x5], #4
    dup v22.4s, w8
    dup v23.4s, w8
    ldr w8, [x5], #4
    dup v24.4s, w8
    dup v25.4s, w8
    ldr w8, [x5], #4
    dup v26.4s, w8
    dup v27.4s, w8
    ldr w8, [x5], #4
    dup v28.4s, w8
    dup v29.4s, w8
    ldr w8, [x5]
    dup v30.4s, w8
    dup v31.4s, w8

    // OC 0
    ldr w10, [x2], #4           // load nnz
    cmp w10, #0
    beq OC_1
LOOP_NNZ0:
    ldr x11, [x3], #8           // load dmap
    add x8, x0, x11
    ld1 {v0.4s, v1.4s}, [x8]    // load inputs
    ldr w8, [x1], #4            // load weight
    dup v2.4s, w8
    // matmul
    fmla v16.4s, v0.4s, v2.4s
    fmla v17.4s, v1.4s, v2.4s
    // loop nnz condition
    subs w10, w10, #1
    bgt LOOP_NNZ0

OC_1:
    ldr w10, [x2], #4
    cmp w10, #0
    beq OC_2
LOOP_NNZ1:
    ldr x11, [x3], #8
    add x8, x0, x11
    ld1 {v0.4s, v1.4s}, [x8]
    ldr w8, [x1], #4
    dup v2.4s, w8
    // matmul
    fmla v18.4s, v0.4s, v2.4s
    fmla v19.4s, v1.4s, v2.4s
    // loop nnz condition
    subs w10, w10, #1
    bgt LOOP_NNZ1

OC_2:
    ldr w10, [x2], #4
    cmp w10, #0
    beq OC_3
LOOP_NNZ2:
    ldr x11, [x3], #8
    add x8, x0, x11
    ld1 {v0.4s, v1.4s}, [x8]
    ldr w8, [x1], #4
    dup v2.4s, w8
    // matmul
    fmla v20.4s, v0.4s, v2.4s
    fmla v21.4s, v1.4s, v2.4s
    // loop nnz condition
    subs w10, w10, #1
    bgt LOOP_NNZ2

OC_3:
    ldr w10, [x2], #4
    cmp w10, #0
    beq OC_4
LOOP_NNZ3:
    ldr x11, [x3], #8
    add x8, x0, x11
    ld1 {v0.4s, v1.4s}, [x8]
    ldr w8, [x1], #4
    dup v2.4s, w8
    // matmul
    fmla v22.4s, v0.4s, v2.4s
    fmla v23.4s, v1.4s, v2.4s
    // loop nnz condition
    subs w10, w10, #1
    bgt LOOP_NNZ3

OC_4:
    ldr w10, [x2], #4
    cmp w10, #0
    beq OC_5
LOOP_NNZ4:
    ldr x11, [x3], #8
    add x8, x0, x11
    ld1 {v0.4s, v1.4s}, [x8]
    ldr w8, [x1], #4
    dup v2.4s, w8
    // matmul
    fmla v24.4s, v0.4s, v2.4s
    fmla v25.4s, v1.4s, v2.4s
    // loop nnz condition
    subs w10, w10, #1
    bgt LOOP_NNZ4

OC_5:
    ldr w10, [x2], #4
    cmp w10, #0
    beq OC_6
LOOP_NNZ5:
    ldr x11, [x3], #8
    add x8, x0, x11
    ld1 {v0.4s, v1.4s}, [x8]
    ldr w8, [x1], #4
    dup v2.4s, w8
    // matmul
    fmla v26.4s, v0.4s, v2.4s
    fmla v27.4s, v1.4s, v2.4s
    // loop nnz condition
    subs w10, w10, #1
    bgt LOOP_NNZ5

OC_6:
    ldr w10, [x2], #4
    cmp w10, #0
    beq OC_7
LOOP_NNZ6:
    ldr x11, [x3], #8
    add x8, x0, x11
    ld1 {v0.4s, v1.4s}, [x8]
    ldr w8, [x1], #4
    dup v2.4s, w8
    // matmul
    fmla v28.4s, v0.4s, v2.4s
    fmla v29.4s, v1.4s, v2.4s
    // loop nnz condition
    subs w10, w10, #1
    bgt LOOP_NNZ6

OC_7:
    ldr w10, [x2], #4
    cmp w10, #0
    beq REORDER_OUT
LOOP_NNZ7:
    ldr x11, [x3], #8
    add x8, x0, x11
    ld1 {v0.4s, v1.4s}, [x8]
    ldr w8, [x1], #4
    dup v2.4s, w8
    // matmul
    fmla v30.4s, v0.4s, v2.4s
    fmla v31.4s, v1.4s, v2.4s
    // loop nnz condition
    subs w10, w10, #1
    bgt LOOP_NNZ7

    // reorder output
// v16[0] v18[0] v20[0] v22[0] v24[0] v26[0] v28[0] v30[0]
// v16[1] v18[1] v20[1] v22[1] v24[1] v26[1] v28[1] v30[1]
// v16[2] v18[2] v20[2] v22[2] v24[2] v26[2] v28[2] v30[2]
// v16[3] v18[3] v20[3] v22[3] v24[3] v26[3] v28[3] v30[3]
// v17[0] v19[0] v21[0] v23[0] v25[0] v27[0] v29[0] v31[0]
// v17[1] v19[1] v21[1] v23[1] v25[1] v27[1] v29[1] v31[1]
// v17[2] v19[2] v21[2] v23[2] v25[2] v27[2] v29[2] v31[2]
// v17[3] v19[3] v21[3] v23[3] v25[3] v27[3] v29[3] v31[3]

// v0[0]  v0[1]  v0[2]  v0[3]  v1[0]  v1[1]  v1[2]  v1[3]
// v2[0]  v2[1]  v2[2]  v2[3]  v3[0]  v3[1]  v3[2]  v3[3]
// v4[0]  v4[1]  v4[2]  v4[3]  v5[0]  v5[1]  v5[2]  v5[3]
// v6[0]  v6[1]  v6[2]  v6[3]  v7[0]  v7[1]  v7[2]  v7[3]
// v8[0]  v8[1]  v8[2]  v8[3]  v9[0]  v9[1]  v9[2]  v9[3]
// v10[0] v10[1] v10[2] v10[3] v11[0] v11[1] v11[2] v11[3]
// v12[0] v12[1] v12[2] v12[3] v13[0] v13[1] v13[2] v13[3]
// v14[0] v14[1] v14[2] v14[3] v15[0] v15[1] v15[2] v15[3]

REORDER_OUT:
    zip1 v1.4s, v16.4s, v18.4s
    zip2 v3.4s, v16.4s, v18.4s
    zip1 v9.4s, v17.4s, v19.4s
    zip2 v11.4s, v17.4s, v19.4s
    zip1 v5.4s, v20.4s, v22.4s
    zip2 v7.4s, v20.4s, v22.4s
    zip1 v13.4s, v21.4s, v23.4s
    zip2 v15.4s, v21.4s, v23.4s
    trn1 v0.2d, v1.2d, v5.2d
    trn2 v2.2d, v1.2d, v5.2d
    trn1 v4.2d, v3.2d, v7.2d
    trn2 v6.2d, v3.2d, v7.2d
    trn1 v8.2d, v9.2d, v13.2d
    trn2 v10.2d, v9.2d, v13.2d
    trn1 v12.2d, v11.2d, v15.2d
    trn2 v14.2d, v11.2d, v15.2d

    zip1 v16.4s, v24.4s, v26.4s
    zip2 v17.4s, v24.4s, v26.4s
    zip1 v20.4s, v25.4s, v27.4s
    zip2 v21.4s, v25.4s, v27.4s
    zip1 v18.4s, v28.4s, v30.4s
    zip2 v19.4s, v28.4s, v30.4s
    zip1 v22.4s, v29.4s, v31.4s
    zip2 v23.4s, v29.4s, v31.4s
    trn1 v1.2d, v16.2d, v18.2d
    trn2 v3.2d, v16.2d, v18.2d
    trn1 v5.2d, v17.2d, v19.2d
    trn2 v7.2d, v17.2d, v19.2d
    trn1 v9.2d, v20.2d, v22.2d
    trn2 v11.2d, v20.2d, v22.2d
    trn1 v13.2d, v21.2d, v23.2d
    trn2 v15.2d, v21.2d, v23.2d

WRITE_OUT:
    st1 {v0.4s, v1.4s}, [x4], x7
    st1 {v2.4s, v3.4s}, [x4], x7
    st1 {v4.4s, v5.4s}, [x4], x7
    st1 {v6.4s, v7.4s}, [x4], x7
    st1 {v8.4s, v9.4s}, [x4], x7
    st1 {v10.4s, v11.4s}, [x4], x7
    st1 {v12.4s, v13.4s}, [x4], x7
    st1 {v14.4s, v15.4s}, [x4]

End:
  sub sp, sp, #144
  ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
  ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
  ldp x19, x20, [sp], #16
  ret
#endif
