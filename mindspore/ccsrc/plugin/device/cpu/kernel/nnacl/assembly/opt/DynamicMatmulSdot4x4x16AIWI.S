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

// void DynamicMatmulSdot4x4x16AIWI(const int8_t *a, const int8_t *b, float *out, size_t deep4, float *multi_scales,
//                                  float *bias, size_t row, size_t col, size_t stride, const int *a_sums,
//                                  const int *b_sums, int64_t a_zp, int64_t b_zp_sum, int64_t act_type);
// x0: a(left matrix ptr)
// x1: b(right matrix ptr)
// x2: out ptr
// x3: deep
// x4: multi_scales
// x5: bias
// x6: row
// x7: col
// x8: stride
// x9: a_sums
// x10: b_sums
// x19/w19: a_zp
// x19/w20: b_zp_sum
// x21: act_type -> 0: none, 1:Relu, 3:Relu6

asm_function DynamicMatmulSdot4x4x16AIWI
    sub sp, sp, #160
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
    stp x19, x20, [sp], #16
    stp x21, x22, [sp], #16
    
    ldr x8, [sp]
    ldr x9, [sp, #8]
    ldr x10, [sp, #16]
    ldr x19, [sp, #24]
    ldr x20, [sp, #32]
    ldr x21, [sp, #40]

    dup v16.4s, wzr // dup:Duplicate general-purpose register to vector.
    dup v17.4s, wzr
    dup v18.4s, wzr
    dup v19.4s, wzr
    dup v20.4s, wzr
    dup v21.4s, wzr
    dup v22.4s, wzr
    dup v23.4s, wzr
    dup v24.4s, wzr
    dup v25.4s, wzr
    dup v26.4s, wzr
    dup v27.4s, wzr
    dup v28.4s, wzr
    dup v29.4s, wzr
    dup v30.4s, wzr
    dup v31.4s, wzr

    mov x11, x1 // reload rhs ptr
    mov x17, x0 // reload lhs ptr
    mov x16, x3 // reload depth

    cmp x7, #4
    ble LoopDepthQuarter
    cmp x7, #8
    ble LoopDepthHalf

LoopDepth:
    ld1 {v0.16b}, [x17], #16
    ld1 {v1.16b, v2.16b, v3.16b, v4.16b}, [x11], #64

    sdot v16.4s, v1.16b, v0.4b[0]
    sdot v17.4s, v2.16b, v0.4b[0]
    sdot v18.4s, v3.16b, v0.4b[0]
    sdot v19.4s, v4.16b, v0.4b[0]
    sdot v20.4s, v1.16b, v0.4b[1]
    sdot v21.4s, v2.16b, v0.4b[1]
    sdot v22.4s, v3.16b, v0.4b[1]
    sdot v23.4s, v4.16b, v0.4b[1]
    sdot v24.4s, v1.16b, v0.4b[2]
    sdot v25.4s, v2.16b, v0.4b[2]
    sdot v26.4s, v3.16b, v0.4b[2]
    sdot v27.4s, v4.16b, v0.4b[2]
    sdot v28.4s, v1.16b, v0.4b[3]
    sdot v29.4s, v2.16b, v0.4b[3]
    sdot v30.4s, v3.16b, v0.4b[3]
    sdot v31.4s, v4.16b, v0.4b[3]

    subs x16, x16, #4
    bgt LoopDepth
    b AddInputSum

LoopDepthHalf:
    ld1 {v0.16b}, [x17], #16
    ld1 {v1.16b, v2.16b}, [x11]
    add x11, x11, #64
    sdot v16.4s, v1.16b, v0.4b[0]
    sdot v17.4s, v2.16b, v0.4b[0]
    sdot v20.4s, v1.16b, v0.4b[1]
    sdot v21.4s, v2.16b, v0.4b[1]
    sdot v24.4s, v1.16b, v0.4b[2]
    sdot v25.4s, v2.16b, v0.4b[2]
    sdot v28.4s, v1.16b, v0.4b[3]
    sdot v29.4s, v2.16b, v0.4b[3]

    subs x16, x16, #4
    bgt LoopDepthHalf
    b AddInputSum

LoopDepthQuarter:
    ld1 {v0.16b}, [x17], #16
    ld1 {v1.16b}, [x11]
    add x11, x11, #64
    sdot v16.4s, v1.16b, v0.4b[0]
    sdot v20.4s, v1.16b, v0.4b[1]
    sdot v24.4s, v1.16b, v0.4b[2]
    sdot v28.4s, v1.16b, v0.4b[3]

    subs x16, x16, #4
    bgt LoopDepthQuarter
    b AddInputSum

AddInputSum:
    cmp w20, #0
    beq AddInputSumEnd
    ld1 {v5.4s}, [x9], #16
    dup v6.4s, v5.s[0]
    dup v7.4s, v5.s[1]
    dup v8.4s, v5.s[2]
    dup v9.4s, v5.s[3]

    sub v16.4s, v16.4s, v6.4s
    sub v17.4s, v17.4s, v6.4s
    sub v18.4s, v18.4s, v6.4s
    sub v19.4s, v19.4s, v6.4s
    sub v20.4s, v20.4s, v7.4s
    sub v21.4s, v21.4s, v7.4s
    sub v22.4s, v22.4s, v7.4s
    sub v23.4s, v23.4s, v7.4s
    sub v24.4s, v24.4s, v8.4s
    sub v25.4s, v25.4s, v8.4s
    sub v26.4s, v26.4s, v8.4s
    sub v27.4s, v27.4s, v8.4s
    sub v28.4s, v28.4s, v9.4s
    sub v29.4s, v29.4s, v9.4s
    sub v30.4s, v30.4s, v9.4s
    sub v31.4s, v31.4s, v9.4s
AddInputSumEnd:

AddWeightSum:
    ld1 {v9.4s},  [x10], #16
    ld1 {v10.4s}, [x10], #16
    ld1 {v11.4s}, [x10], #16
    ld1 {v12.4s}, [x10], #16
    dup v13.4s, w19
    mul v9.4s, v9.4s, v13.4s
    mul v10.4s, v10.4s, v13.4s
    mul v11.4s, v11.4s, v13.4s
    mul v12.4s, v12.4s, v13.4s
    sub v16.4s, v16.4s, v9.4s
    sub v17.4s, v17.4s, v10.4s
    sub v18.4s, v18.4s, v11.4s
    sub v19.4s, v19.4s, v12.4s
    sub v20.4s, v20.4s, v9.4s
    sub v21.4s, v21.4s, v10.4s
    sub v22.4s, v22.4s, v11.4s
    sub v23.4s, v23.4s, v12.4s
    sub v24.4s, v24.4s, v9.4s
    sub v25.4s, v25.4s, v10.4s
    sub v26.4s, v26.4s, v11.4s
    sub v27.4s, v27.4s, v12.4s
    sub v28.4s, v28.4s, v9.4s
    sub v29.4s, v29.4s, v10.4s
    sub v30.4s, v30.4s, v11.4s
    sub v31.4s, v31.4s, v12.4s

AddZpSum:
    mul w15, w19, w20
    cmp w15, #0
    beq AddZpSumEnd
    dup v14.4s, w15
    add v16.4s, v16.4s, v14.4s
    add v17.4s, v17.4s, v14.4s
    add v18.4s, v18.4s, v14.4s
    add v19.4s, v19.4s, v14.4s
    add v20.4s, v20.4s, v14.4s
    add v21.4s, v21.4s, v14.4s
    add v22.4s, v22.4s, v14.4s
    add v23.4s, v23.4s, v14.4s
    add v24.4s, v24.4s, v14.4s
    add v25.4s, v25.4s, v14.4s
    add v26.4s, v26.4s, v14.4s
    add v27.4s, v27.4s, v14.4s
    add v28.4s, v28.4s, v14.4s
    add v29.4s, v29.4s, v14.4s
    add v30.4s, v30.4s, v14.4s
    add v31.4s, v31.4s, v14.4s
AddZpSumEnd:

Convert2Float:
    scvtf v16.4s, v16.4s
    scvtf v17.4s, v17.4s
    scvtf v18.4s, v18.4s
    scvtf v19.4s, v19.4s
    scvtf v20.4s, v20.4s
    scvtf v21.4s, v21.4s
    scvtf v22.4s, v22.4s
    scvtf v23.4s, v23.4s
    scvtf v24.4s, v24.4s
    scvtf v25.4s, v25.4s
    scvtf v26.4s, v26.4s
    scvtf v27.4s, v27.4s
    scvtf v28.4s, v28.4s
    scvtf v29.4s, v29.4s
    scvtf v30.4s, v30.4s
    scvtf v31.4s, v31.4s

MultiplyScale:
    // multi_scale * input_matrix
    ld1 {v1.4s, v2.4s, v3.4s, v4.4s}, [x4]

    fmul v16.4s,v16.4s,v1.4s
    fmul v17.4s,v17.4s,v2.4s
    fmul v18.4s,v18.4s,v3.4s
    fmul v19.4s,v19.4s,v4.4s

    fmul v20.4s,v20.4s,v1.4s
    fmul v21.4s,v21.4s,v2.4s
    fmul v22.4s,v22.4s,v3.4s
    fmul v23.4s,v23.4s,v4.4s

    fmul v24.4s,v24.4s,v1.4s
    fmul v25.4s,v25.4s,v2.4s
    fmul v26.4s,v26.4s,v3.4s
    fmul v27.4s,v27.4s,v4.4s

    fmul v28.4s,v28.4s,v1.4s
    fmul v29.4s,v29.4s,v2.4s
    fmul v30.4s,v30.4s,v3.4s
    fmul v31.4s,v31.4s,v4.4s

AddBias:
    // +bias
    cbz x5, StoreData
    ld1 {v1.4s, v2.4s, v3.4s, v4.4s}, [x5]

    fadd v16.4s,v16.4s,v1.4s
    fadd v17.4s,v17.4s,v2.4s
    fadd v18.4s,v18.4s,v3.4s
    fadd v19.4s,v19.4s,v4.4s

    fadd v20.4s,v20.4s,v1.4s
    fadd v21.4s,v21.4s,v2.4s
    fadd v22.4s,v22.4s,v3.4s
    fadd v23.4s,v23.4s,v4.4s

    fadd v24.4s,v24.4s,v1.4s
    fadd v25.4s,v25.4s,v2.4s
    fadd v26.4s,v26.4s,v3.4s
    fadd v27.4s,v27.4s,v4.4s

    fadd v28.4s,v28.4s,v1.4s
    fadd v29.4s,v29.4s,v2.4s
    fadd v30.4s,v30.4s,v3.4s
    fadd v31.4s,v31.4s,v4.4s

Activate:
    cmp x21, #1
    beq Relu
    cmp x21, #3
    beq Relu6
    b StoreData

Relu:
    dup v1.4s, wzr

    smax v16.4s,v16.4s,v1.4s
    smax v17.4s,v17.4s,v1.4s
    smax v18.4s,v18.4s,v1.4s
    smax v19.4s,v19.4s,v1.4s

    smax v20.4s,v20.4s,v1.4s
    smax v21.4s,v21.4s,v1.4s
    smax v22.4s,v22.4s,v1.4s
    smax v23.4s,v23.4s,v1.4s

    smax v24.4s,v24.4s,v1.4s
    smax v25.4s,v25.4s,v1.4s
    smax v26.4s,v26.4s,v1.4s
    smax v27.4s,v27.4s,v1.4s

    smax v28.4s,v28.4s,v1.4s
    smax v29.4s,v29.4s,v1.4s
    smax v30.4s,v30.4s,v1.4s
    smax v31.4s,v31.4s,v1.4s

    b StoreData

Relu6:
    dup v1.4s, wzr
    movi v2.4s, #6
    scvtf v2.4s, v2.4s

    // max (out, 0)
    smax v16.4s,v16.4s,v1.4s
    smax v17.4s,v17.4s,v1.4s
    smax v18.4s,v18.4s,v1.4s
    smax v19.4s,v19.4s,v1.4s

    smax v20.4s,v20.4s,v1.4s
    smax v21.4s,v21.4s,v1.4s
    smax v22.4s,v22.4s,v1.4s
    smax v23.4s,v23.4s,v1.4s

    smax v24.4s,v24.4s,v1.4s
    smax v25.4s,v25.4s,v1.4s
    smax v26.4s,v26.4s,v1.4s
    smax v27.4s,v27.4s,v1.4s

    smax v28.4s,v28.4s,v1.4s
    smax v29.4s,v29.4s,v1.4s
    smax v30.4s,v30.4s,v1.4s
    smax v31.4s,v31.4s,v1.4s

    // min (out, 6)

    smin v16.4s,v16.4s,v2.4s
    smin v17.4s,v17.4s,v2.4s
    smin v18.4s,v18.4s,v2.4s
    smin v19.4s,v19.4s,v2.4s

    smin v20.4s,v20.4s,v2.4s
    smin v21.4s,v21.4s,v2.4s
    smin v22.4s,v22.4s,v2.4s
    smin v23.4s,v23.4s,v2.4s

    smin v24.4s,v24.4s,v2.4s
    smin v25.4s,v25.4s,v2.4s
    smin v26.4s,v26.4s,v2.4s
    smin v27.4s,v27.4s,v2.4s

    smin v28.4s,v28.4s,v2.4s
    smin v29.4s,v29.4s,v2.4s
    smin v30.4s,v30.4s,v2.4s
    smin v31.4s,v31.4s,v2.4s

    b StoreData

StoreData:
    cmp x7, #16
    beq Write16

    mov x15, x2 // reload out ptr
    add x14, x15, x8
    add x13, x14, x8
    add x12, x13, x8

    cmp x7, #15
    beq Write15
    cmp x7, #14
    beq Write14
    cmp x7, #13
    beq Write13
    cmp x7, #12
    beq Write12
    cmp x7, #11
    beq Write11
    cmp x7, #10
    beq Write10
    cmp x7, #9
    beq Write9
    cmp x7, #8
    beq Write8
    cmp x7, #7
    beq Write7
    cmp x7, #6
    beq Write6
    cmp x7, #5
    beq Write5
    cmp x7, #4
    beq Write4
    cmp x7, #3
    beq Write3
    cmp x7, #2
    beq Write2
    cmp x7, #1
    beq Write1
    b StoreDataEnd

Write16:
    cmp x6, #4
    beq Write16Row4
    cmp x6, #3
    beq Write16Row3
    cmp x6, #2
    beq Write16Row2
    cmp x6, #1
    beq Write16Row1

    Write16Row4:
        st1 {v16.4s,v17.4s,v18.4s,v19.4s}, [x2], x8
        st1 {v20.4s,v21.4s,v22.4s,v23.4s}, [x2], x8
        st1 {v24.4s,v25.4s,v26.4s,v27.4s}, [x2], x8
        st1 {v28.4s,v29.4s,v30.4s,v31.4s}, [x2]
        b StoreDataEnd
    Write16Row3:
        st1 {v16.4s,v17.4s,v18.4s,v19.4s}, [x2], x8
        st1 {v20.4s,v21.4s,v22.4s,v23.4s}, [x2], x8
        st1 {v24.4s,v25.4s,v26.4s,v27.4s}, [x2]
        b StoreDataEnd
    Write16Row2:
        st1 {v16.4s,v17.4s,v18.4s,v19.4s}, [x2], x8
        st1 {v20.4s,v21.4s,v22.4s,v23.4s}, [x2]
        b StoreDataEnd
    Write16Row1:
        st1 {v16.4s,v17.4s,v18.4s,v19.4s}, [x2]
        b StoreDataEnd

Write15:
    st1 {v16.4s,v17.4s,v18.4s}, [x15], #48
    st1 {v19.1d}, [x15], #8
    st1 {v19.s}[2], [x15]
    cmp x6, #1
    beq StoreDataEnd
    st1 {v20.4s,v21.4s,v22.4s}, [x14], #48
    st1 {v23.1d}, [x14], #8
    st1 {v23.s}[2], [x14]
    cmp x6, #2
    beq StoreDataEnd
    st1 {v24.4s,v25.4s,v26.4s}, [x13], #48
    st1 {v27.1d}, [x13], #8
    st1 {v27.s}[2], [x13]
    cmp x6, #3
    beq StoreDataEnd
    st1 {v28.4s,v29.4s,v30.4s}, [x12], #48
    st1 {v31.1d}, [x12], #8
    st1 {v31.s}[2], [x12]
    b StoreDataEnd

Write14:
    st1 {v16.4s,v17.4s,v18.4s}, [x15], #48
    st1 {v19.1d}, [x15]
    cmp x6, #1
    beq StoreDataEnd
    st1 {v20.4s,v21.4s,v22.4s}, [x14], #48
    st1 {v23.1d}, [x14]
    cmp x6, #2
    beq StoreDataEnd
    st1 {v24.4s,v25.4s,v26.4s}, [x13], #48
    st1 {v27.1d}, [x13]
    cmp x6, #3
    beq StoreDataEnd
    st1 {v28.4s,v29.4s,v30.4s}, [x12], #48
    st1 {v31.1d}, [x12]
    b StoreDataEnd

Write13:
    st1 {v16.4s,v17.4s,v18.4s}, [x15], #48
    st1 {v19.s}[0], [x15]
    cmp x6, #1
    beq StoreDataEnd
    st1 {v20.4s,v21.4s,v22.4s}, [x14], #48
    st1 {v23.s}[0], [x14]
    cmp x6, #2
    beq StoreDataEnd
    st1 {v24.4s,v25.4s,v26.4s}, [x13], #48
    st1 {v27.s}[0], [x13]
    cmp x6, #3
    beq StoreDataEnd
    st1 {v28.4s,v29.4s,v30.4s}, [x12], #48
    st1 {v31.s}[0], [x12]
    b StoreDataEnd

Write12:
    st1 {v16.4s,v17.4s,v18.4s}, [x15], #48
    cmp x6, #1
    beq StoreDataEnd
    st1 {v20.4s,v21.4s,v22.4s}, [x14], #48
    cmp x6, #2
    beq StoreDataEnd
    st1 {v24.4s,v25.4s,v26.4s}, [x13], #48
    cmp x6, #3
    beq StoreDataEnd
    st1 {v28.4s,v29.4s,v30.4s}, [x12], #48
    b StoreDataEnd

Write11:
    st1 {v16.4s,v17.4s}, [x15], #32
    st1 {v18.1d}, [x15], #8
    st1 {v18.s}[2], [x15]
    cmp x6, #1
    beq StoreDataEnd
    st1 {v20.4s,v21.4s}, [x14], #32
    st1 {v22.1d}, [x14], #8
    st1 {v22.s}[2], [x14]
    cmp x6, #2
    beq StoreDataEnd
    st1 {v24.4s,v25.4s}, [x13], #32
    st1 {v26.1d}, [x13], #8
    st1 {v26.s}[2], [x13]
    cmp x6, #3
    beq StoreDataEnd
    st1 {v28.4s,v29.4s}, [x12], #32
    st1 {v30.1d}, [x12], #8
    st1 {v30.s}[2], [x12]
    b StoreDataEnd

Write10:
    st1 {v16.4s,v17.4s}, [x15], #32
    st1 {v18.1d}, [x15]
    cmp x6, #1
    beq StoreDataEnd
    st1 {v20.4s,v21.4s}, [x14], #32
    st1 {v22.1d}, [x14]
    cmp x6, #2
    beq StoreDataEnd
    st1 {v24.4s,v25.4s}, [x13], #32
    st1 {v26.1d}, [x13]
    cmp x6, #3
    beq StoreDataEnd
    st1 {v28.4s,v29.4s}, [x12], #32
    st1 {v30.1d}, [x12]
    b StoreDataEnd

Write9:
    st1 {v16.4s,v17.4s}, [x15], #32
    st1 {v18.s}[0], [x15]
    cmp x6, #1
    beq StoreDataEnd
    st1 {v20.4s,v21.4s}, [x14], #32
    st1 {v22.s}[0], [x14]
    cmp x6, #2
    beq StoreDataEnd
    st1 {v24.4s,v25.4s}, [x13], #32
    st1 {v26.s}[0], [x13]
    cmp x6, #3
    beq StoreDataEnd
    st1 {v28.4s,v29.4s}, [x12], #32
    st1 {v30.s}[0], [x12]
    b StoreDataEnd

Write8:
    st1 {v16.4s,v17.4s}, [x15], #32
    cmp x6, #1
    beq StoreDataEnd
    st1 {v20.4s,v21.4s}, [x14], #32
    cmp x6, #2
    beq StoreDataEnd
    st1 {v24.4s,v25.4s}, [x13], #32
    cmp x6, #3
    beq StoreDataEnd
    st1 {v28.4s,v29.4s}, [x12], #32
    b StoreDataEnd

Write7:
    st1 {v16.4s}, [x15], #16
    st1 {v17.1d}, [x15], #8
    st1 {v17.s}[2], [x15]
    cmp x6, #1
    beq StoreDataEnd
    st1 {v20.4s}, [x14], #16
    st1 {v21.1d}, [x14], #8
    st1 {v21.s}[2], [x14]
    cmp x6, #2
    beq StoreDataEnd
    st1 {v24.4s}, [x13], #16
    st1 {v25.1d}, [x13], #8
    st1 {v25.s}[2], [x13]
    cmp x6, #3
    beq StoreDataEnd
    st1 {v28.4s}, [x12], #16
    st1 {v29.1d}, [x12], #8
    st1 {v29.s}[2], [x12]
    b StoreDataEnd

Write6:
    st1 {v16.4s}, [x15], #16
    st1 {v17.1d}, [x15]
    cmp x6, #1
    beq StoreDataEnd
    st1 {v20.4s}, [x14], #16
    st1 {v21.1d}, [x14]
    cmp x6, #2
    beq StoreDataEnd
    st1 {v24.4s}, [x13], #16
    st1 {v25.1d}, [x13]
    cmp x6, #3
    beq StoreDataEnd
    st1 {v28.4s}, [x12], #16
    st1 {v29.1d}, [x12]
    b StoreDataEnd

Write5:
    st1 {v16.4s}, [x15], #16
    st1 {v17.s}[0], [x15]
    cmp x6, #1
    beq StoreDataEnd
    st1 {v20.4s}, [x14], #16
    st1 {v21.s}[0], [x14]
    cmp x6, #2
    beq StoreDataEnd
    st1 {v24.4s}, [x13], #16
    st1 {v25.s}[0], [x13]
    cmp x6, #3
    beq StoreDataEnd
    st1 {v28.4s}, [x12], #16
    st1 {v29.s}[0], [x12]
    b StoreDataEnd

Write4:
    st1 {v16.4s}, [x15]
    cmp x6, #1
    beq StoreDataEnd
    st1 {v20.4s}, [x14]
    cmp x6, #2
    beq StoreDataEnd
    st1 {v24.4s}, [x13]
    cmp x6, #3
    beq StoreDataEnd
    st1 {v28.4s}, [x12]
    b StoreDataEnd

Write3:
    st1 {v16.1d}, [x15], #8
    st1 {v16.s}[2], [x15]
    cmp x6, #1
    beq StoreDataEnd
    st1 {v20.1d}, [x14], #8
    st1 {v20.s}[2], [x14]
    cmp x6, #2
    beq StoreDataEnd
    st1 {v24.1d}, [x13], #8
    st1 {v24.s}[2], [x13]
    cmp x6, #3
    beq StoreDataEnd
    st1 {v28.1d}, [x12], #8
    st1 {v28.s}[2], [x12]
    b StoreDataEnd

Write2:
    st1 {v16.1d}, [x15]
    cmp x6, #1
    beq StoreDataEnd
    st1 {v20.1d}, [x14]
    cmp x6, #2
    beq StoreDataEnd
    st1 {v24.1d}, [x13]
    cmp x6, #3
    beq StoreDataEnd
    st1 {v28.1d}, [x12]
    b StoreDataEnd

Write1:
    st1 {v16.s}[0], [x15]
    cmp x6, #1
    beq StoreDataEnd
    st1 {v20.s}[0], [x14]
    cmp x6, #2
    beq StoreDataEnd
    st1 {v24.s}[0], [x13]
    cmp x6, #3
    beq StoreDataEnd
    st1 {v28.s}[0], [x12]
    b StoreDataEnd
StoreDataEnd:
    sub sp, sp, #160
    ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
    ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
    ldp x19, x20, [sp], #16
    ldp x21, x22, [sp], #16
    ret
#endif
