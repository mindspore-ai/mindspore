#ifdef __aarch64__
#include "nnacl/assembly_global.h"

    .text
    .align 5
    .global TiledC4MatmulFp32
#ifndef __APPLE__
    .type TiledC4MatmulFp32, %function
#endif

asm_function TiledC4MatmulFp32
//void TiledC4MatmulFp32(float* dst, const float* src, const float* weight, size_t ic4, size_t cal_num, size_t oc4)
//x0: dst
//x1: src
//x2: weight
//x3: cal_num
//x4: ic4
//x5: oc4

sub sp, sp, #128
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64

mov x7, #4 //sizeof(float)
mul x3, x3, x7
mov x7, #64
mul x10, x4, x7

cmp x5, #2
blt LoopOcHalf
LoopOc:
    mov x8, x1
    subs x9, x4, #1

    add x6, x2, x10
    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x8], #64
    ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x2], #64
    fmul v16.4s, v8.4s, v0.s[0]
    fmul v17.4s, v8.4s, v1.s[0]
    ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x8], #64
    fmul v18.4s, v8.4s, v2.s[0]
    fmul v19.4s, v8.4s, v3.s[0]
    ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x6], #64
    fmul v20.4s, v8.4s, v4.s[0]
    fmul v21.4s, v8.4s, v5.s[0]
    fmul v22.4s, v8.4s, v6.s[0]
    fmul v23.4s, v8.4s, v7.s[0]
    fmul v24.4s, v12.4s, v0.s[0]
    fmul v25.4s, v12.4s, v1.s[0]
    fmul v26.4s, v12.4s, v2.s[0]
    fmul v27.4s, v12.4s, v3.s[0]
    fmul v28.4s, v12.4s, v4.s[0]
    fmul v29.4s, v12.4s, v5.s[0]
    fmul v30.4s, v12.4s, v6.s[0]
    fmul v31.4s, v12.4s, v7.s[0]

    beq LoopIcEnd
    LoopIc:
        add x2, x2, #128
        prfm pldl1keep, [x2]
        prfm pldl1keep, [x2, x10]
        sub x2, x2, #128
        prfm pldl1keep, [x8, #128]
        prfm pldl1keep, [x8, #192]

        fmla v16.4s, v9.4s, v0.s[1]
        fmla v17.4s, v9.4s, v1.s[1]
        fmla v18.4s, v9.4s, v2.s[1]
        fmla v19.4s, v9.4s, v3.s[1]
        fmla v20.4s, v9.4s, v4.s[1]
        fmla v21.4s, v9.4s, v5.s[1]
        fmla v22.4s, v9.4s, v6.s[1]
        fmla v23.4s, v9.4s, v7.s[1]
        fmla v24.4s, v13.4s, v0.s[1]
        fmla v25.4s, v13.4s, v1.s[1]
        fmla v26.4s, v13.4s, v2.s[1]
        fmla v27.4s, v13.4s, v3.s[1]
        fmla v28.4s, v13.4s, v4.s[1]
        fmla v29.4s, v13.4s, v5.s[1]
        fmla v30.4s, v13.4s, v6.s[1]
        fmla v31.4s, v13.4s, v7.s[1]

        fmla v16.4s, v10.4s, v0.s[2]
        fmla v17.4s, v10.4s, v1.s[2]
        fmla v18.4s, v10.4s, v2.s[2]
        fmla v19.4s, v10.4s, v3.s[2]
        fmla v20.4s, v10.4s, v4.s[2]
        fmla v21.4s, v10.4s, v5.s[2]
        fmla v22.4s, v10.4s, v6.s[2]
        fmla v23.4s, v10.4s, v7.s[2]
        fmla v24.4s, v14.4s, v0.s[2]
        fmla v25.4s, v14.4s, v1.s[2]
        fmla v26.4s, v14.4s, v2.s[2]
        fmla v27.4s, v14.4s, v3.s[2]
        fmla v28.4s, v14.4s, v4.s[2]
        fmla v29.4s, v14.4s, v5.s[2]
        fmla v30.4s, v14.4s, v6.s[2]
        fmla v31.4s, v14.4s, v7.s[2]

        fmla v16.4s, v11.4s, v0.s[3]
        fmla v17.4s, v11.4s, v1.s[3]
        fmla v18.4s, v11.4s, v2.s[3]
        fmla v19.4s, v11.4s, v3.s[3]
        fmla v20.4s, v11.4s, v4.s[3]
        fmla v21.4s, v11.4s, v5.s[3]
        fmla v22.4s, v11.4s, v6.s[3]
        fmla v23.4s, v11.4s, v7.s[3]
        fmla v24.4s, v15.4s, v0.s[3]
        fmla v25.4s, v15.4s, v1.s[3]
        fmla v26.4s, v15.4s, v2.s[3]
        fmla v27.4s, v15.4s, v3.s[3]
        fmla v28.4s, v15.4s, v4.s[3]
        fmla v29.4s, v15.4s, v5.s[3]
        fmla v30.4s, v15.4s, v6.s[3]
        fmla v31.4s, v15.4s, v7.s[3]

        ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x2], #64
        ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x8], #64
        fmla v16.4s, v8.4s, v0.s[0]
        fmla v17.4s, v8.4s, v1.s[0]
        ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x8], #64
        fmla v18.4s, v8.4s, v2.s[0]
        fmla v19.4s, v8.4s, v3.s[0]
        ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x6], #64
        fmla v20.4s, v8.4s, v4.s[0]
        fmla v21.4s, v8.4s, v5.s[0]
        fmla v22.4s, v8.4s, v6.s[0]
        fmla v23.4s, v8.4s, v7.s[0]
        fmla v24.4s, v12.4s, v0.s[0]
        fmla v25.4s, v12.4s, v1.s[0]
        fmla v26.4s, v12.4s, v2.s[0]
        fmla v27.4s, v12.4s, v3.s[0]
        fmla v28.4s, v12.4s, v4.s[0]
        fmla v29.4s, v12.4s, v5.s[0]
        fmla v30.4s, v12.4s, v6.s[0]
        fmla v31.4s, v12.4s, v7.s[0]

        subs x9, x9, #1
        bne LoopIc

    LoopIcEnd:
        fmla v16.4s, v9.4s, v0.s[1]
        fmla v17.4s, v9.4s, v1.s[1]
        fmla v18.4s, v9.4s, v2.s[1]
        fmla v19.4s, v9.4s, v3.s[1]
        fmla v20.4s, v9.4s, v4.s[1]
        fmla v21.4s, v9.4s, v5.s[1]
        fmla v22.4s, v9.4s, v6.s[1]
        fmla v23.4s, v9.4s, v7.s[1]
        fmla v24.4s, v13.4s, v0.s[1]
        fmla v25.4s, v13.4s, v1.s[1]
        fmla v26.4s, v13.4s, v2.s[1]
        fmla v27.4s, v13.4s, v3.s[1]
        fmla v28.4s, v13.4s, v4.s[1]
        fmla v29.4s, v13.4s, v5.s[1]
        fmla v30.4s, v13.4s, v6.s[1]
        fmla v31.4s, v13.4s, v7.s[1]

        fmla v16.4s, v10.4s, v0.s[2]
        fmla v17.4s, v10.4s, v1.s[2]
        fmla v18.4s, v10.4s, v2.s[2]
        fmla v19.4s, v10.4s, v3.s[2]
        fmla v20.4s, v10.4s, v4.s[2]
        fmla v21.4s, v10.4s, v5.s[2]
        fmla v22.4s, v10.4s, v6.s[2]
        fmla v23.4s, v10.4s, v7.s[2]
        fmla v24.4s, v14.4s, v0.s[2]
        fmla v25.4s, v14.4s, v1.s[2]
        fmla v26.4s, v14.4s, v2.s[2]
        fmla v27.4s, v14.4s, v3.s[2]
        fmla v28.4s, v14.4s, v4.s[2]
        fmla v29.4s, v14.4s, v5.s[2]
        fmla v30.4s, v14.4s, v6.s[2]
        fmla v31.4s, v14.4s, v7.s[2]

        add x7, x0, #64

        fmla v16.4s, v11.4s, v0.s[3]
        fmla v17.4s, v11.4s, v1.s[3]
        fmla v18.4s, v11.4s, v2.s[3]
        fmla v19.4s, v11.4s, v3.s[3]
        fmla v20.4s, v11.4s, v4.s[3]
        fmla v21.4s, v11.4s, v5.s[3]
        fmla v22.4s, v11.4s, v6.s[3]
        fmla v23.4s, v11.4s, v7.s[3]
        fmla v24.4s, v15.4s, v0.s[3]
        fmla v25.4s, v15.4s, v1.s[3]
        fmla v26.4s, v15.4s, v2.s[3]
        fmla v27.4s, v15.4s, v3.s[3]
        fmla v28.4s, v15.4s, v4.s[3]
        st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], x3
        fmla v29.4s, v15.4s, v5.s[3]
        st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x7], x3
        fmla v30.4s, v15.4s, v6.s[3]
        st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x0], x3
        mov x2, x6
        fmla v31.4s, v15.4s, v7.s[3]
        st1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x7]

        subs x5, x5, #2
        beq LoopOcEnd
        cmp x5, #2
        bge LoopOc

LoopOcHalf:
    mov x8, x1
    mov x9, x4
    dup v16.4s, wzr
    dup v17.4s, wzr
    dup v18.4s, wzr
    dup v19.4s, wzr
    dup v20.4s, wzr
    dup v21.4s, wzr
    dup v22.4s, wzr
    dup v23.4s, wzr

    LoopIcHalf:
        ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x2], #64
        ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x8], #64
        fmla v16.4s, v8.4s, v0.s[0]
        fmla v17.4s, v8.4s, v1.s[0]
        ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x8], #64
        fmla v18.4s, v8.4s, v2.s[0]
        fmla v19.4s, v8.4s, v3.s[0]
        fmla v20.4s, v8.4s, v4.s[0]
        fmla v21.4s, v8.4s, v5.s[0]
        fmla v22.4s, v8.4s, v6.s[0]
        fmla v23.4s, v8.4s, v7.s[0]

        fmla v16.4s, v9.4s, v0.s[1]
        fmla v17.4s, v9.4s, v1.s[1]
        fmla v18.4s, v9.4s, v2.s[1]
        fmla v19.4s, v9.4s, v3.s[1]
        fmla v20.4s, v9.4s, v4.s[1]
        fmla v21.4s, v9.4s, v5.s[1]
        fmla v22.4s, v9.4s, v6.s[1]
        fmla v23.4s, v9.4s, v7.s[1]

        fmla v16.4s, v10.4s, v0.s[2]
        fmla v17.4s, v10.4s, v1.s[2]
        fmla v18.4s, v10.4s, v2.s[2]
        fmla v19.4s, v10.4s, v3.s[2]
        fmla v20.4s, v10.4s, v4.s[2]
        fmla v21.4s, v10.4s, v5.s[2]
        fmla v22.4s, v10.4s, v6.s[2]
        fmla v23.4s, v10.4s, v7.s[2]

        fmla v16.4s, v11.4s, v0.s[3]
        fmla v17.4s, v11.4s, v1.s[3]
        fmla v18.4s, v11.4s, v2.s[3]
        fmla v19.4s, v11.4s, v3.s[3]
        fmla v20.4s, v11.4s, v4.s[3]
        fmla v21.4s, v11.4s, v5.s[3]
        fmla v22.4s, v11.4s, v6.s[3]
        fmla v23.4s, v11.4s, v7.s[3]

        subs x9, x9, #1
        bne LoopIcHalf

    st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], #64
    st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x0], #64

LoopOcEnd:
    sub sp, sp, #128
    ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
    ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
    ret
#endif
