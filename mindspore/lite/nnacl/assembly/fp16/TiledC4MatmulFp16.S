#include "nnacl/assembly_global.h"

.text
.align 5
.global TiledC4MatmulFp16
#ifndef __APPLE__
.type TiledC4MatmulFp16, %function
#endif

asm_function TiledC4MatmulFp16

sub sp, sp, #128
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64

mov x7, #2 //sizeof(float)
mul x3, x3, x7
mov x7, #32
mul x10, x4, x7

cmp x5, #2
blt LoopOcHalf
LoopOc:
    mov x8, x1
    subs x9, x4, #1

    add x6, x2, x10
    ld1 {v0.4h, v1.4h, v2.4h, v3.4h}, [x8], #32
    ld1 {v8.4h, v9.4h, v10.4h, v11.4h}, [x2], #32
    fmul v16.4h, v8.4h, v0.h[0]
    fmul v17.4h, v8.4h, v1.h[0]
    ld1 {v4.4h, v5.4h, v6.4h, v7.4h}, [x8], #32
    fmul v18.4h, v8.4h, v2.h[0]
    fmul v19.4h, v8.4h, v3.h[0]
    ld1 {v12.4h, v13.4h, v14.4h, v15.4h}, [x6], #32
    fmul v20.4h, v8.4h, v4.h[0]
    fmul v21.4h, v8.4h, v5.h[0]
    fmul v22.4h, v8.4h, v6.h[0]
    fmul v23.4h, v8.4h, v7.h[0]
    fmul v24.4h, v12.4h, v0.h[0]
    fmul v25.4h, v12.4h, v1.h[0]
    fmul v26.4h, v12.4h, v2.h[0]
    fmul v27.4h, v12.4h, v3.h[0]
    fmul v28.4h, v12.4h, v4.h[0]
    fmul v29.4h, v12.4h, v5.h[0]
    fmul v30.4h, v12.4h, v6.h[0]
    fmul v31.4h, v12.4h, v7.h[0]

    beq LoopIcEnd
    LoopIc:
        add x2, x2, #64
        prfm pldl1keep, [x2]
        prfm pldl1keep, [x2, x10]
        sub x2, x2, #64
        prfm pldl1keep, [x8, #64]
        prfm pldl1keep, [x8, #96]

        fmla v16.4h, v9.4h, v0.h[1]
        fmla v17.4h, v9.4h, v1.h[1]
        fmla v18.4h, v9.4h, v2.h[1]
        fmla v19.4h, v9.4h, v3.h[1]
        fmla v20.4h, v9.4h, v4.h[1]
        fmla v21.4h, v9.4h, v5.h[1]
        fmla v22.4h, v9.4h, v6.h[1]
        fmla v23.4h, v9.4h, v7.h[1]
        fmla v24.4h, v13.4h, v0.h[1]
        fmla v25.4h, v13.4h, v1.h[1]
        fmla v26.4h, v13.4h, v2.h[1]
        fmla v27.4h, v13.4h, v3.h[1]
        fmla v28.4h, v13.4h, v4.h[1]
        fmla v29.4h, v13.4h, v5.h[1]
        fmla v30.4h, v13.4h, v6.h[1]
        fmla v31.4h, v13.4h, v7.h[1]

        fmla v16.4h, v10.4h, v0.h[2]
        fmla v17.4h, v10.4h, v1.h[2]
        fmla v18.4h, v10.4h, v2.h[2]
        fmla v19.4h, v10.4h, v3.h[2]
        fmla v20.4h, v10.4h, v4.h[2]
        fmla v21.4h, v10.4h, v5.h[2]
        fmla v22.4h, v10.4h, v6.h[2]
        fmla v23.4h, v10.4h, v7.h[2]
        fmla v24.4h, v14.4h, v0.h[2]
        fmla v25.4h, v14.4h, v1.h[2]
        fmla v26.4h, v14.4h, v2.h[2]
        fmla v27.4h, v14.4h, v3.h[2]
        fmla v28.4h, v14.4h, v4.h[2]
        fmla v29.4h, v14.4h, v5.h[2]
        fmla v30.4h, v14.4h, v6.h[2]
        fmla v31.4h, v14.4h, v7.h[2]

        fmla v16.4h, v11.4h, v0.h[3]
        fmla v17.4h, v11.4h, v1.h[3]
        fmla v18.4h, v11.4h, v2.h[3]
        fmla v19.4h, v11.4h, v3.h[3]
        fmla v20.4h, v11.4h, v4.h[3]
        fmla v21.4h, v11.4h, v5.h[3]
        fmla v22.4h, v11.4h, v6.h[3]
        fmla v23.4h, v11.4h, v7.h[3]
        fmla v24.4h, v15.4h, v0.h[3]
        fmla v25.4h, v15.4h, v1.h[3]
        fmla v26.4h, v15.4h, v2.h[3]
        fmla v27.4h, v15.4h, v3.h[3]
        fmla v28.4h, v15.4h, v4.h[3]
        fmla v29.4h, v15.4h, v5.h[3]
        fmla v30.4h, v15.4h, v6.h[3]
        fmla v31.4h, v15.4h, v7.h[3]

        ld1 {v8.4h, v9.4h, v10.4h, v11.4h}, [x2], #32
        ld1 {v0.4h, v1.4h, v2.4h, v3.4h}, [x8], #32
        fmla v16.4h, v8.4h, v0.h[0]
        fmla v17.4h, v8.4h, v1.h[0]
        ld1 {v4.4h, v5.4h, v6.4h, v7.4h}, [x8], #32
        fmla v18.4h, v8.4h, v2.h[0]
        fmla v19.4h, v8.4h, v3.h[0]
        ld1 {v12.4h, v13.4h, v14.4h, v15.4h}, [x6], #32
        fmla v20.4h, v8.4h, v4.h[0]
        fmla v21.4h, v8.4h, v5.h[0]
        fmla v22.4h, v8.4h, v6.h[0]
        fmla v23.4h, v8.4h, v7.h[0]
        fmla v24.4h, v12.4h, v0.h[0]
        fmla v25.4h, v12.4h, v1.h[0]
        fmla v26.4h, v12.4h, v2.h[0]
        fmla v27.4h, v12.4h, v3.h[0]
        fmla v28.4h, v12.4h, v4.h[0]
        fmla v29.4h, v12.4h, v5.h[0]
        fmla v30.4h, v12.4h, v6.h[0]
        fmla v31.4h, v12.4h, v7.h[0]

        subs x9, x9, #1
        bne LoopIc

    LoopIcEnd:
        fmla v16.4h, v9.4h, v0.h[1]
        fmla v17.4h, v9.4h, v1.h[1]
        fmla v18.4h, v9.4h, v2.h[1]
        fmla v19.4h, v9.4h, v3.h[1]
        fmla v20.4h, v9.4h, v4.h[1]
        fmla v21.4h, v9.4h, v5.h[1]
        fmla v22.4h, v9.4h, v6.h[1]
        fmla v23.4h, v9.4h, v7.h[1]
        fmla v24.4h, v13.4h, v0.h[1]
        fmla v25.4h, v13.4h, v1.h[1]
        fmla v26.4h, v13.4h, v2.h[1]
        fmla v27.4h, v13.4h, v3.h[1]
        fmla v28.4h, v13.4h, v4.h[1]
        fmla v29.4h, v13.4h, v5.h[1]
        fmla v30.4h, v13.4h, v6.h[1]
        fmla v31.4h, v13.4h, v7.h[1]

        fmla v16.4h, v10.4h, v0.h[2]
        fmla v17.4h, v10.4h, v1.h[2]
        fmla v18.4h, v10.4h, v2.h[2]
        fmla v19.4h, v10.4h, v3.h[2]
        fmla v20.4h, v10.4h, v4.h[2]
        fmla v21.4h, v10.4h, v5.h[2]
        fmla v22.4h, v10.4h, v6.h[2]
        fmla v23.4h, v10.4h, v7.h[2]
        fmla v24.4h, v14.4h, v0.h[2]
        fmla v25.4h, v14.4h, v1.h[2]
        fmla v26.4h, v14.4h, v2.h[2]
        fmla v27.4h, v14.4h, v3.h[2]
        fmla v28.4h, v14.4h, v4.h[2]
        fmla v29.4h, v14.4h, v5.h[2]
        fmla v30.4h, v14.4h, v6.h[2]
        fmla v31.4h, v14.4h, v7.h[2]

        add x7, x0, #32

        fmla v16.4h, v11.4h, v0.h[3]
        fmla v17.4h, v11.4h, v1.h[3]
        fmla v18.4h, v11.4h, v2.h[3]
        fmla v19.4h, v11.4h, v3.h[3]
        fmla v20.4h, v11.4h, v4.h[3]
        fmla v21.4h, v11.4h, v5.h[3]
        fmla v22.4h, v11.4h, v6.h[3]
        fmla v23.4h, v11.4h, v7.h[3]
        fmla v24.4h, v15.4h, v0.h[3]
        fmla v25.4h, v15.4h, v1.h[3]
        fmla v26.4h, v15.4h, v2.h[3]
        fmla v27.4h, v15.4h, v3.h[3]
        fmla v28.4h, v15.4h, v4.h[3]
        st1 {v16.4h, v17.4h, v18.4h, v19.4h}, [x0], x3
        fmla v29.4h, v15.4h, v5.h[3]
        st1 {v20.4h, v21.4h, v22.4h, v23.4h}, [x7], x3
        fmla v30.4h, v15.4h, v6.h[3]
        st1 {v24.4h, v25.4h, v26.4h, v27.4h}, [x0], x3
        mov x2, x6
        fmla v31.4h, v15.4h, v7.h[3]
        st1 {v28.4h, v29.4h, v30.4h, v31.4h}, [x7]

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
        ld1 {v8.4h, v9.4h, v10.4h, v11.4h}, [x2], #32
        ld1 {v0.4h, v1.4h, v2.4h, v3.4h}, [x8], #32
        fmla v16.4h, v8.4h, v0.h[0]
        fmla v17.4h, v8.4h, v1.h[0]
        ld1 {v4.4h, v5.4h, v6.4h, v7.4h}, [x8], #32
        fmla v18.4h, v8.4h, v2.h[0]
        fmla v19.4h, v8.4h, v3.h[0]
        fmla v20.4h, v8.4h, v4.h[0]
        fmla v21.4h, v8.4h, v5.h[0]
        fmla v22.4h, v8.4h, v6.h[0]
        fmla v23.4h, v8.4h, v7.h[0]

        fmla v16.4h, v9.4h, v0.h[1]
        fmla v17.4h, v9.4h, v1.h[1]
        fmla v18.4h, v9.4h, v2.h[1]
        fmla v19.4h, v9.4h, v3.h[1]
        fmla v20.4h, v9.4h, v4.h[1]
        fmla v21.4h, v9.4h, v5.h[1]
        fmla v22.4h, v9.4h, v6.h[1]
        fmla v23.4h, v9.4h, v7.h[1]

        fmla v16.4h, v10.4h, v0.h[2]
        fmla v17.4h, v10.4h, v1.h[2]
        fmla v18.4h, v10.4h, v2.h[2]
        fmla v19.4h, v10.4h, v3.h[2]
        fmla v20.4h, v10.4h, v4.h[2]
        fmla v21.4h, v10.4h, v5.h[2]
        fmla v22.4h, v10.4h, v6.h[2]
        fmla v23.4h, v10.4h, v7.h[2]

        fmla v16.4h, v11.4h, v0.h[3]
        fmla v17.4h, v11.4h, v1.h[3]
        fmla v18.4h, v11.4h, v2.h[3]
        fmla v19.4h, v11.4h, v3.h[3]
        fmla v20.4h, v11.4h, v4.h[3]
        fmla v21.4h, v11.4h, v5.h[3]
        fmla v22.4h, v11.4h, v6.h[3]
        fmla v23.4h, v11.4h, v7.h[3]

        subs x9, x9, #1
        bne LoopIcHalf

    st1 {v16.4h, v17.4h, v18.4h, v19.4h}, [x0], #32
    st1 {v20.4h, v21.4h, v22.4h, v23.4h}, [x0], #32

LoopOcEnd:
    sub sp, sp, #128
    ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
    ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
    ret

