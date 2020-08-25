#ifdef __aarch64__
    .text
    .align 5
    .global MatmulFloatNeon64OptRemain
#ifndef __APPLE__
    .type MatmulFloatNeon64OptRemain, %function
#endif

// void MatmulFloatNeon64(const float *a, const float *b, float *c, int depth
//                        int row, int col, size_t stride)
// x0: a
// x1: b
// x2: c
// x3: depth
// x4: row
// x5: col
// x6: stride
// only for winograd
MatmulFloatNeon64OptRemain:
    mov x18, #32 // sizeof(float) * 8
    mul x9, x3, x18 // block stride of lhs/rhs: sizeof(float) * 8 * depth
    mov x18, #4
    mul x8, x5, x6
    mov x11, #8
    mul x11, x11, x6
    mul x8, x8, x18
    mul x11, x11, x18

    cmp x4, #4
    ble LoopH4

    LoopH8:
        mov x10, x4 // reload lhs row
        mov x12, x0 // reload lhs ptr
        mov x18, x2 // reload dst ptr

        LoopW8:
            mov x16, x1 // reload rhs ptr
            mov x13, x3 // reload depth
            dup v16.4s, wzr
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

            LoopD8:
                ld1 {v0.4s, v1.4s, v2.4s}, [x12], #48
                ld1 {v3.4s, v4.4s}, [x16], #32
                fmla v16.4s, v3.4s, v0.s[0]
                fmla v18.4s, v3.4s, v0.s[1]
                fmla v20.4s, v3.4s, v0.s[2]
                fmla v22.4s, v3.4s, v0.s[3]
                fmla v17.4s, v4.4s, v0.s[0]
                fmla v19.4s, v4.4s, v0.s[1]
                fmla v21.4s, v4.4s, v0.s[2]
                fmla v23.4s, v4.4s, v0.s[3]
                fmla v24.4s, v3.4s, v1.s[0]
                fmla v26.4s, v3.4s, v1.s[1]
                fmla v28.4s, v3.4s, v1.s[2]
                fmla v30.4s, v3.4s, v1.s[3]
                fmla v25.4s, v4.4s, v1.s[0]
                fmla v27.4s, v4.4s, v1.s[1]
                fmla v29.4s, v4.4s, v1.s[2]
                fmla v31.4s, v4.4s, v1.s[3]

                subs w13, w13, #1
                bgt LoopD8

                st1 {v16.4s, v17.4s}, [x18], x8
                st1 {v18.4s, v19.4s}, [x18], x8
                st1 {v20.4s, v21.4s}, [x18], x8
                st1 {v22.4s, v23.4s}, [x18], x8
                st1 {v24.4s, v25.4s}, [x18], x8
                st1 {v26.4s, v27.4s}, [x18], x8
                st1 {v28.4s, v29.4s}, [x18], x8
                st1 {v30.4s, v31.4s}, [x18], x8

            subs x10, x10, #8 // lhs row - 8
            bgt LoopW8

        subs x5, x5, #8 // rhs col - 8
        add x1, x1, x9 // rhs ptr + stride
        add x2, x2, x11
        bgt LoopH8

        ret

    LoopH4:
        mov x10, x4 // reload lhs row
        mov x12, x0 // reload lhs ptr
        mov x18, x2 // reload dst ptr

        LoopW4:
            mov x16, x1 // reload rhs ptr
            mov x13, x3 // reload depth
            dup v16.4s, wzr
            dup v17.4s, wzr
            dup v18.4s, wzr
            dup v19.4s, wzr
            dup v20.4s, wzr
            dup v21.4s, wzr
            dup v22.4s, wzr
            dup v23.4s, wzr

            LoopD4:
                ld1 {v0.4s, v1.4s, v2.4s}, [x12], #48
                ld1 {v3.4s, v4.4s}, [x16], #32
                fmla v16.4s, v3.4s, v0.s[0]
                fmla v18.4s, v3.4s, v0.s[1]
                fmla v20.4s, v3.4s, v0.s[2]
                fmla v22.4s, v3.4s, v0.s[3]
                fmla v17.4s, v4.4s, v0.s[0]
                fmla v19.4s, v4.4s, v0.s[1]
                fmla v21.4s, v4.4s, v0.s[2]
                fmla v23.4s, v4.4s, v0.s[3]

                subs x13, x13, #1
                bgt LoopD4

                st1 {v16.4s, v17.4s}, [x18], x8
                st1 {v18.4s, v19.4s}, [x18], x8
                st1 {v20.4s, v21.4s}, [x18], x8
                st1 {v22.4s, v23.4s}, [x18], x8

            subs x10, x10, #4 // lhs row - 4
            bgt LoopW4

        subs x5, x5, #8 // rhs col - 8
        add x1, x1, x9 // rhs ptr + stride
        add x2, x2, x11
        bgt LoopH4
    ret
#endif
