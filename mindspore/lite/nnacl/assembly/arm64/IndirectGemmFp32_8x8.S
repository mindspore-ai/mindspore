#ifdef __aarch64__

.text
.align 5
.global IndirectGemmFp32_8x8
#ifndef __APPLE__
.type IndirectGemmFp32_8x8, %function
#endif

// void IndirectGemmFp32_8x8(float *output, float *input, float *weight, float *bias,
//     size_t kSize, size_t ic4, size_t oc8, size_t offset, size_t mode, size_t writeC4, size_t relu, size_t relu6);
// x0: output, x1: input, x2: weight, x3: bias, x4: kSize, x5: ic4, x6: oc, x7: offset
// x8:mode, x9: writeMode, x10: relu, x11:relu6
// mode = 0 for general convolution, where one conv unit is a row
// mode = 1 for winograd/common gemm, where the total channels of one input is a row
IndirectGemmFp32_8x8:

    .macro INIT_BIAS
        dup v16.4s, wzr
        dup v17.4s, wzr
        cbz x3, InitBias
        ld1 {v16.4s, v17.4s}, [x3]
    InitBias:
        mov v18.16b, v16.16b
        mov v19.16b, v17.16b
        mov v20.16b, v16.16b
        mov v21.16b, v17.16b
        mov v22.16b, v16.16b
        mov v23.16b, v17.16b
        mov v24.16b, v16.16b
        mov v25.16b, v17.16b
        mov v26.16b, v16.16b
        mov v27.16b, v17.16b
        mov v28.16b, v16.16b
        mov v29.16b, v17.16b
        mov v30.16b, v16.16b
        mov v31.16b, v17.16b
    .endm

    .macro INIT_BIAS_HALF
        dup v16.4s, wzr
        cbz x3, InitBiasHalf
        ld1 {v16.4s}, [x3]
    InitBiasHalf:
        mov v18.16b, v16.16b
        mov v20.16b, v16.16b
        mov v22.16b, v16.16b
        mov v24.16b, v16.16b
        mov v26.16b, v16.16b
        mov v28.16b, v16.16b
        mov v30.16b, v16.16b
    .endm

    // registers v8 ~ v15 must be preserved by a callee across subroutine calls, according to
    // https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst#simd-and-floating-point-registers
    // r19 ~ r29 should be also preserved
    // whereas our coding style do not permit such amount of parameters
    sub sp, sp, #128
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64

    ldr x8, [sp, #0]
    ldr x9, [sp, #8]
    ldr x10, [sp, #16]
    ldr x11, [sp, #24]

    cbnz x8, NoStepShuffle
    // step is one for common convolution, where ic8 should multiply by kernel size 
    // step is (a+b-1) for F(a,b) in winograd
    mul x5, x4, x5
    mov x4, #1

NoStepShuffle:
    // x8 is used to store offset now
    // only useful for WriteC4
    mov x8, #16
    mul x8, x8, x4

IndirectGemmStart:

    cmp x6, #4
    ble LoopOcHalf

    LoopOc:

        mov x14, x4
        mov x12, x1

        LoopKsize:

            mov x15, x0
            INIT_BIAS

            // load input for output 1-2
            ld1 {v0.4s, v1.4s}, [x12], #32
            // load weight
            ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x2], #64
            // step for output 1-2
            fmla v16.4s, v8.4s, v0.s[0]
            fmla v17.4s, v9.4s, v0.s[0]
            fmla v18.4s, v8.4s, v1.s[0]
            fmla v19.4s, v9.4s, v1.s[0]
            // load input for output 3-4
            ld1 {v2.4s, v3.4s}, [x12], #32
            // another step for output 1-2
            fmla v16.4s, v10.4s, v0.s[1]
            fmla v17.4s, v11.4s, v0.s[1]
            fmla v18.4s, v10.4s, v1.s[1]
            fmla v19.4s, v11.4s, v1.s[1]
            // load input  for output 5-8
            // input cache should be refreshed after loading
            // ATTENTION: advance is prefered, but advancing too much may lead to invalid prefetching 
            ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x12], #64
            // step for output 3-8
            fmla v20.4s, v8.4s, v2.s[0]
            fmla v21.4s, v9.4s, v2.s[0]
            fmla v22.4s, v8.4s, v3.s[0]
            fmla v23.4s, v9.4s, v3.s[0]

            subs x13, x5, #1
            beq LoopIcEnd

            LoopIc:
                fmla v24.4s, v8.4s, v4.s[0]
                fmla v25.4s, v9.4s, v4.s[0]
                fmla v26.4s, v8.4s, v5.s[0]
                fmla v27.4s, v9.4s, v5.s[0]
                fmla v28.4s, v8.4s, v6.s[0]
                fmla v29.4s, v9.4s, v6.s[0]
                fmla v30.4s, v8.4s, v7.s[0]
                fmla v31.4s, v9.4s, v7.s[0]
                // load weight
                ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x2], #64
                // step for output 3-8
                fmla v20.4s, v10.4s, v2.s[1]
                fmla v21.4s, v11.4s, v2.s[1]
                fmla v22.4s, v10.4s, v3.s[1]
                fmla v23.4s, v11.4s, v3.s[1]
                fmla v24.4s, v10.4s, v4.s[1]
                fmla v25.4s, v11.4s, v4.s[1]
                fmla v26.4s, v10.4s, v5.s[1]
                fmla v27.4s, v11.4s, v5.s[1]
                fmla v28.4s, v10.4s, v6.s[1]
                fmla v29.4s, v11.4s, v6.s[1]
                fmla v30.4s, v10.4s, v7.s[1]
                fmla v31.4s, v11.4s, v7.s[1]
                // another step for output 1-8
                fmla v16.4s, v12.4s, v0.s[2]
                fmla v17.4s, v13.4s, v0.s[2]
                fmla v18.4s, v12.4s, v1.s[2]
                fmla v19.4s, v13.4s, v1.s[2]
                fmla v20.4s, v12.4s, v2.s[2]
                fmla v21.4s, v13.4s, v2.s[2]
                fmla v22.4s, v12.4s, v3.s[2]
                fmla v23.4s, v13.4s, v3.s[2]
                fmla v24.4s, v12.4s, v4.s[2]
                fmla v25.4s, v13.4s, v4.s[2]
                fmla v26.4s, v12.4s, v5.s[2]
                fmla v27.4s, v13.4s, v5.s[2]
                fmla v28.4s, v12.4s, v6.s[2]
                fmla v29.4s, v13.4s, v6.s[2]
                fmla v30.4s, v12.4s, v7.s[2]
                fmla v31.4s, v13.4s, v7.s[2]
                // load weight
                ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x2], #64
                // another step for output 1-8
                fmla v16.4s, v14.4s, v0.s[3]
                fmla v17.4s, v15.4s, v0.s[3]
                fmla v18.4s, v14.4s, v1.s[3]
                fmla v19.4s, v15.4s, v1.s[3]
                fmla v20.4s, v14.4s, v2.s[3]
                fmla v21.4s, v15.4s, v2.s[3]
                fmla v22.4s, v14.4s, v3.s[3]
                fmla v23.4s, v15.4s, v3.s[3]
                // load input for output 1-4
                ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x12], #64
                fmla v24.4s, v14.4s, v4.s[3]
                fmla v25.4s, v15.4s, v4.s[3]
                fmla v26.4s, v14.4s, v5.s[3]
                fmla v27.4s, v15.4s, v5.s[3]
                fmla v28.4s, v14.4s, v6.s[3]
                fmla v29.4s, v15.4s, v6.s[3]
                fmla v30.4s, v14.4s, v7.s[3]
                fmla v31.4s, v15.4s, v7.s[3]
                // load input  for output 5-8
                ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x12], #64
                // step for output 1-8
                fmla v16.4s, v8.4s, v0.s[0]
                fmla v17.4s, v9.4s, v0.s[0]
                fmla v18.4s, v8.4s, v1.s[0]
                fmla v19.4s, v9.4s, v1.s[0]
                fmla v16.4s, v10.4s, v0.s[1]
                fmla v17.4s, v11.4s, v0.s[1]
                fmla v18.4s, v10.4s, v1.s[1]
                fmla v19.4s, v11.4s, v1.s[1]
                fmla v20.4s, v8.4s, v2.s[0]
                fmla v21.4s, v9.4s, v2.s[0]
                fmla v22.4s, v8.4s, v3.s[0]
                fmla v23.4s, v9.4s, v3.s[0]

                subs x13, x13, #1
                bne LoopIc

            LoopIcEnd:
                fmla v24.4s, v8.4s, v4.s[0]
                fmla v25.4s, v9.4s, v4.s[0]
                fmla v26.4s, v8.4s, v5.s[0]
                fmla v27.4s, v9.4s, v5.s[0]
                fmla v28.4s, v8.4s, v6.s[0]
                fmla v29.4s, v9.4s, v6.s[0]
                fmla v30.4s, v8.4s, v7.s[0]
                fmla v31.4s, v9.4s, v7.s[0]
                // load weight
                ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x2], #64
                // step for output 3-8
                fmla v20.4s, v10.4s, v2.s[1]
                fmla v21.4s, v11.4s, v2.s[1]
                fmla v22.4s, v10.4s, v3.s[1]
                fmla v23.4s, v11.4s, v3.s[1]
                fmla v24.4s, v10.4s, v4.s[1]
                fmla v25.4s, v11.4s, v4.s[1]
                fmla v26.4s, v10.4s, v5.s[1]
                fmla v27.4s, v11.4s, v5.s[1]
                fmla v28.4s, v10.4s, v6.s[1]
                fmla v29.4s, v11.4s, v6.s[1]
                fmla v30.4s, v10.4s, v7.s[1]
                fmla v31.4s, v11.4s, v7.s[1]
                // another step for output 1-8
                fmla v16.4s, v12.4s, v0.s[2]
                fmla v17.4s, v13.4s, v0.s[2]
                fmla v18.4s, v12.4s, v1.s[2]
                fmla v19.4s, v13.4s, v1.s[2]
                fmla v20.4s, v12.4s, v2.s[2]
                fmla v21.4s, v13.4s, v2.s[2]
                fmla v22.4s, v12.4s, v3.s[2]
                fmla v23.4s, v13.4s, v3.s[2]
                fmla v24.4s, v12.4s, v4.s[2]
                fmla v25.4s, v13.4s, v4.s[2]
                fmla v26.4s, v12.4s, v5.s[2]
                fmla v27.4s, v13.4s, v5.s[2]
                fmla v28.4s, v12.4s, v6.s[2]
                fmla v29.4s, v13.4s, v6.s[2]
                fmla v30.4s, v12.4s, v7.s[2]
                fmla v31.4s, v13.4s, v7.s[2]
                // another step for output 1-8
                fmla v16.4s, v14.4s, v0.s[3]
                fmla v17.4s, v15.4s, v0.s[3]
                fmla v18.4s, v14.4s, v1.s[3]
                fmla v19.4s, v15.4s, v1.s[3]
                fmla v20.4s, v14.4s, v2.s[3]
                fmla v21.4s, v15.4s, v2.s[3]
                fmla v22.4s, v14.4s, v3.s[3]
                fmla v23.4s, v15.4s, v3.s[3]
                fmla v24.4s, v14.4s, v4.s[3]
                fmla v25.4s, v15.4s, v4.s[3]
                fmla v26.4s, v14.4s, v5.s[3]
                fmla v27.4s, v15.4s, v5.s[3]
                fmla v28.4s, v14.4s, v6.s[3]
                fmla v29.4s, v15.4s, v6.s[3]
                fmla v30.4s, v14.4s, v7.s[3]
                fmla v31.4s, v15.4s, v7.s[3]
                // prefetching is not prefered while writing results in spite of cache missings
                // you could try prfm pstl2strm
                // there are almost no benefits observed though
                cbnz x11, Relu6
                cbnz x10, Relu
                b WriteStart
            Relu6:
                movi v1.4s, #6
                scvtf v1.4s, v1.4s
                fmin v16.4s, v16.4s, v1.4s
                fmin v17.4s, v17.4s, v1.4s
                fmin v18.4s, v18.4s, v1.4s
                fmin v19.4s, v19.4s, v1.4s
                fmin v20.4s, v20.4s, v1.4s
                fmin v21.4s, v21.4s, v1.4s
                fmin v22.4s, v22.4s, v1.4s
                fmin v23.4s, v23.4s, v1.4s
                fmin v24.4s, v24.4s, v1.4s
                fmin v25.4s, v25.4s, v1.4s
                fmin v26.4s, v26.4s, v1.4s
                fmin v27.4s, v27.4s, v1.4s
                fmin v28.4s, v28.4s, v1.4s
                fmin v29.4s, v29.4s, v1.4s
                fmin v30.4s, v30.4s, v1.4s
                fmin v31.4s, v31.4s, v1.4s
            Relu:
                dup v0.4s, wzr
                fmax v16.4s, v16.4s, v0.4s
                fmax v17.4s, v17.4s, v0.4s
                fmax v18.4s, v18.4s, v0.4s
                fmax v19.4s, v19.4s, v0.4s
                fmax v20.4s, v20.4s, v0.4s
                fmax v21.4s, v21.4s, v0.4s
                fmax v22.4s, v22.4s, v0.4s
                fmax v23.4s, v23.4s, v0.4s
                fmax v24.4s, v24.4s, v0.4s
                fmax v25.4s, v25.4s, v0.4s
                fmax v26.4s, v26.4s, v0.4s
                fmax v27.4s, v27.4s, v0.4s
                fmax v28.4s, v28.4s, v0.4s
                fmax v29.4s, v29.4s, v0.4s
                fmax v30.4s, v30.4s, v0.4s
                fmax v31.4s, v31.4s, v0.4s

            WriteStart:
                cbnz x9, WriteC4
                cmp x6, #5
                beq Write5
                cmp x6, #6
                beq Write6
                cmp x6, #7
                beq Write7
                b Write8
            Write5:
                add x17, x15, #16
                st1 {v16.4s}, [x15], x7
                str s17, [x17]
                add x17, x17, x7
                st1 {v18.4s}, [x15], x7
                str s19, [x17]
                add x17, x17, x7
                st1 {v20.4s}, [x15], x7
                str s21, [x17]
                add x17, x17, x7
                st1 {v22.4s}, [x15], x7
                str s23, [x17]
                add x17, x17, x7
                st1 {v24.4s}, [x15], x7
                str s25, [x17]
                add x17, x17, x7
                st1 {v26.4s}, [x15], x7
                str s27, [x17]
                add x17, x17, x7
                st1 {v28.4s}, [x15], x7
                str s29, [x17]
                add x17, x17, x7
                st1 {v30.4s}, [x15]
                str s31, [x17]
                add x0, x0, #20
                b WriteEnd
            Write6:
                add x17, x15, #16
                st1 {v16.4s}, [x15], x7
                dup s16, v17.s[1]
                stp s17, s16, [x17]
                add x17, x17, x7
                st1 {v18.4s}, [x15], x7
                dup s18, v19.s[1]
                stp s19, s18, [x17]
                add x17, x17, x7
                st1 {v20.4s}, [x15], x7
                dup s20, v21.s[1]
                stp s21, s20, [x17]
                add x17, x17, x7
                st1 {v22.4s}, [x15], x7
                dup s22, v23.s[1]
                stp s23, s22, [x17]
                add x17, x17, x7
                st1 {v24.4s}, [x15], x7
                dup s24, v25.s[1]
                stp s25, s24, [x17]
                add x17, x17, x7
                st1 {v26.4s}, [x15], x7
                dup s26, v27.s[1]
                stp s27, s26, [x17]
                add x17, x17, x7
                st1 {v28.4s}, [x15], x7
                dup s28, v29.s[1]
                stp s29, s28, [x17]
                add x17, x17, x7
                st1 {v30.4s}, [x15]
                dup s30, v31.s[1]
                stp s31, s30, [x17]
                add x0, x0, #24
                b WriteEnd
            Write7:
                add x17, x15, #16
                add x16, x15, #24
                st1 {v16.4s}, [x15], x7
                dup s16, v17.s[1]
                stp s17, s16, [x17]
                add x17, x17, x7
                st1 {v17.s}[2], [x16], x7
                st1 {v18.4s}, [x15], x7
                dup s18, v19.s[1]
                stp s19, s18, [x17]
                add x17, x17, x7
                st1 {v19.s}[2], [x16], x7
                st1 {v20.4s}, [x15], x7
                dup s20, v21.s[1]
                stp s21, s20, [x17]
                add x17, x17, x7
                st1 {v21.s}[2], [x16], x7
                st1 {v22.4s}, [x15], x7
                dup s22, v23.s[1]
                stp s23, s22, [x17]
                add x17, x17, x7
                st1 {v23.s}[2], [x16], x7
                st1 {v24.4s}, [x15], x7
                dup s24, v25.s[1]
                stp s25, s24, [x17]
                add x17, x17, x7
                st1 {v25.s}[2], [x16], x7
                st1 {v26.4s}, [x15], x7
                dup s26, v27.s[1]
                stp s27, s26, [x17]
                add x17, x17, x7
                st1 {v27.s}[2], [x16], x7
                st1 {v28.4s}, [x15], x7
                dup s28, v29.s[1]
                stp s29, s28, [x17]
                add x17, x17, x7
                st1 {v29.s}[2], [x16], x7
                st1 {v30.4s}, [x15], x7
                dup s30, v31.s[1]
                stp s31, s30, [x17]
                add x17, x17, x7
                st1 {v31.s}[2], [x16], x7
                add x0, x0, #28
                b WriteEnd
            WriteC4:
                st1 {v16.4s}, [x15], x7
                st1 {v18.4s}, [x15], x7
                st1 {v20.4s}, [x15], x7
                st1 {v22.4s}, [x15], x7
                st1 {v24.4s}, [x15], x7
                st1 {v26.4s}, [x15], x7
                st1 {v28.4s}, [x15], x7
                st1 {v30.4s}, [x15]
                add x15, x8, x0
                st1 {v17.4s}, [x15], x7
                st1 {v19.4s}, [x15], x7
                st1 {v21.4s}, [x15], x7
                st1 {v23.4s}, [x15], x7
                st1 {v25.4s}, [x15], x7
                st1 {v27.4s}, [x15], x7
                st1 {v29.4s}, [x15], x7
                st1 {v31.4s}, [x15]
                add x0, x0, #16
                b WriteEnd
            Write8:
                st1 {v16.4s, v17.4s}, [x15], x7
                st1 {v18.4s, v19.4s}, [x15], x7
                st1 {v20.4s, v21.4s}, [x15], x7
                st1 {v22.4s, v23.4s}, [x15], x7
                st1 {v24.4s, v25.4s}, [x15], x7
                st1 {v26.4s, v27.4s}, [x15], x7
                st1 {v28.4s, v29.4s}, [x15], x7
                st1 {v30.4s, v31.4s}, [x15]
                add x0, x0, #32

        WriteEnd:

            subs x14, x14, #1
            bne LoopKsize

        subs x6, x6, #8
        ble LoopOcEnd
        cbz x9, NoStepC4Block
        add x0, x0, x8
    NoStepC4Block:
        cbz x3, NoStepForward
        add x3, x3, #32
    NoStepForward:
        cmp x6, #4
        bgt LoopOc

    LoopOcHalf:
        mov x18, #32

        mov x14, x4
        mov x12, x1

        LoopKsizeHalf:

            mov x15, x0
            INIT_BIAS_HALF

            // load input for output 1-2
            ld1 {v0.4s, v1.4s}, [x12], #32
            // load weight
            ld1 {v8.4s}, [x2], x18
            ld1 {v10.4s}, [x2], x18
            // step for output 1-2
            fmla v16.4s, v8.4s, v0.s[0]
            fmla v18.4s, v8.4s, v1.s[0]
            // load input for output 3-4
            ld1 {v2.4s, v3.4s}, [x12], #32
            // another step for output 1-2
            fmla v16.4s, v10.4s, v0.s[1]
            fmla v18.4s, v10.4s, v1.s[1]
            // load input  for output 5-8
            // input cache should be refreshed after loading
            // ATTENTION: advance is prefered, but advancing too much may lead to invalid prefetching 
            ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x12], #64
            // step for output 3-8
            fmla v20.4s, v8.4s, v2.s[0]
            fmla v22.4s, v8.4s, v3.s[0]

            subs x13, x5, #1
            beq LoopIcEndHalf

            LoopIcHalf:
                fmla v24.4s, v8.4s, v4.s[0]
                fmla v26.4s, v8.4s, v5.s[0]
                fmla v28.4s, v8.4s, v6.s[0]
                fmla v30.4s, v8.4s, v7.s[0]
                // load weight
                ld1 {v12.4s}, [x2], x18
                // step for output 3-8
                fmla v20.4s, v10.4s, v2.s[1]
                fmla v22.4s, v10.4s, v3.s[1]
                // load weight
                ld1 {v14.4s}, [x2], x18
                fmla v24.4s, v10.4s, v4.s[1]
                fmla v26.4s, v10.4s, v5.s[1]
                fmla v28.4s, v10.4s, v6.s[1]
                fmla v30.4s, v10.4s, v7.s[1]
                // another step for output 1-8
                fmla v16.4s, v12.4s, v0.s[2]
                fmla v18.4s, v12.4s, v1.s[2]
                fmla v20.4s, v12.4s, v2.s[2]
                fmla v22.4s, v12.4s, v3.s[2]
                fmla v24.4s, v12.4s, v4.s[2]
                fmla v26.4s, v12.4s, v5.s[2]
                fmla v28.4s, v12.4s, v6.s[2]
                fmla v30.4s, v12.4s, v7.s[2]
                // load weight
                ld1 {v8.4s}, [x2], x18
                // another step for output 1-8
                fmla v16.4s, v14.4s, v0.s[3]
                fmla v18.4s, v14.4s, v1.s[3]
                // load weight
                ld1 {v10.4s}, [x2], x18
                fmla v20.4s, v14.4s, v2.s[3]
                fmla v22.4s, v14.4s, v3.s[3]
                // load input for output 1-4
                ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x12], #64
                fmla v24.4s, v14.4s, v4.s[3]
                fmla v26.4s, v14.4s, v5.s[3]
                fmla v28.4s, v14.4s, v6.s[3]
                fmla v30.4s, v14.4s, v7.s[3]
                // load input  for output 5-8
                ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x12], #64
                // step for output 1-8
                fmla v16.4s, v8.4s, v0.s[0]
                fmla v18.4s, v8.4s, v1.s[0]
                fmla v16.4s, v10.4s, v0.s[1]
                fmla v18.4s, v10.4s, v1.s[1]
                fmla v20.4s, v8.4s, v2.s[0]
                fmla v22.4s, v8.4s, v3.s[0]

                subs x13, x13, #1
                bne LoopIcHalf

            LoopIcEndHalf:
                fmla v24.4s, v8.4s, v4.s[0]
                fmla v26.4s, v8.4s, v5.s[0]
                fmla v28.4s, v8.4s, v6.s[0]
                fmla v30.4s, v8.4s, v7.s[0]
                // load weight
                ld1 {v12.4s}, [x2], x18
                // step for output 3-8
                fmla v20.4s, v10.4s, v2.s[1]
                fmla v22.4s, v10.4s, v3.s[1]
                // load weight
                ld1 {v14.4s}, [x2], x18
                fmla v24.4s, v10.4s, v4.s[1]
                fmla v26.4s, v10.4s, v5.s[1]
                fmla v28.4s, v10.4s, v6.s[1]
                fmla v30.4s, v10.4s, v7.s[1]
                // another step for output 1-8
                fmla v16.4s, v12.4s, v0.s[2]
                fmla v18.4s, v12.4s, v1.s[2]
                fmla v20.4s, v12.4s, v2.s[2]
                fmla v22.4s, v12.4s, v3.s[2]
                fmla v24.4s, v12.4s, v4.s[2]
                fmla v26.4s, v12.4s, v5.s[2]
                fmla v28.4s, v12.4s, v6.s[2]
                fmla v30.4s, v12.4s, v7.s[2]
                // another step for output 1-8
                fmla v16.4s, v14.4s, v0.s[3]
                fmla v18.4s, v14.4s, v1.s[3]
                fmla v20.4s, v14.4s, v2.s[3]
                fmla v22.4s, v14.4s, v3.s[3]
                fmla v24.4s, v14.4s, v4.s[3]
                fmla v26.4s, v14.4s, v5.s[3]
                fmla v28.4s, v14.4s, v6.s[3]
                fmla v30.4s, v14.4s, v7.s[3]

                cbnz x11, Relu6Half
                cbnz x10, ReluHalf
                b WriteStartHalf
            Relu6Half:
                movi v1.4s, #6
                scvtf v1.4s, v1.4s
                fmin v16.4s, v16.4s, v1.4s
                fmin v18.4s, v18.4s, v1.4s
                fmin v20.4s, v20.4s, v1.4s
                fmin v22.4s, v22.4s, v1.4s
                fmin v24.4s, v24.4s, v1.4s
                fmin v26.4s, v26.4s, v1.4s
                fmin v28.4s, v28.4s, v1.4s
                fmin v30.4s, v30.4s, v1.4s
            ReluHalf:
                dup v0.4s, wzr
                fmax v16.4s, v16.4s, v0.4s
                fmax v18.4s, v18.4s, v0.4s
                fmax v20.4s, v20.4s, v0.4s
                fmax v22.4s, v22.4s, v0.4s
                fmax v24.4s, v24.4s, v0.4s
                fmax v26.4s, v26.4s, v0.4s
                fmax v28.4s, v28.4s, v0.4s
                fmax v30.4s, v30.4s, v0.4s

            WriteStartHalf:
                cbnz x9, Write4
                cmp x6, #1
                beq Write1
                cmp x6, #2
                beq Write2
                cmp x6, #3
                beq Write3
                b Write4
            Write1:
                str s16, [x15]
                add x15, x15, x7
                str s18, [x15]
                add x15, x15, x7
                str s20, [x15]
                add x15, x15, x7
                str s22, [x15]
                add x15, x15, x7
                str s24, [x15]
                add x15, x15, x7
                str s26, [x15]
                add x15, x15, x7
                str s28, [x15]
                add x15, x15, x7
                str s30, [x15]
                add x0, x0, #4
                b WriteEndHalf
            Write2:
                dup s17, v16.s[1]
                stp s16, s17, [x15]
                add x15, x15, x7
                dup s19, v18.s[1]
                stp s18, s19, [x15]
                add x15, x15, x7
                dup s21, v20.s[1]
                stp s20, s21, [x15]
                add x15, x15, x7
                dup s23, v22.s[1]
                stp s22, s23, [x15]
                add x15, x15, x7
                dup s25, v24.s[1]
                stp s24, s25, [x15]
                add x15, x15, x7
                dup s27, v26.s[1]
                stp s26, s27, [x15]
                add x15, x15, x7
                dup s29, v28.s[1]
                stp s28, s29, [x15]
                add x15, x15, x7
                dup s31, v30.s[1]
                stp s30, s31, [x15]
                add x0, x0, #8
                b WriteEndHalf
            Write3:
                add x17, x15, #8
                dup s17, v16.s[1]
                stp s16, s17, [x15]
                add x15, x15, x7
                st1 {v16.s}[2], [x17], x7
                dup s19, v18.s[1]
                stp s18, s19, [x15]
                add x15, x15, x7
                st1 {v18.s}[2], [x17], x7
                dup s21, v20.s[1]
                stp s20, s21, [x15]
                add x15, x15, x7
                st1 {v20.s}[2], [x17], x7
                dup s23, v22.s[1]
                stp s22, s23, [x15]
                add x15, x15, x7
                st1 {v22.s}[2], [x17], x7
                dup s25, v24.s[1]
                stp s24, s25, [x15]
                add x15, x15, x7
                st1 {v24.s}[2], [x17], x7
                dup s27, v26.s[1]
                stp s26, s27, [x15]
                add x15, x15, x7
                st1 {v26.s}[2], [x17], x7
                dup s29, v28.s[1]
                stp s28, s29, [x15]
                add x15, x15, x7
                st1 {v28.s}[2], [x17], x7
                dup s31, v30.s[1]
                stp s30, s31, [x15]
                st1 {v30.s}[2], [x17]
                add x0, x0, #12
                b WriteEndHalf
            Write4:
                // prefetching is not prefered while writing results in spite of cache missings
                // you could try prfm pstl2strm
                // there are almost no benefits observed though
                st1 {v16.4s}, [x15], x7
                st1 {v18.4s}, [x15], x7
                st1 {v20.4s}, [x15], x7
                st1 {v22.4s}, [x15], x7
                st1 {v24.4s}, [x15], x7
                st1 {v26.4s}, [x15], x7
                st1 {v28.4s}, [x15], x7
                st1 {v30.4s}, [x15]
                add x0, x0, #16

        WriteEndHalf:

            subs x14, x14, #1
            bne LoopKsizeHalf

LoopOcEnd:

    sub sp, sp, #128
    ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
    ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
    ret
#endif
