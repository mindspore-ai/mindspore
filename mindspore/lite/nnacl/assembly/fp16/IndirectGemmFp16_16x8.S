#ifdef __aarch64__
#include "nnacl/assembly_global.h"

.text
.align 5
.global IndirectGemmFp16_16x8
#ifndef __APPLE__
.type IndirectGemmFp16_16x8, %function
#endif

// void IndirectGemmFp16_16x8(float16_t *output, float16_t *input, float16_t *weight, float16_t *bias,
//     size_t step, size_t ic4, size_t oc8, size_t offset, size_t mode, size_t writeC4, size_t relu, size_t relu6);
// x0: output, x1: input, x2: weight, x3: bias, x4: step, x5: ic4, x6: oc8, x7: offset, 
// x8:mode, x9: writeC4, x10:relu, x11: relu6
// compute 8 channel for 16 outputs
asm_function IndirectGemmFp16_16x8

    .macro INIT_BIAS
        dup v16.4s, wzr
        cbz x3, InitBias
        ld1 {v16.8h}, [x3]
    InitBias:
        mov v17.16b, v16.16b
        mov v18.16b, v16.16b
        mov v19.16b, v16.16b
        mov v20.16b, v16.16b
        mov v21.16b, v16.16b
        mov v22.16b, v16.16b
        mov v23.16b, v16.16b
        mov v24.16b, v16.16b
        mov v25.16b, v16.16b
        mov v26.16b, v16.16b
        mov v27.16b, v16.16b
        mov v28.16b, v16.16b
        mov v29.16b, v16.16b
        mov v30.16b, v16.16b
        mov v31.16b, v16.16b
    .endm

    // registers v8 ~ v15 must be preserved by a callee across subroutine calls, according to
    // https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst#simd-and-floating-point-registers
    // x19 ~ r29 should be also preserved
    // whereas our coding style do not permit such amount of parameters
    sub sp, sp, #128
    // performance between storing 4 registers at the same time and separately storing them on in-order cores
    // is not tested yet
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64

    ldr x8, [sp, #0]
    ldr x9, [sp, #8]
    ldr x10, [sp, #16]
    ldr x11, [sp, #24]

    cbnz x8, IndirectGemmStart
    // step is one for common convolution, where ic8 should multiply by kernel size 
    // step is (a+b-1) for F(a,b) in winograd
    mul x5, x4, x5
    mov x4, #1

IndirectGemmStart:

    LoopOc:

        mov x14, x4
        mov x12, x1

        LoopKsize:

            mov x15, x0
            INIT_BIAS
            // load input for output 1-8
            ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x12], #64
            // load weight
            ld1 {v8.8h, v9.8h}, [x2], #32
            //  first 2 steps for output 1 and 3
            fmla v16.8h, v8.8h, v0.h[0]
            fmla v18.8h, v8.8h, v1.h[0]
            fmla v16.8h, v9.8h, v0.h[1]
            fmla v18.8h, v9.8h, v1.h[1]
            // load weight
            ld1 {v10.8h, v11.8h}, [x2], #32
            //  first 2 steps for output 2 and 4
            fmla v17.8h, v8.8h, v0.h[4]
            fmla v19.8h, v8.8h, v1.h[4]
            fmla v17.8h, v9.8h, v0.h[5]
            fmla v19.8h, v9.8h, v1.h[5]
            // load input  for output 9-16
            // input cache should be refreshed after loading
            // ATTENTION: advance is preferred, but advancing too much may lead to invalid prefetching
            ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x12], #64
            //  last 2 steps for output 1 and 3
            fmla v16.8h, v10.8h, v0.h[2]
            fmla v18.8h, v10.8h, v1.h[2]
            fmla v16.8h, v11.8h, v0.h[3]
            fmla v18.8h, v11.8h, v1.h[3]

            // check if ic4=1
            subs x13, x5, #1
            beq LoopIcEnd

            LoopIc:
                //  last 2 steps for output 2 and 4
                fmla v17.8h, v10.8h, v0.h[6]
                fmla v19.8h, v10.8h, v1.h[6]
                fmla v17.8h, v11.8h, v0.h[7]
                fmla v19.8h, v11.8h, v1.h[7]
                //  steps for output 5-8
                fmla v20.8h, v8.8h, v2.h[0]
                fmla v22.8h, v8.8h, v3.h[0]
                fmla v20.8h, v9.8h, v2.h[1]
                fmla v22.8h, v9.8h, v3.h[1]
                fmla v21.8h, v8.8h, v2.h[4]
                fmla v23.8h, v8.8h, v3.h[4]
                fmla v21.8h, v9.8h, v2.h[5]
                fmla v23.8h, v9.8h, v3.h[5]
                fmla v20.8h, v10.8h, v2.h[2]
                fmla v22.8h, v10.8h, v3.h[2]
                fmla v20.8h, v11.8h, v2.h[3]
                fmla v22.8h, v11.8h, v3.h[3]
                fmla v21.8h, v10.8h, v2.h[6]
                fmla v23.8h, v10.8h, v3.h[6]
                fmla v21.8h, v11.8h, v2.h[7]
                fmla v23.8h, v11.8h, v3.h[7]
                // load input for output 1-8
                ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x12], #64
                //  steps for output 9-12
                fmla v24.8h, v8.8h, v4.h[0]
                fmla v26.8h, v8.8h, v5.h[0]
                fmla v24.8h, v9.8h, v4.h[1]
                fmla v26.8h, v9.8h, v5.h[1]
                fmla v25.8h, v8.8h, v4.h[4]
                fmla v27.8h, v8.8h, v5.h[4]
                fmla v25.8h, v9.8h, v4.h[5]
                fmla v27.8h, v9.8h, v5.h[5]
                fmla v24.8h, v10.8h, v4.h[2]
                fmla v26.8h, v10.8h, v5.h[2]
                fmla v24.8h, v11.8h, v4.h[3]
                fmla v26.8h, v11.8h, v5.h[3]
                fmla v25.8h, v10.8h, v4.h[6]
                fmla v27.8h, v10.8h, v5.h[6]
                fmla v25.8h, v11.8h, v4.h[7]
                fmla v27.8h, v11.8h, v5.h[7]
                //  steps for output 13-16
                fmla v28.8h, v8.8h, v6.h[0]
                fmla v30.8h, v8.8h, v7.h[0]
                fmla v28.8h, v9.8h, v6.h[1]
                fmla v30.8h, v9.8h, v7.h[1]
                fmla v29.8h, v8.8h, v6.h[4]
                fmla v31.8h, v8.8h, v7.h[4]
                fmla v29.8h, v9.8h, v6.h[5]
                fmla v31.8h, v9.8h, v7.h[5]
                // load weight
                ld1 {v8.8h, v9.8h}, [x2], #32
                fmla v28.8h, v10.8h, v6.h[2]
                fmla v30.8h, v10.8h, v7.h[2]
                fmla v28.8h, v11.8h, v6.h[3]
                fmla v30.8h, v11.8h, v7.h[3]
                fmla v29.8h, v10.8h, v6.h[6]
                fmla v31.8h, v10.8h, v7.h[6]
                fmla v29.8h, v11.8h, v6.h[7]
                fmla v31.8h, v11.8h, v7.h[7]
                // load weight
                ld1 {v10.8h, v11.8h}, [x2], #32
                // first 2 steps for output 1-4
                fmla v16.8h, v8.8h, v0.h[0]
                fmla v18.8h, v8.8h, v1.h[0]
                fmla v16.8h, v9.8h, v0.h[1]
                fmla v18.8h, v9.8h, v1.h[1]
                fmla v17.8h, v8.8h, v0.h[4]
                fmla v19.8h, v8.8h, v1.h[4]
                fmla v17.8h, v9.8h, v0.h[5]
                fmla v19.8h, v9.8h, v1.h[5]
                // load input  for output 9-16
                ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x12], #64
                //  last 2 steps for output 1 and 3
                fmla v16.8h, v10.8h, v0.h[2]
                fmla v18.8h, v10.8h, v1.h[2]
                fmla v16.8h, v11.8h, v0.h[3]
                fmla v18.8h, v11.8h, v1.h[3]

                subs x13, x13, #1
                bne LoopIc

            LoopIcEnd:
                fmla v17.8h, v10.8h, v0.h[6]
                fmla v19.8h, v10.8h, v1.h[6]
                fmla v17.8h, v11.8h, v0.h[7]
                fmla v19.8h, v11.8h, v1.h[7]
                //  steps for output 5-8
                fmla v20.8h, v8.8h, v2.h[0]
                fmla v22.8h, v8.8h, v3.h[0]
                fmla v20.8h, v9.8h, v2.h[1]
                fmla v22.8h, v9.8h, v3.h[1]
                fmla v21.8h, v8.8h, v2.h[4]
                fmla v23.8h, v8.8h, v3.h[4]
                fmla v21.8h, v9.8h, v2.h[5]
                fmla v23.8h, v9.8h, v3.h[5]
                fmla v20.8h, v10.8h, v2.h[2]
                fmla v22.8h, v10.8h, v3.h[2]
                fmla v20.8h, v11.8h, v2.h[3]
                fmla v22.8h, v11.8h, v3.h[3]
                fmla v21.8h, v10.8h, v2.h[6]
                fmla v23.8h, v10.8h, v3.h[6]
                fmla v21.8h, v11.8h, v2.h[7]
                fmla v23.8h, v11.8h, v3.h[7]
                //  steps for output 9-12
                fmla v24.8h, v8.8h, v4.h[0]
                fmla v26.8h, v8.8h, v5.h[0]
                fmla v24.8h, v9.8h, v4.h[1]
                fmla v26.8h, v9.8h, v5.h[1]
                fmla v25.8h, v8.8h, v4.h[4]
                fmla v27.8h, v8.8h, v5.h[4]
                fmla v25.8h, v9.8h, v4.h[5]
                fmla v27.8h, v9.8h, v5.h[5]
                fmla v24.8h, v10.8h, v4.h[2]
                fmla v26.8h, v10.8h, v5.h[2]
                fmla v24.8h, v11.8h, v4.h[3]
                fmla v26.8h, v11.8h, v5.h[3]
                fmla v25.8h, v10.8h, v4.h[6]
                fmla v27.8h, v10.8h, v5.h[6]
                fmla v25.8h, v11.8h, v4.h[7]
                fmla v27.8h, v11.8h, v5.h[7]
                //  steps for output 13-16
                fmla v28.8h, v8.8h, v6.h[0]
                fmla v30.8h, v8.8h, v7.h[0]
                fmla v28.8h, v9.8h, v6.h[1]
                fmla v30.8h, v9.8h, v7.h[1]
                fmla v29.8h, v8.8h, v6.h[4]
                fmla v31.8h, v8.8h, v7.h[4]
                fmla v29.8h, v9.8h, v6.h[5]
                fmla v31.8h, v9.8h, v7.h[5]
                fmla v28.8h, v10.8h, v6.h[2]
                fmla v30.8h, v10.8h, v7.h[2]
                fmla v28.8h, v11.8h, v6.h[3]
                fmla v30.8h, v11.8h, v7.h[3]
                fmla v29.8h, v10.8h, v6.h[6]
                fmla v31.8h, v10.8h, v7.h[6]
                fmla v29.8h, v11.8h, v6.h[7]
                fmla v31.8h, v11.8h, v7.h[7]

                cbnz x11, Relu6
                cbnz x10, Relu
                b WriteStart
            Relu6:
                movi v9.8h, #0x46, lsl #8
                fmin v16.8h, v16.8h, v9.8h
                fmin v17.8h, v17.8h, v9.8h
                fmin v18.8h, v18.8h, v9.8h
                fmin v19.8h, v19.8h, v9.8h
                fmin v20.8h, v20.8h, v9.8h
                fmin v21.8h, v21.8h, v9.8h
                fmin v22.8h, v22.8h, v9.8h
                fmin v23.8h, v23.8h, v9.8h
                fmin v24.8h, v24.8h, v9.8h
                fmin v25.8h, v25.8h, v9.8h
                fmin v26.8h, v26.8h, v9.8h
                fmin v27.8h, v27.8h, v9.8h
                fmin v28.8h, v28.8h, v9.8h
                fmin v29.8h, v29.8h, v9.8h
                fmin v30.8h, v30.8h, v9.8h
                fmin v31.8h, v31.8h, v9.8h
            Relu:
                dup v8.4s, wzr
                fmax v16.8h, v16.8h, v8.8h
                fmax v17.8h, v17.8h, v8.8h
                fmax v18.8h, v18.8h, v8.8h
                fmax v19.8h, v19.8h, v8.8h
                fmax v20.8h, v20.8h, v8.8h
                fmax v21.8h, v21.8h, v8.8h
                fmax v22.8h, v22.8h, v8.8h
                fmax v23.8h, v23.8h, v8.8h
                fmax v24.8h, v24.8h, v8.8h
                fmax v25.8h, v25.8h, v8.8h
                fmax v26.8h, v26.8h, v8.8h
                fmax v27.8h, v27.8h, v8.8h
                fmax v28.8h, v28.8h, v8.8h
                fmax v29.8h, v29.8h, v8.8h
                fmax v30.8h, v30.8h, v8.8h
                fmax v31.8h, v31.8h, v8.8h

            WriteStart:
                cbnz x9, Write8
                cmp x6, #1
                beq Write1
                cmp x6, #2
                beq Write2
                cmp x6, #3
                beq Write3
                cmp x6, #4
                beq Write4
                cmp x6, #5
                beq Write5
                cmp x6, #6
                beq Write6
                cmp x6, #7
                beq Write7
                b Write8
                // prefetching is not preferred while writing results in spite of cache missing
                // you could try prfm pstl2strm
                // there are almost no benefits observed though
            Write1:
                str h16, [x15]
                add x15, x15, x7
                str h17, [x15]
                add x15, x15, x7
                str h18, [x15]
                add x15, x15, x7
                str h19, [x15]
                add x15, x15, x7
                str h20, [x15]
                add x15, x15, x7
                str h21, [x15]
                add x15, x15, x7
                str h22, [x15]
                add x15, x15, x7
                str h23, [x15]
                add x15, x15, x7
                str h24, [x15]
                add x15, x15, x7
                str h25, [x15]
                add x15, x15, x7
                str h26, [x15]
                add x15, x15, x7
                str h27, [x15]
                add x15, x15, x7
                str h28, [x15]
                add x15, x15, x7
                str h29, [x15]
                add x15, x15, x7
                str h30, [x15]
                add x15, x15, x7
                str h31, [x15]
                add x0, x0, #2
                b WriteEnd
            Write2:
                add x17, x15, #2
                st1 {v16.h}[0], [x15], x7
                st1 {v16.h}[1], [x17], x7
                st1 {v17.h}[0], [x15], x7
                st1 {v17.h}[1], [x17], x7
                st1 {v18.h}[0], [x15], x7
                st1 {v18.h}[1], [x17], x7
                st1 {v19.h}[0], [x15], x7
                st1 {v19.h}[1], [x17], x7
                st1 {v20.h}[0], [x15], x7
                st1 {v20.h}[1], [x17], x7
                st1 {v21.h}[0], [x15], x7
                st1 {v21.h}[1], [x17], x7
                st1 {v22.h}[0], [x15], x7
                st1 {v22.h}[1], [x17], x7
                st1 {v23.h}[0], [x15], x7
                st1 {v23.h}[1], [x17], x7
                st1 {v24.h}[0], [x15], x7
                st1 {v24.h}[1], [x17], x7
                st1 {v25.h}[0], [x15], x7
                st1 {v25.h}[1], [x17], x7
                st1 {v26.h}[0], [x15], x7
                st1 {v26.h}[1], [x17], x7
                st1 {v27.h}[0], [x15], x7
                st1 {v27.h}[1], [x17], x7
                st1 {v28.h}[0], [x15], x7
                st1 {v28.h}[1], [x17], x7
                st1 {v29.h}[0], [x15], x7
                st1 {v29.h}[1], [x17], x7
                st1 {v30.h}[0], [x15], x7
                st1 {v30.h}[1], [x17], x7
                st1 {v31.h}[0], [x15]
                st1 {v31.h}[1], [x17]
                add x0, x0, #4
                b WriteEnd
            Write3:
                add x17, x15, #4
                add x16, x15, #2
                st1 {v16.h}[0], [x15], x7
                st1 {v16.h}[1], [x16], x7
                st1 {v16.h}[2], [x17], x7
                st1 {v17.h}[0], [x15], x7
                st1 {v17.h}[1], [x16], x7
                st1 {v17.h}[2], [x17], x7
                st1 {v18.h}[0], [x15], x7
                st1 {v18.h}[1], [x16], x7
                st1 {v18.h}[2], [x17], x7
                st1 {v19.h}[0], [x15], x7
                st1 {v19.h}[1], [x16], x7
                st1 {v19.h}[2], [x17], x7
                st1 {v20.h}[0], [x15], x7
                st1 {v20.h}[1], [x16], x7
                st1 {v20.h}[2], [x17], x7
                st1 {v21.h}[0], [x15], x7
                st1 {v21.h}[1], [x16], x7
                st1 {v21.h}[2], [x17], x7
                st1 {v22.h}[0], [x15], x7
                st1 {v22.h}[1], [x16], x7
                st1 {v22.h}[2], [x17], x7
                st1 {v23.h}[0], [x15], x7
                st1 {v23.h}[1], [x16], x7
                st1 {v23.h}[2], [x17], x7
                st1 {v24.h}[0], [x15], x7
                st1 {v24.h}[1], [x16], x7
                st1 {v24.h}[2], [x17], x7
                st1 {v25.h}[0], [x15], x7
                st1 {v25.h}[1], [x16], x7
                st1 {v25.h}[2], [x17], x7
                st1 {v26.h}[0], [x15], x7
                st1 {v26.h}[1], [x16], x7
                st1 {v26.h}[2], [x17], x7
                st1 {v27.h}[0], [x15], x7
                st1 {v27.h}[1], [x16], x7
                st1 {v27.h}[2], [x17], x7
                st1 {v28.h}[0], [x15], x7
                st1 {v28.h}[1], [x16], x7
                st1 {v28.h}[2], [x17], x7
                st1 {v29.h}[0], [x15], x7
                st1 {v29.h}[1], [x16], x7
                st1 {v29.h}[2], [x17], x7
                st1 {v30.h}[0], [x15], x7
                st1 {v30.h}[1], [x16], x7
                st1 {v30.h}[2], [x17], x7
                st1 {v31.h}[0], [x15]
                st1 {v31.h}[1], [x16]
                st1 {v31.h}[2], [x17]
                add x0, x0, #6
                b WriteEnd
            Write4:
                st1 {v16.4h}, [x15], x7
                st1 {v17.4h}, [x15], x7
                st1 {v18.4h}, [x15], x7
                st1 {v19.4h}, [x15], x7
                st1 {v20.4h}, [x15], x7
                st1 {v21.4h}, [x15], x7
                st1 {v22.4h}, [x15], x7
                st1 {v23.4h}, [x15], x7
                st1 {v24.4h}, [x15], x7
                st1 {v25.4h}, [x15], x7
                st1 {v26.4h}, [x15], x7
                st1 {v27.4h}, [x15], x7
                st1 {v28.4h}, [x15], x7
                st1 {v29.4h}, [x15], x7
                st1 {v30.4h}, [x15], x7
                st1 {v31.4h}, [x15]
                add x0, x0, #8
                b WriteEnd
            Write5:
                add x17, x15, #8
                st1 {v16.4h}, [x15], x7
                st1 {v16.h}[4], [x17], x7
                st1 {v17.4h}, [x15], x7
                st1 {v17.h}[4], [x17], x7
                st1 {v18.4h}, [x15], x7
                st1 {v18.h}[4], [x17], x7
                st1 {v19.4h}, [x15], x7
                st1 {v19.h}[4], [x17], x7
                st1 {v20.4h}, [x15], x7
                st1 {v20.h}[4], [x17], x7
                st1 {v21.4h}, [x15], x7
                st1 {v21.h}[4], [x17], x7
                st1 {v22.4h}, [x15], x7
                st1 {v22.h}[4], [x17], x7
                st1 {v23.4h}, [x15], x7
                st1 {v23.h}[4], [x17], x7
                st1 {v24.4h}, [x15], x7
                st1 {v24.h}[4], [x17], x7
                st1 {v25.4h}, [x15], x7
                st1 {v25.h}[4], [x17], x7
                st1 {v26.4h}, [x15], x7
                st1 {v26.h}[4], [x17], x7
                st1 {v27.4h}, [x15], x7
                st1 {v27.h}[4], [x17], x7
                st1 {v28.4h}, [x15], x7
                st1 {v28.h}[4], [x17], x7
                st1 {v29.4h}, [x15], x7
                st1 {v29.h}[4], [x17], x7
                st1 {v30.4h}, [x15], x7
                st1 {v30.h}[4], [x17], x7
                st1 {v31.4h}, [x15]
                st1 {v31.h}[4], [x17]
                add x0, x0, #10
                b WriteEnd
            Write6:
                add x17, x15, #8
                add x16, x15, #10
                st1 {v16.4h}, [x15], x7
                ins v0.s[0], v16.s[2]
                st1 {v0.h}[0], [x17], x7
                st1 {v0.h}[1], [x16], x7
                st1 {v17.4h}, [x15], x7
                ins v1.s[0], v17.s[2]
                st1 {v1.h}[0], [x17], x7
                st1 {v1.h}[1], [x16], x7
                st1 {v18.4h}, [x15], x7
                ins v2.s[0], v18.s[2]
                st1 {v2.h}[0], [x17], x7
                st1 {v2.h}[1], [x16], x7
                st1 {v19.4h}, [x15], x7
                ins v3.s[0], v19.s[2]
                st1 {v3.h}[0], [x17], x7
                st1 {v3.h}[1], [x16], x7
                st1 {v20.4h}, [x15], x7
                ins v4.s[0], v20.s[2]
                st1 {v4.h}[0], [x17], x7
                st1 {v4.h}[1], [x16], x7
                st1 {v21.4h}, [x15], x7
                ins v5.s[0], v21.s[2]
                st1 {v5.h}[0], [x17], x7
                st1 {v5.h}[1], [x16], x7
                st1 {v22.4h}, [x15], x7
                ins v6.s[0], v22.s[2]
                st1 {v6.h}[0], [x17], x7
                st1 {v6.h}[1], [x16], x7
                st1 {v23.4h}, [x15], x7
                ins v7.s[0], v23.s[2]
                st1 {v7.h}[0], [x17], x7
                st1 {v7.h}[1], [x16], x7
                st1 {v24.4h}, [x15], x7
                ins v8.s[0], v24.s[2]
                st1 {v8.h}[0], [x17], x7
                st1 {v8.h}[1], [x16], x7
                st1 {v25.4h}, [x15], x7
                ins v9.s[0], v25.s[2]
                st1 {v9.h}[0], [x17], x7
                st1 {v9.h}[1], [x16], x7
                st1 {v26.4h}, [x15], x7
                ins v10.s[0], v26.s[2]
                st1 {v10.h}[0], [x17], x7
                st1 {v10.h}[1], [x16], x7
                st1 {v27.4h}, [x15], x7
                ins v11.s[0], v27.s[2]
                st1 {v11.h}[0], [x17], x7
                st1 {v11.h}[1], [x16], x7
                st1 {v28.4h}, [x15], x7
                ins v12.s[0], v28.s[2]
                st1 {v12.h}[0], [x17], x7
                st1 {v12.h}[1], [x16], x7
                st1 {v29.4h}, [x15], x7
                ins v13.s[0], v29.s[2]
                st1 {v13.h}[0], [x17], x7
                st1 {v13.h}[1], [x16], x7
                st1 {v30.4h}, [x15], x7
                ins v14.s[0], v30.s[2]
                st1 {v14.h}[0], [x17], x7
                st1 {v14.h}[1], [x16], x7
                st1 {v31.4h}, [x15]
                ins v15.s[0], v31.s[2]
                st1 {v14.h}[0], [x17]
                st1 {v14.h}[1], [x16]
                add x0, x0, #12
                b WriteEnd
            Write7:
                add x17, x15, #8
                add x18, x15, #10
                add x16, x15, #12
                st1 {v16.4h}, [x15], x7
                ins v0.s[0], v16.s[2]
                st1 {v0.h}[0], [x17], x7
                st1 {v0.h}[1], [x18], x7
                st1 {v16.h}[6], [x16], x7
                st1 {v17.4h}, [x15], x7
                ins v1.s[0], v17.s[2]
                st1 {v1.h}[0], [x17], x7
                st1 {v1.h}[1], [x18], x7
                st1 {v17.h}[6], [x16], x7
                st1 {v18.4h}, [x15], x7
                ins v2.s[0], v18.s[2]
                st1 {v2.h}[0], [x17], x7
                st1 {v2.h}[1], [x18], x7
                st1 {v18.h}[6], [x16], x7
                st1 {v19.4h}, [x15], x7
                ins v3.s[0], v19.s[2]
                st1 {v3.h}[0], [x17], x7
                st1 {v3.h}[1], [x18], x7
                st1 {v19.h}[6], [x16], x7
                st1 {v20.4h}, [x15], x7
                ins v4.s[0], v20.s[2]
                st1 {v4.h}[0], [x17], x7
                st1 {v4.h}[1], [x18], x7
                st1 {v20.h}[6], [x16], x7
                st1 {v21.4h}, [x15], x7
                ins v5.s[0], v21.s[2]
                st1 {v5.h}[0], [x17], x7
                st1 {v5.h}[1], [x18], x7
                st1 {v21.h}[6], [x16], x7
                st1 {v22.4h}, [x15], x7
                ins v6.s[0], v22.s[2]
                st1 {v6.h}[0], [x17], x7
                st1 {v6.h}[1], [x18], x7
                st1 {v22.h}[6], [x16], x7
                st1 {v23.4h}, [x15], x7
                ins v7.s[0], v23.s[2]
                st1 {v7.h}[0], [x17], x7
                st1 {v7.h}[1], [x18], x7
                st1 {v23.h}[6], [x16], x7
                st1 {v24.4h}, [x15], x7
                ins v8.s[0], v24.s[2]
                st1 {v8.h}[0], [x17], x7
                st1 {v8.h}[1], [x18], x7
                st1 {v24.h}[6], [x16], x7
                st1 {v25.4h}, [x15], x7
                ins v9.s[0], v25.s[2]
                st1 {v9.h}[0], [x17], x7
                st1 {v9.h}[1], [x18], x7
                st1 {v25.h}[6], [x16], x7
                st1 {v26.4h}, [x15], x7
                ins v10.s[0], v26.s[2]
                st1 {v10.h}[0], [x17], x7
                st1 {v10.h}[1], [x18], x7
                st1 {v26.h}[6], [x16], x7
                st1 {v27.4h}, [x15], x7
                ins v11.s[0], v27.s[2]
                st1 {v11.h}[0], [x17], x7
                st1 {v11.h}[1], [x18], x7
                st1 {v27.h}[6], [x16], x7
                st1 {v28.4h}, [x15], x7
                ins v12.s[0], v28.s[2]
                st1 {v12.h}[0], [x17], x7
                st1 {v12.h}[1], [x18], x7
                st1 {v28.h}[6], [x16], x7
                st1 {v29.4h}, [x15], x7
                ins v13.s[0], v29.s[2]
                st1 {v13.h}[0], [x17], x7
                st1 {v13.h}[1], [x18], x7
                st1 {v29.h}[6], [x16], x7
                st1 {v30.4h}, [x15], x7
                ins v14.s[0], v30.s[2]
                st1 {v14.h}[0], [x17], x7
                st1 {v14.h}[1], [x18], x7
                st1 {v30.h}[6], [x16], x7
                st1 {v31.4h}, [x15]
                ins v15.s[0], v31.s[2]
                st1 {v15.h}[0], [x17]
                st1 {v15.h}[1], [x18]
                st1 {v31.h}[6], [x16]
                add x0, x0, #14
                b WriteEnd
            Write8:
                st1 {v16.8h}, [x15], x7
                st1 {v17.8h}, [x15], x7
                st1 {v18.8h}, [x15], x7
                st1 {v19.8h}, [x15], x7
                st1 {v20.8h}, [x15], x7
                st1 {v21.8h}, [x15], x7
                st1 {v22.8h}, [x15], x7
                st1 {v23.8h}, [x15], x7
                st1 {v24.8h}, [x15], x7
                st1 {v25.8h}, [x15], x7
                st1 {v26.8h}, [x15], x7
                st1 {v27.8h}, [x15], x7
                st1 {v28.8h}, [x15], x7
                st1 {v29.8h}, [x15], x7
                st1 {v30.8h}, [x15], x7
                st1 {v31.8h}, [x15]
                add x0, x0, #16

        WriteEnd:
            subs x14, x14, #1
            bne LoopKsize

        subs x6, x6, #8
        cbz x3, NoStepForward
        add x3, x3, #16
    NoStepForward:
        bgt LoopOc

    sub sp, sp, #128
    ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
    ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
    ret
#endif

