#ifdef __aarch64__

.text
.align 5
.global IndirectGemmInt8_24x4_dp
#ifndef __APPLE__
.type IndirectGemmInt8_24x4_dp, %function
#endif

// void IndirectGemmInt8_24x4_dp(int8_t *output, int8_t *input, int8_t *weight, int32_t *bias, size_t ksize, size_t ic4,
// size_t oc, size_t offset, int32_t *input_sum, size_t act_min, size_t act_max, size_t out_zp, size_t out_multiplier,
// size_t shift_before, size_t shift_after);
// x0: output, x1: input, x2: weight, x3: bias, x4: kSize, x5: ic4, x6: oc, x7: offset
// we use sdot intrinsic on cores that supports dotprod(Armv8.2-A w/dp or later)
// mrs intrinsic could read system register ID_AA64ISAR0_EL1(or s3_0_c0_c6_0 on Armv8.2-A)
// the 44-48 bits indicates whether dotprod is supported
IndirectGemmInt8_24x4_dp:

    .macro INIT_BIAS
        mov x20, x15
        ld1r {v8.4s}, [x20], #4
        ld1r {v9.4s}, [x20], #4
        ld1r {v10.4s}, [x20], #4
        ld1r {v11.4s}, [x20], #4
        ld1r {v12.4s}, [x20], #4
        ld1r {v13.4s}, [x20], #4
        ld1r {v14.4s}, [x20], #4
        ld1r {v15.4s}, [x20], #4
        ld1r {v16.4s}, [x20], #4
        ld1r {v17.4s}, [x20], #4
        ld1r {v18.4s}, [x20], #4
        ld1r {v19.4s}, [x20], #4
        ld1r {v20.4s}, [x20], #4
        ld1r {v21.4s}, [x20], #4
        ld1r {v22.4s}, [x20], #4
        ld1r {v23.4s}, [x20], #4
        ld1r {v24.4s}, [x20], #4
        ld1r {v25.4s}, [x20], #4
        ld1r {v26.4s}, [x20], #4
        ld1r {v27.4s}, [x20], #4
        ld1r {v28.4s}, [x20], #4
        ld1r {v29.4s}, [x20], #4
        ld1r {v30.4s}, [x20], #4
        ld1r {v31.4s}, [x20], #4
        dup v7.4s, wzr
        cbz x3, InitBias
        ld1 {v7.4s}, [x3]
    InitBias:
        sub v8.4s, v7.4s, v8.4s
        sub v9.4s, v7.4s, v9.4s
        sub v10.4s, v7.4s, v10.4s
        sub v11.4s, v7.4s, v11.4s
        sub v12.4s, v7.4s, v12.4s
        sub v13.4s, v7.4s, v13.4s
        sub v14.4s, v7.4s, v14.4s
        sub v15.4s, v7.4s, v15.4s
        sub v16.4s, v7.4s, v16.4s
        sub v17.4s, v7.4s, v17.4s
        sub v18.4s, v7.4s, v18.4s
        sub v19.4s, v7.4s, v19.4s
        sub v20.4s, v7.4s, v20.4s
        sub v21.4s, v7.4s, v21.4s
        sub v22.4s, v7.4s, v22.4s
        sub v23.4s, v7.4s, v23.4s
        sub v24.4s, v7.4s, v24.4s
        sub v25.4s, v7.4s, v25.4s
        sub v26.4s, v7.4s, v26.4s
        sub v27.4s, v7.4s, v27.4s
        sub v28.4s, v7.4s, v28.4s
        sub v29.4s, v7.4s, v29.4s
        sub v30.4s, v7.4s, v30.4s
        sub v31.4s, v7.4s, v31.4s
    .endm

    // registers v8 ~ v15 must be preserved by a callee across subroutine calls, according to
    // https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst#simd-and-floating-point-registers
    // r19 ~ r29 should be also preserved
    // whereas our coding style do not permit such amount of parameters
    sub sp, sp, #144
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
    stp x19, x20, [sp], #16

    ldr x15, [sp]
    ldr w8, [sp, #8]
    ldr w9, [sp, #16]
    ldr w16, [sp, #24]
    ldr w17, [sp, #32]
    ldr w18, [sp, #40]
    ldr w19, [sp, #48]

    mul x5, x4, x5
    mov x4, #1

    LoopOc:

        mov x10, x4
        mov x12, x1

        LoopKsize:
            INIT_BIAS
            mov x11, x0
            
            // as some processors do not support sdot intrinsic, we use instruction word
            // dp support is stilled judged dymaticly, instruction word is just used to ensure compilation
            // according to https://static.docs.arm.com/ddi0596/g/ISA_A64_xml_v86A-2020-03_OPT.pdf
            // the instruction word of sdot vd.4s, vn.16b, vm.4b[index] is
            // 0100 1111 10Lm mmmm 1110 H0nn nnnd dddd
            // mmmmm/nnnnn/ddddd is the number of neon register, HL is the high/low bit of index

            // load input for output 1-8
            ld1 {v0.16b, v1.16b}, [x12], #32
            // load weight
            ld1 {v6.16b}, [x2], #16
            // step for output 1-4
            .inst 0x4f80e0c8 // sdot v8.4s, v6.16b, v0.4b[0]
            .inst 0x4fa0e0c9 // sdot v9.4s, v6.16b, v0.4b[1]
            .inst 0x4f80e8ca // sdot v10.4s, v6.16b, v0.4b[2]
            .inst 0x4fa0e8cb // sdot v11.4s, v6.16b, v0.4b[3]
            // load input for output 9-16
            ld1 {v2.16b, v3.16b, v4.16b, v5.16b}, [x12], #64
            // another step for output 5-8
            .inst 0x4f81e0cc // sdot v12.4s, v6.16b, v1.4b[0]
            .inst 0x4fa1e0cd // sdot v13.4s, v6.16b, v1.4b[1]
            .inst 0x4f81e8ce // sdot v14.4s, v6.16b, v1.4b[2]
            .inst 0x4fa1e8cf // sdot v15.4s, v6.16b, v1.4b[3]

            subs x13, x5, #1
            beq LoopIcEndOne
            // load weight
            ld1 {v7.16b}, [x2], #16
            cmp x13, #1
            beq LoopIcEnd

            LoopIc:
                // load input for output 1-8
                ld1 {v0.16b, v1.16b}, [x12], #32
                .inst 0x4f82e0d0 // sdot v16.4s, v6.16b, v2.4b[0]
                .inst 0x4fa2e0d1 // sdot v17.4s, v6.16b, v2.4b[1]
                .inst 0x4f82e8d2 // sdot v18.4s, v6.16b, v2.4b[2]
                .inst 0x4fa2e8d3 // sdot v19.4s, v6.16b, v2.4b[3]
                .inst 0x4f83e0d4 // sdot v20.4s, v6.16b, v3.4b[0]
                .inst 0x4fa3e0d5 // sdot v21.4s, v6.16b, v3.4b[1]
                .inst 0x4f83e8d6 // sdot v22.4s, v6.16b, v3.4b[2]
                .inst 0x4fa3e8d7 // sdot v23.4s, v6.16b, v3.4b[3]
                ld1 {v2.16b, v3.16b}, [x12], #32
                .inst 0x4f84e0d8 // sdot v24.4s, v6.16b, v4.4b[0]
                .inst 0x4fa4e0d9 // sdot v25.4s, v6.16b, v4.4b[1]
                .inst 0x4f84e8da // sdot v26.4s, v6.16b, v4.4b[2]
                .inst 0x4fa4e8db // sdot v27.4s, v6.16b, v4.4b[3]
                .inst 0x4f85e0dc // sdot v28.4s, v6.16b, v5.4b[0]
                .inst 0x4fa5e0dd // sdot v29.4s, v6.16b, v5.4b[1]
                .inst 0x4f85e8de // sdot v30.4s, v6.16b, v5.4b[2]
                .inst 0x4fa5e8df // sdot v31.4s, v6.16b, v5.4b[3]
                // load input for output 9-16
                ld1 {v4.4s, v5.4s}, [x12], #32
                .inst 0x4f80e0e8 // sdot v8.4s, v7.16b, v0.4b[0]
                .inst 0x4fa0e0e9 // sdot v9.4s, v7.16b, v0.4b[1]
                .inst 0x4f80e8ea // sdot v10.4s, v7.16b, v0.4b[2]
                .inst 0x4fa0e8eb // sdot v11.4s, v7.16b, v0.4b[3]
                // another step for output 5-8
                .inst 0x4f81e0ec // sdot v12.4s, v7.16b, v1.4b[0]
                .inst 0x4fa1e0ed // sdot v13.4s, v7.16b, v1.4b[1]
                .inst 0x4f81e8ee // sdot v14.4s, v7.16b, v1.4b[2]
                .inst 0x4fa1e8ef // sdot v15.4s, v7.16b, v1.4b[3]
                // load input for output 1-8
                ld1 {v0.16b, v1.16b}, [x12], #32
                .inst 0x4f82e0f0 // sdot v16.4s, v7.16b, v2.4b[0]
                .inst 0x4fa2e0f1 // sdot v17.4s, v7.16b, v2.4b[1]
                .inst 0x4f82e8f2 // sdot v18.4s, v7.16b, v2.4b[2]
                .inst 0x4fa2e8f3 // sdot v19.4s, v7.16b, v2.4b[3]
                .inst 0x4f83e0f4 // sdot v20.4s, v7.16b, v3.4b[0]
                .inst 0x4fa3e0f5 // sdot v21.4s, v7.16b, v3.4b[1]
                .inst 0x4f83e8f6 // sdot v22.4s, v7.16b, v3.4b[2]
                .inst 0x4fa3e8f7 // sdot v23.4s, v7.16b, v3.4b[3]
                // load weight
                ld1 {v6.16b}, [x2], #16
                .inst 0x4f84e0f8 // sdot v24.4s, v7.16b, v4.4b[0]
                .inst 0x4fa4e0f9 // sdot v25.4s, v7.16b, v4.4b[1]
                .inst 0x4f84e8fa // sdot v26.4s, v7.16b, v4.4b[2]
                .inst 0x4fa4e8fb // sdot v27.4s, v7.16b, v4.4b[3]
                .inst 0x4f85e0fc // sdot v28.4s, v7.16b, v5.4b[0]
                .inst 0x4fa5e0fd // sdot v29.4s, v7.16b, v5.4b[1]
                .inst 0x4f85e8fe // sdot v30.4s, v7.16b, v5.4b[2]
                .inst 0x4fa5e8ff // sdot v31.4s, v7.16b, v5.4b[3]
                // load input for output 9-16
                ld1 {v2.4s, v3.4s}, [x12], #32
                .inst 0x4f80e0c8 // sdot v8.4s, v6.16b, v0.4b[0]
                .inst 0x4fa0e0c9 // sdot v9.4s, v6.16b, v0.4b[1]
                .inst 0x4f80e8ca // sdot v10.4s, v6.16b, v0.4b[2]
                .inst 0x4fa0e8cb // sdot v11.4s, v6.16b, v0.4b[3]
                // another step for output 5-8
                .inst 0x4f81e0cc // sdot v12.4s, v6.16b, v1.4b[0]
                .inst 0x4fa1e0cd // sdot v13.4s, v6.16b, v1.4b[1]
                .inst 0x4f81e8ce // sdot v14.4s, v6.16b, v1.4b[2]
                .inst 0x4fa1e8cf // sdot v15.4s, v6.16b, v1.4b[3]
                // load input for output 9-16
                ld1 {v4.4s, v5.4s}, [x12], #32

                subs x13, x13, #2
                beq LoopIcEndOne
                // load weight
                ld1 {v7.16b}, [x2], #16
                cmp x13, #1
                beq LoopIcEnd
                b LoopIc

            LoopIcEnd:
                mov x20, x15
                // load input for output 1-8
                ld1 {v0.16b, v1.16b}, [x12], #32
                .inst 0x4f82e0d0 // sdot v16.4s, v6.16b, v2.4b[0]
                .inst 0x4fa2e0d1 // sdot v17.4s, v6.16b, v2.4b[1]
                .inst 0x4f82e8d2 // sdot v18.4s, v6.16b, v2.4b[2]
                .inst 0x4fa2e8d3 // sdot v19.4s, v6.16b, v2.4b[3]
                .inst 0x4f83e0d4 // sdot v20.4s, v6.16b, v3.4b[0]
                .inst 0x4fa3e0d5 // sdot v21.4s, v6.16b, v3.4b[1]
                .inst 0x4f83e8d6 // sdot v22.4s, v6.16b, v3.4b[2]
                .inst 0x4fa3e8d7 // sdot v23.4s, v6.16b, v3.4b[3]
                ld1 {v2.16b, v3.16b}, [x12], #32
                .inst 0x4f84e0d8 // sdot v24.4s, v6.16b, v4.4b[0]
                .inst 0x4fa4e0d9 // sdot v25.4s, v6.16b, v4.4b[1]
                .inst 0x4f84e8da // sdot v26.4s, v6.16b, v4.4b[2]
                .inst 0x4fa4e8db // sdot v27.4s, v6.16b, v4.4b[3]
                .inst 0x4f85e0dc // sdot v28.4s, v6.16b, v5.4b[0]
                .inst 0x4fa5e0dd // sdot v29.4s, v6.16b, v5.4b[1]
                .inst 0x4f85e8de // sdot v30.4s, v6.16b, v5.4b[2]
                .inst 0x4fa5e8df // sdot v31.4s, v6.16b, v5.4b[3]
                // load input for output 9-16
                ld1 {v4.4s, v5.4s}, [x12], #32
                .inst 0x4f80e0e8 // sdot v8.4s, v7.16b, v0.4b[0]
                .inst 0x4fa0e0e9 // sdot v9.4s, v7.16b, v0.4b[1]
                .inst 0x4f80e8ea // sdot v10.4s, v7.16b, v0.4b[2]
                .inst 0x4fa0e8eb // sdot v11.4s, v7.16b, v0.4b[3]
                .inst 0x4f81e0ec // sdot v12.4s, v7.16b, v1.4b[0]
                .inst 0x4fa1e0ed // sdot v13.4s, v7.16b, v1.4b[1]
                .inst 0x4f81e8ee // sdot v14.4s, v7.16b, v1.4b[2]
                .inst 0x4fa1e8ef // sdot v15.4s, v7.16b, v1.4b[3]

                .inst 0x4f82e0f0 // sdot v16.4s, v7.16b, v2.4b[0]
                .inst 0x4fa2e0f1 // sdot v17.4s, v7.16b, v2.4b[1]
                .inst 0x4f82e8f2 // sdot v18.4s, v7.16b, v2.4b[2]
                .inst 0x4fa2e8f3 // sdot v19.4s, v7.16b, v2.4b[3]
                .inst 0x4f83e0f4 // sdot v20.4s, v7.16b, v3.4b[0]
                .inst 0x4fa3e0f5 // sdot v21.4s, v7.16b, v3.4b[1]
                .inst 0x4f83e8f6 // sdot v22.4s, v7.16b, v3.4b[2]
                .inst 0x4fa3e8f7 // sdot v23.4s, v7.16b, v3.4b[3]

                .inst 0x4f84e0f8 // sdot v24.4s, v7.16b, v4.4b[0]
                .inst 0x4fa4e0f9 // sdot v25.4s, v7.16b, v4.4b[1]
                .inst 0x4f84e8fa // sdot v26.4s, v7.16b, v4.4b[2]
                .inst 0x4fa4e8fb // sdot v27.4s, v7.16b, v4.4b[3]
                .inst 0x4f85e0fc // sdot v28.4s, v7.16b, v5.4b[0]
                .inst 0x4fa5e0fd // sdot v29.4s, v7.16b, v5.4b[1]
                .inst 0x4f85e8fe // sdot v30.4s, v7.16b, v5.4b[2]
                .inst 0x4fa5e8ff // sdot v31.4s, v7.16b, v5.4b[3]
                b Quantization

            LoopIcEndOne:
                .inst 0x4f82e0d0 // sdot v16.4s, v6.16b, v2.4b[0]
                .inst 0x4fa2e0d1 // sdot v17.4s, v6.16b, v2.4b[1]
                .inst 0x4f82e8d2 // sdot v18.4s, v6.16b, v2.4b[2]
                .inst 0x4fa2e8d3 // sdot v19.4s, v6.16b, v2.4b[3]
                .inst 0x4f83e0d4 // sdot v20.4s, v6.16b, v3.4b[0]
                .inst 0x4fa3e0d5 // sdot v21.4s, v6.16b, v3.4b[1]
                .inst 0x4f83e8d6 // sdot v22.4s, v6.16b, v3.4b[2]
                .inst 0x4fa3e8d7 // sdot v23.4s, v6.16b, v3.4b[3]

                .inst 0x4f84e0d8 // sdot v24.4s, v6.16b, v4.4b[0]
                .inst 0x4fa4e0d9 // sdot v25.4s, v6.16b, v4.4b[1]
                .inst 0x4f84e8da // sdot v26.4s, v6.16b, v4.4b[2]
                .inst 0x4fa4e8db // sdot v27.4s, v6.16b, v4.4b[3]
                .inst 0x4f85e0dc // sdot v28.4s, v6.16b, v5.4b[0]
                .inst 0x4fa5e0dd // sdot v29.4s, v6.16b, v5.4b[1]
                .inst 0x4f85e8de // sdot v30.4s, v6.16b, v5.4b[2]
                .inst 0x4fa5e8df // sdot v31.4s, v6.16b, v5.4b[3]

            Quantization:
                dup v2.4s, w18
                sqshl v8.4s, v8.4s, v2.4s
                sqshl v9.4s, v9.4s, v2.4s
                sqshl v10.4s, v10.4s, v2.4s
                sqshl v11.4s, v11.4s, v2.4s
                sqshl v12.4s, v12.4s, v2.4s
                sqshl v13.4s, v13.4s, v2.4s
                sqshl v14.4s, v14.4s, v2.4s
                sqshl v15.4s, v15.4s, v2.4s
                sqshl v16.4s, v16.4s, v2.4s
                sqshl v17.4s, v17.4s, v2.4s
                sqshl v18.4s, v18.4s, v2.4s
                sqshl v19.4s, v19.4s, v2.4s
                sqshl v20.4s, v20.4s, v2.4s
                sqshl v21.4s, v21.4s, v2.4s
                sqshl v22.4s, v22.4s, v2.4s
                sqshl v23.4s, v23.4s, v2.4s
                sqshl v24.4s, v24.4s, v2.4s
                sqshl v25.4s, v25.4s, v2.4s
                sqshl v26.4s, v26.4s, v2.4s
                sqshl v27.4s, v27.4s, v2.4s
                sqshl v28.4s, v28.4s, v2.4s
                sqshl v29.4s, v29.4s, v2.4s
                sqshl v30.4s, v30.4s, v2.4s
                sqshl v31.4s, v31.4s, v2.4s

                dup v3.4s, w17
                sqrdmulh v8.4s, v8.4s, v3.4s
                sqrdmulh v9.4s, v9.4s, v3.4s
                sqrdmulh v10.4s, v10.4s, v3.4s
                sqrdmulh v11.4s, v11.4s, v3.4s
                sqrdmulh v12.4s, v12.4s, v3.4s
                sqrdmulh v13.4s, v13.4s, v3.4s
                sqrdmulh v14.4s, v14.4s, v3.4s
                sqrdmulh v15.4s, v15.4s, v3.4s
                sqrdmulh v16.4s, v16.4s, v3.4s
                sqrdmulh v17.4s, v17.4s, v3.4s
                sqrdmulh v18.4s, v18.4s, v3.4s
                sqrdmulh v19.4s, v19.4s, v3.4s
                sqrdmulh v20.4s, v20.4s, v3.4s
                sqrdmulh v21.4s, v21.4s, v3.4s
                sqrdmulh v22.4s, v22.4s, v3.4s
                sqrdmulh v23.4s, v23.4s, v3.4s
                sqrdmulh v24.4s, v24.4s, v3.4s
                sqrdmulh v25.4s, v25.4s, v3.4s
                sqrdmulh v26.4s, v26.4s, v3.4s
                sqrdmulh v27.4s, v27.4s, v3.4s
                sqrdmulh v28.4s, v28.4s, v3.4s
                sqrdmulh v29.4s, v29.4s, v3.4s
                sqrdmulh v30.4s, v30.4s, v3.4s
                sqrdmulh v31.4s, v31.4s, v3.4s

                dup v4.4s, w19
                add v0.16b, v4.16b, v8.16b
                sshr v0.4s, v0.4s, #31
                sqadd v8.4s, v8.4s, v0.4s
                srshl v8.4s, v8.4s, v4.4s
                add v0.16b, v4.16b, v9.16b
                sshr v1.4s, v1.4s, #31
                sqadd v9.4s, v9.4s, v1.4s
                srshl v9.4s, v9.4s, v4.4s
                add v2.16b, v4.16b, v10.16b
                sshr v2.4s, v2.4s, #31
                sqadd v10.4s, v10.4s, v2.4s
                srshl v10.4s, v10.4s, v4.4s
                add v3.16b, v4.16b, v11.16b
                sshr v3.4s, v3.4s, #31
                sqadd v11.4s, v11.4s, v3.4s
                srshl v11.4s, v11.4s, v4.4s
                add v0.16b, v4.16b, v12.16b
                sshr v0.4s, v0.4s, #31
                sqadd v12.4s, v12.4s, v0.4s
                srshl v12.4s, v12.4s, v4.4s
                add v0.16b, v4.16b, v13.16b
                sshr v1.4s, v1.4s, #31
                sqadd v13.4s, v13.4s, v1.4s
                srshl v13.4s, v13.4s, v4.4s
                add v2.16b, v4.16b, v14.16b
                sshr v2.4s, v2.4s, #31
                sqadd v14.4s, v14.4s, v2.4s
                srshl v14.4s, v14.4s, v4.4s
                add v3.16b, v4.16b, v15.16b
                sshr v3.4s, v3.4s, #31
                sqadd v15.4s, v15.4s, v3.4s
                srshl v15.4s, v15.4s, v4.4s
                add v0.16b, v4.16b, v16.16b
                sshr v0.4s, v0.4s, #31
                sqadd v16.4s, v16.4s, v0.4s
                srshl v16.4s, v16.4s, v4.4s
                add v0.16b, v4.16b, v17.16b
                sshr v1.4s, v1.4s, #31
                sqadd v17.4s, v17.4s, v1.4s
                srshl v17.4s, v17.4s, v4.4s
                add v2.16b, v4.16b, v18.16b
                sshr v2.4s, v2.4s, #31
                sqadd v18.4s, v18.4s, v2.4s
                srshl v18.4s, v18.4s, v4.4s
                add v3.16b, v4.16b, v19.16b
                sshr v3.4s, v3.4s, #31
                sqadd v19.4s, v19.4s, v3.4s
                srshl v19.4s, v19.4s, v4.4s
                add v0.16b, v4.16b, v20.16b
                sshr v0.4s, v0.4s, #31
                sqadd v20.4s, v20.4s, v0.4s
                srshl v20.4s, v20.4s, v4.4s
                add v0.16b, v4.16b, v21.16b
                sshr v1.4s, v1.4s, #31
                sqadd v21.4s, v21.4s, v1.4s
                srshl v21.4s, v21.4s, v4.4s
                add v2.16b, v4.16b, v22.16b
                sshr v2.4s, v2.4s, #31
                sqadd v22.4s, v22.4s, v2.4s
                srshl v10.4s, v10.4s, v4.4s
                add v3.16b, v4.16b, v23.16b
                sshr v3.4s, v3.4s, #31
                sqadd v23.4s, v23.4s, v3.4s
                srshl v23.4s, v23.4s, v4.4s
                add v0.16b, v4.16b, v24.16b
                sshr v0.4s, v0.4s, #31
                sqadd v24.4s, v24.4s, v0.4s
                srshl v24.4s, v24.4s, v4.4s
                add v0.16b, v4.16b, v25.16b
                sshr v1.4s, v1.4s, #31
                sqadd v25.4s, v25.4s, v1.4s
                srshl v25.4s, v25.4s, v4.4s
                add v2.16b, v4.16b, v26.16b
                sshr v2.4s, v2.4s, #31
                sqadd v26.4s, v26.4s, v2.4s
                srshl v26.4s, v26.4s, v4.4s
                add v3.16b, v4.16b, v27.16b
                sshr v3.4s, v3.4s, #31
                sqadd v27.4s, v27.4s, v3.4s
                srshl v27.4s, v27.4s, v4.4s
                add v0.16b, v4.16b, v28.16b
                sshr v0.4s, v0.4s, #31
                sqadd v28.4s, v28.4s, v0.4s
                srshl v28.4s, v28.4s, v4.4s
                add v0.16b, v4.16b, v29.16b
                sshr v1.4s, v1.4s, #31
                sqadd v29.4s, v29.4s, v1.4s
                srshl v29.4s, v29.4s, v4.4s
                add v2.16b, v4.16b, v30.16b
                sshr v2.4s, v2.4s, #31
                sqadd v30.4s, v30.4s, v2.4s
                srshl v30.4s, v30.4s, v4.4s
                add v3.16b, v4.16b, v31.16b
                sshr v3.4s, v3.4s, #31
                sqadd v31.4s, v31.4s, v3.4s
                srshl v31.4s, v31.4s, v4.4s

                dup v5.4s, w16
                add v8.4s, v8.4s, v5.4s
                add v9.4s, v9.4s, v5.4s
                add v10.4s, v10.4s, v5.4s
                add v11.4s, v11.4s, v5.4s
                add v12.4s, v12.4s, v5.4s
                add v13.4s, v13.4s, v5.4s
                add v14.4s, v14.4s, v5.4s
                add v15.4s, v15.4s, v5.4s
                add v16.4s, v16.4s, v5.4s
                add v17.4s, v17.4s, v5.4s
                add v18.4s, v18.4s, v5.4s
                add v19.4s, v19.4s, v5.4s
                add v20.4s, v20.4s, v5.4s
                add v21.4s, v21.4s, v5.4s
                add v22.4s, v22.4s, v5.4s
                add v23.4s, v23.4s, v5.4s
                add v24.4s, v24.4s, v5.4s
                add v25.4s, v25.4s, v5.4s
                add v26.4s, v26.4s, v5.4s
                add v27.4s, v27.4s, v5.4s
                add v28.4s, v28.4s, v5.4s
                add v29.4s, v29.4s, v5.4s
                add v30.4s, v30.4s, v5.4s
                add v31.4s, v31.4s, v5.4s

                dup v0.4s, w8
                smax v8.4s, v8.4s, v0.4s
                smax v9.4s, v9.4s, v0.4s
                smax v10.4s, v10.4s, v0.4s
                smax v11.4s, v11.4s, v0.4s
                smax v12.4s, v12.4s, v0.4s
                smax v13.4s, v13.4s, v0.4s
                smax v14.4s, v14.4s, v0.4s
                smax v15.4s, v15.4s, v0.4s
                smax v16.4s, v16.4s, v0.4s
                smax v17.4s, v17.4s, v0.4s
                smax v18.4s, v18.4s, v0.4s
                smax v19.4s, v19.4s, v0.4s
                smax v20.4s, v20.4s, v0.4s
                smax v21.4s, v21.4s, v0.4s
                smax v22.4s, v22.4s, v0.4s
                smax v23.4s, v23.4s, v0.4s
                smax v24.4s, v24.4s, v0.4s
                smax v25.4s, v25.4s, v0.4s
                smax v26.4s, v26.4s, v0.4s
                smax v27.4s, v27.4s, v0.4s
                smax v28.4s, v28.4s, v0.4s
                smax v29.4s, v29.4s, v0.4s
                smax v30.4s, v30.4s, v0.4s
                smax v31.4s, v31.4s, v0.4s

                dup v1.4s, w9
                smin v8.4s, v8.4s, v1.4s
                smin v9.4s, v9.4s, v1.4s
                smin v10.4s, v10.4s, v1.4s
                smin v11.4s, v11.4s, v1.4s
                smin v12.4s, v12.4s, v1.4s
                smin v13.4s, v13.4s, v1.4s
                smin v14.4s, v14.4s, v1.4s
                smin v15.4s, v15.4s, v1.4s
                smin v16.4s, v16.4s, v1.4s
                smin v17.4s, v17.4s, v1.4s
                smin v18.4s, v18.4s, v1.4s
                smin v19.4s, v19.4s, v1.4s
                smin v20.4s, v20.4s, v1.4s
                smin v21.4s, v21.4s, v1.4s
                smin v22.4s, v22.4s, v1.4s
                smin v23.4s, v23.4s, v1.4s
                smin v24.4s, v24.4s, v1.4s
                smin v25.4s, v25.4s, v1.4s
                smin v26.4s, v26.4s, v1.4s
                smin v27.4s, v27.4s, v1.4s
                smin v28.4s, v28.4s, v1.4s
                smin v29.4s, v29.4s, v1.4s
                smin v30.4s, v30.4s, v1.4s
                smin v31.4s, v31.4s, v1.4s

                sqxtn v6.4h, v8.4s
                sqxtn2 v6.8h, v9.4s
                sqxtn v0.8b, v6.8h
                sqxtn v7.4h, v10.4s
                sqxtn2 v7.8h, v11.4s
                sqxtn2 v0.16b, v7.8h

                sqxtn v6.4h, v12.4s
                sqxtn2 v6.8h, v13.4s
                sqxtn v1.8b, v6.8h
                sqxtn v7.4h, v14.4s
                sqxtn2 v7.8h, v15.4s
                sqxtn2 v1.16b, v7.8h

                sqxtn v6.4h, v16.4s
                sqxtn2 v6.8h, v17.4s
                sqxtn v2.8b, v6.8h
                sqxtn v7.4h, v18.4s
                sqxtn2 v7.8h, v19.4s
                sqxtn2 v2.16b, v7.8h

                sqxtn v6.4h, v20.4s
                sqxtn2 v6.8h, v21.4s
                sqxtn v3.8b, v6.8h
                sqxtn v7.4h, v22.4s
                sqxtn2 v7.8h, v23.4s
                sqxtn2 v3.16b, v7.8h

                sqxtn v6.4h, v24.4s
                sqxtn2 v6.8h, v25.4s
                sqxtn v4.8b, v6.8h
                sqxtn v7.4h, v26.4s
                sqxtn2 v7.8h, v27.4s
                sqxtn2 v4.16b, v7.8h

                sqxtn v6.4h, v28.4s
                sqxtn2 v6.8h, v29.4s
                sqxtn v5.8b, v6.8h
                sqxtn v7.4h, v30.4s
                sqxtn2 v7.8h, v31.4s
                sqxtn2 v5.16b, v7.8h
            // prefetching is not prefered while writing results in spite of cache missings
            // you could try prfm pstl2strm
            WriteStart:
                cmp x6, #1
                beq Write1
                cmp x6, #2
                beq Write2
                cmp x6, #3
                beq Write3
                b Write4
            Write1:
                st1 {v0.b}[0], [x11], x7
                st1 {v0.b}[4], [x11], x7
                st1 {v0.b}[8], [x11], x7
                st1 {v0.b}[12], [x11], x7
                st1 {v1.b}[0], [x11], x7
                st1 {v1.b}[4], [x11], x7
                st1 {v1.b}[8], [x11], x7
                st1 {v1.b}[12], [x11], x7
                st1 {v2.b}[0], [x11], x7
                st1 {v2.b}[4], [x11], x7
                st1 {v2.b}[8], [x11], x7
                st1 {v2.b}[12], [x11], x7
                st1 {v3.b}[0], [x11], x7
                st1 {v3.b}[4], [x11], x7
                st1 {v3.b}[8], [x11], x7
                st1 {v3.b}[12], [x11], x7
                st1 {v4.b}[0], [x11], x7
                st1 {v4.b}[4], [x11], x7
                st1 {v4.b}[8], [x11], x7
                st1 {v4.b}[12], [x11], x7
                st1 {v5.b}[0], [x11], x7
                st1 {v5.b}[4], [x11], x7
                st1 {v5.b}[8], [x11], x7
                st1 {v5.b}[12], [x11]
                add x0, x0, #1
                b WriteEnd
            Write2:
                st1 {v0.h}[0], [x11], x7
                st1 {v0.h}[2], [x11], x7
                st1 {v0.h}[4], [x11], x7
                st1 {v0.h}[6], [x11], x7
                st1 {v1.h}[0], [x11], x7
                st1 {v1.h}[2], [x11], x7
                st1 {v1.h}[4], [x11], x7
                st1 {v1.h}[6], [x11], x7
                st1 {v2.h}[0], [x11], x7
                st1 {v2.h}[2], [x11], x7
                st1 {v2.h}[4], [x11], x7
                st1 {v2.h}[6], [x11], x7
                st1 {v3.h}[0], [x11], x7
                st1 {v3.h}[2], [x11], x7
                st1 {v3.h}[4], [x11], x7
                st1 {v3.h}[6], [x11], x7
                st1 {v4.h}[0], [x11], x7
                st1 {v4.h}[2], [x11], x7
                st1 {v4.h}[4], [x11], x7
                st1 {v4.h}[6], [x11], x7
                st1 {v5.h}[0], [x11], x7
                st1 {v5.h}[2], [x11], x7
                st1 {v5.h}[4], [x11], x7
                st1 {v5.h}[6], [x11]
                add x0, x0, #2
                b WriteEnd
            Write3:
                add x14, x11, #2
                st1 {v0.h}[0], [x11], x7
                st1 {v0.b}[2], [x14], x7
                st1 {v0.h}[2], [x11], x7
                st1 {v0.b}[6], [x14], x7
                st1 {v0.h}[4], [x11], x7
                st1 {v0.b}[10], [x14], x7
                st1 {v0.h}[6], [x11], x7
                st1 {v0.b}[14], [x14], x7
                st1 {v1.h}[0], [x11], x7
                st1 {v1.b}[2], [x14], x7
                st1 {v1.h}[2], [x11], x7
                st1 {v1.b}[6], [x14], x7
                st1 {v1.h}[4], [x11], x7
                st1 {v1.b}[10], [x14], x7
                st1 {v1.h}[6], [x11], x7
                st1 {v1.b}[14], [x14], x7
                st1 {v2.h}[0], [x11], x7
                st1 {v2.b}[2], [x14], x7
                st1 {v2.h}[2], [x11], x7
                st1 {v2.b}[6], [x14], x7
                st1 {v2.h}[4], [x11], x7
                st1 {v2.b}[10], [x14], x7
                st1 {v2.h}[6], [x11], x7
                st1 {v2.b}[14], [x14], x7
                st1 {v3.h}[0], [x11], x7
                st1 {v3.b}[2], [x14], x7
                st1 {v3.h}[2], [x11], x7
                st1 {v3.b}[6], [x14], x7
                st1 {v3.h}[4], [x11], x7
                st1 {v3.b}[10], [x14], x7
                st1 {v3.h}[6], [x11], x7
                st1 {v3.b}[14], [x14], x7
                st1 {v4.h}[0], [x11], x7
                st1 {v4.b}[2], [x14], x7
                st1 {v4.h}[2], [x11], x7
                st1 {v4.b}[6], [x14], x7
                st1 {v4.h}[4], [x11], x7
                st1 {v4.b}[10], [x14], x7
                st1 {v4.h}[6], [x11], x7
                st1 {v4.b}[14], [x14], x7
                st1 {v5.h}[0], [x11], x7
                st1 {v5.b}[2], [x14], x7
                st1 {v5.h}[2], [x11], x7
                st1 {v5.b}[6], [x14], x7
                st1 {v5.h}[4], [x11], x7
                st1 {v5.b}[10], [x14], x7
                st1 {v5.h}[6], [x11], x7
                st1 {v5.b}[14], [x14], x7
                add x0, x0, #3
                b WriteEnd
            Write4:
                st1 {v0.s}[0], [x11], x7
                st1 {v0.s}[1], [x11], x7
                st1 {v0.s}[2], [x11], x7
                st1 {v0.s}[3], [x11], x7
                st1 {v1.s}[0], [x11], x7
                st1 {v1.s}[1], [x11], x7
                st1 {v1.s}[2], [x11], x7
                st1 {v1.s}[3], [x11], x7
                st1 {v2.s}[0], [x11], x7
                st1 {v2.s}[1], [x11], x7
                st1 {v2.s}[2], [x11], x7
                st1 {v2.s}[3], [x11], x7
                st1 {v3.s}[0], [x11], x7
                st1 {v3.s}[1], [x11], x7
                st1 {v3.s}[2], [x11], x7
                st1 {v3.s}[3], [x11], x7
                st1 {v4.s}[0], [x11], x7
                st1 {v4.s}[1], [x11], x7
                st1 {v4.s}[2], [x11], x7
                st1 {v4.s}[3], [x11], x7
                st1 {v5.s}[0], [x11], x7
                st1 {v5.s}[1], [x11], x7
                st1 {v5.s}[2], [x11], x7
                st1 {v5.s}[3], [x11]
                add x0, x0, #4

        WriteEnd:

            subs x10, x10, #1
            bne LoopKsize

        subs x6, x6, #4
        cbz x3, NoStepFowrard
        add x3, x3, #16
    NoStepFowrard:
        bgt LoopOc

    sub sp, sp, #144
    ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
    ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
    ldp x19, x20, [sp], #16
    ret
#endif

