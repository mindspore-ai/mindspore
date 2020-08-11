#ifdef __aarch64__

.text
.align 5
.global IndirectGemmInt8_4x4
#ifndef __APPLE__
.type IndirectGemmInt8_4x4, %function
#endif

// void IndirectGemmInt8_4x4(int8_t *output, int8_t *input, int8_t *weight, int32_t *bias, size_t ksize, size_t ic4,
// size_t oc, size_t offset, int32_t *input_sum, size_t act_min, size_t act_max, size_t out_zp, size_t out_multiplier,
// size_t shift_before, size_t shift_after);
// x0: output, x1: input, x2: weight, x3: bias, x4: kSize, x5: ic4, x6: oc, x7: offset
IndirectGemmInt8_4x4:

    .macro INIT_BIAS
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
            ld1 {v4.16b, v5.16b}, [x2], #32
            // step for output 1-4
            smull v8.8h, v0.8b, v4.8b
            smull v9.8h, v0.8b, v5.8b
            smlal2 v8.8h, v0.16b, v4.16b
            smlal2 v9.8h, v0.16b, v5.16b
            // load input for output 9-16
            ld1 {v6.16b, v7.16b}, [x2], #32
            // another step for output 5-8
            smull v12.8h, v1.8b, v4.8b
            smull v13.8h, v1.8b, v5.8b
            smlal2 v12.8h, v1.16b, v4.16b
            smlal2 v13.8h, v1.16b, v5.16b
            ld1 {v2.16b, v3.16b}, [x12], #32
            smull v10.8h, v0.8b, v6.8b
            smull v11.8h, v0.8b, v7.8b
            saddlp v16.4s, v8.8h
            smlal2 v10.8h, v0.16b, v6.16b
            smlal2 v11.8h, v0.16b, v7.16b
            saddlp v17.4s, v9.8h
            smull v14.8h, v1.8b, v6.8b
            smull v15.8h, v1.8b, v7.8b
            saddlp v18.4s, v10.8h
            smlal2 v14.8h, v1.16b, v6.16b
            smlal2 v15.8h, v1.16b, v7.16b

            subs x13, x5, #1
            beq LoopIcEnd

            LoopIc:
                // load input for output 1-8
                ld1 {v0.16b, v1.16b}, [x12], #32
                sadalp v19.4s, v11.8h
                smull v8.8h, v2.8b, v4.8b
                smull v9.8h, v2.8b, v5.8b
                sadalp v20.4s, v12.8h
                smlal2 v8.8h, v2.16b, v4.16b
                smlal2 v9.8h, v2.16b, v5.16b
                sadalp v21.4s, v13.8h
                smull v10.8h, v2.8b, v6.8b
                smull v11.8h, v2.8b, v7.8b
                sadalp v22.4s, v14.8h
                smlal2 v10.8h, v2.16b, v6.16b
                smlal2 v11.8h, v2.16b, v7.16b
                sadalp v23.4s, v15.8h
                smull v12.8h, v3.8b, v4.8b
                smull v13.8h, v3.8b, v5.8b
                sadalp v24.4s, v8.8h
                smlal2 v12.8h, v3.16b, v4.16b
                smlal2 v13.8h, v3.16b, v5.16b
                ld1 {v4.16b, v5.16b}, [x2], #32
                sadalp v25.4s, v9.8h
                smull v14.8h, v3.8b, v6.8b
                smull v15.8h, v3.8b, v7.8b
                sadalp v26.4s, v10.8h
                smlal2 v14.8h, v3.16b, v6.16b
                smlal2 v15.8h, v3.16b, v7.16b
                ld1 {v6.16b, v7.16b}, [x2], #32
                sadalp v27.4s, v11.8h
                smull v8.8h, v0.8b, v4.8b
                smull v9.8h, v0.8b, v5.8b
                sadalp v28.4s, v12.8h
                smlal2 v8.8h, v0.16b, v4.16b
                smlal2 v9.8h, v0.16b, v5.16b
                ld1 {v2.16b, v3.16b}, [x12], #32
                sadalp v29.4s, v13.8h
                smull v12.8h, v1.8b, v4.8b
                smull v13.8h, v1.8b, v5.8b
                sadalp v30.4s, v14.8h
                smlal2 v12.8h, v1.16b, v4.16b
                smlal2 v13.8h, v1.16b, v5.16b
                sadalp v31.4s, v15.8h
                smull v10.8h, v0.8b, v6.8b
                smull v11.8h, v0.8b, v7.8b
                sadalp v16.4s, v8.8h
                smlal2 v10.8h, v0.16b, v6.16b
                smlal2 v11.8h, v0.16b, v7.16b
                sadalp v17.4s, v9.8h
                smull v14.8h, v1.8b, v6.8b
                smull v15.8h, v1.8b, v7.8b
                sadalp v18.4s, v10.8h
                smlal2 v14.8h, v1.16b, v6.16b
                smlal2 v15.8h, v1.16b, v7.16b

                subs x13, x13, #1
                bne LoopIc

            LoopIcEnd:
                sadalp v19.4s, v11.8h
                smull v8.8h, v2.8b, v4.8b
                smull v9.8h, v2.8b, v5.8b
                sadalp v20.4s, v12.8h
                smlal2 v8.8h, v2.16b, v4.16b
                smlal2 v9.8h, v2.16b, v5.16b
                sadalp v21.4s, v13.8h
                smull v10.8h, v2.8b, v6.8b
                smull v11.8h, v2.8b, v7.8b
                sadalp v22.4s, v14.8h
                smlal2 v10.8h, v2.16b, v6.16b
                smlal2 v11.8h, v2.16b, v7.16b
                sadalp v23.4s, v15.8h
                smull v12.8h, v3.8b, v4.8b
                smull v13.8h, v3.8b, v5.8b
                sadalp v24.4s, v8.8h
                smlal2 v12.8h, v3.16b, v4.16b
                smlal2 v13.8h, v3.16b, v5.16b
                sadalp v25.4s, v9.8h
                smull v14.8h, v3.8b, v6.8b
                smull v15.8h, v3.8b, v7.8b
                sadalp v26.4s, v10.8h
                smlal2 v14.8h, v3.16b, v6.16b
                smlal2 v15.8h, v3.16b, v7.16b
                sadalp v27.4s, v11.8h
                sadalp v28.4s, v12.8h
                sadalp v29.4s, v13.8h
                sadalp v30.4s, v14.8h
                sadalp v31.4s, v15.8h

                // load sum
                mov x20, x15
                ld1r {v8.4s}, [x20], #4
                ld1r {v9.4s}, [x20], #4
                ld1r {v10.4s}, [x20], #4
                ld1r {v11.4s}, [x20]
                // pairwise add
                addp v16.4s, v16.4s, v17.4s
                addp v18.4s, v18.4s, v19.4s
                addp v20.4s, v20.4s, v21.4s
                addp v22.4s, v22.4s, v23.4s
                addp v24.4s, v24.4s, v25.4s
                addp v26.4s, v26.4s, v27.4s
                addp v28.4s, v28.4s, v29.4s
                addp v30.4s, v30.4s, v31.4s
                dup v12.4s, wzr
                cbz x3, NoReadBias
                ld1 {v12.4s}, [x3]            
            NoReadBias:
                addp v16.4s, v16.4s, v18.4s
                addp v20.4s, v20.4s, v22.4s
                addp v24.4s, v24.4s, v26.4s
                addp v28.4s, v28.4s, v30.4s
                sub v16.4s, v16.4s, v8.4s
                sub v20.4s, v20.4s, v9.4s
                sub v24.4s, v24.4s, v10.4s
                sub v28.4s, v28.4s, v11.4s
                add v16.4s, v16.4s, v12.4s
                add v20.4s, v20.4s, v12.4s
                add v24.4s, v24.4s, v12.4s
                add v28.4s, v28.4s, v12.4s

                dup v2.4s, w18
                sqshl v16.4s, v16.4s, v2.4s
                sqshl v20.4s, v20.4s, v2.4s
                sqshl v24.4s, v24.4s, v2.4s
                sqshl v28.4s, v28.4s, v2.4s

                dup v3.4s, w17
                sqrdmulh v16.4s, v16.4s, v3.4s
                sqrdmulh v20.4s, v20.4s, v3.4s
                sqrdmulh v24.4s, v24.4s, v3.4s
                sqrdmulh v28.4s, v28.4s, v3.4s

                dup v4.4s, w19
                and v0.16b, v4.16b, v16.16b
                sshr v0.4s, v0.4s, #31
                sqadd v16.4s, v16.4s, v0.4s
                srshl v16.4s, v16.4s, v4.4s
                and v1.16b, v4.16b, v20.16b
                sshr v1.4s, v1.4s, #31
                sqadd v20.4s, v20.4s, v1.4s
                srshl v20.4s, v20.4s, v4.4s
                and v2.16b, v4.16b, v24.16b
                sshr v2.4s, v2.4s, #31
                sqadd v24.4s, v24.4s, v2.4s
                srshl v24.4s, v24.4s, v4.4s
                and v3.16b, v4.16b, v28.16b
                sshr v3.4s, v3.4s, #31
                sqadd v28.4s, v28.4s, v3.4s
                srshl v28.4s, v28.4s, v4.4s
                
                dup v5.4s, w16
                add v16.4s, v16.4s, v5.4s
                add v20.4s, v20.4s, v5.4s
                add v24.4s, v24.4s, v5.4s
                add v28.4s, v28.4s, v5.4s

                dup v0.4s, w8
                smax v16.4s, v16.4s, v0.4s
                smax v20.4s, v20.4s, v0.4s
                smax v24.4s, v24.4s, v0.4s
                smax v28.4s, v28.4s, v0.4s

                dup v1.4s, w9
                smin v16.4s, v16.4s, v1.4s
                smin v20.4s, v20.4s, v1.4s
                smin v24.4s, v24.4s, v1.4s
                smin v28.4s, v28.4s, v1.4s

                sqxtn v13.4h, v16.4s
                sqxtn2 v13.8h, v20.4s
                sqxtn v15.8b, v13.8h
                sqxtn v14.4h, v24.4s
                sqxtn2 v14.8h, v28.4s
                sqxtn2 v15.16b, v14.8h

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
                st1 {v15.b}[0], [x11], x7
                st1 {v15.b}[4], [x11], x7
                st1 {v15.b}[8], [x11], x7
                st1 {v15.b}[12], [x11]
                add x0, x0, #1
                b WriteEnd
            Write2:
                st1 {v15.h}[0], [x11], x7
                st1 {v15.h}[2], [x11], x7
                st1 {v15.h}[4], [x11], x7
                st1 {v15.h}[6], [x11]
                add x0, x0, #2
                b WriteEnd
            Write3:
                add x14, x11, #2
                st1 {v15.h}[0], [x11], x7
                st1 {v15.b}[2], [x14], x7
                st1 {v15.h}[2], [x11], x7
                st1 {v15.b}[6], [x14], x7
                st1 {v15.h}[4], [x11], x7
                st1 {v15.b}[10], [x14], x7
                st1 {v15.h}[6], [x11]
                st1 {v15.b}[14], [x14]
                add x0, x0, #3
                b WriteEnd
            Write4:
                st1 {v15.s}[0], [x11], x7
                st1 {v15.s}[1], [x11], x7
                st1 {v15.s}[2], [x11], x7
                st1 {v15.s}[3], [x11]
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

