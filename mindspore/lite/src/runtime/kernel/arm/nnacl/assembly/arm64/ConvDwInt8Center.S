#ifdef __aarch64__

.text
.align 5
.global ConvDwInt8Center
#ifndef __APPLE__
.type ConvDwInt8Center, %function
#endif

// void ConvDwInt8Center(int8_t *dst, const int16_t *src, const int16_t *weight, const int32_t *bias, size_t height, size_t width,
//                      size_t kernel_h, size_t kernel_w, size_t out_h_step, size_t block_channel, size_t in_sh_step, size_t in_sw_step,
//                      size_t in_kh_step, size_t in_kw_step, int out_multiplier, int left_shift,
//                      int right_shift, int32_t out_zp, int32_t acc_min, int32_t acc_max);
// x0: dst, x1: src, x2: weight, x3: bias, x4: height, x5: weight, x6: kernel_h, x7: kernel_w, 
// x8: out_h_step, x9: block_channel, x10: in_sh_step, x11: in_sw_step, x12: in_kh_step, x13: in_kw_step
// x14: out_multiplier, #56: left_shift, #64: right_shift, #72:out_zp, #80: acc_min, #88: acc_max
ConvDwInt8Center:
    // registers v8 ~ v15 must be preserved by a callee across subroutine calls, according to
    // https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst#simd-and-floating-point-registers
    // x19 ~ x29 should be also preserved
    // whereas our coding style do not permit such amount of parameters
    sub sp, sp, #48
    stp x19, x20, [sp], #16
    stp x21, x22, [sp], #16
    stp x23, x24, [sp], #16

    ldr x8, [sp]
    ldr x9, [sp, #8]
    ldr x10, [sp, #16]
    ldr x11, [sp, #24]
    ldr x12, [sp, #32]
    ldr x13, [sp, #40]

    ldr w14, [sp, #56]
    dup v26.4s, w14

    ldr x15, [sp, #48]
    dup v27.4s, w15

    ldr w16, [sp, #64]
    dup v28.4s, w16

    ldr w17, [sp, #72]
    dup v29.4s, w17
                
    ldr w18, [sp, #80]
    dup v30.4s, w18

    ldr w19, [sp, #88]
    dup v31.4s, w19

    ld1 {v24.4s}, [x3]

    LoopH:
        mov x23, x1
        mov x24, x5
        mov x3, x0
        cmp x24, #8
        blt LoopW
        cmp x24, #16
        blt LoopW8

        LoopW16:
            mov x19, #16
            mul x19, x19, x11
            mov x16, x23
            mov x17, x2
            mov x20, x6
            mov v0.16b, v24.16b
            mov v1.16b, v24.16b
            mov v2.16b, v24.16b
            mov v3.16b, v24.16b
            mov v4.16b, v24.16b
            mov v5.16b, v24.16b
            mov v6.16b, v24.16b
            mov v7.16b, v24.16b
            mov v8.16b, v24.16b
            mov v9.16b, v24.16b
            mov v10.16b, v24.16b
            mov v11.16b, v24.16b
            mov v12.16b, v24.16b
            mov v13.16b, v24.16b
            mov v14.16b, v24.16b
            mov v15.16b, v24.16b
            LoopKh16:
                mov x18, x7
                mov x21, x16
                LoopKw16:
                    mov x22, x21
                    ld1 {v25.4h}, [x17], #8
                    ld1 {v16.4h}, [x22], x13
                    ld1 {v17.4h}, [x22], x13
                    smlal v0.4s, v16.4h, v25.4h
                    smlal v1.4s, v17.4h, v25.4h
                    ld1 {v18.4h}, [x22], x13
                    ld1 {v19.4h}, [x22], x13
                    smlal v2.4s, v18.4h, v25.4h
                    smlal v3.4s, v19.4h, v25.4h
                    ld1 {v20.4h}, [x22], x13
                    ld1 {v21.4h}, [x22], x13
                    smlal v4.4s, v20.4h, v25.4h
                    smlal v5.4s, v21.4h, v25.4h
                    ld1 {v22.4h}, [x22], x13
                    ld1 {v23.4h}, [x22], x13
                    smlal v6.4s, v22.4h, v25.4h
                    smlal v7.4s, v23.4h, v25.4h
                    ld1 {v16.4h}, [x22], x13
                    ld1 {v17.4h}, [x22], x13
                    smlal v8.4s, v16.4h, v25.4h
                    smlal v9.4s, v17.4h, v25.4h
                    ld1 {v18.4h}, [x22], x13
                    ld1 {v19.4h}, [x22], x13
                    smlal v10.4s, v18.4h, v25.4h
                    smlal v11.4s, v19.4h, v25.4h
                    ld1 {v20.4h}, [x22], x13
                    ld1 {v21.4h}, [x22], x13
                    smlal v12.4s, v20.4h, v25.4h
                    smlal v13.4s, v21.4h, v25.4h
                    ld1 {v22.4h}, [x22], x13
                    ld1 {v23.4h}, [x22], x13
                    smlal v14.4s, v22.4h, v25.4h
                    smlal v15.4s, v23.4h, v25.4h
                    subs x18, x18, #1
                    add x21, x21, x13
                    bne LoopKw16
                add x16, x16, x12
                subs x20, x20, #1
                bne LoopKh16

            sqshl v0.4s, v0.4s, v26.4s
            sqshl v1.4s, v1.4s, v26.4s
            sqshl v2.4s, v2.4s, v26.4s
            sqshl v3.4s, v3.4s, v26.4s
            sqshl v4.4s, v4.4s, v26.4s
            sqshl v5.4s, v5.4s, v26.4s
            sqshl v6.4s, v6.4s, v26.4s
            sqshl v7.4s, v7.4s, v26.4s
            sqshl v8.4s, v8.4s, v26.4s
            sqshl v9.4s, v9.4s, v26.4s
            sqshl v10.4s, v10.4s, v26.4s
            sqshl v11.4s, v11.4s, v26.4s
            sqshl v12.4s, v12.4s, v26.4s
            sqshl v13.4s, v13.4s, v26.4s
            sqshl v14.4s, v14.4s, v26.4s
            sqshl v15.4s, v15.4s, v26.4s
            sqrdmulh v0.4s, v0.4s, v27.4s
            sqrdmulh v1.4s, v1.4s, v27.4s
            sqrdmulh v2.4s, v2.4s, v27.4s
            sqrdmulh v3.4s, v3.4s, v27.4s
            sqrdmulh v4.4s, v4.4s, v27.4s
            sqrdmulh v5.4s, v5.4s, v27.4s
            sqrdmulh v6.4s, v6.4s, v27.4s
            sqrdmulh v7.4s, v7.4s, v27.4s
            sqrdmulh v8.4s, v8.4s, v27.4s
            sqrdmulh v9.4s, v9.4s, v27.4s
            sqrdmulh v10.4s, v10.4s, v27.4s
            sqrdmulh v11.4s, v11.4s, v27.4s
            sqrdmulh v12.4s, v12.4s, v27.4s
            sqrdmulh v13.4s, v13.4s, v27.4s
            sqrdmulh v14.4s, v14.4s, v27.4s
            sqrdmulh v15.4s, v15.4s, v27.4s

            and v16.16b, v28.16b, v0.16b
            sshr v16.4s, v16.4s, #31
            sqadd v0.4s, v0.4s, v16.4s
            srshl v0.4s, v0.4s, v28.4s
            and v17.16b, v28.16b, v1.16b
            sshr v17.4s, v17.4s, #31
            sqadd v1.4s, v1.4s, v17.4s
            srshl v1.4s, v1.4s, v28.4s
            and v18.16b, v28.16b, v2.16b
            sshr v18.4s, v18.4s, #31
            sqadd v2.4s, v2.4s, v18.4s
            srshl v2.4s, v2.4s, v28.4s
            and v19.16b, v28.16b, v3.16b
            sshr v19.4s, v19.4s, #31
            sqadd v3.4s, v3.4s, v19.4s
            srshl v3.4s, v3.4s, v28.4s
            and v20.16b, v28.16b, v4.16b
            sshr v20.4s, v20.4s, #31
            sqadd v4.4s, v4.4s, v20.4s
            srshl v4.4s, v4.4s, v28.4s
            and v21.16b, v28.16b, v5.16b
            sshr v21.4s, v21.4s, #31
            sqadd v5.4s, v5.4s, v21.4s
            srshl v5.4s, v5.4s, v28.4s
            and v22.16b, v28.16b, v6.16b
            sshr v22.4s, v22.4s, #31
            sqadd v6.4s, v6.4s, v22.4s
            srshl v6.4s, v6.4s, v28.4s
            and v23.16b, v28.16b, v7.16b
            sshr v23.4s, v23.4s, #31
            sqadd v7.4s, v7.4s, v23.4s
            srshl v7.4s, v7.4s, v28.4s
            and v16.16b, v28.16b, v8.16b
            sshr v16.4s, v16.4s, #31
            sqadd v8.4s, v8.4s, v16.4s
            srshl v8.4s, v8.4s, v28.4s
            and v17.16b, v28.16b, v9.16b
            sshr v17.4s, v17.4s, #31
            sqadd v9.4s, v9.4s, v17.4s
            srshl v9.4s, v9.4s, v28.4s
            and v18.16b, v28.16b, v10.16b
            sshr v18.4s, v18.4s, #31
            sqadd v10.4s, v10.4s, v18.4s
            srshl v10.4s, v10.4s, v28.4s
            and v19.16b, v28.16b, v11.16b
            sshr v19.4s, v19.4s, #31
            sqadd v11.4s, v11.4s, v19.4s
            srshl v11.4s, v11.4s, v28.4s
            and v20.16b, v28.16b, v12.16b
            sshr v20.4s, v20.4s, #31
            sqadd v12.4s, v12.4s, v20.4s
            srshl v12.4s, v12.4s, v28.4s
            and v21.16b, v28.16b, v13.16b
            sshr v21.4s, v21.4s, #31
            sqadd v13.4s, v13.4s, v21.4s
            srshl v13.4s, v13.4s, v28.4s
            and v22.16b, v28.16b, v14.16b
            sshr v22.4s, v22.4s, #31
            sqadd v14.4s, v14.4s, v22.4s
            srshl v14.4s, v14.4s, v28.4s
            and v23.16b, v28.16b, v15.16b
            sshr v23.4s, v23.4s, #31
            sqadd v15.4s, v15.4s, v23.4s
            srshl v15.4s, v15.4s, v28.4s

            add v0.4s, v0.4s, v29.4s
            add v1.4s, v1.4s, v29.4s
            add v2.4s, v2.4s, v29.4s
            add v3.4s, v3.4s, v29.4s
            add v4.4s, v4.4s, v29.4s
            add v5.4s, v5.4s, v29.4s
            add v6.4s, v6.4s, v29.4s
            add v7.4s, v7.4s, v29.4s
            add v8.4s, v8.4s, v29.4s
            add v9.4s, v9.4s, v29.4s
            add v10.4s, v10.4s, v29.4s
            add v11.4s, v11.4s, v29.4s
            add v12.4s, v12.4s, v29.4s
            add v13.4s, v13.4s, v29.4s
            add v14.4s, v14.4s, v29.4s
            add v15.4s, v15.4s, v29.4s
            smax v0.4s, v0.4s, v30.4s
            smax v1.4s, v1.4s, v30.4s
            smax v2.4s, v2.4s, v30.4s
            smax v3.4s, v3.4s, v30.4s
            smax v4.4s, v4.4s, v30.4s
            smax v5.4s, v5.4s, v30.4s
            smax v6.4s, v6.4s, v30.4s
            smax v7.4s, v7.4s, v30.4s
            smax v8.4s, v8.4s, v30.4s
            smax v9.4s, v9.4s, v30.4s
            smax v10.4s, v10.4s, v30.4s
            smax v11.4s, v11.4s, v30.4s
            smax v12.4s, v12.4s, v30.4s
            smax v13.4s, v13.4s, v30.4s
            smax v14.4s, v14.4s, v30.4s
            smax v15.4s, v15.4s, v30.4s
            smin v0.4s, v0.4s, v31.4s
            smin v1.4s, v1.4s, v31.4s
            smin v2.4s, v2.4s, v31.4s
            smin v3.4s, v3.4s, v31.4s
            smin v4.4s, v4.4s, v31.4s
            smin v5.4s, v5.4s, v31.4s
            smin v6.4s, v6.4s, v31.4s
            smin v7.4s, v7.4s, v31.4s
            smin v8.4s, v8.4s, v31.4s
            smin v9.4s, v9.4s, v31.4s
            smin v10.4s, v10.4s, v31.4s
            smin v11.4s, v11.4s, v31.4s
            smin v12.4s, v12.4s, v31.4s
            smin v13.4s, v13.4s, v31.4s
            smin v14.4s, v14.4s, v31.4s
            smin v15.4s, v15.4s, v31.4s

            sqxtn v0.4h, v0.4s
            sqxtn v1.4h, v1.4s
            sqxtn v2.4h, v2.4s
            sqxtn v3.4h, v3.4s
            sqxtn v4.4h, v4.4s
            sqxtn v5.4h, v5.4s
            sqxtn v6.4h, v6.4s
            sqxtn v7.4h, v7.4s
            sqxtn v8.4h, v8.4s
            sqxtn v9.4h, v9.4s
            sqxtn v10.4h, v10.4s
            sqxtn v11.4h, v11.4s
            sqxtn v12.4h, v12.4s
            sqxtn v13.4h, v13.4s
            sqxtn v14.4h, v14.4s
            sqxtn v15.4h, v15.4s
            sqxtn v0.8b, v0.8h
            sqxtn v1.8b, v1.8h
            sqxtn v2.8b, v2.8h
            sqxtn v3.8b, v3.8h
            sqxtn v4.8b, v4.8h
            sqxtn v5.8b, v5.8h
            sqxtn v6.8b, v6.8h
            sqxtn v7.8b, v7.8h
            sqxtn v8.8b, v8.8h
            sqxtn v9.8b, v9.8h
            sqxtn v10.8b, v10.8h
            sqxtn v11.8b, v11.8h
            sqxtn v12.8b, v12.8h
            sqxtn v13.8b, v13.8h
            sqxtn v14.8b, v14.8h
            sqxtn v15.8b, v15.8h

            add x17, x3, #1
            add x18, x3, #2
            add x21, x3, #3
            st1 {v0.b}[0], [x3], x9
            st1 {v0.b}[1], [x17], x9
            st1 {v0.b}[2], [x18], x9
            st1 {v0.b}[3], [x21], x9

            st1 {v1.b}[0], [x3], x9
            st1 {v1.b}[1], [x17], x9
            st1 {v1.b}[2], [x18], x9
            st1 {v1.b}[3], [x21], x9

            st1 {v2.b}[0], [x3], x9
            st1 {v2.b}[1], [x17], x9
            st1 {v2.b}[2], [x18], x9
            st1 {v2.b}[3], [x21], x9

            st1 {v3.b}[0], [x3], x9
            st1 {v3.b}[1], [x17], x9
            st1 {v3.b}[2], [x18], x9
            st1 {v3.b}[3], [x21], x9

            st1 {v4.b}[0], [x3], x9
            st1 {v4.b}[1], [x17], x9
            st1 {v4.b}[2], [x18], x9
            st1 {v4.b}[3], [x21], x9

            st1 {v5.b}[0], [x3], x9
            st1 {v5.b}[1], [x17], x9
            st1 {v5.b}[2], [x18], x9
            st1 {v5.b}[3], [x21], x9

            st1 {v6.b}[0], [x3], x9
            st1 {v6.b}[1], [x17], x9
            st1 {v6.b}[2], [x18], x9
            st1 {v6.b}[3], [x21], x9

            st1 {v7.b}[0], [x3], x9
            st1 {v7.b}[1], [x17], x9
            st1 {v7.b}[2], [x18], x9
            st1 {v7.b}[3], [x21], x9

            st1 {v8.b}[0], [x3], x9
            st1 {v8.b}[1], [x17], x9
            st1 {v8.b}[2], [x18], x9
            st1 {v8.b}[3], [x21], x9

            st1 {v9.b}[0], [x3], x9
            st1 {v9.b}[1], [x17], x9
            st1 {v9.b}[2], [x18], x9
            st1 {v9.b}[3], [x21], x9

            st1 {v10.b}[0], [x3], x9
            st1 {v10.b}[1], [x17], x9
            st1 {v10.b}[2], [x18], x9
            st1 {v10.b}[3], [x21], x9

            st1 {v11.b}[0], [x3], x9
            st1 {v11.b}[1], [x17], x9
            st1 {v11.b}[2], [x18], x9
            st1 {v11.b}[3], [x21], x9

            st1 {v12.b}[0], [x3], x9
            st1 {v12.b}[1], [x17], x9
            st1 {v12.b}[2], [x18], x9
            st1 {v12.b}[3], [x21], x9

            st1 {v13.b}[0], [x3], x9
            st1 {v13.b}[1], [x17], x9
            st1 {v13.b}[2], [x18], x9
            st1 {v13.b}[3], [x21], x9

            st1 {v14.b}[0], [x3], x9
            st1 {v14.b}[1], [x17], x9
            st1 {v14.b}[2], [x18], x9
            st1 {v14.b}[3], [x21], x9

            st1 {v15.b}[0], [x3], x9
            st1 {v15.b}[1], [x17], x9
            st1 {v15.b}[2], [x18], x9
            st1 {v15.b}[3], [x21], x9

            add x23, x23, x19
            sub x24, x24, #16
            cmp x24, #0
            ble LoopWEnd
            cmp x24, #8
            blt LoopW
            cmp x24, #16
            bge LoopW16
        LoopW8:
            mov x19, #8
            mul x19, x19, x11
            mov x16, x23
            mov x17, x2
            mov x20, x6
            mov v0.16b, v24.16b
            mov v1.16b, v24.16b
            mov v2.16b, v24.16b
            mov v3.16b, v24.16b
            mov v4.16b, v24.16b
            mov v5.16b, v24.16b
            mov v6.16b, v24.16b
            mov v7.16b, v24.16b
            LoopKh8:
                mov x18, x7
                mov x21, x16
                LoopKw8:
                    mov x22, x21
                    ld1 {v25.4h}, [x17], #8
                    ld1 {v16.4h}, [x22], x13
                    ld1 {v17.4h}, [x22], x13
                    smlal v0.4s, v16.4h, v25.4h
                    smlal v1.4s, v17.4h, v25.4h
                    ld1 {v18.4h}, [x22], x13
                    ld1 {v19.4h}, [x22], x13
                    smlal v2.4s, v18.4h, v25.4h
                    smlal v3.4s, v19.4h, v25.4h
                    ld1 {v20.4h}, [x22], x13
                    ld1 {v21.4h}, [x22], x13
                    smlal v4.4s, v20.4h, v25.4h
                    smlal v5.4s, v21.4h, v25.4h
                    ld1 {v22.4h}, [x22], x13
                    ld1 {v23.4h}, [x22], x13
                    smlal v6.4s, v22.4h, v25.4h
                    smlal v7.4s, v23.4h, v25.4h
                    subs x18, x18, #1
                    add x21, x21, x13
                    bne LoopKw8
                add x16, x16, x12
                subs x20, x20, #1
                bne LoopKh8

            sqshl v0.4s, v0.4s, v26.4s
            sqshl v1.4s, v1.4s, v26.4s
            sqshl v2.4s, v2.4s, v26.4s
            sqshl v3.4s, v3.4s, v26.4s
            sqshl v4.4s, v4.4s, v26.4s
            sqshl v5.4s, v5.4s, v26.4s
            sqshl v6.4s, v6.4s, v26.4s
            sqshl v7.4s, v7.4s, v26.4s
            sqrdmulh v0.4s, v0.4s, v27.4s
            sqrdmulh v1.4s, v1.4s, v27.4s
            sqrdmulh v2.4s, v2.4s, v27.4s
            sqrdmulh v3.4s, v3.4s, v27.4s
            sqrdmulh v4.4s, v4.4s, v27.4s
            sqrdmulh v5.4s, v5.4s, v27.4s
            sqrdmulh v6.4s, v6.4s, v27.4s
            sqrdmulh v7.4s, v7.4s, v27.4s

            and v16.16b, v28.16b, v0.16b
            sshr v16.4s, v16.4s, #31
            sqadd v0.4s, v0.4s, v16.4s
            srshl v0.4s, v0.4s, v28.4s
            and v17.16b, v28.16b, v1.16b
            sshr v17.4s, v17.4s, #31
            sqadd v1.4s, v1.4s, v17.4s
            srshl v1.4s, v1.4s, v28.4s
            and v18.16b, v28.16b, v2.16b
            sshr v18.4s, v18.4s, #31
            sqadd v2.4s, v2.4s, v18.4s
            srshl v2.4s, v2.4s, v28.4s
            and v19.16b, v28.16b, v3.16b
            sshr v19.4s, v19.4s, #31
            sqadd v3.4s, v3.4s, v19.4s
            srshl v3.4s, v3.4s, v28.4s
            and v20.16b, v28.16b, v4.16b
            sshr v20.4s, v20.4s, #31
            sqadd v4.4s, v4.4s, v20.4s
            srshl v4.4s, v4.4s, v28.4s
            and v21.16b, v28.16b, v5.16b
            sshr v21.4s, v21.4s, #31
            sqadd v5.4s, v5.4s, v21.4s
            srshl v5.4s, v5.4s, v28.4s
            and v22.16b, v28.16b, v6.16b
            sshr v22.4s, v22.4s, #31
            sqadd v6.4s, v6.4s, v22.4s
            srshl v6.4s, v6.4s, v28.4s
            and v23.16b, v28.16b, v7.16b
            sshr v23.4s, v23.4s, #31
            sqadd v7.4s, v7.4s, v23.4s
            srshl v7.4s, v7.4s, v28.4s

            add v0.4s, v0.4s, v29.4s
            add v1.4s, v1.4s, v29.4s
            add v2.4s, v2.4s, v29.4s
            add v3.4s, v3.4s, v29.4s
            add v4.4s, v4.4s, v29.4s
            add v5.4s, v5.4s, v29.4s
            add v6.4s, v6.4s, v29.4s
            add v7.4s, v7.4s, v29.4s
            smax v0.4s, v0.4s, v30.4s
            smax v1.4s, v1.4s, v30.4s
            smax v2.4s, v2.4s, v30.4s
            smax v3.4s, v3.4s, v30.4s
            smax v4.4s, v4.4s, v30.4s
            smax v5.4s, v5.4s, v30.4s
            smax v6.4s, v6.4s, v30.4s
            smax v7.4s, v7.4s, v30.4s
            smin v0.4s, v0.4s, v31.4s
            smin v1.4s, v1.4s, v31.4s
            smin v2.4s, v2.4s, v31.4s
            smin v3.4s, v3.4s, v31.4s
            smin v4.4s, v4.4s, v31.4s
            smin v5.4s, v5.4s, v31.4s
            smin v6.4s, v6.4s, v31.4s
            smin v7.4s, v7.4s, v31.4s

            sqxtn v0.4h, v0.4s
            sqxtn v1.4h, v1.4s
            sqxtn v2.4h, v2.4s
            sqxtn v3.4h, v3.4s
            sqxtn v4.4h, v4.4s
            sqxtn v5.4h, v5.4s
            sqxtn v6.4h, v6.4s
            sqxtn v7.4h, v7.4s
            sqxtn v0.8b, v0.8h
            sqxtn v1.8b, v1.8h
            sqxtn v2.8b, v2.8h
            sqxtn v3.8b, v3.8h
            sqxtn v4.8b, v4.8h
            sqxtn v5.8b, v5.8h
            sqxtn v6.8b, v6.8h
            sqxtn v7.8b, v7.8h

            add x17, x3, #1
            add x18, x3, #2
            add x21, x3, #3
            st1 {v0.b}[0], [x3], x9
            st1 {v0.b}[1], [x17], x9
            st1 {v0.b}[2], [x18], x9
            st1 {v0.b}[3], [x21], x9

            st1 {v1.b}[0], [x3], x9
            st1 {v1.b}[1], [x17], x9
            st1 {v1.b}[2], [x18], x9
            st1 {v1.b}[3], [x21], x9

            st1 {v2.b}[0], [x3], x9
            st1 {v2.b}[1], [x17], x9
            st1 {v2.b}[2], [x18], x9
            st1 {v2.b}[3], [x21], x9

            st1 {v3.b}[0], [x3], x9
            st1 {v3.b}[1], [x17], x9
            st1 {v3.b}[2], [x18], x9
            st1 {v3.b}[3], [x21], x9

            st1 {v4.b}[0], [x3], x9
            st1 {v4.b}[1], [x17], x9
            st1 {v4.b}[2], [x18], x9
            st1 {v4.b}[3], [x21], x9

            st1 {v5.b}[0], [x3], x9
            st1 {v5.b}[1], [x17], x9
            st1 {v5.b}[2], [x18], x9
            st1 {v5.b}[3], [x21], x9

            st1 {v6.b}[0], [x3], x9
            st1 {v6.b}[1], [x17], x9
            st1 {v6.b}[2], [x18], x9
            st1 {v6.b}[3], [x21], x9

            st1 {v7.b}[0], [x3], x9
            st1 {v7.b}[1], [x17], x9
            st1 {v7.b}[2], [x18], x9
            st1 {v7.b}[3], [x21], x9

            add x23, x23, x19
            sub x24, x24, #8
            cmp x24, #0
            ble LoopWEnd
            cmp x24, #8
            bge LoopW8
        LoopW:
            mov x16, x23
            mov x17, x2
            mov x20, x6
            mov v0.16b, v24.16b
            LoopKh:
                mov x18, x7
                mov x22, x16
                LoopKw:
                    ld1 {v16.4h}, [x22], x13
                    ld1 {v25.4h}, [x17], #8
                    smlal v0.4s, v16.4h, v25.4h
                    subs x18, x18, #1
                    bne LoopKw
                add x16, x16, x12
                subs x20, x20, #1
                bne LoopKh

            sqshl v0.4s, v0.4s, v26.4s
            sqrdmulh v0.4s, v0.4s, v27.4s

            and v16.16b, v28.16b, v0.16b
            sshr v16.4s, v16.4s, #31
            sqadd v0.4s, v0.4s, v16.4s
            srshl v0.4s, v0.4s, v28.4s

            add v0.4s, v0.4s, v29.4s
            smax v0.4s, v0.4s, v30.4s
            smin v0.4s, v0.4s, v31.4s

            sqxtn v0.4h, v0.4s
            sqxtn v0.8b, v0.8h

            mov x17, x3
            st1 {v0.b}[0], [x17], #1
            st1 {v0.b}[1], [x17], #1
            st1 {v0.b}[2], [x17], #1
            st1 {v0.b}[3], [x17], #1
            add x3, x3, x9

            add x23, x23, x11
            subs x24, x24, #1
            bne LoopW
    LoopWEnd:
        add x0, x0, x8
        add x1, x1, x10
        subs x4, x4, #1
        bne LoopH

    sub sp, sp, #48
    ldp x19, x20, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x23, x24, [sp], #16
    ret
#endif
