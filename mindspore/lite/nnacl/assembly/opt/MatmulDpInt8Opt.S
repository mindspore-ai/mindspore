#ifdef __aarch64__
#include "nnacl/assembly_global.h"
    .text
    .align 5
    .global MatmulInt8DpOpt
#ifndef __APPLE__
    .type MatmulInt8DpOpt, %function
#endif

//void MatmulInt8DpOpt(const int8_t *a, const int8_t *b, int8_t *dst, int row, int col, int deep4, const int *a_sums,
//                     const int *bias, int act_min, int act_max, int out_zp, int32_t *multiplier, int32_t *left_shift,
//                     int32_t *right_shift, size_t stride, size_t filter_peroc, int32_t *filter_zp)

// x0: a(left matrix ptr)
// x1: b(right matrix ptr)
// x2: out ptr
// x3: row
// x4: col
// x5: deep4
// x6: a_sums
// x7: bias
// w8: act_min
// w9: act_max
// w10: out_zp
// x11: multiplier
// x12: left_shift
// x13: right_shift
// x14: stride
// x15: filter_peroc
// x28: filter_zp

asm_function MatmulInt8DpOpt
  sub sp, sp, #208
  st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
  st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
  stp x19, x20, [sp], #16
  stp x21, x22, [sp], #16
  stp x23, x24, [sp], #16
  stp x25, x26, [sp], #16
  stp x27, x28, [sp], #16

  ldr w8, [sp]
  ldr w9, [sp, #8]
  ldr w10, [sp, #16]
  ldr x11, [sp, #24]
  ldr x12, [sp, #32]
  ldr x13, [sp, #40]
  ldr x14, [sp, #48]
  ldr x15, [sp, #56]

  mov x23, #4
  mul x23, x23, x5 // lhs step
  mov x24, #4
  mul x24, x24, x14 // dst step

LoopRow:
    mov x16, x1 // reload rhs ptr
    mov x17, x4 // reload rhs col
    mov x18, x7 // reload bias ptr
    mov x25, x6 // reload input_sum ptr
    mov x27, x2 // reload dst ptr
    ldr x28, [sp, #64] // reload filter_zp

    LoopCol:
        mov x19, x27 // reload dst ptr
        mov x20, x0 // reload lhs ptr
        mov x21, x5 // reload depth

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

        cmp x17, #4
        ble LoopDepthQuarter
        cmp x17, #8
        ble LoopDepthHalf

        LoopDepth:
            ld1 {v0.16b}, [x20], #16
            ld1 {v1.16b, v2.16b, v3.16b, v4.16b}, [x16], #64
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

            subs x21, x21, #4
            bgt LoopDepth

        Bias:
            cbz x7, NoReadBias
            ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x18], #64
            add v16.4s, v16.4s, v0.4s
            add v17.4s, v17.4s, v1.4s
            add v18.4s, v18.4s, v2.4s
            add v19.4s, v19.4s, v3.4s
            add v20.4s, v20.4s, v0.4s
            add v21.4s, v21.4s, v1.4s
            add v22.4s, v22.4s, v2.4s
            add v23.4s, v23.4s, v3.4s
            add v24.4s, v24.4s, v0.4s
            add v25.4s, v25.4s, v1.4s
            add v26.4s, v26.4s, v2.4s
            add v27.4s, v27.4s, v3.4s
            add v28.4s, v28.4s, v0.4s
            add v29.4s, v29.4s, v1.4s
            add v30.4s, v30.4s, v2.4s
            add v31.4s, v31.4s, v3.4s

        NoReadBias:
            ld1r {v12.4s}, [x25], #4
            ld1r {v13.4s}, [x25], #4
            ld1r {v14.4s}, [x25], #4
            ld1r {v15.4s}, [x25], #4
            cbnz x15, PerChannelSum

        PerTensorSum:
            sub v16.4s, v16.4s, v12.4s
            sub v17.4s, v17.4s, v12.4s
            sub v18.4s, v18.4s, v12.4s
            sub v19.4s, v19.4s, v12.4s
            sub v20.4s, v20.4s, v13.4s
            sub v21.4s, v21.4s, v13.4s
            sub v22.4s, v22.4s, v13.4s
            sub v23.4s, v23.4s, v13.4s
            sub v24.4s, v24.4s, v14.4s
            sub v25.4s, v25.4s, v14.4s
            sub v26.4s, v26.4s, v14.4s
            sub v27.4s, v27.4s, v14.4s
            sub v28.4s, v28.4s, v15.4s
            sub v29.4s, v29.4s, v15.4s
            sub v30.4s, v30.4s, v15.4s
            sub v31.4s, v31.4s, v15.4s

            b PerTensor

        PerChannelSum:
            ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x28], #64
            mul v0.4s, v8.4s, v12.4s
            mul v1.4s, v9.4s, v12.4s
            mul v2.4s, v10.4s, v12.4s
            mul v3.4s, v11.4s, v12.4s
            mul v4.4s, v8.4s, v13.4s
            mul v5.4s, v9.4s, v13.4s
            mul v6.4s, v10.4s, v13.4s
            mul v7.4s, v11.4s, v13.4s
            sub v16.4s, v16.4s, v0.4s
            sub v17.4s, v17.4s, v1.4s
            sub v18.4s, v18.4s, v2.4s
            sub v19.4s, v19.4s, v3.4s
            sub v20.4s, v20.4s, v4.4s
            sub v21.4s, v21.4s, v5.4s
            sub v22.4s, v22.4s, v6.4s
            sub v23.4s, v23.4s, v7.4s
            mul v0.4s, v8.4s, v14.4s
            mul v1.4s, v9.4s, v14.4s
            mul v2.4s, v10.4s, v14.4s
            mul v3.4s, v11.4s, v14.4s
            mul v4.4s, v8.4s, v15.4s
            mul v5.4s, v9.4s, v15.4s
            mul v6.4s, v10.4s, v15.4s
            mul v7.4s, v11.4s, v15.4s
            sub v24.4s, v24.4s, v0.4s
            sub v25.4s, v25.4s, v1.4s
            sub v26.4s, v26.4s, v2.4s
            sub v27.4s, v27.4s, v3.4s
            sub v28.4s, v28.4s, v4.4s
            sub v29.4s, v29.4s, v5.4s
            sub v30.4s, v30.4s, v6.4s
            sub v31.4s, v31.4s, v7.4s

        PerTensor:
            cbnz x15, PerChannel
            ld1r {v0.4s}, [x12]
            mov v1.16b, v0.16b
            mov v2.16b, v0.16b
            mov v3.16b, v0.16b
            ld1r {v4.4s}, [x11]
            mov v5.16b, v4.16b
            mov v6.16b, v4.16b
            mov v7.16b, v4.16b
            ld1r {v8.4s}, [x13]
            mov v9.16b, v8.16b
            mov v10.16b, v8.16b
            mov v11.16b, v8.16b

            b Quantization

        PerChannel:
            ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x12], #64
            ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x11], #64
            ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x13], #64

        Quantization:
            sqshl v16.4s, v16.4s, v0.4s
            sqshl v17.4s, v17.4s, v1.4s
            sqshl v18.4s, v18.4s, v2.4s
            sqshl v19.4s, v19.4s, v3.4s
            sqshl v20.4s, v20.4s, v0.4s
            sqshl v21.4s, v21.4s, v1.4s
            sqshl v22.4s, v22.4s, v2.4s
            sqshl v23.4s, v23.4s, v3.4s
            sqshl v24.4s, v24.4s, v0.4s
            sqshl v25.4s, v25.4s, v1.4s
            sqshl v26.4s, v26.4s, v2.4s
            sqshl v27.4s, v27.4s, v3.4s
            sqshl v28.4s, v28.4s, v0.4s
            sqshl v29.4s, v29.4s, v1.4s
            sqshl v30.4s, v30.4s, v2.4s
            sqshl v31.4s, v31.4s, v3.4s

            sqrdmulh v16.4s, v16.4s, v4.4s
            sqrdmulh v17.4s, v17.4s, v5.4s
            sqrdmulh v18.4s, v18.4s, v6.4s
            sqrdmulh v19.4s, v19.4s, v7.4s
            sqrdmulh v20.4s, v20.4s, v4.4s
            sqrdmulh v21.4s, v21.4s, v5.4s
            sqrdmulh v22.4s, v22.4s, v6.4s
            sqrdmulh v23.4s, v23.4s, v7.4s
            sqrdmulh v24.4s, v24.4s, v4.4s
            sqrdmulh v25.4s, v25.4s, v5.4s
            sqrdmulh v26.4s, v26.4s, v6.4s
            sqrdmulh v27.4s, v27.4s, v7.4s
            sqrdmulh v28.4s, v28.4s, v4.4s
            sqrdmulh v29.4s, v29.4s, v5.4s
            sqrdmulh v30.4s, v30.4s, v6.4s
            sqrdmulh v31.4s, v31.4s, v7.4s

            and v0.16b, v8.16b, v16.16b
            sshr v0.4s, v0.4s, #31
            sqadd v16.4s, v16.4s, v0.4s
            srshl v16.4s, v16.4s, v8.4s
            and v1.16b, v9.16b, v17.16b
            sshr v1.4s, v1.4s, #31
            sqadd v17.4s, v17.4s, v1.4s
            srshl v17.4s, v17.4s, v9.4s
            and v2.16b, v10.16b, v18.16b
            sshr v2.4s, v2.4s, #31
            sqadd v18.4s, v18.4s, v2.4s
            srshl v18.4s, v18.4s, v10.4s
            and v3.16b, v11.16b, v19.16b
            sshr v3.4s, v3.4s, #31
            sqadd v19.4s, v19.4s, v3.4s
            srshl v19.4s, v19.4s, v11.4s

            and v0.16b, v8.16b, v20.16b
            sshr v0.4s, v0.4s, #31
            sqadd v20.4s, v20.4s, v0.4s
            srshl v20.4s, v20.4s, v8.4s
            and v1.16b, v9.16b, v21.16b
            sshr v1.4s, v1.4s, #31
            sqadd v21.4s, v21.4s, v1.4s
            srshl v21.4s, v21.4s, v9.4s
            and v2.16b, v10.16b, v22.16b
            sshr v2.4s, v2.4s, #31
            sqadd v22.4s, v22.4s, v2.4s
            srshl v22.4s, v22.4s, v10.4s
            and v3.16b, v11.16b, v23.16b
            sshr v3.4s, v3.4s, #31
            sqadd v23.4s, v23.4s, v3.4s
            srshl v23.4s, v23.4s, v11.4s

            and v0.16b, v8.16b, v24.16b
            sshr v0.4s, v0.4s, #31
            sqadd v24.4s, v24.4s, v0.4s
            srshl v24.4s, v24.4s, v8.4s
            and v1.16b, v9.16b, v25.16b
            sshr v1.4s, v1.4s, #31
            sqadd v25.4s, v25.4s, v1.4s
            srshl v25.4s, v25.4s, v9.4s
            and v2.16b, v10.16b, v26.16b
            sshr v2.4s, v2.4s, #31
            sqadd v26.4s, v26.4s, v2.4s
            srshl v26.4s, v26.4s, v10.4s
            and v3.16b, v11.16b, v27.16b
            sshr v3.4s, v3.4s, #31
            sqadd v27.4s, v27.4s, v3.4s
            srshl v27.4s, v27.4s, v11.4s

            and v0.16b, v8.16b, v28.16b
            sshr v0.4s, v0.4s, #31
            sqadd v28.4s, v28.4s, v0.4s
            srshl v28.4s, v28.4s, v8.4s
            and v1.16b, v9.16b, v29.16b
            sshr v1.4s, v1.4s, #31
            sqadd v29.4s, v29.4s, v1.4s
            srshl v29.4s, v29.4s, v9.4s
            and v2.16b, v10.16b, v30.16b
            sshr v2.4s, v2.4s, #31
            sqadd v30.4s, v30.4s, v2.4s
            srshl v30.4s, v30.4s, v10.4s
            and v3.16b, v11.16b, v31.16b
            sshr v3.4s, v3.4s, #31
            sqadd v31.4s, v31.4s, v3.4s
            srshl v31.4s, v31.4s, v11.4s

            // zp
            dup v6.4s, w10
            add v16.4s, v16.4s, v6.4s
            add v17.4s, v17.4s, v6.4s
            add v18.4s, v18.4s, v6.4s
            add v19.4s, v19.4s, v6.4s
            add v20.4s, v20.4s, v6.4s
            add v21.4s, v21.4s, v6.4s
            add v22.4s, v22.4s, v6.4s
            add v23.4s, v23.4s, v6.4s
            add v24.4s, v24.4s, v6.4s
            add v25.4s, v25.4s, v6.4s
            add v26.4s, v26.4s, v6.4s
            add v27.4s, v27.4s, v6.4s
            add v28.4s, v28.4s, v6.4s
            add v29.4s, v29.4s, v6.4s
            add v30.4s, v30.4s, v6.4s
            add v31.4s, v31.4s, v6.4s

            // min
            dup v0.4s, w8
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

            // max
            dup v1.4s, w9
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

            sqxtn v16.4h, v16.4s
            sqxtn2 v16.8h, v17.4s
            sqxtn v0.8b, v16.8h
            sqxtn v18.4h, v18.4s
            sqxtn2 v18.8h, v19.4s
            sqxtn2 v0.16b, v18.8h

            sqxtn v20.4h, v20.4s
            sqxtn2 v20.8h, v21.4s
            sqxtn v1.8b, v20.8h
            sqxtn v22.4h, v22.4s
            sqxtn2 v22.8h, v23.4s
            sqxtn2 v1.16b, v22.8h

            sqxtn v24.4h, v24.4s
            sqxtn2 v24.8h, v25.4s
            sqxtn v2.8b, v24.8h
            sqxtn v26.4h, v26.4s
            sqxtn2 v26.8h, v27.4s
            sqxtn2 v2.16b, v26.8h

            sqxtn v28.4h, v28.4s
            sqxtn2 v28.8h, v29.4s
            sqxtn v3.8b, v28.8h
            sqxtn v30.4h, v30.4s
            sqxtn2 v30.8h, v31.4s
            sqxtn2 v3.16b, v30.8h

            b WriteStart

        LoopDepthHalf:
            ld1 {v0.16b}, [x20], #16
            ld1 {v1.16b, v2.16b}, [x16]
            add x16, x16, #64
            sdot v16.4s, v1.16b, v0.4b[0]
            sdot v17.4s, v2.16b, v0.4b[0]
            sdot v20.4s, v1.16b, v0.4b[1]
            sdot v21.4s, v2.16b, v0.4b[1]
            sdot v24.4s, v1.16b, v0.4b[2]
            sdot v25.4s, v2.16b, v0.4b[2]
            sdot v28.4s, v1.16b, v0.4b[3]
            sdot v29.4s, v2.16b, v0.4b[3]

            subs x21, x21, #4
            bgt LoopDepthHalf

        BiasHalf:
            cbz x7, NoReadBiasHalf
            ld1 {v0.4s, v1.4s}, [x18]
            add x18, x18, #64
            add v16.4s, v16.4s, v0.4s
            add v17.4s, v17.4s, v1.4s
            add v20.4s, v20.4s, v0.4s
            add v21.4s, v21.4s, v1.4s
            add v24.4s, v24.4s, v0.4s
            add v25.4s, v25.4s, v1.4s
            add v28.4s, v28.4s, v0.4s
            add v29.4s, v29.4s, v1.4s

        NoReadBiasHalf:
            ld1r {v12.4s}, [x25], #4
            ld1r {v13.4s}, [x25], #4
            ld1r {v14.4s}, [x25], #4
            ld1r {v15.4s}, [x25], #4
            cbnz x15, PerChannelSumHalf

        PerTensorSumHalf:
            sub v16.4s, v16.4s, v12.4s
            sub v17.4s, v17.4s, v12.4s
            sub v20.4s, v20.4s, v13.4s
            sub v21.4s, v21.4s, v13.4s
            sub v24.4s, v24.4s, v14.4s
            sub v25.4s, v25.4s, v14.4s
            sub v28.4s, v28.4s, v15.4s
            sub v29.4s, v29.4s, v15.4s

            b PerTensorHalf

        PerChannelSumHalf:
            ld1 {v8.4s, v9.4s}, [x28]
            add x28, x28, #64
            mul v0.4s, v8.4s, v12.4s
            mul v1.4s, v9.4s, v12.4s
            mul v4.4s, v8.4s, v13.4s
            mul v5.4s, v9.4s, v13.4s
            sub v16.4s, v16.4s, v0.4s
            sub v17.4s, v17.4s, v1.4s
            sub v20.4s, v20.4s, v4.4s
            sub v21.4s, v21.4s, v5.4s
            mul v2.4s, v8.4s, v14.4s
            mul v3.4s, v9.4s, v14.4s
            mul v6.4s, v8.4s, v15.4s
            mul v7.4s, v9.4s, v15.4s
            sub v24.4s, v24.4s, v2.4s
            sub v25.4s, v25.4s, v3.4s
            sub v28.4s, v28.4s, v6.4s
            sub v29.4s, v29.4s, v7.4s

        PerTensorHalf:
            cbnz x15, PerChannelHalf
            ld1r {v0.4s}, [x12]
            mov v1.16b, v0.16b
            ld1r {v4.4s}, [x11]
            mov v5.16b, v4.16b
            ld1r {v8.4s}, [x13]
            mov v9.16b, v8.16b

            b QuantizationHalf

        PerChannelHalf:
            ld1 {v0.4s, v1.4s}, [x12]
            add x12, x12, #64
            ld1 {v4.4s, v5.4s}, [x11]
            add x11, x11, #64
            ld1 {v8.4s, v9.4s}, [x13]
            add x13, x13, #64

        QuantizationHalf:
            sqshl v16.4s, v16.4s, v0.4s
            sqshl v17.4s, v17.4s, v1.4s
            sqshl v20.4s, v20.4s, v0.4s
            sqshl v21.4s, v21.4s, v1.4s
            sqshl v24.4s, v24.4s, v0.4s
            sqshl v25.4s, v25.4s, v1.4s
            sqshl v28.4s, v28.4s, v0.4s
            sqshl v29.4s, v29.4s, v1.4s

            sqrdmulh v16.4s, v16.4s, v4.4s
            sqrdmulh v17.4s, v17.4s, v5.4s
            sqrdmulh v20.4s, v20.4s, v4.4s
            sqrdmulh v21.4s, v21.4s, v5.4s
            sqrdmulh v24.4s, v24.4s, v4.4s
            sqrdmulh v25.4s, v25.4s, v5.4s
            sqrdmulh v28.4s, v28.4s, v4.4s
            sqrdmulh v29.4s, v29.4s, v5.4s

            and v0.16b, v8.16b, v16.16b
            sshr v0.4s, v0.4s, #31
            sqadd v16.4s, v16.4s, v0.4s
            srshl v16.4s, v16.4s, v8.4s
            and v1.16b, v9.16b, v17.16b
            sshr v1.4s, v1.4s, #31
            sqadd v17.4s, v17.4s, v1.4s
            srshl v17.4s, v17.4s, v9.4s

            and v0.16b, v8.16b, v20.16b
            sshr v0.4s, v0.4s, #31
            sqadd v20.4s, v20.4s, v0.4s
            srshl v20.4s, v20.4s, v8.4s
            and v1.16b, v9.16b, v21.16b
            sshr v1.4s, v1.4s, #31
            sqadd v21.4s, v21.4s, v1.4s
            srshl v21.4s, v21.4s, v9.4s

            and v0.16b, v8.16b, v24.16b
            sshr v0.4s, v0.4s, #31
            sqadd v24.4s, v24.4s, v0.4s
            srshl v24.4s, v24.4s, v8.4s
            and v1.16b, v9.16b, v25.16b
            sshr v1.4s, v1.4s, #31
            sqadd v25.4s, v25.4s, v1.4s
            srshl v25.4s, v25.4s, v9.4s

            and v0.16b, v8.16b, v28.16b
            sshr v0.4s, v0.4s, #31
            sqadd v28.4s, v28.4s, v0.4s
            srshl v28.4s, v28.4s, v8.4s
            and v1.16b, v9.16b, v29.16b
            sshr v1.4s, v1.4s, #31
            sqadd v29.4s, v29.4s, v1.4s
            srshl v29.4s, v29.4s, v9.4s

            // zp
            dup v6.4s, w10
            add v16.4s, v16.4s, v6.4s
            add v17.4s, v17.4s, v6.4s
            add v20.4s, v20.4s, v6.4s
            add v21.4s, v21.4s, v6.4s
            add v24.4s, v24.4s, v6.4s
            add v25.4s, v25.4s, v6.4s
            add v28.4s, v28.4s, v6.4s
            add v29.4s, v29.4s, v6.4s

            // min
            dup v0.4s, w8
            smax v16.4s, v16.4s, v0.4s
            smax v17.4s, v17.4s, v0.4s
            smax v20.4s, v20.4s, v0.4s
            smax v21.4s, v21.4s, v0.4s
            smax v24.4s, v24.4s, v0.4s
            smax v25.4s, v25.4s, v0.4s
            smax v28.4s, v28.4s, v0.4s
            smax v29.4s, v29.4s, v0.4s

            // max
            dup v1.4s, w9
            smin v16.4s, v16.4s, v1.4s
            smin v17.4s, v17.4s, v1.4s
            smin v20.4s, v20.4s, v1.4s
            smin v21.4s, v21.4s, v1.4s
            smin v24.4s, v24.4s, v1.4s
            smin v25.4s, v25.4s, v1.4s
            smin v28.4s, v28.4s, v1.4s
            smin v29.4s, v29.4s, v1.4s

            sqxtn v16.4h, v16.4s
            sqxtn2 v16.8h, v17.4s
            sqxtn v0.8b, v16.8h

            sqxtn v20.4h, v20.4s
            sqxtn2 v20.8h, v21.4s
            sqxtn v1.8b, v20.8h

            sqxtn v24.4h, v24.4s
            sqxtn2 v24.8h, v25.4s
            sqxtn v2.8b, v24.8h

            sqxtn v28.4h, v28.4s
            sqxtn2 v28.8h, v29.4s
            sqxtn v3.8b, v28.8h

            b WriteStart

        LoopDepthQuarter:
            ld1 {v0.16b}, [x20], #16
            ld1 {v1.16b}, [x16]
            add x16, x16, #64
            sdot v16.4s, v1.16b, v0.4b[0]
            sdot v20.4s, v1.16b, v0.4b[1]
            sdot v24.4s, v1.16b, v0.4b[2]
            sdot v28.4s, v1.16b, v0.4b[3]

            subs x21, x21, #4
            bgt LoopDepthQuarter

        BiasQuarter:
            cbz x7, NoReadBiasQuarter
            ld1 {v0.4s}, [x18]
            add x18, x18, #64
            add v16.4s, v16.4s, v0.4s
            add v20.4s, v20.4s, v0.4s
            add v24.4s, v24.4s, v0.4s
            add v28.4s, v28.4s, v0.4s

        NoReadBiasQuarter:
            ld1r {v12.4s}, [x25], #4
            ld1r {v13.4s}, [x25], #4
            ld1r {v14.4s}, [x25], #4
            ld1r {v15.4s}, [x25], #4
            cbnz x15, PerChannelSumQuarter

        PerTensorSumQuarter:
            sub v16.4s, v16.4s, v12.4s
            sub v20.4s, v20.4s, v13.4s
            sub v24.4s, v24.4s, v14.4s
            sub v28.4s, v28.4s, v15.4s

            b PerTensorQuarter

        PerChannelSumQuarter:
            ld1 {v8.4s}, [x28]
            add x28, x28, #64
            mul v0.4s, v8.4s, v12.4s
            mul v4.4s, v8.4s, v13.4s
            sub v16.4s, v16.4s, v0.4s
            sub v20.4s, v20.4s, v4.4s
            mul v2.4s, v8.4s, v14.4s
            mul v6.4s, v8.4s, v15.4s
            sub v24.4s, v24.4s, v2.4s
            sub v28.4s, v28.4s, v6.4s

        PerTensorQuarter:
            cbnz x15, PerChannelQuarter
            ld1r {v0.4s}, [x12]
            ld1r {v4.4s}, [x11]
            ld1r {v8.4s}, [x13]

            b QuantizationHalf

        PerChannelQuarter:
            ld1 {v0.4s}, [x12]
            add x12, x12, #64
            ld1 {v4.4s}, [x11]
            add x11, x11, #64
            ld1 {v8.4s}, [x13]
            add x13, x13, #64

        QuantizationQuarter:
            sqshl v16.4s, v16.4s, v0.4s
            sqshl v20.4s, v20.4s, v0.4s
            sqshl v24.4s, v24.4s, v0.4s
            sqshl v28.4s, v28.4s, v0.4s

            sqrdmulh v16.4s, v16.4s, v4.4s
            sqrdmulh v20.4s, v20.4s, v4.4s
            sqrdmulh v24.4s, v24.4s, v4.4s
            sqrdmulh v28.4s, v28.4s, v4.4s

            and v0.16b, v8.16b, v16.16b
            sshr v0.4s, v0.4s, #31
            sqadd v16.4s, v16.4s, v0.4s
            srshl v16.4s, v16.4s, v8.4s

            and v0.16b, v8.16b, v20.16b
            sshr v0.4s, v0.4s, #31
            sqadd v20.4s, v20.4s, v0.4s
            srshl v20.4s, v20.4s, v8.4s

            and v0.16b, v8.16b, v24.16b
            sshr v0.4s, v0.4s, #31
            sqadd v24.4s, v24.4s, v0.4s
            srshl v24.4s, v24.4s, v8.4s

            and v0.16b, v8.16b, v28.16b
            sshr v0.4s, v0.4s, #31
            sqadd v28.4s, v28.4s, v0.4s
            srshl v28.4s, v28.4s, v8.4s

            // zp
            dup v6.4s, w10
            add v16.4s, v16.4s, v6.4s
            add v20.4s, v20.4s, v6.4s
            add v24.4s, v24.4s, v6.4s
            add v28.4s, v28.4s, v6.4s

            // min
            dup v0.4s, w8
            smax v16.4s, v16.4s, v0.4s
            smax v20.4s, v20.4s, v0.4s
            smax v24.4s, v24.4s, v0.4s
            smax v28.4s, v28.4s, v0.4s

            // max
            dup v1.4s, w9
            smin v16.4s, v16.4s, v1.4s
            smin v20.4s, v20.4s, v1.4s
            smin v24.4s, v24.4s, v1.4s
            smin v28.4s, v28.4s, v1.4s

            sqxtn v16.4h, v16.4s
            sqxtn v0.8b, v16.8h

            sqxtn v20.4h, v20.4s
            sqxtn v1.8b, v20.8h

            sqxtn v24.4h, v24.4s
            sqxtn v2.8b, v24.8h

            sqxtn v28.4h, v28.4s
            sqxtn v3.8b, v28.8h

            b WriteStart

        WriteStart:
            cmp x17, #1
            beq Write1
            cmp x17, #2
            beq Write2
            cmp x17, #3
            beq Write3
            cmp x17, #4
            beq Write4
            cmp x17, #5
            beq Write5
            cmp x17, #6
            beq Write6
            cmp x17, #7
            beq Write7
            cmp x17, #8
            beq Write8
            cmp x17, #9
            beq Write9
            cmp x17, #10
            beq Write10
            cmp x17, #11
            beq Write11
            cmp x17, #12
            beq Write12
            cmp x17, #13
            beq Write13
            cmp x17, #14
            beq Write14
            cmp x17, #15
            beq Write15
            b Write16

        Write1:
            add x27, x27, #1
            st1 {v0.b}[0], [x19], x14
            cmp x3, #1
            beq WriteEnd
            st1 {v1.b}[0], [x19], x14
            cmp x3, #2
            beq WriteEnd
            st1 {v2.b}[0], [x19], x14
            cmp x3, #3
            beq WriteEnd
            st1 {v3.b}[0], [x19], x14
            b WriteEnd
        Write2:
            add x27, x27, #2
            st1 {v0.h}[0], [x19], x14
            cmp x3, #1
            beq WriteEnd
            st1 {v1.h}[0], [x19], x14
            cmp x3, #2
            beq WriteEnd
            st1 {v2.h}[0], [x19], x14
            cmp x3, #3
            beq WriteEnd
            st1 {v3.h}[0], [x19], x14
            b WriteEnd
        Write3:
            add x27, x27, #3
            add x22, x19, #2
            st1 {v0.h}[0], [x19], x14
            st1 {v0.b}[2], [x22], x14
            cmp x3, #1
            beq WriteEnd
            st1 {v1.h}[0], [x19], x14
            st1 {v1.b}[2], [x22], x14
            cmp x3, #2
            beq WriteEnd
            st1 {v2.h}[0], [x19], x14
            st1 {v2.b}[2], [x22], x14
            cmp x3, #3
            beq WriteEnd
            st1 {v3.h}[0], [x19], x14
            st1 {v3.b}[2], [x22], x14
            b WriteEnd
        Write4:
            add x27, x27, #4
            st1 {v0.s}[0], [x19], x14
            cmp x3, #1
            beq WriteEnd
            st1 {v1.s}[0], [x19], x14
            cmp x3, #2
            beq WriteEnd
            st1 {v2.s}[0], [x19], x14
            cmp x3, #3
            beq WriteEnd
            st1 {v3.s}[0], [x19], x14
            b WriteEnd
        Write5:
            add x27, x27, #5
            add x22, x19, #4
            st1 {v0.s}[0], [x19], x14
            st1 {v0.b}[4], [x22], x14
            cmp x3, #1
            beq WriteEnd
            st1 {v1.s}[0], [x19], x14
            st1 {v1.b}[4], [x22], x14
            cmp x3, #2
            beq WriteEnd
            st1 {v2.s}[0], [x19], x14
            st1 {v2.b}[4], [x22], x14
            cmp x3, #3
            beq WriteEnd
            st1 {v3.s}[0], [x19], x14
            st1 {v3.b}[4], [x22], x14
            b WriteEnd
        Write6:
            add x27, x27, #6
            add x22, x19, #4
            st1 {v0.s}[0], [x19], x14
            st1 {v0.h}[2], [x22], x14
            cmp x3, #1
            beq WriteEnd
            st1 {v1.s}[0], [x19], x14
            st1 {v1.h}[2], [x22], x14
            cmp x3, #2
            beq WriteEnd
            st1 {v2.s}[0], [x19], x14
            st1 {v2.h}[2], [x22], x14
            cmp x3, #3
            beq WriteEnd
            st1 {v3.s}[0], [x19], x14
            st1 {v3.h}[2], [x22], x14
            b WriteEnd
        Write7:
            add x27, x27, #7
            add x22, x19, #4
            add x26, x19, #6
            st1 {v0.s}[0], [x19], x14
            st1 {v0.h}[2], [x22], x14
            st1 {v0.b}[6], [x26], x14
            cmp x3, #1
            beq WriteEnd
            st1 {v1.s}[0], [x19], x14
            st1 {v1.h}[2], [x22], x14
            st1 {v1.b}[6], [x26], x14
            cmp x3, #2
            beq WriteEnd
            st1 {v2.s}[0], [x19], x14
            st1 {v2.h}[2], [x22], x14
            st1 {v2.b}[6], [x26], x14
            cmp x3, #3
            beq WriteEnd
            st1 {v3.s}[0], [x19], x14
            st1 {v3.h}[2], [x22], x14
            st1 {v3.b}[6], [x26], x14
            b WriteEnd
        Write8:
            add x27, x27, #8
            st1 {v0.8b}, [x19], x14
            cmp x3, #1
            beq WriteEnd
            st1 {v1.8b}, [x19], x14
            cmp x3, #2
            beq WriteEnd
            st1 {v2.8b}, [x19], x14
            cmp x3, #3
            beq WriteEnd
            st1 {v3.8b}, [x19], x14
            b WriteEnd
        Write9:
            add x27, x27, #9
            add x22, x19, #8
            st1 {v0.8b}, [x19], x14
            st1 {v0.b}[8], [x22], x14
            cmp x3, #1
            beq WriteEnd
            st1 {v1.8b}, [x19], x14
            st1 {v1.b}[8], [x22], x14
            cmp x3, #2
            beq WriteEnd
            st1 {v2.8b}, [x19], x14
            st1 {v2.b}[8], [x22], x14
            cmp x3, #3
            beq WriteEnd
            st1 {v3.8b}, [x19], x14
            st1 {v3.b}[8], [x22], x14
            b WriteEnd
        Write10:
            add x27, x27, #10
            add x22, x19, #8
            st1 {v0.8b}, [x19], x14
            st1 {v0.h}[4], [x22], x14
            cmp x3, #1
            beq WriteEnd
            st1 {v1.8b}, [x19], x14
            st1 {v1.h}[4], [x22], x14
            cmp x3, #2
            beq WriteEnd
            st1 {v2.8b}, [x19], x14
            st1 {v2.h}[4], [x22], x14
            cmp x3, #3
            beq WriteEnd
            st1 {v3.8b}, [x19], x14
            st1 {v3.h}[4], [x22], x14
            b WriteEnd
        Write11:
            add x27, x27, #11
            add x22, x19, #8
            add x26, x19, #10
            st1 {v0.8b}, [x19], x14
            st1 {v0.h}[4], [x22], x14
            st1 {v0.b}[10], [x26], x14
            cmp x3, #1
            beq WriteEnd
            st1 {v1.8b}, [x19], x14
            st1 {v1.h}[4], [x22], x14
            st1 {v1.b}[10], [x26], x14
            cmp x3, #2
            beq WriteEnd
            st1 {v2.8b}, [x19], x14
            st1 {v2.h}[4], [x22], x14
            st1 {v2.b}[10], [x26], x14
            cmp x3, #3
            beq WriteEnd
            st1 {v3.8b}, [x19], x14
            st1 {v3.h}[4], [x22], x14
            st1 {v3.b}[10], [x26], x14
            b WriteEnd
        Write12:
            add x27, x27, #12
            add x22, x19, #8
            st1 {v0.8b}, [x19], x14
            st1 {v0.s}[2], [x22], x14
            cmp x3, #1
            beq WriteEnd
            st1 {v1.8b}, [x19], x14
            st1 {v1.s}[2], [x22], x14
            cmp x3, #2
            beq WriteEnd
            st1 {v2.8b}, [x19], x14
            st1 {v2.s}[2], [x22], x14
            cmp x3, #3
            beq WriteEnd
            st1 {v3.8b}, [x19], x14
            st1 {v3.s}[2], [x22], x14
            b WriteEnd
        Write13:
            add x27, x27, #13
            add x22, x19, #8
            add x26, x19, #12
            st1 {v0.8b}, [x19], x14
            st1 {v0.s}[2], [x22], x14
            st1 {v0.b}[12], [x26], x14
            cmp x3, #1
            beq WriteEnd
            st1 {v1.8b}, [x19], x14
            st1 {v1.s}[2], [x22], x14
            st1 {v1.b}[12], [x26], x14
            cmp x3, #2
            beq WriteEnd
            st1 {v2.8b}, [x19], x14
            st1 {v2.s}[2], [x22], x14
            st1 {v2.b}[12], [x26], x14
            cmp x3, #3
            beq WriteEnd
            st1 {v3.8b}, [x19], x14
            st1 {v3.s}[2], [x22], x14
            st1 {v3.b}[12], [x26], x14
            b WriteEnd
        Write14:
            add x27, x27, #14
            add x22, x19, #8
            add x26, x19, #12
            st1 {v0.8b}, [x19], x14
            st1 {v0.s}[2], [x22], x14
            st1 {v0.h}[6], [x26], x14
            cmp x3, #1
            beq WriteEnd
            st1 {v1.8b}, [x19], x14
            st1 {v1.s}[2], [x22], x14
            st1 {v1.h}[6], [x26], x14
            cmp x3, #2
            beq WriteEnd
            st1 {v2.8b}, [x19], x14
            st1 {v2.s}[2], [x22], x14
            st1 {v2.h}[6], [x26], x14
            cmp x3, #3
            beq WriteEnd
            st1 {v3.8b}, [x19], x14
            st1 {v3.s}[2], [x22], x14
            st1 {v3.h}[6], [x26], x14
            b WriteEnd
        Write15:
            add x27, x27, #15
            add x22, x19, #8
            add x26, x19, #12
            add x21, x19, #14
            st1 {v0.8b}, [x19], x14
            st1 {v0.s}[2], [x22], x14
            st1 {v0.h}[6], [x26], x14
            st1 {v0.b}[14], [x21], x14
            cmp x3, #1
            beq WriteEnd
            st1 {v1.8b}, [x19], x14
            st1 {v1.s}[2], [x22], x14
            st1 {v1.h}[6], [x26], x14
            st1 {v1.b}[14], [x21], x14
            cmp x3, #2
            beq WriteEnd
            st1 {v2.8b}, [x19], x14
            st1 {v2.s}[2], [x22], x14
            st1 {v2.h}[6], [x26], x14
            st1 {v2.b}[14], [x21], x14
            cmp x3, #3
            beq WriteEnd
            st1 {v3.8b}, [x19], x14
            st1 {v3.s}[2], [x22], x14
            st1 {v3.h}[6], [x26], x14
            st1 {v3.b}[14], [x21], x14
            b WriteEnd
        Write16:
            add x27, x27, #16
            st1 {v0.16b}, [x19], x14
            cmp x3, #1
            beq WriteEnd
            st1 {v1.16b}, [x19], x14
            cmp x3, #2
            beq WriteEnd
            st1 {v2.16b}, [x19], x14
            cmp x3, #3
            beq WriteEnd
            st1 {v3.16b}, [x19], x14

    WriteEnd:
        subs x17, x17, #16
        ble LoopColEnd
        mov x25, x6
        b LoopCol

LoopColEnd:
    subs x3, x3, #4
    ble LoopRowEnd
    ldr x11, [sp, #24]
    ldr x12, [sp, #32]
    ldr x13, [sp, #40]
    add x6, x6, #16
    add x0, x0, x23
    add x2, x2, x24
    b LoopRow

LoopRowEnd:
  sub sp, sp, #208
  ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
  ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
  ldp x19, x20, [sp], #16
  ldp x21, x22, [sp], #16
  ldp x23, x24, [sp], #16
  ldp x25, x26, [sp], #16
  ldp x27, x28, [sp], #16
  ret
#endif
