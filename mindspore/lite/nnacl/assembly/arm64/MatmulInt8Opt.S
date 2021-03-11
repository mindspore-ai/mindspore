#ifdef __aarch64__
#include "nnacl/assembly_global.h"
    .text
    .align 5
    .global MatmulInt8Opt
#ifndef __APPLE__
    .type MatmulInt8Opt, %function
#endif

//void MatmulInt8Opt(const int8_t *a, const int8_t *b, int8_t *dst, int row, int col, int deep16, const int *a_sums,
//                   const int *bias, int act_min, int act_max, int out_zp, int32_t *multiplier, int32_t *left_shift,
//                   int32_t *right_shift, int stride, int filter_peroc, int32_t *filter_zp)

// x0: a(left matrix ptr)
// x1: b(right matrix ptr)
// x2: out ptr
// x3: row4
// x4: col4
// x5: deep16
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

asm_function MatmulInt8Opt
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
    mul x23, x23, x5  // lhs step
    mov x24, #4
    mul x24, x24, x14 // dst step
LoopRow:
    mov x16, x1 // reload rhs ptr
    mov x17, x4 // reload rhs col
    mov x18, x7 // reload bias ptr
    mov x27, x2 // reload dst ptr
    ldr x28, [sp, #64] // reload filter_zp

    LoopCol:
        mov x25, x6 // reload a_sums ptr
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

        LoopDepth:
            ld1 {v0.16b, v1.16b}, [x20], #32
            ld1 {v4.16b, v5.16b}, [x16], #32
            smull v8.8h, v4.8b, v0.8b
            smull v9.8h, v5.8b, v0.8b
            smull v12.8h, v4.8b, v1.8b
            smull v13.8h, v5.8b, v1.8b
            ld1 {v6.16b, v7.16b}, [x16], #32
            smlal2 v8.8h, v4.16b, v0.16b
            smlal2 v9.8h, v5.16b, v0.16b
            smlal2 v12.8h, v4.16b, v1.16b
            smlal2 v13.8h, v5.16b, v1.16b
            ld1 {v2.16b, v3.16b}, [x20], #32
            smull v10.8h, v6.8b, v0.8b
            smull v11.8h, v7.8b, v0.8b
            smull v14.8h, v6.8b, v1.8b
            smull v15.8h, v7.8b, v1.8b
            smlal2 v10.8h, v6.16b, v0.16b
            smlal2 v11.8h, v7.16b, v0.16b
            smlal2 v14.8h, v6.16b, v1.16b
            smlal2 v15.8h, v7.16b, v1.16b

            sadalp v16.4s, v8.8h
            sadalp v17.4s, v9.8h
            sadalp v18.4s, v10.8h
            sadalp v19.4s, v11.8h
            sadalp v20.4s, v12.8h
            sadalp v21.4s, v13.8h
            sadalp v22.4s, v14.8h
            sadalp v23.4s, v15.8h

            smull v8.8h, v4.8b, v2.8b
            smull v9.8h, v5.8b, v2.8b
            smull v10.8h, v6.8b, v2.8b
            smull v11.8h, v7.8b, v2.8b
            smull v12.8h, v4.8b, v3.8b
            smull v13.8h, v5.8b, v3.8b
            smull v14.8h, v6.8b, v3.8b
            smull v15.8h, v7.8b, v3.8b

            smlal2 v8.8h, v4.16b, v2.16b
            smlal2 v9.8h, v5.16b, v2.16b
            smlal2 v10.8h, v6.16b, v2.16b
            smlal2 v11.8h, v7.16b, v2.16b
            smlal2 v12.8h, v4.16b, v3.16b
            smlal2 v13.8h, v5.16b, v3.16b
            smlal2 v14.8h, v6.16b, v3.16b
            smlal2 v15.8h, v7.16b, v3.16b

            sadalp v24.4s, v8.8h
            sadalp v25.4s, v9.8h
            sadalp v26.4s, v10.8h
            sadalp v27.4s, v11.8h
            sadalp v28.4s, v12.8h
            sadalp v29.4s, v13.8h
            sadalp v30.4s, v14.8h
            sadalp v31.4s, v15.8h
            subs x21, x21, #16  // depth - 16
            bgt LoopDepth

        addp v16.4s, v16.4s, v17.4s
        addp v18.4s, v18.4s, v19.4s
        addp v20.4s, v20.4s, v21.4s
        addp v22.4s, v22.4s, v23.4s
        addp v24.4s, v24.4s, v25.4s
        addp v26.4s, v26.4s, v27.4s
        addp v28.4s, v28.4s, v29.4s
        addp v30.4s, v30.4s, v31.4s

        addp v16.4s, v16.4s, v18.4s
        addp v17.4s, v20.4s, v22.4s
        addp v18.4s, v24.4s, v26.4s
        addp v19.4s, v28.4s, v30.4s

        Bias:
            cbz x7, NoBias
            ld1 {v15.4s}, [x18], #16
            add v16.4s, v16.4s, v15.4s
            add v17.4s, v17.4s, v15.4s
            add v18.4s, v18.4s, v15.4s
            add v19.4s, v19.4s, v15.4s

        NoBias:
            ld1r {v20.4s}, [x25], #4
            ld1r {v21.4s}, [x25], #4
            ld1r {v22.4s}, [x25], #4
            ld1r {v23.4s}, [x25], #4
            cbz x15, ApplySum

            ld1 {v14.4s}, [x28], #16
            mul v20.4s, v20.4s, v14.4s
            mul v21.4s, v21.4s, v14.4s
            mul v22.4s, v22.4s, v14.4s
            mul v23.4s, v23.4s, v14.4s

        ApplySum:
            sub v16.4s, v16.4s, v20.4s
            sub v17.4s, v17.4s, v21.4s
            sub v18.4s, v18.4s, v22.4s
            sub v19.4s, v19.4s, v23.4s

        cbnz x15, PerCLoad

        ld1r {v13.4s}, [x12]
        ld1r {v12.4s}, [x11]
        ld1r {v11.4s}, [x13]
        b Quantize

    PerCLoad:
        ld1 {v13.4s}, [x12], #16
        ld1 {v12.4s}, [x11], #16
        ld1 {v11.4s}, [x13], #16

    Quantize:
        sqshl v16.4s, v16.4s, v13.4s
        sqshl v17.4s, v17.4s, v13.4s
        sqshl v18.4s, v18.4s, v13.4s
        sqshl v19.4s, v19.4s, v13.4s

        sqrdmulh v16.4s, v16.4s, v12.4s
        sqrdmulh v17.4s, v17.4s, v12.4s
        sqrdmulh v18.4s, v18.4s, v12.4s
        sqrdmulh v19.4s, v19.4s, v12.4s

        and v20.16b, v11.16b, v16.16b
        sshr v20.4s, v20.4s, #31
        sqadd v16.4s, v16.4s, v20.4s
        srshl v16.4s, v16.4s, v11.4s
        and v21.16b, v11.16b, v17.16b
        sshr v21.4s, v21.4s, #31
        sqadd v17.4s, v17.4s, v21.4s
        srshl v17.4s, v17.4s, v11.4s
        and v22.16b, v11.16b, v18.16b
        sshr v22.4s, v22.4s, #31
        sqadd v18.4s, v18.4s, v22.4s
        srshl v18.4s, v18.4s, v11.4s
        and v23.16b, v11.16b, v19.16b
        sshr v23.4s, v23.4s, #31
        sqadd v19.4s, v19.4s, v23.4s
        srshl v19.4s, v19.4s, v11.4s

        dup v10.4s, w10
        add v16.4s, v16.4s, v10.4s
        add v17.4s, v17.4s, v10.4s
        add v18.4s, v18.4s, v10.4s
        add v19.4s, v19.4s, v10.4s

        dup v9.4s, w8
        smax v16.4s, v16.4s, v9.4s
        smax v17.4s, v17.4s, v9.4s
        smax v18.4s, v18.4s, v9.4s
        smax v19.4s, v19.4s, v9.4s

        dup v8.4s, w9
        smin v16.4s, v16.4s, v8.4s
        smin v17.4s, v17.4s, v8.4s
        smin v18.4s, v18.4s, v8.4s
        smin v19.4s, v19.4s, v8.4s

        sqxtn v13.4h, v16.4s
        sqxtn2 v13.8h, v17.4s
        sqxtn v14.4h, v18.4s
        sqxtn2 v14.8h, v19.4s

        sqxtn v15.8b, v13.8h
        sqxtn2 v15.16b, v14.8h

        cmp x17, #1
        beq Write1
        cmp x17, #2
        beq Write2
        cmp x17, #3
        beq Write3
        b Write4

    Write1:
        add x27, x27, #1
        st1 {v15.b}[0], [x19], x14
        cmp x3, #1
        beq WriteEnd
        st1 {v15.b}[4], [x19], x14
        cmp x3, #2
        beq WriteEnd
        st1 {v15.b}[8], [x19], x14
        cmp x3, #3
        beq WriteEnd
        st1 {v15.b}[12], [x19], x14
        b WriteEnd
    Write2:
        add x27, x27, #2
        st1 {v15.h}[0], [x19], x14
        cmp x3, #1
        beq WriteEnd
        st1 {v15.h}[2], [x19], x14
        cmp x3, #2
        beq WriteEnd
        st1 {v15.h}[4], [x19], x14
        cmp x3, #3
        beq WriteEnd
        st1 {v15.h}[6], [x19], x14
        b WriteEnd
    Write3:
        add x27, x27, #3
        add x22, x19, #2
        st1 {v15.h}[0], [x19], x14
        st1 {v15.b}[2], [x22], x14
        cmp x3, #1
        beq WriteEnd
        st1 {v15.h}[2], [x19], x14
        st1 {v15.b}[6], [x22], x14
        cmp x3, #2
        beq WriteEnd
        st1 {v15.h}[4], [x19], x14
        st1 {v15.b}[10], [x22], x14
        cmp x3, #3
        beq WriteEnd
        st1 {v15.h}[6], [x19], x14
        st1 {v15.b}[14], [x22], x14
        b WriteEnd
    Write4:
        add x27, x27, #4
        st1 {v15.s}[0], [x19], x14
        cmp x3, #1
        beq WriteEnd
        st1 {v15.s}[1], [x19], x14
        cmp x3, #2
        beq WriteEnd
        st1 {v15.s}[2], [x19], x14
        cmp x3, #3
        beq WriteEnd
        st1 {v15.s}[3], [x19], x14

    WriteEnd:
        subs x17, x17, #4
        bgt LoopCol

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
