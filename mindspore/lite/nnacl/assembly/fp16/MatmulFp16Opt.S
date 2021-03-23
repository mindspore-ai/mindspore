#ifdef __aarch64__
#include "nnacl/assembly_global.h"
    .text
    .align 5
    .global MatmulFp16Neon64Opt
#ifndef __APPLE__
    .type MatmulFp16Neon64Opt, %function
#endif

// void MatmulFp16Neon64Opt(const float16_t *a, const float16_t *b, float16_t *c, const float16_t *bias, int act_type,
//                          int depth, int row, int col, size_t stride, size_t writeMode)
// x0: a
// x1: b
// x2: c
// x3: bias
// x4: act_type
// x5: depth
// x6: row
// x7: col
// x8: stride
// x9: writeMode

asm_function MatmulFp16Neon64Opt
    sub sp, sp, #80
    st1 {v8.8h, v9.8h, v10.8h, v11.8h}, [sp], #64
    stp x19, x20, [sp], #16

    ldr x8, [sp]
    ldr x9, [sp, #8]

    mov x18, #32 // sizeof(float16_t) * 16
    mul x17, x5, x18 // block stride of lhs/rhs: sizeof(float16_t) * 16 * depth
    cbnz x9, NoC8Steps
    mov x11, x2
    mov x18, #16
    mul x16, x6, x18 // row * 8 * sizeof(float16_t)
NoC8Steps:
    cmp x9, #2
    bne NoWinoSteps
    mov x18, #2
    mul x15, x7, x8
    mul x15, x15, x18 // kernel_size * col *sizeof(float16_t)
    mov x18, #16
    mul x16, x8, x18 // kernel_size * 8 * sizeof(float16_t)
NoWinoSteps:
    mov x18, #2
    mul x8, x8, x18

LoopRowStart:
    cmp x6, #1
    ble LoopRow
    cmp x6, #2
    ble LoopRow2
    cmp x6, #4
    ble LoopRow4
    cmp x6, #8
    ble LoopRow8

LoopRow16:
    mov x14, x1 // reload rhs ptr
    mov x13, x7 // reload rhs col
    mov x12, x3 // reload bias

    LoopCol16:
        cbz x9, NoReloadDst16
        mov x11, x2
    NoReloadDst16:
        mov x10, x0 // reload lhs ptr
        mov x19, x5 // reload depth

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

        cmp x19, #4
        blt LoopDepth16One

    LoopDepth16:
        ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x10], #64
        ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x14], #64
        fmla v16.8h, v8.8h, v0.h[0]
        fmla v17.8h, v8.8h, v0.h[1]
        fmla v18.8h, v8.8h, v0.h[2]
        fmla v19.8h, v8.8h, v0.h[3]
        fmla v20.8h, v8.8h, v0.h[4]
        fmla v21.8h, v8.8h, v0.h[5]
        fmla v22.8h, v8.8h, v0.h[6]
        fmla v23.8h, v8.8h, v0.h[7]
        fmla v24.8h, v8.8h, v1.h[0]
        fmla v25.8h, v8.8h, v1.h[1]
        fmla v26.8h, v8.8h, v1.h[2]
        fmla v27.8h, v8.8h, v1.h[3]
        fmla v28.8h, v8.8h, v1.h[4]
        fmla v29.8h, v8.8h, v1.h[5]
        fmla v30.8h, v8.8h, v1.h[6]
        fmla v31.8h, v8.8h, v1.h[7]
        ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x10], #64
        fmla v16.8h, v9.8h, v2.h[0]
        fmla v17.8h, v9.8h, v2.h[1]
        fmla v18.8h, v9.8h, v2.h[2]
        fmla v19.8h, v9.8h, v2.h[3]
        fmla v20.8h, v9.8h, v2.h[4]
        fmla v21.8h, v9.8h, v2.h[5]
        fmla v22.8h, v9.8h, v2.h[6]
        fmla v23.8h, v9.8h, v2.h[7]
        fmla v24.8h, v9.8h, v3.h[0]
        fmla v25.8h, v9.8h, v3.h[1]
        fmla v26.8h, v9.8h, v3.h[2]
        fmla v27.8h, v9.8h, v3.h[3]
        fmla v28.8h, v9.8h, v3.h[4]
        fmla v29.8h, v9.8h, v3.h[5]
        fmla v30.8h, v9.8h, v3.h[6]
        fmla v31.8h, v9.8h, v3.h[7]
        fmla v16.8h, v10.8h, v4.h[0]
        fmla v17.8h, v10.8h, v4.h[1]
        fmla v18.8h, v10.8h, v4.h[2]
        fmla v19.8h, v10.8h, v4.h[3]
        fmla v20.8h, v10.8h, v4.h[4]
        fmla v21.8h, v10.8h, v4.h[5]
        fmla v22.8h, v10.8h, v4.h[6]
        fmla v23.8h, v10.8h, v4.h[7]
        fmla v24.8h, v10.8h, v5.h[0]
        fmla v25.8h, v10.8h, v5.h[1]
        fmla v26.8h, v10.8h, v5.h[2]
        fmla v27.8h, v10.8h, v5.h[3]
        fmla v28.8h, v10.8h, v5.h[4]
        fmla v29.8h, v10.8h, v5.h[5]
        fmla v30.8h, v10.8h, v5.h[6]
        fmla v31.8h, v10.8h, v5.h[7]
        fmla v16.8h, v11.8h, v6.h[0]
        fmla v17.8h, v11.8h, v6.h[1]
        fmla v18.8h, v11.8h, v6.h[2]
        fmla v19.8h, v11.8h, v6.h[3]
        fmla v20.8h, v11.8h, v6.h[4]
        fmla v21.8h, v11.8h, v6.h[5]
        fmla v22.8h, v11.8h, v6.h[6]
        fmla v23.8h, v11.8h, v6.h[7]
        fmla v24.8h, v11.8h, v7.h[0]
        fmla v25.8h, v11.8h, v7.h[1]
        fmla v26.8h, v11.8h, v7.h[2]
        fmla v27.8h, v11.8h, v7.h[3]
        fmla v28.8h, v11.8h, v7.h[4]
        fmla v29.8h, v11.8h, v7.h[5]
        fmla v30.8h, v11.8h, v7.h[6]
        fmla v31.8h, v11.8h, v7.h[7]

        subs x19, x19, #4
        beq Bias16
        cmp x19, #4
        bge LoopDepth16

        LoopDepth16One:
            ld1 {v0.8h, v1.8h}, [x10], #32
            ld1 {v2.8h}, [x14], #16
            fmla v16.8h, v2.8h, v0.h[0]
            fmla v17.8h, v2.8h, v0.h[1]
            fmla v18.8h, v2.8h, v0.h[2]
            fmla v19.8h, v2.8h, v0.h[3]
            fmla v20.8h, v2.8h, v0.h[4]
            fmla v21.8h, v2.8h, v0.h[5]
            fmla v22.8h, v2.8h, v0.h[6]
            fmla v23.8h, v2.8h, v0.h[7]
            fmla v24.8h, v2.8h, v1.h[0]
            fmla v25.8h, v2.8h, v1.h[1]
            fmla v26.8h, v2.8h, v1.h[2]
            fmla v27.8h, v2.8h, v1.h[3]
            fmla v28.8h, v2.8h, v1.h[4]
            fmla v29.8h, v2.8h, v1.h[5]
            fmla v30.8h, v2.8h, v1.h[6]
            fmla v31.8h, v2.8h, v1.h[7]

            subs x19, x19, #1
            bgt LoopDepth16One

        Bias16:
            cbz x3, Activation16
            ld1 {v0.8h}, [x12], #16
            fadd v16.8h, v16.8h, v0.8h
            fadd v17.8h, v17.8h, v0.8h
            fadd v18.8h, v18.8h, v0.8h
            fadd v19.8h, v19.8h, v0.8h
            fadd v20.8h, v20.8h, v0.8h
            fadd v21.8h, v21.8h, v0.8h
            fadd v22.8h, v22.8h, v0.8h
            fadd v23.8h, v23.8h, v0.8h
            fadd v24.8h, v24.8h, v0.8h
            fadd v25.8h, v25.8h, v0.8h
            fadd v26.8h, v26.8h, v0.8h
            fadd v27.8h, v27.8h, v0.8h
            fadd v28.8h, v28.8h, v0.8h
            fadd v29.8h, v29.8h, v0.8h
            fadd v30.8h, v30.8h, v0.8h
            fadd v31.8h, v31.8h, v0.8h

        Activation16:
            cmp x4, #3
            beq Relu616
            cmp x4, #1
            beq Relu16
            b Write

        Relu616:
            movi v2.8h, #0x46, lsl #8
            fmin v16.8h, v16.8h, v2.8h
            fmin v17.8h, v17.8h, v2.8h
            fmin v18.8h, v18.8h, v2.8h
            fmin v19.8h, v19.8h, v2.8h
            fmin v20.8h, v20.8h, v2.8h
            fmin v21.8h, v21.8h, v2.8h
            fmin v22.8h, v22.8h, v2.8h
            fmin v23.8h, v23.8h, v2.8h
            fmin v24.8h, v24.8h, v2.8h
            fmin v25.8h, v25.8h, v2.8h
            fmin v26.8h, v26.8h, v2.8h
            fmin v27.8h, v27.8h, v2.8h
            fmin v28.8h, v28.8h, v2.8h
            fmin v29.8h, v29.8h, v2.8h
            fmin v30.8h, v30.8h, v2.8h
            fmin v31.8h, v31.8h, v2.8h
        
        Relu16:
            dup v2.8h, wzr
            fmax v16.8h, v16.8h, v2.8h
            fmax v17.8h, v17.8h, v2.8h
            fmax v18.8h, v18.8h, v2.8h
            fmax v19.8h, v19.8h, v2.8h
            fmax v20.8h, v20.8h, v2.8h
            fmax v21.8h, v21.8h, v2.8h
            fmax v22.8h, v22.8h, v2.8h
            fmax v23.8h, v23.8h, v2.8h
            fmax v24.8h, v24.8h, v2.8h
            fmax v25.8h, v25.8h, v2.8h
            fmax v26.8h, v26.8h, v2.8h
            fmax v27.8h, v27.8h, v2.8h
            fmax v28.8h, v28.8h, v2.8h
            fmax v29.8h, v29.8h, v2.8h
            fmax v30.8h, v30.8h, v2.8h
            fmax v31.8h, v31.8h, v2.8h
            b Write

LoopRow8:
    mov x14, x1 // reload rhs ptr
    mov x13, x7 // reload rhs col
    mov x12, x3 // reload bias

    LoopCol8:
        cbz x9, NoReloadDst8
        mov x11, x2
    NoReloadDst8:
        mov x10, x0 // reload lhs ptr
        mov x19, x5 // reload depth

        dup v16.4s, wzr
        dup v17.4s, wzr
        dup v18.4s, wzr
        dup v19.4s, wzr
        dup v20.4s, wzr
        dup v21.4s, wzr
        dup v22.4s, wzr
        dup v23.4s, wzr

        cmp x19, #4
        blt LoopDepth8One

    LoopDepth8:
        ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x10], #64
        ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x14], #64
        fmla v16.8h, v8.8h, v0.h[0]
        fmla v17.8h, v8.8h, v0.h[1]
        fmla v18.8h, v8.8h, v0.h[2]
        fmla v19.8h, v8.8h, v0.h[3]
        fmla v20.8h, v8.8h, v0.h[4]
        fmla v21.8h, v8.8h, v0.h[5]
        fmla v22.8h, v8.8h, v0.h[6]
        fmla v23.8h, v8.8h, v0.h[7]
        ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x10], #64
        fmla v16.8h, v9.8h, v2.h[0]
        fmla v17.8h, v9.8h, v2.h[1]
        fmla v18.8h, v9.8h, v2.h[2]
        fmla v19.8h, v9.8h, v2.h[3]
        fmla v20.8h, v9.8h, v2.h[4]
        fmla v21.8h, v9.8h, v2.h[5]
        fmla v22.8h, v9.8h, v2.h[6]
        fmla v23.8h, v9.8h, v2.h[7]
        fmla v16.8h, v10.8h, v4.h[0]
        fmla v17.8h, v10.8h, v4.h[1]
        fmla v18.8h, v10.8h, v4.h[2]
        fmla v19.8h, v10.8h, v4.h[3]
        fmla v20.8h, v10.8h, v4.h[4]
        fmla v21.8h, v10.8h, v4.h[5]
        fmla v22.8h, v10.8h, v4.h[6]
        fmla v23.8h, v10.8h, v4.h[7]
        fmla v16.8h, v11.8h, v6.h[0]
        fmla v17.8h, v11.8h, v6.h[1]
        fmla v18.8h, v11.8h, v6.h[2]
        fmla v19.8h, v11.8h, v6.h[3]
        fmla v20.8h, v11.8h, v6.h[4]
        fmla v21.8h, v11.8h, v6.h[5]
        fmla v22.8h, v11.8h, v6.h[6]
        fmla v23.8h, v11.8h, v6.h[7]

        subs x19, x19, #4
        beq Bias8
        cmp x19, #4
        bge LoopDepth8

        LoopDepth8One:
            ld1 {v0.8h, v1.8h}, [x10], #32
            ld1 {v2.8h}, [x14], #16
            fmla v16.8h, v2.8h, v0.h[0]
            fmla v17.8h, v2.8h, v0.h[1]
            fmla v18.8h, v2.8h, v0.h[2]
            fmla v19.8h, v2.8h, v0.h[3]
            fmla v20.8h, v2.8h, v0.h[4]
            fmla v21.8h, v2.8h, v0.h[5]
            fmla v22.8h, v2.8h, v0.h[6]
            fmla v23.8h, v2.8h, v0.h[7]

            subs x19, x19, #1
            bgt LoopDepth8One

        Bias8:
            cbz x3, Activation8
            ld1 {v0.8h}, [x12], #16
            fadd v16.8h, v16.8h, v0.8h
            fadd v17.8h, v17.8h, v0.8h
            fadd v18.8h, v18.8h, v0.8h
            fadd v19.8h, v19.8h, v0.8h
            fadd v20.8h, v20.8h, v0.8h
            fadd v21.8h, v21.8h, v0.8h
            fadd v22.8h, v22.8h, v0.8h
            fadd v23.8h, v23.8h, v0.8h

        Activation8:
            cmp x4, #3
            beq Relu68
            cmp x4, #1
            beq Relu8
            b Write

        Relu68:
            movi v2.8h, #0x46, lsl #8
            fmin v16.8h, v16.8h, v2.8h
            fmin v17.8h, v17.8h, v2.8h
            fmin v18.8h, v18.8h, v2.8h
            fmin v19.8h, v19.8h, v2.8h
            fmin v20.8h, v20.8h, v2.8h
            fmin v21.8h, v21.8h, v2.8h
            fmin v22.8h, v22.8h, v2.8h
            fmin v23.8h, v23.8h, v2.8h
        
        Relu8:
            dup v2.8h, wzr
            fmax v16.8h, v16.8h, v2.8h
            fmax v17.8h, v17.8h, v2.8h
            fmax v18.8h, v18.8h, v2.8h
            fmax v19.8h, v19.8h, v2.8h
            fmax v20.8h, v20.8h, v2.8h
            fmax v21.8h, v21.8h, v2.8h
            fmax v22.8h, v22.8h, v2.8h
            fmax v23.8h, v23.8h, v2.8h
            b Write

LoopRow4:
    mov x14, x1 // reload rhs ptr
    mov x13, x7 // reload rhs col
    mov x12, x3 // reload bias

    LoopCol4:
        cbz x9, NoReloadDst4
        mov x11, x2
    NoReloadDst4:
        mov x10, x0 // reload lhs ptr
        mov x19, x5 // reload depth

        dup v16.4s, wzr
        dup v17.4s, wzr
        dup v18.4s, wzr
        dup v19.4s, wzr

        cmp x19, #4
        blt LoopDepth4One

    LoopDepth4:
        ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x10], #64
        ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x14], #64
        fmla v16.8h, v8.8h, v0.h[0]
        fmla v17.8h, v8.8h, v0.h[1]
        fmla v18.8h, v8.8h, v0.h[2]
        fmla v19.8h, v8.8h, v0.h[3]
        ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x10], #64
        fmla v16.8h, v9.8h, v2.h[0]
        fmla v17.8h, v9.8h, v2.h[1]
        fmla v18.8h, v9.8h, v2.h[2]
        fmla v19.8h, v9.8h, v2.h[3]
        fmla v16.8h, v10.8h, v4.h[0]
        fmla v17.8h, v10.8h, v4.h[1]
        fmla v18.8h, v10.8h, v4.h[2]
        fmla v19.8h, v10.8h, v4.h[3]
        fmla v16.8h, v11.8h, v6.h[0]
        fmla v17.8h, v11.8h, v6.h[1]
        fmla v18.8h, v11.8h, v6.h[2]
        fmla v19.8h, v11.8h, v6.h[3]

        subs x19, x19, #4
        beq Bias4
        cmp x19, #4
        bge LoopDepth4

        LoopDepth4One:
            ld1 {v0.8h, v1.8h}, [x10], #32
            ld1 {v2.8h}, [x14], #16
            fmla v16.8h, v2.8h, v0.h[0]
            fmla v17.8h, v2.8h, v0.h[1]
            fmla v18.8h, v2.8h, v0.h[2]
            fmla v19.8h, v2.8h, v0.h[3]

            subs x19, x19, #1
            bgt LoopDepth4One

        Bias4:
            cbz x3, Activation4
            ld1 {v0.8h}, [x12], #16
            fadd v16.8h, v16.8h, v0.8h
            fadd v17.8h, v17.8h, v0.8h
            fadd v18.8h, v18.8h, v0.8h
            fadd v19.8h, v19.8h, v0.8h

        Activation4:
            cmp x4, #3
            beq Relu64
            cmp x4, #1
            beq Relu4
            b Write

        Relu64:
            movi v2.8h, #0x46, lsl #8
            fmin v16.8h, v16.8h, v2.8h
            fmin v17.8h, v17.8h, v2.8h
            fmin v18.8h, v18.8h, v2.8h
            fmin v19.8h, v19.8h, v2.8h
        
        Relu4:
            dup v2.8h, wzr
            fmax v16.8h, v16.8h, v2.8h
            fmax v17.8h, v17.8h, v2.8h
            fmax v18.8h, v18.8h, v2.8h
            fmax v19.8h, v19.8h, v2.8h
            b Write
  
LoopRow2:
    mov x14, x1 // reload rhs ptr
    mov x13, x7 // reload rhs col
    mov x12, x3 // reload bias

    LoopCol2:
        cbz x9, NoReloadDst2
        mov x11, x2
    NoReloadDst2:
        mov x10, x0 // reload lhs ptr
        mov x19, x5 // reload depth

        dup v16.4s, wzr
        dup v17.4s, wzr

        cmp x19, #4
        blt LoopDepth2One

    LoopDepth2:
        ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x10], #64
        ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x14], #64
        fmla v16.8h, v8.8h, v0.h[0]
        fmla v17.8h, v8.8h, v0.h[1]
        ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x10], #64
        fmla v16.8h, v9.8h, v2.h[0]
        fmla v17.8h, v9.8h, v2.h[1]
        fmla v16.8h, v10.8h, v4.h[0]
        fmla v17.8h, v10.8h, v4.h[1]
        fmla v16.8h, v11.8h, v6.h[0]
        fmla v17.8h, v11.8h, v6.h[1]

        subs x19, x19, #4
        beq Bias2
        cmp x19, #4
        bge LoopDepth2

        LoopDepth2One:
            ld1 {v0.8h, v1.8h}, [x10], #32
            ld1 {v2.8h}, [x14], #16
            fmla v16.8h, v2.8h, v0.h[0]
            fmla v17.8h, v2.8h, v0.h[1]

            subs x19, x19, #1
            bgt LoopDepth2One

        Bias2:
            cbz x3, Activation2
            ld1 {v0.8h}, [x12], #16
            fadd v16.8h, v16.8h, v0.8h
            fadd v17.8h, v17.8h, v0.8h

        Activation2:
            cmp x4, #3
            beq Relu62
            cmp x4, #1
            beq Relu2
            b Write

        Relu62:
            movi v2.8h, #0x46, lsl #8
            fmin v16.8h, v16.8h, v2.8h
            fmin v17.8h, v17.8h, v2.8h
        
        Relu2:
            dup v2.8h, wzr
            fmax v16.8h, v16.8h, v2.8h
            fmax v17.8h, v17.8h, v2.8h
            b Write

LoopRow:
    mov x14, x1 // reload rhs ptr
    mov x13, x7 // reload rhs col
    mov x12, x3 // reload bias

    LoopCol:
        cbz x9, NoReloadDst
        mov x11, x2
    NoReloadDst:
        mov x10, x0 // reload lhs ptr
        mov x19, x5 // reload depth

        dup v16.4s, wzr

        cmp x19, #4
        blt LoopDepthOne

    LoopDepth:
        ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x10], #64
        ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x14], #64
        fmla v16.8h, v8.8h, v0.h[0]
        ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x10], #64
        fmla v16.8h, v9.8h, v2.h[0]
        fmla v16.8h, v10.8h, v4.h[0]
        fmla v16.8h, v11.8h, v6.h[0]

        subs x19, x19, #4
        beq Bias
        cmp x19, #4
        bge LoopDepth

        LoopDepthOne:
            ld1 {v0.8h, v1.8h}, [x10], #32
            ld1 {v2.8h}, [x14], #16
            fmla v16.8h, v2.8h, v0.h[0]

            subs x19, x19, #1
            bgt LoopDepthOne

        Bias:
            cbz x3, Activation
            ld1 {v0.8h}, [x12], #16
            fadd v16.8h, v16.8h, v0.8h

        Activation:
            cmp x4, #3
            beq Relu6
            cmp x4, #1
            beq Relu
            b Write

        Relu6:
            movi v2.8h, #0x46, lsl #8
            fmin v16.8h, v16.8h, v2.8h
        
        Relu:
            dup v2.8h, wzr
            fmax v16.8h, v16.8h, v2.8h

        Write:
            cmp x9, #2
            beq WriteWino
            cbz x9, WriteC8
            cmp x13, #1
            beq Write1
            cmp x13, #2
            beq Write2
            cmp x13, #3
            beq Write3
            cmp x13, #4
            beq Write4
            cmp x13, #5
            beq Write5
            cmp x13, #6
            beq Write6
            cmp x13, #7
            beq Write7
            b Write8

        Write1:
            add x2, x2, #2
            str h16, [x11]
            cmp x6, #1
            beq WriteEnd
            add x11, x11, x8
            str h17, [x11]
            cmp x6, #2
            beq WriteEnd
            add x11, x11, x8
            str h18, [x11]
            cmp x6, #3
            beq WriteEnd
            add x11, x11, x8
            str h19, [x11]
            cmp x6, #4
            beq WriteEnd
            add x11, x11, x8
            str h20, [x11]
            cmp x6, #5
            beq WriteEnd
            add x11, x11, x8
            str h21, [x11]
            cmp x6, #6
            beq WriteEnd
            add x11, x11, x8
            str h22, [x11]
            cmp x6, #7
            beq WriteEnd
            add x11, x11, x8
            str h23, [x11]
            cmp x6, #8
            beq WriteEnd
            add x11, x11, x8
            str h24, [x11]
            cmp x6, #9
            beq WriteEnd
            add x11, x11, x8
            str h25, [x11]
            cmp x6, #10
            beq WriteEnd
            add x11, x11, x8
            str h26, [x11]
            cmp x6, #11
            beq WriteEnd
            add x11, x11, x8
            str h27, [x11]
            cmp x6, #12
            beq WriteEnd
            add x11, x11, x8
            str h28, [x11]
            cmp x6, #13
            beq WriteEnd
            add x11, x11, x8
            str h29, [x11]
            cmp x6, #14
            beq WriteEnd
            add x11, x11, x8
            str h30, [x11]
            cmp x6, #15
            beq WriteEnd
            add x11, x11, x8
            str h31, [x11]
            add x11, x11, x8
            add x11, x11, #2
            b WriteEnd
        Write2:
            add x2, x2, #4
            add x19, x11, #2
            st1 {v16.h}[0], [x11], x8
            st1 {v16.h}[1], [x19], x8
            cmp x6, #1
            beq WriteEnd
            st1 {v17.h}[0], [x11], x8
            st1 {v17.h}[1], [x19], x8
            cmp x6, #2
            beq WriteEnd
            st1 {v18.h}[0], [x11], x8
            st1 {v18.h}[1], [x19], x8
            cmp x6, #3
            beq WriteEnd
            st1 {v19.h}[0], [x11], x8
            st1 {v19.h}[1], [x19], x8
            cmp x6, #4
            beq WriteEnd
            st1 {v20.h}[0], [x11], x8
            st1 {v20.h}[1], [x19], x8
            cmp x6, #5
            beq WriteEnd
            st1 {v21.h}[0], [x11], x8
            st1 {v21.h}[1], [x19], x8
            cmp x6, #6
            beq WriteEnd
            st1 {v22.h}[0], [x11], x8
            st1 {v22.h}[1], [x19], x8
            cmp x6, #7
            beq WriteEnd
            st1 {v23.h}[0], [x11], x8
            st1 {v23.h}[1], [x19], x8
            cmp x6, #8
            beq WriteEnd
            st1 {v24.h}[0], [x11], x8
            st1 {v24.h}[1], [x19], x8
            cmp x6, #9
            beq WriteEnd
            st1 {v25.h}[0], [x11], x8
            st1 {v25.h}[1], [x19], x8
            cmp x6, #10
            beq WriteEnd
            st1 {v26.h}[0], [x11], x8
            st1 {v26.h}[1], [x19], x8
            cmp x6, #11
            beq WriteEnd
            st1 {v27.h}[0], [x11], x8
            st1 {v27.h}[1], [x19], x8
            cmp x6, #12
            beq WriteEnd
            st1 {v28.h}[0], [x11], x8
            st1 {v28.h}[1], [x19], x8
            cmp x6, #13
            beq WriteEnd
            st1 {v29.h}[0], [x11], x8
            st1 {v29.h}[1], [x19], x8
            cmp x6, #14
            beq WriteEnd
            st1 {v30.h}[0], [x11], x8
            st1 {v30.h}[1], [x19], x8
            cmp x6, #15
            beq WriteEnd
            st1 {v31.h}[0], [x11], x8
            st1 {v31.h}[1], [x19]
            add x11, x11, #4
            b WriteEnd
        Write3:
            add x2, x2, #6
            add x19, x11, #4
            add x20, x11, #2
            st1 {v16.h}[0], [x11], x8
            st1 {v16.h}[1], [x20], x8
            st1 {v16.h}[2], [x19], x8
            cmp x6, #1
            beq WriteEnd
            st1 {v17.h}[0], [x11], x8
            st1 {v17.h}[1], [x20], x8
            st1 {v17.h}[2], [x19], x8
            cmp x6, #2
            beq WriteEnd
            st1 {v18.h}[0], [x11], x8
            st1 {v18.h}[1], [x20], x8
            st1 {v18.h}[2], [x19], x8
            cmp x6, #3
            beq WriteEnd
            st1 {v19.h}[0], [x11], x8
            st1 {v19.h}[1], [x20], x8
            st1 {v19.h}[2], [x19], x8
            cmp x6, #4
            beq WriteEnd
            st1 {v20.h}[0], [x11], x8
            st1 {v20.h}[1], [x20], x8
            st1 {v20.h}[2], [x19], x8
            cmp x6, #5
            beq WriteEnd
            st1 {v21.h}[0], [x11], x8
            st1 {v21.h}[1], [x20], x8
            st1 {v21.h}[2], [x19], x8
            cmp x6, #6
            beq WriteEnd
            st1 {v22.h}[0], [x11], x8
            st1 {v22.h}[1], [x20], x8
            st1 {v22.h}[2], [x19], x8
            cmp x6, #7
            beq WriteEnd
            st1 {v23.h}[0], [x11], x8
            st1 {v23.h}[1], [x20], x8
            st1 {v23.h}[2], [x19], x8
            cmp x6, #8
            beq WriteEnd
            st1 {v24.h}[0], [x11], x8
            st1 {v24.h}[1], [x20], x8
            st1 {v24.h}[2], [x19], x8
            cmp x6, #9
            beq WriteEnd
            st1 {v25.h}[0], [x11], x8
            st1 {v25.h}[1], [x20], x8
            st1 {v25.h}[2], [x19], x8
            cmp x6, #10
            beq WriteEnd
            st1 {v26.h}[0], [x11], x8
            st1 {v26.h}[1], [x20], x8
            st1 {v26.h}[2], [x19], x8
            cmp x6, #11
            beq WriteEnd
            st1 {v27.h}[0], [x11], x8
            st1 {v27.h}[1], [x20], x8
            st1 {v27.h}[2], [x19], x8
            cmp x6, #12
            beq WriteEnd
            st1 {v28.h}[0], [x11], x8
            st1 {v28.h}[1], [x20], x8
            st1 {v28.h}[2], [x19], x8
            cmp x6, #13
            beq WriteEnd
            st1 {v29.h}[0], [x11], x8
            st1 {v29.h}[1], [x20], x8
            st1 {v29.h}[2], [x19], x8
            cmp x6, #14
            beq WriteEnd
            st1 {v30.h}[0], [x11], x8
            st1 {v30.h}[1], [x20], x8
            st1 {v30.h}[2], [x19], x8
            cmp x6, #15
            beq WriteEnd
            st1 {v31.h}[0], [x11], x8
            st1 {v31.h}[1], [x20]
            st1 {v31.h}[2], [x19]
            add x11, x11, #6
            b WriteEnd
        Write4:
            add x2, x2, #8
            st1 {v16.4h}, [x11], x8
            cmp x6, #1
            beq WriteEnd
            st1 {v17.4h}, [x11], x8
            cmp x6, #2
            beq WriteEnd
            st1 {v18.4h}, [x11], x8
            cmp x6, #3
            beq WriteEnd
            st1 {v19.4h}, [x11], x8
            cmp x6, #4
            beq WriteEnd
            st1 {v20.4h}, [x11], x8
            cmp x6, #5
            beq WriteEnd
            st1 {v21.4h}, [x11], x8
            cmp x6, #6
            beq WriteEnd
            st1 {v22.4h}, [x11], x8
            cmp x6, #7
            beq WriteEnd
            st1 {v23.4h}, [x11], x8
            cmp x6, #8
            beq WriteEnd
            st1 {v24.4h}, [x11], x8
            cmp x6, #9
            beq WriteEnd
            st1 {v25.4h}, [x11], x8
            cmp x6, #10
            beq WriteEnd
            st1 {v26.4h}, [x11], x8
            cmp x6, #11
            beq WriteEnd
            st1 {v27.4h}, [x11], x8
            cmp x6, #12
            beq WriteEnd
            st1 {v28.4h}, [x11], x8
            cmp x6, #13
            beq WriteEnd
            st1 {v29.4h}, [x11], x8
            cmp x6, #14
            beq WriteEnd
            st1 {v30.4h}, [x11], x8
            cmp x6, #15
            beq WriteEnd
            st1 {v31.4h}, [x11], x8
            add x11, x11, #8
            b WriteEnd
        Write5:
            add x2, x2, #10
            add x19, x11, #8
            st1 {v16.4h}, [x11], x8
            st1 {v16.h}[4], [x19], x8
            cmp x6, #1
            beq WriteEnd
            st1 {v17.4h}, [x11], x8
            st1 {v17.h}[4], [x19], x8
            cmp x6, #2
            beq WriteEnd
            st1 {v18.4h}, [x11], x8
            st1 {v18.h}[4], [x19], x8
            cmp x6, #3
            beq WriteEnd
            st1 {v19.4h}, [x11], x8
            st1 {v19.h}[4], [x19], x8
            cmp x6, #4
            beq WriteEnd
            st1 {v20.4h}, [x11], x8
            st1 {v20.h}[4], [x19], x8
            cmp x6, #5
            beq WriteEnd
            st1 {v21.4h}, [x11], x8
            st1 {v21.h}[4], [x19], x8
            cmp x6, #6
            beq WriteEnd
            st1 {v22.4h}, [x11], x8
            st1 {v22.h}[4], [x19], x8
            cmp x6, #7
            beq WriteEnd
            st1 {v23.4h}, [x11], x8
            st1 {v23.h}[4], [x19], x8
            cmp x6, #8
            beq WriteEnd
            st1 {v24.4h}, [x11], x8
            st1 {v24.h}[4], [x19], x8
            cmp x6, #9
            beq WriteEnd
            st1 {v25.4h}, [x11], x8
            st1 {v25.h}[4], [x19], x8
            cmp x6, #10
            beq WriteEnd
            st1 {v26.4h}, [x11], x8
            st1 {v26.h}[4], [x19], x8
            cmp x6, #11
            beq WriteEnd
            st1 {v27.4h}, [x11], x8
            st1 {v27.h}[4], [x19], x8
            cmp x6, #12
            beq WriteEnd
            st1 {v28.4h}, [x11], x8
            st1 {v28.h}[4], [x19], x8
            cmp x6, #13
            beq WriteEnd
            st1 {v29.4h}, [x11], x8
            st1 {v29.h}[4], [x19], x8
            cmp x6, #14
            beq WriteEnd
            st1 {v30.4h}, [x11], x8
            st1 {v30.h}[4], [x19], x8
            cmp x6, #15
            beq WriteEnd
            st1 {v31.4h}, [x11], x8
            st1 {v31.h}[4], [x19]
            add x11, x11, #10
            b WriteEnd
        Write6:
            add x2, x2, #12
            add x19, x11, #8
            add x20, x11, #10
            st1 {v16.4h}, [x11], x8
            st1 {v16.h}[4], [x19], x8
            st1 {v16.h}[5], [x20], x8
            cmp x6, #1
            beq WriteEnd
            st1 {v17.4h}, [x11], x8
            st1 {v17.h}[4], [x19], x8
            st1 {v17.h}[5], [x20], x8
            cmp x6, #2
            beq WriteEnd
            st1 {v18.4h}, [x11], x8
            st1 {v18.h}[4], [x19], x8
            st1 {v18.h}[5], [x20], x8
            cmp x6, #3
            beq WriteEnd
            st1 {v19.4h}, [x11], x8
            st1 {v19.h}[4], [x19], x8
            st1 {v19.h}[5], [x20], x8
            cmp x6, #4
            beq WriteEnd
            st1 {v20.4h}, [x11], x8
            st1 {v20.h}[4], [x19], x8
            st1 {v20.h}[5], [x20], x8
            cmp x6, #5
            beq WriteEnd
            st1 {v21.4h}, [x11], x8
            st1 {v21.h}[4], [x19], x8
            st1 {v21.h}[5], [x20], x8
            cmp x6, #6
            beq WriteEnd
            st1 {v22.4h}, [x11], x8
            st1 {v22.h}[4], [x19], x8
            st1 {v22.h}[5], [x20], x8
            cmp x6, #7
            beq WriteEnd
            st1 {v23.4h}, [x11], x8
            st1 {v23.h}[4], [x19], x8
            st1 {v23.h}[5], [x20], x8
            cmp x6, #8
            beq WriteEnd
            st1 {v24.4h}, [x11], x8
            st1 {v24.h}[4], [x19], x8
            st1 {v24.h}[5], [x20], x8
            cmp x6, #9
            beq WriteEnd
            st1 {v25.4h}, [x11], x8
            st1 {v25.h}[4], [x19], x8
            st1 {v25.h}[5], [x20], x8
            cmp x6, #10
            beq WriteEnd
            st1 {v26.4h}, [x11], x8
            st1 {v26.h}[4], [x19], x8
            st1 {v26.h}[5], [x20], x8
            cmp x6, #11
            beq WriteEnd
            st1 {v27.4h}, [x11], x8
            st1 {v27.h}[4], [x19], x8
            st1 {v27.h}[5], [x20], x8
            cmp x6, #12
            beq WriteEnd
            st1 {v28.4h}, [x11], x8
            st1 {v28.h}[4], [x19], x8
            st1 {v28.h}[5], [x20], x8
            cmp x6, #13
            beq WriteEnd
            st1 {v29.4h}, [x11], x8
            st1 {v29.h}[4], [x19], x8
            st1 {v29.h}[5], [x20], x8
            cmp x6, #14
            beq WriteEnd
            st1 {v30.4h}, [x11], x8
            st1 {v30.h}[4], [x19], x8
            st1 {v30.h}[5], [x20], x8
            cmp x6, #15
            beq WriteEnd
            st1 {v31.4h}, [x11], x8
            st1 {v31.h}[4], [x19]
            st1 {v31.h}[5], [x20]
            add x11, x11, #12
            b WriteEnd
        Write7:
            add x2, x2, #14
            add x19, x11, #8
            add x20, x11, #10
            add x10, x11, #12
            st1 {v16.4h}, [x11], x8
            st1 {v16.h}[4], [x19], x8
            st1 {v16.h}[5], [x20], x8
            st1 {v16.h}[6], [x10], x8
            cmp x6, #1
            beq WriteEnd
            st1 {v17.4h}, [x11], x8
            st1 {v17.h}[4], [x19], x8
            st1 {v17.h}[5], [x20], x8
            st1 {v17.h}[6], [x10], x8
            cmp x6, #2
            beq WriteEnd
            st1 {v18.4h}, [x11], x8
            st1 {v18.h}[4], [x19], x8
            st1 {v18.h}[5], [x20], x8
            st1 {v18.h}[6], [x10], x8
            cmp x6, #3
            beq WriteEnd
            st1 {v19.4h}, [x11], x8
            st1 {v19.h}[4], [x19], x8
            st1 {v19.h}[5], [x20], x8
            st1 {v19.h}[6], [x10], x8
            cmp x6, #4
            beq WriteEnd
            st1 {v20.4h}, [x11], x8
            st1 {v20.h}[4], [x19], x8
            st1 {v20.h}[5], [x20], x8
            st1 {v20.h}[6], [x10], x8
            cmp x6, #5
            beq WriteEnd
            st1 {v21.4h}, [x11], x8
            st1 {v21.h}[4], [x19], x8
            st1 {v21.h}[5], [x20], x8
            st1 {v21.h}[6], [x10], x8
            cmp x6, #6
            beq WriteEnd
            st1 {v22.4h}, [x11], x8
            st1 {v22.h}[4], [x19], x8
            st1 {v22.h}[5], [x20], x8
            st1 {v22.h}[6], [x10], x8
            cmp x6, #7
            beq WriteEnd
            st1 {v23.4h}, [x11], x8
            st1 {v23.h}[4], [x19], x8
            st1 {v23.h}[5], [x20], x8
            st1 {v23.h}[6], [x10], x8
            cmp x6, #8
            beq WriteEnd
            st1 {v24.4h}, [x11], x8
            st1 {v24.h}[4], [x19], x8
            st1 {v24.h}[5], [x20], x8
            st1 {v24.h}[6], [x10], x8
            cmp x6, #9
            beq WriteEnd
            st1 {v25.4h}, [x11], x8
            st1 {v25.h}[4], [x19], x8
            st1 {v25.h}[5], [x20], x8
            st1 {v25.h}[6], [x10], x8
            cmp x6, #10
            beq WriteEnd
            st1 {v26.4h}, [x11], x8
            st1 {v26.h}[4], [x19], x8
            st1 {v26.h}[5], [x20], x8
            st1 {v26.h}[6], [x10], x8
            cmp x6, #11
            beq WriteEnd
            st1 {v27.4h}, [x11], x8
            st1 {v27.h}[4], [x19], x8
            st1 {v27.h}[5], [x20], x8
            st1 {v27.h}[6], [x10], x8
            cmp x6, #12
            beq WriteEnd
            st1 {v28.4h}, [x11], x8
            st1 {v28.h}[4], [x19], x8
            st1 {v28.h}[5], [x20], x8
            st1 {v28.h}[6], [x10], x8
            cmp x6, #13
            beq WriteEnd
            st1 {v29.4h}, [x11], x8
            st1 {v29.h}[4], [x19], x8
            st1 {v29.h}[5], [x20], x8
            st1 {v29.h}[6], [x10], x8
            cmp x6, #14
            beq WriteEnd
            st1 {v30.4h}, [x11], x8
            st1 {v30.h}[4], [x19], x8
            st1 {v30.h}[5], [x20], x8
            st1 {v30.h}[6], [x10], x8
            cmp x6, #15
            beq WriteEnd
            st1 {v31.4h}, [x11], x8
            st1 {v31.h}[4], [x19]
            st1 {v31.h}[5], [x20]
            st1 {v31.h}[6], [x10]
            add x11, x11, #14
            b WriteEnd
        WriteC8:
            mov x19, x11
            st1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x19], #64
            st1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x19], #64
            st1 {v24.8h, v25.8h, v26.8h, v27.8h}, [x19], #64
            st1 {v28.8h, v29.8h, v30.8h, v31.8h}, [x19], #64
            add x11, x11, x16
            b WriteEnd
        WriteWino:
            add x2, x11, x16
            st1 {v16.8h}, [x11], x15
            st1 {v17.8h}, [x11], x15
            st1 {v18.8h}, [x11], x15
            st1 {v19.8h}, [x11], x15
            st1 {v20.8h}, [x11], x15
            st1 {v21.8h}, [x11], x15
            st1 {v22.8h}, [x11], x15
            st1 {v23.8h}, [x11], x15
            st1 {v24.8h}, [x11], x15
            st1 {v25.8h}, [x11], x15
            st1 {v26.8h}, [x11], x15
            st1 {v27.8h}, [x11], x15
            st1 {v28.8h}, [x11], x15
            st1 {v29.8h}, [x11], x15
            st1 {v30.8h}, [x11], x15
            st1 {v31.8h}, [x11], x15
            b WriteEnd
        Write8:
            add x2, x2, #16
            st1 {v16.8h}, [x11], x8
            cmp x6, #1
            beq WriteEnd
            st1 {v17.8h}, [x11], x8
            cmp x6, #2
            beq WriteEnd
            st1 {v18.8h}, [x11], x8
            cmp x6, #3
            beq WriteEnd
            st1 {v19.8h}, [x11], x8
            cmp x6, #4
            beq WriteEnd
            st1 {v20.8h}, [x11], x8
            cmp x6, #5
            beq WriteEnd
            st1 {v21.8h}, [x11], x8
            cmp x6, #6
            beq WriteEnd
            st1 {v22.8h}, [x11], x8
            cmp x6, #7
            beq WriteEnd
            st1 {v23.8h}, [x11], x8
            cmp x6, #8
            beq WriteEnd
            st1 {v24.8h}, [x11], x8
            cmp x6, #9
            beq WriteEnd
            st1 {v25.8h}, [x11], x8
            cmp x6, #10
            beq WriteEnd
            st1 {v26.8h}, [x11], x8
            cmp x6, #11
            beq WriteEnd
            st1 {v27.8h}, [x11], x8
            cmp x6, #12
            beq WriteEnd
            st1 {v28.8h}, [x11], x8
            cmp x6, #13
            beq WriteEnd
            st1 {v29.8h}, [x11], x8
            cmp x6, #14
            beq WriteEnd
            st1 {v30.8h}, [x11], x8
            cmp x6, #15
            beq WriteEnd
            st1 {v31.8h}, [x11], x8
            add x11, x11, #16

        WriteEnd:
            subs x13, x13, #8 // rhs col - 8
            ble LoopColEnd
            cmp x6, #1
            ble LoopCol
            cmp x6, #2
            ble LoopCol2
            cmp x6, #4
            ble LoopCol4
            cmp x6, #8
            ble LoopCol8
            b LoopCol16

LoopColEnd:
        add x0, x0, x17
        cbz x9, C8DstStep
        mov x18, #2
        mul x18, x18, x7
        sub x11, x11, x18
        mov x2, x11
        b NoDstStep
    C8DstStep:
        add x2, x2, #256
        mov x11, x2
    NoDstStep:
        subs x6, x6, #16
        bgt LoopRowStart

    sub sp, sp, #80
    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [sp], #64
    ldp x19, x20, [sp], #16
    ret
#endif
