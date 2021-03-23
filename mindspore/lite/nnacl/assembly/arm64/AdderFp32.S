#ifdef __aarch64__
#include "nnacl/assembly_global.h"
    .text
    .align 5
    .global AdderFloatNeon64
#ifndef __APPLE__
    .type AdderFloatNeon64, %function
#endif

// void AdderFloatNeon64(const float *a, const float *b, float *c, const float *bias, int act_type, int depth
//                        int row, int col, size_t stride)
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

asm_function AdderFloatNeon64
    sub sp, sp, #144
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
    stp x19, x20, [sp], #16

    ldr x8, [sp]

    mov x18, #48 // sizeof(float) * 12
    mul x17, x5, x18 // block stride of lhs/rhs: sizeof(float) * 12 * depth

    mov x18, #4
    mul x8, x8, x18

LoopRowStart:
    cmp x6, #4
    ble LoopRow4
    cmp x6, #8
    blt LoopRow8

LoopRow:
    mov x14, x1 // reload rhs ptr
    mov x13, x7 // reload rhs col
    mov x12, x3 // reload bias

    LoopCol:
        mov x11, x2
        mov x10, x0 // reload lhs ptr
        mov x19, x5 // reload depth

    LoopDepthStart:
        ld1 {v0.4s, v1.4s, v2.4s}, [x10], #48
        ld1 {v3.4s}, [x14], #16

        dup  v8.4s, v0.s[0]
        fabd v9.4s, v3.4s, v8.4s
        dup  v10.4s, v0.s[1]
        fabd v11.4s, v3.4s, v10.4s
        dup  v12.4s, v0.s[2]
        fabd v13.4s, v3.4s, v12.4s
        dup  v14.4s, v0.s[3]
        fabd v15.4s, v3.4s, v14.4s

        dup  v16.4s, v1.s[0]
        fabd v17.4s, v3.4s, v16.4s
        dup  v18.4s, v1.s[1]
        fabd v19.4s, v3.4s, v18.4s
        dup  v20.4s, v1.s[2]
        fabd v21.4s, v3.4s, v20.4s
        dup  v22.4s, v1.s[3]
        fabd v23.4s, v3.4s, v22.4s

        dup  v24.4s, v2.s[0]
        fabd v25.4s, v3.4s, v24.4s
        dup  v26.4s, v2.s[1]
        fabd v27.4s, v3.4s, v26.4s
        dup  v28.4s, v2.s[2]
        fabd v29.4s, v3.4s, v28.4s
        dup  v30.4s, v2.s[3]
        fabd v31.4s, v3.4s, v30.4s

        subs x19, x19, #1
        beq Bias

        LoopDepth:
            ld1 {v0.4s, v1.4s, v2.4s}, [x10], #48
            ld1 {v3.4s}, [x14], #16

            dup v8.4s, v0.s[0]
            fabd v8.4s, v3.4s, v8.4s
            fadd v9.4s, v9.4s, v8.4s
            dup v10.4s, v0.s[1]
            fabd v10.4s, v3.4s, v10.4s
            fadd v11.4s, v11.4s, v10.4s
            dup v12.4s, v0.s[2]
            fabd v12.4s, v3.4s, v12.4s
            fadd v13.4s, v13.4s, v12.4s
            dup v14.4s, v0.s[3]
            fabd v14.4s, v3.4s, v14.4s
            fadd v15.4s, v15.4s, v14.4s

            dup v16.4s, v1.s[0]
            fabd v16.4s, v3.4s, v16.4s
            fadd v17.4s, v17.4s, v16.4s
            dup v18.4s, v1.s[1]
            fabd v18.4s, v3.4s, v18.4s
            fadd v19.4s, v19.4s, v18.4s
            dup v20.4s, v1.s[2]
            fabd v20.4s, v3.4s, v20.4s
            fadd v21.4s, v21.4s, v20.4s
            dup v22.4s, v1.s[3]
            fabd v22.4s, v3.4s, v22.4s
            fadd v23.4s, v23.4s, v22.4s

            dup v24.4s, v2.s[0]
            fabd v24.4s, v3.4s, v24.4s
            fadd v25.4s, v25.4s, v24.4s
            dup v26.4s, v2.s[1]
            fabd v26.4s, v3.4s, v26.4s
            fadd v27.4s, v27.4s, v26.4s
            dup v28.4s, v2.s[2]
            fabd v28.4s, v3.4s, v28.4s
            fadd v29.4s, v29.4s, v28.4s
            dup v30.4s, v2.s[3]
            fabd v30.4s, v3.4s, v30.4s
            fadd v31.4s, v31.4s, v30.4s

            subs x19, x19, #1
            bgt LoopDepth

        Bias:
            fneg v9.4s, v9.4s
            fneg v11.4s, v11.4s
            fneg v13.4s, v13.4s
            fneg v15.4s, v15.4s
            fneg v17.4s, v17.4s
            fneg v19.4s, v19.4s
            fneg v21.4s, v21.4s
            fneg v23.4s, v23.4s
            fneg v25.4s, v25.4s
            fneg v27.4s, v27.4s
            fneg v29.4s, v29.4s
            fneg v31.4s, v31.4s
            cbz x3, Activation
            ld1 {v0.4s}, [x12], #16
            fadd v9.4s, v9.4s, v0.4s
            fadd v11.4s, v11.4s, v0.4s
            fadd v13.4s, v13.4s, v0.4s
            fadd v15.4s, v15.4s, v0.4s
            fadd v17.4s, v17.4s, v0.4s
            fadd v19.4s, v19.4s, v0.4s
            fadd v21.4s, v21.4s, v0.4s
            fadd v23.4s, v23.4s, v0.4s
            fadd v25.4s, v25.4s, v0.4s
            fadd v27.4s, v27.4s, v0.4s
            fadd v29.4s, v29.4s, v0.4s
            fadd v31.4s, v31.4s, v0.4s

        Activation:
            cmp x4, #3
            beq Relu6
            cmp x4, #1
            beq Relu
            b Write

        Relu6:
            mov w19, #6
            dup v2.4s, w19
            scvtf v2.4s, v2.4s
            fmin v9.4s, v9.4s, v2.4s
            fmin v11.4s, v11.4s, v2.4s
            fmin v13.4s, v13.4s, v2.4s
            fmin v15.4s, v15.4s, v2.4s
            fmin v17.4s, v17.4s, v2.4s
            fmin v19.4s, v19.4s, v2.4s
            fmin v21.4s, v21.4s, v2.4s
            fmin v23.4s, v23.4s, v2.4s
            fmin v25.4s, v25.4s, v2.4s
            fmin v27.4s, v27.4s, v2.4s
            fmin v29.4s, v29.4s, v2.4s
            fmin v31.4s, v31.4s, v2.4s

        Relu:
            dup v3.4s, wzr
            fmax v9.4s, v9.4s, v3.4s
            fmax v11.4s, v11.4s, v3.4s
            fmax v13.4s, v13.4s, v3.4s
            fmax v15.4s, v15.4s, v3.4s
            fmax v17.4s, v17.4s, v3.4s
            fmax v19.4s, v19.4s, v3.4s
            fmax v21.4s, v21.4s, v3.4s
            fmax v23.4s, v23.4s, v3.4s
            fmax v25.4s, v25.4s, v3.4s
            fmax v27.4s, v27.4s, v3.4s
            fmax v29.4s, v29.4s, v3.4s
            fmax v31.4s, v31.4s, v3.4s
            b Write

LoopRow8:
    mov x14, x1 // reload rhs ptr
    mov x13, x7 // reload rhs col
    mov x12, x3 // reload bias

    LoopCol8:
        mov x11, x2
        mov x10, x0 // reload lhs ptr
        mov x19, x5 // reload depth

    LoopDepthStart8:
        ld1 {v0.4s, v1.4s, v2.4s}, [x10], #48
        ld1 {v3.4s}, [x14], #16

        dup  v8.4s, v0.s[0]
        fabd v9.4s, v3.4s, v8.4s
        dup  v10.4s, v0.s[1]
        fabd v11.4s, v3.4s, v10.4s
        dup  v12.4s, v0.s[2]
        fabd v13.4s, v3.4s, v12.4s
        dup  v14.4s, v0.s[3]
        fabd v15.4s, v3.4s, v14.4s

        dup  v16.4s, v1.s[0]
        fabd v17.4s, v3.4s, v16.4s
        dup  v18.4s, v1.s[1]
        fabd v19.4s, v3.4s, v18.4s
        dup  v20.4s, v1.s[2]
        fabd v21.4s, v3.4s, v20.4s
        dup  v22.4s, v1.s[3]
        fabd v23.4s, v3.4s, v22.4s

        subs x19, x19, #1
        beq Bias8

        LoopDepth8:
            ld1 {v0.4s, v1.4s, v2.4s}, [x10], #48
            ld1 {v3.4s}, [x14], #16
            dup v8.4s, v0.s[0]
            fabd v8.4s, v3.4s, v8.4s
            fadd v9.4s, v9.4s, v8.4s
            dup v10.4s, v0.s[1]
            fabd v10.4s, v3.4s, v10.4s
            fadd v11.4s, v11.4s, v10.4s
            dup v12.4s, v0.s[2]
            fabd v12.4s, v3.4s, v12.4s
            fadd v13.4s, v13.4s, v12.4s
            dup v14.4s, v0.s[3]
            fabd v14.4s, v3.4s, v14.4s
            fadd v15.4s, v15.4s, v14.4s

            dup v16.4s, v1.s[0]
            fabd v16.4s, v3.4s, v16.4s
            fadd v17.4s, v17.4s, v16.4s
            dup v18.4s, v1.s[1]
            fabd v18.4s, v3.4s, v18.4s
            fadd v19.4s, v19.4s, v18.4s
            dup v20.4s, v1.s[2]
            fabd v20.4s, v3.4s, v20.4s
            fadd v21.4s, v21.4s, v20.4s
            dup v22.4s, v1.s[3]
            fabd v22.4s, v3.4s, v22.4s
            fadd v23.4s, v23.4s, v22.4s

            subs x19, x19, #1
            bgt LoopDepth8

        Bias8:
            fneg v9.4s, v9.4s
            fneg v11.4s, v11.4s
            fneg v13.4s, v13.4s
            fneg v15.4s, v15.4s
            fneg v17.4s, v17.4s
            fneg v19.4s, v19.4s
            fneg v21.4s, v21.4s
            fneg v23.4s, v23.4s
            cbz x3, Activation8
            ld1 {v0.4s}, [x12], #16
            fadd v9.4s, v9.4s, v0.4s
            fadd v11.4s, v11.4s, v0.4s
            fadd v13.4s, v13.4s, v0.4s
            fadd v15.4s, v15.4s, v0.4s
            fadd v17.4s, v17.4s, v0.4s
            fadd v19.4s, v19.4s, v0.4s
            fadd v21.4s, v21.4s, v0.4s
            fadd v23.4s, v23.4s, v0.4s

        Activation8:
            cmp x4, #3
            beq Relu68
            cmp x4, #1
            beq Relu8
            b Write

        Relu68:
            mov w19, #6
            dup v2.4s, w19
            scvtf v2.4s, v2.4s
            fmin v9.4s, v9.4s, v2.4s
            fmin v11.4s, v11.4s, v2.4s
            fmin v13.4s, v13.4s, v2.4s
            fmin v15.4s, v15.4s, v2.4s
            fmin v17.4s, v17.4s, v2.4s
            fmin v19.4s, v19.4s, v2.4s
            fmin v21.4s, v21.4s, v2.4s
            fmin v23.4s, v23.4s, v2.4s
        Relu8:
            dup v3.4s, wzr
            fmax v9.4s, v9.4s, v3.4s
            fmax v11.4s, v11.4s, v3.4s
            fmax v13.4s, v13.4s, v3.4s
            fmax v15.4s, v15.4s, v3.4s
            fmax v17.4s, v17.4s, v3.4s
            fmax v19.4s, v19.4s, v3.4s
            fmax v21.4s, v21.4s, v3.4s
            fmax v23.4s, v23.4s, v3.4s
            b Write

LoopRow4:
    mov x14, x1 // reload rhs ptr
    mov x13, x7 // reload rhs col
    mov x12, x3 // reload bias

    LoopCol4:
        mov x11, x2
        mov x10, x0 // reload lhs ptr
        mov x19, x5 // reload depth

    LoopDepthStart4:
        ld1 {v0.4s, v1.4s, v2.4s}, [x10], #48
        ld1 {v3.4s}, [x14], #16
        dup  v8.4s, v0.s[0]
        fabd v9.4s, v3.4s, v8.4s
        dup  v10.4s, v0.s[1]
        fabd v11.4s, v3.4s, v10.4s
        dup  v12.4s, v0.s[2]
        fabd v13.4s, v3.4s, v12.4s
        dup  v14.4s, v0.s[3]
        fabd v15.4s, v3.4s, v14.4s

        subs x19, x19, #1
        beq Bias4

        LoopDepth4:
            ld1 {v0.4s, v1.4s, v2.4s}, [x10], #48
            ld1 {v3.4s}, [x14], #16
            dup v8.4s, v0.s[0]
            fabd v8.4s, v3.4s, v8.4s
            fadd v9.4s, v9.4s, v8.4s
            dup v10.4s, v0.s[1]
            fabd v10.4s, v3.4s, v10.4s
            fadd v11.4s, v11.4s, v10.4s
            dup v12.4s, v0.s[2]
            fabd v12.4s, v3.4s, v12.4s
            fadd v13.4s, v13.4s, v12.4s
            dup v14.4s, v0.s[3]
            fabd v14.4s, v3.4s, v14.4s
            fadd v15.4s, v15.4s, v14.4s

            subs x19, x19, #1
            bgt LoopDepth4

        Bias4:
            fneg v9.4s, v9.4s
            fneg v11.4s, v11.4s
            fneg v13.4s, v13.4s
            fneg v15.4s, v15.4s
            cbz x3, Activation4
            ld1 {v0.4s}, [x12], #16

            fadd v9.4s, v9.4s, v0.4s
            fadd v11.4s, v11.4s, v0.4s
            fadd v13.4s, v13.4s, v0.4s
            fadd v15.4s, v15.4s, v0.4s

        Activation4:
            cmp x4, #3
            beq Relu64
            cmp x4, #1
            beq Relu4
            b Write

        Relu64:
            mov w19, #6
            dup v2.4s, w19
            scvtf v2.4s, v2.4s
            fmin v9.4s, v9.4s, v2.4s
            fmin v11.4s, v11.4s, v2.4s
            fmin v13.4s, v13.4s, v2.4s
            fmin v15.4s, v15.4s, v2.4s

        Relu4:
            dup v3.4s, wzr
            fmax v9.4s, v9.4s, v2.4s
            fmax v11.4s, v11.4s, v2.4s
            fmax v13.4s, v13.4s, v2.4s
            fmax v15.4s, v15.4s, v2.4s
            b Write

        Write:
            cmp x13, #1
            beq Write1
            cmp x13, #2
            beq Write2
            cmp x13, #3
            beq Write3
            b Write4

        Write1:
            add x2, x2, #4
            str s9, [x11]
            cmp x6, #1
            beq WriteEnd
            add x11, x11, x8
            str s11, [x11]
            cmp x6, #2
            beq WriteEnd
            add x11, x11, x8
            str s13, [x11]
            cmp x6, #3
            beq WriteEnd
            add x11, x11, x8
            str s15, [x11]
            cmp x6, #4
            beq WriteEnd
            add x11, x11, x8
            str s17, [x11]
            cmp x6, #5
            beq WriteEnd
            add x11, x11, x8
            str s19, [x11]
            cmp x6, #6
            beq WriteEnd
            add x11, x11, x8
            str s21, [x11]
            cmp x6, #7
            beq WriteEnd
            add x11, x11, x8
            str s23, [x11]
            cmp x6, #8
            beq WriteEnd
            add x11, x11, x8
            str s25, [x11]
            cmp x6, #9
            beq WriteEnd
            add x11, x11, x8
            str s27, [x11]
            cmp x6, #10
            beq WriteEnd
            add x11, x11, x8
            str s29, [x11]
            cmp x6, #11
            beq WriteEnd
            add x11, x11, x8
            str s31, [x11]
            add x11, x11, x8
            add x11, x11, #4
            b WriteEnd
        Write2:
            add x2, x2, #8
            st1 {v9.2s}, [x11], x8
            cmp x6, #1
            beq WriteEnd
            st1 {v11.2s}, [x11], x8
            cmp x6, #2
            beq WriteEnd
            st1 {v13.2s}, [x11], x8
            cmp x6, #3
            beq WriteEnd
            st1 {v15.2s}, [x11], x8
            cmp x6, #4
            beq WriteEnd
            st1 {v17.2s}, [x11], x8
            cmp x6, #5
            beq WriteEnd
            st1 {v19.2s}, [x11], x8
            cmp x6, #6
            beq WriteEnd
            st1 {v21.2s}, [x11], x8
            cmp x6, #7
            beq WriteEnd
            st1 {v23.2s}, [x11], x8
            cmp x6, #8
            beq WriteEnd
            st1 {v25.2s}, [x11], x8
            cmp x6, #9
            beq WriteEnd
            st1 {v27.2s}, [x11], x8
            cmp x6, #10
            beq WriteEnd
            st1 {v29.2s}, [x11], x8
            cmp x6, #11
            beq WriteEnd
            st1 {v31.2s}, [x11], x8
            add x11, x11, #8
            b WriteEnd
        Write3:
            add x2, x2, #12
            add x19, x11, #8
            st1 {v9.2s}, [x11], x8
            st1 {v9.s}[2], [x19], x8
            cmp x6, #1
            beq WriteEnd
            st1 {v11.2s}, [x11], x8
            st1 {v11.s}[2], [x19], x8
            cmp x6, #2
            beq WriteEnd
            st1 {v13.2s}, [x11], x8
            st1 {v13.s}[2], [x19], x8
            cmp x6, #3
            beq WriteEnd
            st1 {v15.2s}, [x11], x8
            st1 {v15.s}[2], [x19], x8
            cmp x6, #4
            beq WriteEnd
            st1 {v17.2s}, [x11], x8
            st1 {v17.s}[2], [x19], x8
            cmp x6, #5
            beq WriteEnd
            st1 {v19.2s}, [x11], x8
            st1 {v19.s}[2], [x19], x8
            cmp x6, #6
            beq WriteEnd
            st1 {v21.2s}, [x11], x8
            st1 {v21.s}[2], [x19], x8
            cmp x6, #7
            beq WriteEnd
            st1 {v23.2s}, [x11], x8
            st1 {v23.s}[2], [x19], x8
            cmp x6, #8
            beq WriteEnd
            st1 {v25.2s}, [x11], x8
            st1 {v25.s}[2], [x19], x8
            cmp x6, #9
            beq WriteEnd
            st1 {v27.2s}, [x11], x8
            st1 {v27.s}[2], [x19], x8
            cmp x6, #10
            beq WriteEnd
            st1 {v29.2s}, [x11], x8
            st1 {v29.s}[2], [x19], x8
            cmp x6, #11
            beq WriteEnd
            st1 {v31.2s}, [x11], x8
            st1 {v31.s}[2], [x19]
            add x11, x11, #12
            b WriteEnd
        Write4:
            add x2, x2, #16
            st1 {v9.4s}, [x11], x8
            cmp x6, #1
            beq WriteEnd
            st1 {v11.4s}, [x11], x8
            cmp x6, #2
            beq WriteEnd
            st1 {v13.4s}, [x11], x8
            cmp x6, #3
            beq WriteEnd
            st1 {v15.4s}, [x11], x8
            cmp x6, #4
            beq WriteEnd
            st1 {v17.4s}, [x11], x8
            cmp x6, #5
            beq WriteEnd
            st1 {v19.4s}, [x11], x8
            cmp x6, #6
            beq WriteEnd
            st1 {v21.4s}, [x11], x8
            cmp x6, #7
            beq WriteEnd
            st1 {v23.4s}, [x11], x8
            cmp x6, #8
            beq WriteEnd
            st1 {v25.4s}, [x11], x8
            cmp x6, #9
            beq WriteEnd
            st1 {v27.4s}, [x11], x8
            cmp x6, #10
            beq WriteEnd
            st1 {v29.4s}, [x11], x8
            cmp x6, #11
            beq WriteEnd
            st1 {v31.4s}, [x11], x8
            add x11, x11, #16
            b WriteEnd

        WriteEnd:
            subs x13, x13, #4 // rhs col - 4
            ble LoopColEnd
            cmp x6, #4
            ble LoopCol4
            cmp x6, #8
            ble LoopCol8
            b LoopCol

LoopColEnd:
        add x0, x0, x17
        mov x18, #4
        mul x18, x18, x7
        sub x11, x11, x18
        mov x2, x11
        subs x6, x6, #12
        bgt LoopRowStart

  sub sp, sp, #144
  ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
  ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
  ldp x19, x20, [sp], #16
  ret
#endif
