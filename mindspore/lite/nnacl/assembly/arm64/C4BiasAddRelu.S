#ifdef __aarch64__

    .text
    .align 5
    //.p2align 5,,15
    .global C4BiasAddRelu
#ifndef __APPLE__
    .type C4BiasAddRelu, %function
#endif

//void C4BiasAddRelu(float *dst, const float *input, const float* bias, size_t oc, size_t plane_size, size_t stride)
//x0: dst, x1: input, x2: bias, x3: oc, x4: plane_size, x5: stride

C4BiasAddRelu:
    dup v5.4s, wzr
    LoopOc:
        ld1 {v4.4s}, [x2], #16
        mov x6, x4
        mov x7, x0
        cmp x6, #4
        blt Loop1

        Loop4:
            ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
            fadd v0.4s, v0.4s, v4.4s
            fadd v1.4s, v1.4s, v4.4s
            fadd v2.4s, v2.4s, v4.4s
            fadd v3.4s, v3.4s, v4.4s

            fmax v0.4s, v0.4s, v5.4s
            fmax v1.4s, v1.4s, v5.4s
            fmax v2.4s, v2.4s, v5.4s
            fmax v3.4s, v3.4s, v5.4s

            cmp x3, #4
            bge Write4x4
            cmp x3, #3
            beq Write3x4
            cmp x3, #2
            beq Write2x4
         
        Write1x4:
            str s0, [x7]
            add x7, x7, x5
            str s1, [x7]
            add x7, x7, x5
            str s2, [x7]
            add x7, x7, x5
            str s3, [x7]
            add x7, x7, x5
            b WriteEndx4
        Write2x4:
            dup s16, v0.s[1]
            stp s0, s16, [x7]
            add x7, x7, x5
            dup s17, v1.s[1]
            stp s1, s17, [x7]
            add x7, x7, x5
            dup s18, v2.s[1]
            stp s2, s18, [x7]
            add x7, x7, x5
            dup s19, v3.s[1]
            stp s3, s19, [x7]
            add x7, x7, x5
            b WriteEndx4
        Write3x4:
            add x8, x7, #8
            dup s16, v0.s[1]
            stp s0, s16, [x7]
            add x7, x7, x5
            st1 {v0.s}[2], [x8], x5
            dup s17, v1.s[1]
            stp s1, s17, [x7]
            add x7, x7, x5
            st1 {v1.s}[2], [x8], x5
            dup s18, v2.s[1]
            stp s2, s18, [x7]
            add x7, x7, x5
            st1 {v2.s}[2], [x8], x5
            dup s19, v3.s[1]
            stp s3, s19, [x7]
            add x7, x7, x5
            st1 {v3.s}[2], [x8], x5
            b WriteEndx4
        Write4x4:
            st1 {v0.4s}, [x7], x5
            st1 {v1.4s}, [x7], x5
            st1 {v2.4s}, [x7], x5
            st1 {v3.4s}, [x7], x5

        WriteEndx4:
            subs x6, x6, #4
            beq LoopOcEnd
            cmp x6, #4
            blt Loop1
            b Loop4

        Loop1:
            ld1 {v0.4s}, [x1], #16
            fadd v0.4s, v0.4s, v4.4s
            fmax v0.4s, v0.4s, v5.4s

            cmp x3, #4
            bge Write4
            cmp x3, #3
            beq Write3
            cmp x3, #2
            beq Write2
         
        Write1:
            str s0, [x7]
            add x7, x7, x5
            b WriteEnd
        Write2:
            dup s16, v0.s[1]
            stp s0, s16, [x7]
            add x7, x7, x5
            b WriteEnd
        Write3:
            add x8, x7, #8
            dup s16, v0.s[1]
            stp s0, s16, [x7]
            add x7, x7, x5
            st1 {v0.s}[2], [x8], x5
            b WriteEnd
        Write4:
            st1 {v0.4s}, [x7], x5
        WriteEnd:
            subs x6, x6, #1
            bne Loop1
    LoopOcEnd:
        subs x3, x3, #4
        add x0, x0, #16
        bgt LoopOc

    ret
#endif
