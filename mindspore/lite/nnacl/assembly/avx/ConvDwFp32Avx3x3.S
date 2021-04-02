#ifdef ENABLE_AVX
.text
.align 4
.global ConvDwFp32Avx3x3
#ifndef __APPLE__
#ifndef WIN32
.type ConvDwFp32Avx3x3, %function
#endif
#endif

// void ConvDwFp32Avx3x3(float *output, float **input, const float *weights, const float *bias, size_t channels, size_t output_width,
//                       size_t input_stride, size_t relum, szie_t relu6)
// in linux x64 platform:
// rdi: output
// rsi: input
// rdx: weights
// rcx: bias
// r8: channels
// r9: output_width
// 8: input_stride
// 16: relu
// 24: relu6

// in win x64 platform: "shadow space" needs to be opened up for first four parameters ==> 32 bites
// rcx: output
// rdx: input
// r8: weights
// r9: bias
// 40: channels
// 48: output_width
// 56: input_stride
// 64: relu
// 72: relu6
ConvDwFp32Avx3x3:
    pushq %r15
    pushq %r14
    pushq %r13
    pushq %r12
    pushq %rbx
    pushq %rbp
    pushq %r9  // -56
    pushq %r8  // -64
    pushq %rcx  // -72
    pushq %rdx  // -80
    pushq %rsi  // -88
    pushq %rdi  // -96
    addq $96, %rsp

#ifdef WIN32
    movq %rcx, %rdi
    movq %rdx, %rsi
    movq %r8, %rdx
    movq %r9, %rcx
    movq 40(%rsp), %r8  // channels
    movq 48(%rsp), %r9  // output_width

    mov %rdx, -80(%rsp)
    mov %rcx, -72(%rsp)
    mov %r9, -56(%rsp)
    mov %r8, -64(%rsp)
    movq 56(%rsp), %rbp // input_stride
    movq %rbp, 8(%rsp)
    movq 64(%rsp), %rbp  // relu
    movq %rbp, 16(%rsp)
    movq 72(%rsp), %rbp  // relu6
    movq %rbp, 24(%rsp)
#endif

    movq $6, %rax
    vcvtsi2ss %rax, %xmm15, %xmm15
    vshufps $0, %xmm15, %xmm15, %xmm15
    vinsertf128 $1, %xmm15, %ymm15, %ymm15
    vxorps %ymm14, %ymm14, %ymm14   

    LoopPixel:
        movq -80(%rsp), %rdx
        movq -72(%rsp), %rcx
        movq -64(%rsp), %r8
        movq (%rsi), %r9
        movq 8(%rsi), %r10
        movq 16(%rsi), %r11
        movq 24(%rsi), %r12
        movq 32(%rsi), %r13
        movq 40(%rsi), %r14
        movq 48(%rsi), %r15
        movq 56(%rsi), %rbp
        movq 64(%rsi), %rbx

        vmovups (%r9), %ymm0
        addq $32, %r9
        vmovups (%r10), %ymm1
        addq $32, %r10
        vmovups (%r11), %ymm2
        addq $32, %r11

        vmovups (%rdx), %ymm11
        addq $32, %rdx
        vmovups (%rdx), %ymm12
        addq $32, %rdx
        vmovups (%rdx), %ymm13
        addq $32, %rdx

        vmovups (%rcx), %ymm10
        addq $32, %rcx

        cmpq $8, %r8
        jbe LeftLoop
        LoopC8:
            vfmadd231ps %ymm11, %ymm0, %ymm10
            vmovups (%r12), %ymm3
            addq $32, %r12
            vmovups (%rdx), %ymm11
            addq $32, %rdx
            vfmadd231ps %ymm12, %ymm1, %ymm10
            vmovups (%r13), %ymm4
            addq $32, %r13
            vmovups (%rdx), %ymm12
            addq $32, %rdx
            vfmadd231ps %ymm13, %ymm2, %ymm10
            vmovups (%r14), %ymm5
            addq $32, %r14
            vmovups (%rdx), %ymm13
            addq $32, %rdx
            vfmadd231ps %ymm11, %ymm3, %ymm10
            vmovups (%r15), %ymm6
            addq $32, %r15
            vmovups (%rdx), %ymm11
            addq $32, %rdx
            vfmadd231ps %ymm12, %ymm4, %ymm10
            vmovups (%rbp), %ymm7
            addq $32, %rbp
            vmovups (%rdx), %ymm12
            addq $32, %rdx
            vfmadd231ps %ymm13, %ymm5, %ymm10
            vmovups (%rbx), %ymm8
            addq $32, %rbx
            vmovups (%rdx), %ymm13
            addq $32, %rdx
            vfmadd231ps %ymm11, %ymm6, %ymm10
            vmovups (%r9), %ymm0
            addq $32, %r9
            vmovups (%rdx), %ymm11
            addq $32, %rdx
            vfmadd231ps %ymm12, %ymm7, %ymm10
            vmovups (%r10), %ymm1
            addq $32, %r10
            vmovups (%rdx), %ymm12
            addq $32, %rdx
            vfmadd231ps %ymm13, %ymm8, %ymm10
            vmovups (%r11), %ymm2
            addq $32, %r11
            vmovups (%rdx), %ymm13
            addq $32, %rdx

            movq 24(%rsp), %rax
            cmpq $0, %rax
            jne Relu6
            movq 16(%rsp), %rax
            cmpq $0, %rax
            jne Relu
            jmp Write
            Relu6:
                vminps %ymm15, %ymm10, %ymm10
            Relu:
                vmaxps %ymm14, %ymm10, %ymm10
            Write:
                vmovups %ymm10, (%rdi)
                addq $32, %rdi

            vmovups (%rcx), %ymm10
            addq $32, %rcx
            subq $8, %r8
            cmpq $8, %r8
            ja LoopC8

        LeftLoop:
            vfmadd231ps %ymm11, %ymm0, %ymm10
            vmovups (%r12), %ymm3
            addq $32, %r12
            vmovups (%rdx), %ymm11
            addq $32, %rdx
            vfmadd231ps %ymm12, %ymm1, %ymm10
            vmovups (%r13), %ymm4
            addq $32, %r13
            vmovups (%rdx), %ymm12
            addq $32, %rdx
            vfmadd231ps %ymm13, %ymm2, %ymm10
            vmovups (%r14), %ymm5
            addq $32, %r14
            vmovups (%rdx), %ymm13
            addq $32, %rdx
            vfmadd231ps %ymm11, %ymm3, %ymm10
            vmovups (%r15), %ymm6
            addq $32, %r15
            vmovups (%rdx), %ymm11
            addq $32, %rdx
            vfmadd231ps %ymm12, %ymm4, %ymm10
            vmovups (%rbp), %ymm7
            addq $32, %rbp
            vmovups (%rdx), %ymm12
            addq $32, %rdx
            vfmadd231ps %ymm13, %ymm5, %ymm10
            vmovups (%rbx), %ymm8
            addq $32, %rbx
            vmovups (%rdx), %ymm13
            addq $32, %rdx
            vfmadd231ps %ymm11, %ymm6, %ymm10
            vfmadd231ps %ymm12, %ymm7, %ymm10
            vfmadd231ps %ymm13, %ymm8, %ymm10

            movq 24(%rsp), %rax
            cmpq $0, %rax
            jne LeftRelu6
            movq 16(%rsp), %rax
            cmpq $0, %rax
            jne LeftRelu
            jmp LeftWrite
            LeftRelu6:
                vminps %ymm15, %ymm10, %ymm10
            LeftRelu:
                vmaxps %ymm14, %ymm10, %ymm10
            LeftWrite:
                cmpq $1, %r8
                je Write1
                cmpq $2, %r8
                je Write2
                cmpq $3, %r8
                je Write3
                cmpq $4, %r8
                je Write4
                cmpq $5, %r8
                je Write5
                cmpq $6, %r8
                je Write6
                cmpq $7, %r8
                je Write7
                jmp Write8
            Write1:
                vmovss %xmm10, (%rdi)
                addq $4, %rdi
                jmp NextPixel
            Write2:
                vmovsd %xmm10, (%rdi)
                addq $8, %rdi
                jmp NextPixel
            Write3:
                vmovsd %xmm10, (%rdi)
                movhlps %xmm10, %xmm10
                vmovss %xmm10, 8(%rdi)
                addq $12, %rdi
                jmp NextPixel
            Write4:
                vmovups %xmm10, (%rdi)
                addq $16, %rdi
                jmp NextPixel
            Write5:
                vmovups %xmm10, (%rdi)
                vextractf128 $1, %ymm10, %xmm9
                vmovss %xmm9, 16(%rdi)
                addq $20, %rdi
                jmp NextPixel
            Write6:
                vmovups %xmm10, (%rdi)
                vextractf128 $1, %ymm10, %xmm9
                vmovsd %xmm9, 16(%rdi)
                addq $24, %rdi
                jmp NextPixel
            Write7:
                vmovups %xmm10, (%rdi)
                vextractf128 $1, %ymm10, %xmm9
                vmovsd %xmm9, 16(%rdi)
                movhlps %xmm9, %xmm9
                vmovss %xmm9, 24(%rdi)
                addq $28, %rdi
                jmp NextPixel
            Write8:
                vmovups %ymm10, (%rdi)
                add $32, %rdi

    NextPixel:
        movq 8(%rsp), %rbp
        addq %rbp, %rsi
        movq -56(%rsp), %rax
        subq $1, %rax
        movq %rax, -56(%rsp)
        cmpq $0, %rax
        ja LoopPixel
End:
    subq $96, %rsp
    popq %rdi
    popq %rsi
    popq %rdx
    popq %rcx
    popq %r8
    popq %r9
    popq %rbp
    popq %rbx
    popq %r12
    popq %r13
    popq %r14
    popq %r15
    retq
#endif
