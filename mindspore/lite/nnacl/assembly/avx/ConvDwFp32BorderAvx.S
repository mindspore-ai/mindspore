#ifdef ENABLE_AVX
    .text
    .align 4
    .global ConvDwFp32Border
#ifndef __APPLE__
#ifndef WIN32
    .type ConvDwFp32Border, %function
#endif
#endif

// void ConvDwFp32Border(float *dst, const float *src, const float *weight, const float *bias, size_t height,
//                       size_t width, size_t in_kh_step, size_t in_kw_step, size_t kernel_w, size_t relu,
//                       size_t relu6);

ConvDwFp32Border:
    pushq %r15
    pushq %r14
    pushq %r13
    pushq %r12
    pushq %rbx
    pushq %rbp
    pushq %r9
    pushq %r8  // -64
    pushq %rcx  // -72
    pushq %rdx  // -80
    pushq %rsi
    pushq %rdi
    addq $96, %rsp

    movq %rdi, %rdx
#ifdef WIN32
    movq %rcx, %rdx
#endif
    movq 8(%rdx), %r12  // src
    movq 16(%rdx), %r13  // weight
    movq 24(%rdx), %rbp  // bias
    movq 32(%rdx), %r11  // height
    movq 40(%rdx), %r10
    movq %r10, -72(%rsp)  // width
    movq 48(%rdx), %r10
    movq %r10, -80(%rsp)  // in_kh_step
    movq 56(%rdx), %r10  // in_kw_step
    movq 64(%rdx), %rax  // kernel_w
    movq 72(%rdx), %rcx  // relu
    movq 80(%rdx), %rbx  // reul6
    movq $6, -64(%rsp)
    movq (%rdx), %rdx
    cmpq $0, %r11
    je End
    
    xorps %xmm8, %xmm8
    LoopHeight:
        movq %r12, %rsi // src_kh, src_kw
        movq %r13, %rdi // weight_kh, weight_kw
        movq -72(%rsp), %r8 // width
        
        cmpq $6, %r8
        jae LoopWidth6
        cmpq $4, %r8
        jae LoopWidth4
        cmpq $1, %r8
        jae LoopWidth1
        jmp LoopWidthEnd
        
        LoopWidth6:
            xorps %xmm6, %xmm6
            xorps %xmm7, %xmm7
            imul $3, %r10, %r9
            addq %rsi, %r9
            vmovups (%rsi), %xmm0  // src_kw
            vmovups (%rsi, %r10), %xmm1
            vmovups (%rsi, %r10, 2), %xmm2
            vmovups (%r9), %xmm3
            vmovups (%rsi, %r10, 4), %xmm4
            vmovups (%r9, %r10, 2), %xmm5
            
            vfmadd231ps (%rdi), %xmm0, %xmm6
            vfmadd231ps 16(%rdi), %xmm1, %xmm7
            vfmadd231ps 32(%rdi), %xmm2, %xmm8
            vfmadd231ps 48(%rdi), %xmm3, %xmm6
            vfmadd231ps 64(%rdi), %xmm4, %xmm7
            vfmadd231ps 80(%rdi), %xmm5, %xmm8
            
            addps %xmm6, %xmm7
            imul $6, %r10, %r15
            addq $96, %rdi
            addps %xmm7, %xmm8
            addq %r15, %rsi

            subq $6, %r8
            cmpq $6, %r8
            jae LoopWidth6
            cmpq $4, %r8
            jae LoopWidth4
            cmpq $0, %r8
            je LoopWidthEnd
            jmp LoopWidth1

        LoopWidth4:
            xorps %xmm6, %xmm6
            xorps %xmm7, %xmm7
            imul $3, %r10, %r9
            addq %rsi, %r9
            vmovups (%rsi), %xmm0  // src_kw
            vmovups (%rsi, %r10, 1), %xmm1
            vmovups (%rsi, %r10, 2), %xmm2
            vmovups (%r9), %xmm3

            vfmadd231ps (%rdi), %xmm0, %xmm6
            vfmadd231ps 16(%rdi), %xmm1, %xmm7
            vfmadd231ps 32(%rdi), %xmm2, %xmm8
            vfmadd231ps 48(%rdi), %xmm3, %xmm6

            addps %xmm6, %xmm7
            imul $4, %r10, %r15
            addq $64, %rdi
            addps %xmm7, %xmm8
            addq %r15, %rsi

            subq $4, %r8
            cmpq $4, %r8
            jae LoopWidth4
            cmpq $0, %r8
            je LoopWidthEnd
            jmp LoopWidth1
            
        LoopWidth1:
            vmovups (%rsi), %xmm0  // input_tmp
            addq %r10, %rsi
            vfmadd231ps (%rdi), %xmm0, %xmm8
            addq $16, %rdi

            subq $1, %r8
            cmpq $0, %r8
            ja LoopWidth1
            jmp LoopWidthEnd
    
        LoopWidthEnd:
            subq $1, %r11
            cmpq $0, %r11
            je LoopHeightEnd
            addq -80(%rsp), %r12  // in_kh_step
            addq %rax, %r13  // kernel_w_step
            jmp LoopHeight
    
    LoopHeightEnd:
        xorps %xmm10, %xmm10
        vbroadcastss -64(%rsp), %xmm9

        addps (%rbp), %xmm8
        cmpq $1, %rbx
        je Relu6
        cmpq $1, %rcx
        je Relu
        jmp Write
        Relu6:
            minps %xmm9, %xmm8
        Relu:
            maxps %xmm10, %xmm8
        Write:
            movups %xmm8, (%rdx)
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
