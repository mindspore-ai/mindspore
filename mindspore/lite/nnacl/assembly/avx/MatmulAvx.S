#ifdef ENABLE_AVX
    .text
    .align 4
    .global MatmulFloatAvxOpt
#ifndef __APPLE__
#ifndef WIN32
    .type MatmulFloatAvxOpt, %function
#endif
#endif

// void MatmulFloatAvxOpt(const float *a, const float *b, float *c, const float *bias, int act_type, int depth
//                        int row, int col, size_t stride, size_t writeMode)
// parameters pass in Linux x86 platform:
// rdi: a
// rsi: b
// rdx: c
// rcx: bias
// r8: act_type
// r9: depth
// 8: row
// 16: col
// 24: stride
// 32: writeNhwc/writeWino

// parameters pass in win x64 platform: "shadow space" needs to be opened up for first four parameters ==> 32 bites
// rcx: a
// rdx: b
// r8: c
// r9: bias
// 40: act_type
// 48: depth
// 56: row
// 64: col
// 72: stride
// 80: writeMode

MatmulFloatAvxOpt:
    // rbx, rsp, rbp, r12-r15 must be saved according to x86 calling convention
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
    pushq %rsi  // -104  rsi
    pushq %rdi  // -112  rdi
    addq $112, %rsp
#ifdef WIN32
    movq %rcx, %rdi
    movq %rdx, %rsi
    movq %r8, %rdx
    movq %r9, %rcx
    movq 40(%rsp), %r8  // act_type
    movq 48(%rsp), %r9  // depth
    movq %r9, -56(%rsp)  // r9
    movq %rcx, -72(%rsp)  // rcx
    movq %rdx, -80(%rsp)  // rdx
    movq %rsi, -88(%rsp)  // rsi
    movq %rdi, -96(%rsp)  // rdi

    movq 56(%rsp), %rbp // row
    movq %rbp, 8(%rsp)
    movq 64(%rsp), %rbp  // col
    movq %rbp, 16(%rsp)
    movq 72(%rsp), %rbp  // stride
    movq %rbp, 24(%rsp)
    movq 80(%rsp), %rbp  // weiteMode
    movq %rbp, 32(%rsp)
#endif
    movq 8(%rsp), %rbp
    movq 16(%rsp), %rbx
    movq 24(%rsp), %r10
    movq 32(%rsp), %r14

    movq $24, %r11
    imul %r9, %r11
    cmpq $0, %r14
    jne NoC8Steps
    movq $48, %r13
    imul %rbp, %r13
NoC8Steps:
    cmpq $2, %r14
    jne NoWinoSteps
    movq $4, %r12
    imul %r10, %r12
    imul %rbx, %r12
    movq $32, %r13
    imul %r10, %r13
NoWinoSteps:
    movq $4, %rax
    imul %rax, %r10

LoopRow:
    movq -88(%rsp), %rsi
    movq 16(%rsp), %rbx
    movq -72(%rsp), %rcx

    LoopCol:
        cmpq $0, %r14
        je NoReloadDst
        movq -80(%rsp), %rdx
    NoReloadDst:
        movq -96(%rsp), %rdi
        movq -56(%rsp), %r9

        vmovups (%rsi), %ymm0
        vmovups 32(%rsi), %ymm1
        vbroadcastss (%rdi), %ymm10
        vbroadcastss 4(%rdi), %ymm11
        vbroadcastss 8(%rdi), %ymm12
        vbroadcastss 12(%rdi), %ymm13
        vbroadcastss 16(%rdi), %ymm2
        vbroadcastss 20(%rdi), %ymm3
        addq $64, %rsi
        vmulps %ymm0, %ymm10, %ymm4
        vmulps %ymm1, %ymm10, %ymm5
        vmulps %ymm0, %ymm11, %ymm6
        vmulps %ymm1, %ymm11, %ymm7
        vmulps %ymm0, %ymm12, %ymm8
        vmulps %ymm1, %ymm12, %ymm9
        vmulps %ymm0, %ymm13, %ymm10
        vmulps %ymm1, %ymm13, %ymm11
        add $24, %rdi
        vmulps %ymm0, %ymm2, %ymm12
        vmulps %ymm1, %ymm2, %ymm13
        vmulps %ymm0, %ymm3, %ymm14
        vmulps %ymm1, %ymm3, %ymm15

        subq $1, %r9
        cmpq $0, %r9
        je Bias

        LoopDepth:
            vmovups (%rsi), %ymm0
            vmovups 32(%rsi), %ymm1
            vbroadcastss (%rdi), %ymm2
            vbroadcastss 4(%rdi), %ymm3
            vfmadd231ps %ymm0, %ymm2, %ymm4
            addq $64, %rsi
            vfmadd231ps %ymm1, %ymm2, %ymm5
            vbroadcastss 8(%rdi), %ymm2
            vfmadd231ps %ymm0, %ymm3, %ymm6
            vfmadd231ps %ymm1, %ymm3, %ymm7
            vbroadcastss 12(%rdi), %ymm3
            vfmadd231ps %ymm0, %ymm2, %ymm8
            prefetcht0 384(%rsi)
            vfmadd231ps %ymm1, %ymm2, %ymm9
            vbroadcastss 16(%rdi), %ymm2
            vfmadd231ps %ymm0, %ymm3, %ymm10
            vfmadd231ps %ymm1, %ymm3, %ymm11
            vbroadcastss 20(%rdi), %ymm3
            vfmadd231ps %ymm0, %ymm2, %ymm12
            vfmadd231ps %ymm1, %ymm2, %ymm13
            addq $24, %rdi
            vfmadd231ps %ymm0, %ymm3, %ymm14
            vfmadd231ps %ymm1, %ymm3, %ymm15

            subq $1, %r9
            cmpq $0, %r9
            ja LoopDepth

        Bias:
            cmpq $0, %rcx
            je Activation
            vmovups (%rcx), %ymm0
            vmovups 32(%rcx), %ymm1
            add $64, %rcx
            vaddps %ymm0, %ymm4, %ymm4
            vaddps %ymm1, %ymm5, %ymm5
            vaddps %ymm0, %ymm6, %ymm6
            vaddps %ymm1, %ymm7, %ymm7
            vaddps %ymm0, %ymm8, %ymm8
            vaddps %ymm1, %ymm9, %ymm9
            vaddps %ymm0, %ymm10, %ymm10
            vaddps %ymm1, %ymm11, %ymm11
            vaddps %ymm0, %ymm12, %ymm12
            vaddps %ymm1, %ymm13, %ymm13
            vaddps %ymm0, %ymm14, %ymm14
            vaddps %ymm1, %ymm15, %ymm15

        Activation:
            cmpq $3, %r8
            je Relu6
            cmpq $1, %r8
            je Relu
            jmp Write

        Relu6:
            movq $6, %rax
            vcvtsi2ss %rax, %xmm0, %xmm0
            vshufps $0, %xmm0, %xmm0, %xmm0
            vinsertf128 $1, %xmm0, %ymm0, %ymm0
            vminps %ymm0, %ymm4, %ymm4
            vminps %ymm0, %ymm5, %ymm5
            vminps %ymm0, %ymm6, %ymm6
            vminps %ymm0, %ymm7, %ymm7
            vminps %ymm0, %ymm8, %ymm8
            vminps %ymm0, %ymm9, %ymm9
            vminps %ymm0, %ymm10, %ymm10
            vminps %ymm0, %ymm11, %ymm11
            vminps %ymm0, %ymm12, %ymm12
            vminps %ymm0, %ymm13, %ymm13
            vminps %ymm0, %ymm14, %ymm14
            vminps %ymm0, %ymm15, %ymm15

        Relu:
            vxorps %ymm1, %ymm1, %ymm1            
            vmaxps %ymm1, %ymm4, %ymm4
            vmaxps %ymm1, %ymm5, %ymm5
            vmaxps %ymm1, %ymm6, %ymm6
            vmaxps %ymm1, %ymm7, %ymm7
            vmaxps %ymm1, %ymm8, %ymm8
            vmaxps %ymm1, %ymm9, %ymm9
            vmaxps %ymm1, %ymm10, %ymm10
            vmaxps %ymm1, %ymm11, %ymm11
            vmaxps %ymm1, %ymm12, %ymm12
            vmaxps %ymm1, %ymm13, %ymm13
            vmaxps %ymm1, %ymm14, %ymm14
            vmaxps %ymm1, %ymm15, %ymm15

        Write:
            cmpq $2, %r14
            je WriteWino
            cmpq $0, %r14
            je WriteC8
            cmpq $1, %rbx
            je Write1
            cmpq $2, %rbx
            je Write2
            cmpq $3, %rbx
            je Write3
            cmpq $4, %rbx
            je Write4
            cmpq $5, %rbx
            je Write5
            cmpq $6, %rbx
            je Write6
            cmpq $7, %rbx
            je Write7
            cmpq $8, %rbx
            je Write8
            cmpq $9, %rbx
            je Write9
            cmpq $10, %rbx
            je Write10
            cmpq $11, %rbx
            je Write11
            cmpq $12, %rbx
            je Write12
            cmpq $13, %rbx
            je Write13
            cmpq $14, %rbx
            je Write14
            cmpq $15, %rbx
            je Write15
            jmp Write16

        Write1:
            movq %rdx, %rax
            addq $4, %rax
            movq %rax, -80(%rsp)
            vmovss %xmm4, (%rdx)
            cmpq $1, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovss %xmm6, (%rdx)
            cmpq $2, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovss %xmm8, (%rdx)
            cmpq $3, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovss %xmm10, (%rdx)
            cmpq $4, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovss %xmm12, (%rdx)
            cmpq $5, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovss %xmm14, (%rdx)
            addq %r10, %rdx
            addq $4, %rdx
            jmp WriteEnd
        Write2:
            movq %rdx, %rax
            addq $8, %rax
            movq %rax, -80(%rsp)
            vmovsd %xmm4, (%rdx)
            cmpq $1, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovsd %xmm6, (%rdx)
            cmpq $2, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovsd %xmm8, (%rdx)
            cmpq $3, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovsd %xmm10, (%rdx)
            cmpq $4, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovsd %xmm12, (%rdx)
            cmpq $5, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovsd %xmm14, (%rdx)
            addq %r10, %rdx
            addq $8, %rdx
            jmp WriteEnd
        Write3:
            movq %rdx, %rax
            addq $12, %rax
            movq %rax, -80(%rsp)
            vmovsd %xmm4, (%rdx)
            movhlps %xmm4, %xmm4
            vmovss %xmm4, 8(%rdx)
            cmpq $1, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovsd %xmm6, (%rdx)
            movhlps %xmm6, %xmm6
            vmovss %xmm6, 8(%rdx)
            cmpq $2, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovsd %xmm8, (%rdx)
            movhlps %xmm8, %xmm8
            vmovss %xmm8, 8(%rdx)
            cmpq $3, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovsd %xmm10, (%rdx)
            movhlps %xmm10, %xmm10
            vmovss %xmm10, 8(%rdx)
            cmpq $4, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovsd %xmm12, (%rdx)
            movhlps %xmm12, %xmm12
            vmovss %xmm12, 8(%rdx)
            cmpq $5, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovsd %xmm14, (%rdx)
            movhlps %xmm14, %xmm14
            vmovss %xmm14, 8(%rdx)
            addq %r10, %rdx
            addq $12, %rdx
            jmp WriteEnd
        Write4:
            movq %rdx, %rax
            addq $16, %rax
            movq %rax, -80(%rsp)
            vmovups %xmm4, (%rdx)
            cmpq $1, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %xmm6, (%rdx)
            cmpq $2, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %xmm8, (%rdx)
            cmpq $3, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %xmm10, (%rdx)
            cmpq $4, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %xmm12, (%rdx)
            cmpq $5, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %xmm14, (%rdx)
            addq %r10, %rdx
            addq $16, %rdx
            jmp WriteEnd
        Write5:
            movq %rdx, %rax
            addq $20, %rax
            movq %rax, -80(%rsp)
            vmovups %xmm4, (%rdx)
            vextractf128 $1, %ymm4, %xmm4
            vmovss %xmm4, 16(%rdx)
            cmpq $1, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %xmm6, (%rdx)
            vextractf128 $1, %ymm6, %xmm6
            vmovss %xmm6, 16(%rdx)
            cmpq $2, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %xmm8, (%rdx)
            vextractf128 $1, %ymm8, %xmm8
            vmovss %xmm8, 16(%rdx)
            cmpq $3, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %xmm10, (%rdx)
            vextractf128 $1, %ymm10, %xmm10
            vmovss %xmm10, 16(%rdx)
            cmpq $4, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %xmm12, (%rdx)
            vextractf128 $1, %ymm12, %xmm12
            vmovss %xmm12, 16(%rdx)
            cmpq $5, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %xmm14, (%rdx)
            vextractf128 $1, %ymm14, %xmm14
            vmovss %xmm14, 16(%rdx)
            addq %r10, %rdx
            addq $20, %rdx
            jmp WriteEnd
        Write6:
            movq %rdx, %rax
            addq $24, %rax
            movq %rax, -80(%rsp)
            vmovups %xmm4, (%rdx)
            vextractf128 $1, %ymm4, %xmm4
            vmovsd %xmm4, 16(%rdx)
            cmpq $1, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %xmm6, (%rdx)
            vextractf128 $1, %ymm6, %xmm6
            vmovsd %xmm6, 16(%rdx)
            cmpq $2, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %xmm8, (%rdx)
            vextractf128 $1, %ymm8, %xmm8
            vmovsd %xmm8, 16(%rdx)
            cmpq $3, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %xmm10, (%rdx)
            vextractf128 $1, %ymm10, %xmm10
            vmovsd %xmm10, 16(%rdx)
            cmpq $4, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %xmm12, (%rdx)
            vextractf128 $1, %ymm12, %xmm12
            vmovsd %xmm12, 16(%rdx)
            cmpq $5, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %xmm14, (%rdx)
            vextractf128 $1, %ymm14, %xmm14
            vmovsd %xmm14, 16(%rdx)
            addq %r10, %rdx
            addq $24, %rdx
            jmp WriteEnd
        Write7:
            movq %rdx, %rax
            addq $28, %rax
            movq %rax, -80(%rsp)
            vmovups %xmm4, (%rdx)
            vextractf128 $1, %ymm4, %xmm4
            vmovsd %xmm4, 16(%rdx)
            movhlps %xmm4, %xmm4
            vmovss %xmm4, 24(%rdx)
            cmpq $1, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %xmm6, (%rdx)
            vextractf128 $1, %ymm6, %xmm6
            vmovsd %xmm6, 16(%rdx)
            movhlps %xmm6, %xmm6
            vmovss %xmm6, 24(%rdx)
            cmpq $2, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %xmm8, (%rdx)
            vextractf128 $1, %ymm8, %xmm8
            vmovsd %xmm8, 16(%rdx)
            movhlps %xmm8, %xmm8
            vmovss %xmm8, 24(%rdx)
            cmpq $3, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %xmm10, (%rdx)
            vextractf128 $1, %ymm10, %xmm10
            vmovsd %xmm10, 16(%rdx)
            movhlps %xmm10, %xmm10
            vmovss %xmm10, 24(%rdx)
            cmpq $4, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %xmm12, (%rdx)
            vextractf128 $1, %ymm12, %xmm12
            vmovsd %xmm12, 16(%rdx)
            movhlps %xmm12, %xmm12
            vmovss %xmm12, 24(%rdx)
            cmpq $5, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %xmm14, (%rdx)
            vextractf128 $1, %ymm14, %xmm14
            vmovsd %xmm14, 16(%rdx)
            movhlps %xmm14, %xmm14
            vmovss %xmm14, 24(%rdx)
            addq %r10, %rdx
            addq $28, %rdx
            jmp WriteEnd
        Write8:
            movq %rdx, %rax
            addq $32, %rax
            movq %rax, -80(%rsp)
            vmovups %ymm4, (%rdx)
            cmpq $1, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm6, (%rdx)
            cmpq $2, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm8, (%rdx)
            cmpq $3, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm10, (%rdx)
            cmpq $4, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm12, (%rdx)
            cmpq $5, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm14, (%rdx)
            addq %r10, %rdx
            addq $32, %rdx
            jmp WriteEnd
        Write9:
            movq %rdx, %rax
            addq $36, %rax
            movq %rax, -80(%rsp)
            vmovups %ymm4, (%rdx)
            vmovss %xmm5, 32(%rdx)
            cmpq $1, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm6, (%rdx)
            vmovss %xmm7, 32(%rdx)
            cmpq $2, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm8, (%rdx)
            vmovss %xmm9, 32(%rdx)
            cmpq $3, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm10, (%rdx)
            vmovss %xmm11, 32(%rdx)
            cmpq $4, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm12, (%rdx)
            vmovss %xmm13, 32(%rdx)
            cmpq $5, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm14, (%rdx)
            vmovss %xmm15, 32(%rdx)
            addq %r10, %rdx
            addq $36, %rdx
            jmp WriteEnd
        Write10:
            movq %rdx, %rax
            addq $40, %rax
            movq %rax, -80(%rsp)
            vmovups %ymm4, (%rdx)
            vmovsd %xmm5, 32(%rdx)
            cmpq $1, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm6, (%rdx)
            vmovsd %xmm7, 32(%rdx)
            cmpq $2, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm8, (%rdx)
            vmovsd %xmm9, 32(%rdx)
            cmpq $3, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm10, (%rdx)
            vmovsd %xmm11, 32(%rdx)
            cmpq $4, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm12, (%rdx)
            vmovsd %xmm13, 32(%rdx)
            cmpq $5, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm14, (%rdx)
            vmovsd %xmm15, 32(%rdx)
            addq %r10, %rdx
            addq $40, %rdx
            jmp WriteEnd
        Write11:
            movq %rdx, %rax
            addq $44, %rax
            movq %rax, -80(%rsp)
            vmovups %ymm4, (%rdx)
            vmovsd %xmm5, 32(%rdx)
            movhlps %xmm5, %xmm5
            vmovss %xmm5, 40(%rdx)
            cmpq $1, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm6, (%rdx)
            vmovsd %xmm7, 32(%rdx)
            movhlps %xmm7, %xmm7
            vmovss %xmm7, 40(%rdx)
            cmpq $2, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm8, (%rdx)
            vmovsd %xmm9, 32(%rdx)
            movhlps %xmm9, %xmm9
            vmovss %xmm9, 40(%rdx)
            cmpq $3, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm10, (%rdx)
            vmovsd %xmm11, 32(%rdx)
            movhlps %xmm11, %xmm11
            vmovss %xmm11, 40(%rdx)
            cmpq $4, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm12, (%rdx)
            vmovsd %xmm13, 32(%rdx)
            movhlps %xmm13, %xmm13
            vmovss %xmm13, 40(%rdx)
            cmpq $5, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm14, (%rdx)
            vmovsd %xmm15, 32(%rdx)
            movhlps %xmm15, %xmm15
            vmovss %xmm15, 40(%rdx)
            addq %r10, %rdx
            addq $44, %rdx
            jmp WriteEnd
        Write12:
            movq %rdx, %rax
            addq $48, %rax
            movq %rax, -80(%rsp)
            vmovups %ymm4, (%rdx)
            vmovups %xmm5, 32(%rdx)
            cmpq $1, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm6, (%rdx)
            vmovups %xmm7, 32(%rdx)
            cmpq $2, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm8, (%rdx)
            vmovups %xmm9, 32(%rdx)
            cmpq $3, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm10, (%rdx)
            vmovups %xmm11, 32(%rdx)
            cmpq $4, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm12, (%rdx)
            vmovups %xmm13, 32(%rdx)
            cmpq $5, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm14, (%rdx)
            vmovups %xmm15, 32(%rdx)
            addq %r10, %rdx
            addq $48, %rdx
            jmp WriteEnd
        Write13:
            movq %rdx, %rax
            addq $52, %rax
            movq %rax, -80(%rsp)
            vmovups %ymm4, (%rdx)
            vmovups %xmm5, 32(%rdx)
            vextractf128 $1, %ymm5, %xmm5
            vmovss %xmm5, 48(%rdx)
            cmpq $1, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm6, (%rdx)
            vmovups %xmm7, 32(%rdx)
            vextractf128 $1, %ymm7, %xmm7
            vmovss %xmm7, 48(%rdx)
            cmpq $2, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm8, (%rdx)
            vmovups %xmm9, 32(%rdx)
            vextractf128 $1, %ymm9, %xmm9
            vmovss %xmm9, 48(%rdx)
            cmpq $3, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm10, (%rdx)
            vmovups %xmm11, 32(%rdx)
            vextractf128 $1, %ymm11, %xmm11
            vmovss %xmm11, 48(%rdx)
            cmpq $4, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm12, (%rdx)
            vmovups %xmm13, 32(%rdx)
            vextractf128 $1, %ymm13, %xmm13
            vmovss %xmm13, 48(%rdx)
            cmpq $5, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm14, (%rdx)
            vmovups %xmm15, 32(%rdx)
            vextractf128 $1, %ymm15, %xmm15
            vmovss %xmm15, 48(%rdx)
            addq %r10, %rdx
            addq $52, %rdx
            jmp WriteEnd
        Write14:
            movq %rdx, %rax
            addq $56, %rax
            movq %rax, -80(%rsp)
            vmovups %ymm4, (%rdx)
            vmovups %xmm5, 32(%rdx)
            vextractf128 $1, %ymm5, %xmm5
            vmovsd %xmm5, 48(%rdx)
            cmpq $1, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm6, (%rdx)
            vmovups %xmm7, 32(%rdx)
            vextractf128 $1, %ymm7, %xmm7
            vmovsd %xmm7, 48(%rdx)
            cmpq $2, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm8, (%rdx)
            vmovups %xmm9, 32(%rdx)
            vextractf128 $1, %ymm9, %xmm9
            vmovsd %xmm9, 48(%rdx)
            cmpq $3, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm10, (%rdx)
            vmovups %xmm11, 32(%rdx)
            vextractf128 $1, %ymm11, %xmm11
            vmovsd %xmm11, 48(%rdx)
            cmpq $4, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm12, (%rdx)
            vmovups %xmm13, 32(%rdx)
            vextractf128 $1, %ymm13, %xmm13
            vmovsd %xmm13, 48(%rdx)
            cmpq $5, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm14, (%rdx)
            vmovups %xmm15, 32(%rdx)
            vextractf128 $1, %ymm15, %xmm15
            vmovsd %xmm15, 48(%rdx)
            addq %r10, %rdx
            addq $56, %rdx
            jmp WriteEnd
        Write15:
            movq %rdx, %rax
            addq $60, %rax
            movq %rax, -80(%rsp)
            vmovups %ymm4, (%rdx)
            vmovups %xmm5, 32(%rdx)
            vextractf128 $1, %ymm5, %xmm5
            vmovsd %xmm5, 48(%rdx)
            movhlps %xmm5, %xmm5
            vmovss %xmm5, 56(%rdx)
            cmpq $1, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm6, (%rdx)
            vmovups %xmm7, 32(%rdx)
            vextractf128 $1, %ymm7, %xmm7
            vmovsd %xmm7, 48(%rdx)
            movhlps %xmm7, %xmm7
            vmovss %xmm7, 56(%rdx)
            cmpq $2, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm8, (%rdx)
            vmovups %xmm9, 32(%rdx)
            vextractf128 $1, %ymm9, %xmm9
            vmovsd %xmm9, 48(%rdx)
            movhlps %xmm9, %xmm9
            vmovss %xmm9, 56(%rdx)
            cmpq $3, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm10, (%rdx)
            vmovups %xmm11, 32(%rdx)
            vextractf128 $1, %ymm11, %xmm11
            vmovsd %xmm11, 48(%rdx)
            movhlps %xmm11, %xmm11
            vmovss %xmm11, 56(%rdx)
            cmpq $4, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm12, (%rdx)
            vmovups %xmm13, 32(%rdx)
            vextractf128 $1, %ymm13, %xmm13
            vmovsd %xmm13, 48(%rdx)
            movhlps %xmm13, %xmm13
            vmovss %xmm13, 56(%rdx)
            cmpq $5, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm14, (%rdx)
            vmovups %xmm15, 32(%rdx)
            vextractf128 $1, %ymm15, %xmm15
            vmovsd %xmm15, 48(%rdx)
            movhlps %xmm15, %xmm15
            vmovss %xmm15, 56(%rdx)
            addq %r10, %rdx
            addq $60, %rdx
            jmp WriteEnd
        WriteC8:
            movq %rdx, %rax
            addq %r11, %rdx
            movq %rdx, %r15
            addq %r11, %rdx
            movq %rdx, -80(%rsp)
            vmovups %ymm4, (%rax)
            vmovups %ymm6, 32(%rax)
            vmovups %ymm8, 64(%rax)
            vmovups %ymm10, 96(%rax)
            vmovups %ymm12, 128(%rax)
            vmovups %ymm14, 160(%rax)
            vmovups %ymm5, (%r15)
            vmovups %ymm7, 32(%r15)
            vmovups %ymm9, 64(%r15)
            vmovups %ymm11, 96(%r15)
            vmovups %ymm13, 128(%r15)
            vmovups %ymm15, 160(%r15)
            jmp WriteEnd
        WriteWino:
            movq %rdx, %rax
            addq %r13, %rax
            movq %rax, -80(%rsp)
            vmovups %ymm4, (%rdx)
            addq %r12, %rdx
            vmovups %ymm6, (%rdx)
            addq %r12, %rdx
            vmovups %ymm8, (%rdx)
            addq %r12, %rdx
            vmovups %ymm10, (%rdx)
            addq %r12, %rdx
            vmovups %ymm12, (%rdx)
            addq %r12, %rdx
            vmovups %ymm14, (%rdx)
            cmpq $8, %rbx
            je WriteEnd
            movq %rax, %rdx
            addq %r13, %rax
            movq %rax, -80(%rsp)
            vmovups %ymm5, (%rdx)
            addq %r12, %rdx
            vmovups %ymm7, (%rdx)
            addq %r12, %rdx
            vmovups %ymm9, (%rdx)
            addq %r12, %rdx
            vmovups %ymm11, (%rdx)
            addq %r12, %rdx
            vmovups %ymm13, (%rdx)
            addq %r12, %rdx
            vmovups %ymm15, (%rdx)
            jmp WriteEnd
        Write16:
            movq %rdx, %rax
            addq $64, %rax
            movq %rax, -80(%rsp)
            vmovups %ymm4, (%rdx)
            vmovups %ymm5, 32(%rdx)
            cmpq $1, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm6, (%rdx)
            vmovups %ymm7, 32(%rdx)
            cmpq $2, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm8, (%rdx)
            vmovups %ymm9, 32(%rdx)
            cmpq $3, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm10, (%rdx)
            vmovups %ymm11, 32(%rdx)
            cmpq $4, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm12, (%rdx)
            vmovups %ymm13, 32(%rdx)
            cmpq $5, %rbp
            je WriteEnd
            addq %r10, %rdx
            vmovups %ymm14, (%rdx)
            vmovups %ymm15, 32(%rdx)
            addq %r10, %rdx
            addq $64, %rdx

        WriteEnd:
            cmpq $16, %rbx
            jbe LoopColEnd
            subq $16, %rbx
            jmp LoopCol

    LoopColEnd:
        movq -96(%rsp), %rdi
        addq %r11, %rdi
        movq %rdi, -96(%rsp)
        cmpq $0, %r14
        je C8DstStep
        cmpq $2, %r14
        je WinoDstStep
        movq $4, %rax
        movq 16(%rsp), %rbx
        imul %rbx, %rax
        subq %rax, %rdx
        movq %rdx, -80(%rsp)
        jmp NoDstStep
    C8DstStep:
        movq -80(%rsp), %rax
        addq $384, %rax
        movq %rax, -80(%rsp)
        jmp NoDstStep
    WinoDstStep:
        addq %r13, %rdx
        movq %rdx, -80(%rsp)
    NoDstStep:
        cmpq $6, %rbp
        jbe LoopRowEnd
        subq $6, %rbp
        jmp LoopRow

LoopRowEnd:
    subq $112, %rsp
    popq %rdi
    popq %rsi
    popq %rdx
    popq %rdx
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
