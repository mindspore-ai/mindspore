#ifdef ENABLE_AVX
    .text
    .align 4
    .global ConvDwFp32Row
#ifndef __APPLE__
#ifndef WIN32
    .type ConvDwFp32Row, %function
#endif
#endif

// void ConvDwFp32Row(float *output_ptr, const float *input_tmp, const float *weight_ptr, size_t num_pixels,
//                    size_t output_channel, size_t input_step);
// in linux x64 platform:
// rdi: output_ptr
// rsi: input_ptr
// rdx: weight_ptr
// rcx: num_pixels
// r8: output_channel
// r9: input_step

// in win x64 platform: "shadow space" needs to be opened up for first four parameters ==> 32 bites
// rcx: output_ptr
// rdx: input_ptr
// r8: weight_ptr
// r9: num_pixels
// 40: output_channel
// 48: input_step

ConvDwFp32Row:
    pushq %r15
    pushq %r14
    pushq %r13
    pushq %r12
    pushq %rsi
    pushq %rdi
    addq $48, %rsp

#ifdef WIN32
    movq %rcx, %rdi // output_ptr
    movq %rdx, %rsi // input_ptr
    movq %r8, %rdx // weight_ptr
    movq %r9, %rcx // num_pixels
    movq 40(%rsp), %r8 // output_channel
    movq 48(%rsp), %r9 // input_step
#endif

    movq $4, %r13
    imul %r13, %r9
    movq %rsi, %r13 // input_ptr
    movq %rdx, %r14 // weight_ptr
    movq %r8, %r15 // output_channel
    cmpq $0, %rcx
    je End

    LoopPixel:
        movq %r13, %rsi // input_tmp
        movq %r14, %rdx // weight_tmp
        movq %r15, %r8 // channel_tmp
        
        cmpq $32, %r8
        jae LoopC32
        cmpq $16, %r8
        jae LoopC16
        cmpq $8, %r8
        jae LoopC8
        cmpq $0, %r8
        ja LoopC
        jmp LoopCEnd
        
        LoopC32:
            vmovups (%rsi), %ymm0 // input_tmp
            vmovups 32(%rsi), %ymm1
            vmovups 64(%rsi), %ymm2
            vmovups 96(%rsi), %ymm3

            vmovups (%rdi), %ymm8 // output_tmp
            vmovups 32(%rdi), %ymm9
            vmovups 64(%rdi), %ymm10
            vmovups 96(%rdi), %ymm11

            addq $128, %rsi
            vfmadd231ps (%rdx), %ymm0, %ymm8
            vfmadd231ps 32(%rdx), %ymm1, %ymm9
            vfmadd231ps 64(%rdx), %ymm2, %ymm10
            vfmadd231ps 96(%rdx), %ymm3, %ymm11
            
            vmovups %ymm8, (%rdi) // output_ptr
            vmovups %ymm9, 32(%rdi)
            vmovups %ymm10, 64(%rdi)
            vmovups %ymm11, 96(%rdi)
            addq $128, %rdi
            addq $128, %rdx

            subq $32, %r8
            cmpq $32, %r8
            jae LoopC32
            cmpq $16, %r8
            jae LoopC16
            cmpq $8, %r8
            jae LoopC8
            cmpq $0, %r8
            ja LoopC
            jmp LoopCEnd

        LoopC16:
            vmovups (%rsi), %ymm0 // input_tmp
            vmovups (%rdi), %ymm8 // output_tmp
            vmovups 32(%rsi), %ymm1
            vmovups 32(%rdi), %ymm9
            addq $64, %rsi

            vfmadd231ps (%rdx), %ymm0, %ymm8
            vfmadd231ps 32(%rdx), %ymm1, %ymm9

            vmovups %ymm8, (%rdi) // output_ptr
            addq $64, %rdx
            vmovups %ymm9, 32(%rdi)
            addq $64, %rdi

            subq $16, %r8
            cmpq $16, %r8
            jae LoopC16
            cmpq $8, %r8
            jae LoopC8
            cmpq $0, %r8
            ja LoopC
            jmp LoopCEnd

        LoopC8:
            vmovups (%rsi), %ymm0 // input_tmp
            vmovups (%rdi), %ymm8 // output_tmp
            addq $32, %rsi

            vfmadd231ps (%rdx), %ymm0, %ymm8

            addq $32, %rdx
            vmovups %ymm8, (%rdi)
            addq $32, %rdi

            subq $8, %r8
            cmpq $8, %r8
            jae LoopC8
            cmpq $0, %r8
            ja LoopC
            jmp LoopCEnd

        LoopC:
            vmovss (%rsi), %xmm0  // input_tmp
            vmovss (%rdi), %xmm8  // output_ptr

            vfmadd231ss (%rdx), %xmm0, %xmm8

            addq $4, %rsi
            addq $4, %rdx
            vmovss %xmm8, (%rdi)
            addq $4, %rdi

            subq $1, %r8
            cmpq $0, %r8
            ja LoopC
            jmp LoopCEnd

        LoopCEnd:
            subq $1, %rcx // num_pixel -= 1
            cmpq $0, %rcx
            je End
            addq %r9, %r13
            jmp LoopPixel
End:
    subq $48, %rsp
    popq %rdi
    popq %rsi
    popq %r12
    popq %r13
    popq %r14
    popq %r15
    retq
#endif
