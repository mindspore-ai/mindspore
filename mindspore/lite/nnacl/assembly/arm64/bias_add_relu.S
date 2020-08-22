
#ifdef __aarch64__
    .text
    .align 5
    //.p2align 5,,15
    .global BiasAddRelu
#ifndef __APPLE__
    .type BiasAddRelu, %function
#endif


//void BiasAddRelu(const float* bias, float* data, size_t oc4, size_t plan_size)

//Auto: x0:bias, x1: data, x2:oc4,x3: plan_size,

BiasAddRelu:
cmp x2, #0
beq BiasAddEnd

cmp x3, #0
beq BiasAddEnd

dup v16.4s, wzr

LoopOc4:
ld1 {v0.4s}, [x0], #16
mov x6, x3
mov x5, x1

Loop16LineIn:
cmp x6, #4
blt L4
sub x6, x6, #4

ld1 {v1.4s, v2.4s}, [x5], #32

fadd v21.4s, v0.4s, v1.4s
fadd v22.4s, v0.4s, v2.4s
ld1 {v3.4s, v4.4s}, [x5], #32

fmax v23.4s, v21.4s, v16.4s
fmax v24.4s, v22.4s, v16.4s

cmp x6, #4
blt Loop16LineOut

Loop16:
st1 {v23.4s, v24.4s}, [x1], #32
fadd v25.4s, v0.4s, v3.4s
fadd v26.4s, v0.4s, v4.4s
ld1 {v1.4s, v2.4s}, [x5], #32

fmax v27.4s, v25.4s, v16.4s
fmax v28.4s, v26.4s, v16.4s
fadd v21.4s, v0.4s, v1.4s
fadd v22.4s, v0.4s, v2.4s

st1 {v27.4s, v28.4s}, [x1], #32
ld1 {v3.4s, v4.4s}, [x5], #32
fmax v23.4s, v21.4s, v16.4s
fmax v24.4s, v22.4s, v16.4s
sub x6, x6, #4
cmp x6, #4
bge Loop16

Loop16LineOut:
st1 {v23.4s, v24.4s}, [x1], #32
fadd v25.4s, v0.4s, v3.4s
fadd v26.4s, v0.4s, v4.4s

fmax v27.4s, v25.4s, v16.4s
fmax v28.4s, v26.4s, v16.4s
st1 {v27.4s, v28.4s}, [x1], #32

L4:
cmp x6, #0
beq Loop16LineEnd
Loop4:
ld1 {v1.4s}, [x5], #16
fadd v1.4s, v1.4s, v0.4s
fmax v1.4s, v1.4s, v16.4s

subs x6, x6, #1
st1 {v1.4s}, [x1], #16
bne Loop4

Loop16LineEnd:
subs x2, x2, #1
bne LoopOc4

BiasAddEnd:

ret
#endif
