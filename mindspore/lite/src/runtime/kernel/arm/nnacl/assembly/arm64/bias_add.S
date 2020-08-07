
#ifdef __aarch64__
    .text
    .align 5
    //.p2align 5,,15
    .global BiasAdd
#ifndef __APPLE__
    .type BiasAdd, %function
#endif



//void BiasAdd(const float* bias, float* data, size_t oc4, size_t plan_size)

//Auto: x0:bias, x1: data, x2:oc4,x3: plan_size,

BiasAdd:
cmp x2, #0
beq BiasAddEnd

cmp x3, #0
beq BiasAddEnd

LoopOc4:
ld1 {v0.4s}, [x0], #16
mov x6, x3
mov x5, x1

Loop16LineIn:
cmp x6, #4
blt L4
sub x6, x6, #4

ld1 {v1.4s, v2.4s}, [x5], #32

fadd v5.4s, v0.4s, v1.4s
fadd v6.4s, v0.4s, v2.4s
ld1 {v3.4s, v4.4s}, [x5], #32

cmp x6, #4
blt Loop16LineOut

Loop16:
st1 {v5.4s, v6.4s}, [x1], #32
fadd v7.4s, v0.4s, v3.4s
fadd v8.4s, v0.4s, v4.4s
ld1 {v1.4s, v2.4s}, [x5], #32

st1 {v7.4s, v8.4s}, [x1], #32
fadd v5.4s, v0.4s, v1.4s
fadd v6.4s, v0.4s, v2.4s
ld1 {v3.4s, v4.4s}, [x5], #32

sub x6, x6, #4
cmp x6, #4
bge Loop16

Loop16LineOut:
st1 {v5.4s, v6.4s}, [x1], #32
fadd v7.4s, v0.4s, v3.4s
fadd v8.4s, v0.4s, v4.4s

st1 {v7.4s, v8.4s}, [x1], #32

L4:
cmp x6, #0
beq Loop16LineEnd
Loop4:
ld1 {v1.4s}, [x5], #16
fadd v2.4s, v1.4s, v0.4s
subs x6, x6, #1
st1 {v2.4s}, [x1], #16
bne Loop4

Loop16LineEnd:
subs x2, x2, #1
bne LoopOc4

BiasAddEnd:

ret
#endif
