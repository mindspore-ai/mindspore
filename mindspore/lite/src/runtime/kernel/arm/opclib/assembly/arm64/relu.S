
#ifdef __aarch64__
    .text
    .align 5
    //.p2align 5,,15
    .global Relu
#ifndef __APPLE__
    .type Relu, %function
#endif


//void Relu(float* data, size_t element4)

//Auto: x0:data, x1: element4

Relu:
cmp x1, #0
beq ReluEnd

dup v16.4s, wzr

mov x5, x0

Loop16LineIn:
cmp x1, #4
blt L4
sub x1, x1, #4

ld1 {v1.4s, v2.4s}, [x5], #32

fmax v5.4s, v16.4s, v1.4s
fmax v6.4s, v16.4s, v2.4s
ld1 {v3.4s, v4.4s}, [x5], #32

cmp x1, #4
blt Loop16LineOut

Loop16:
st1 {v5.4s, v6.4s}, [x0], #32
fmax v7.4s, v16.4s, v3.4s
fmax v8.4s, v16.4s, v4.4s
ld1 {v1.4s, v2.4s}, [x5], #32

st1 {v7.4s, v8.4s}, [x0], #32
fmax v5.4s, v16.4s, v1.4s
fmax v6.4s, v16.4s, v2.4s
ld1 {v3.4s, v4.4s}, [x5], #32

sub x1, x1, #4
cmp x1, #4
bge Loop16

Loop16LineOut:
st1 {v5.4s, v6.4s}, [x0], #32
fmax v7.4s, v16.4s, v3.4s
fmax v8.4s, v16.4s, v4.4s

st1 {v7.4s, v8.4s}, [x0], #32

L4:
cmp x1, #0
beq ReluEnd
Loop4:
ld1 {v1.4s}, [x5], #16
fmax v2.4s, v16.4s, v0.4s
subs x1, x1, #1
st1 {v2.4s}, [x0], #16
bne Loop4

ReluEnd:

ret
#endif
