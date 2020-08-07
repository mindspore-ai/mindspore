
#ifdef __aarch64__
    .text
    .align 5
    //.p2align 5,,15
    .global Relu6
#ifndef __APPLE__
    .type Relu6, %function
#endif


//void Relu6(float* data, size_t element4)

//Auto: x0:data, x1: element4

Relu6:
cmp x1, #0
beq Relu6End

dup v16.4s, wzr
movi v17.4s, #6
scvtf v17.4s, v17.4s

mov x5, x0

Loop16LineIn:
cmp x1, #4
blt L4
sub x1, x1, #4

ld1 {v1.4s, v2.4s}, [x5], #32

fmax v21.4s, v1.4s, v16.4s
fmax v22.4s, v2.4s, v16.4s
ld1 {v3.4s, v4.4s}, [x5], #32

fmin v23.4s, v21.4s, v17.4s
fmin v24.4s, v22.4s, v17.4s


cmp x1, #4
blt Loop16LineOut

Loop16:
st1 {v23.4s, v24.4s}, [x0], #32
fmax v25.4s, v3.4s, v16.4s
fmax v26.4s, v4.4s, v16.4s
ld1 {v1.4s, v2.4s}, [x5], #32

fmin v27.4s, v25.4s, v17.4s
fmin v28.4s, v26.4s, v17.4s
fmax v21.4s, v1.4s, v16.4s
fmax v22.4s, v2.4s, v16.4s

st1 {v27.4s, v28.4s}, [x0], #32
ld1 {v3.4s, v4.4s}, [x5], #32
fmin v23.4s, v21.4s, v17.4s
fmin v24.4s, v22.4s, v17.4s

sub x1, x1, #4
cmp x1, #4
bge Loop16

Loop16LineOut:
st1 {v23.4s, v24.4s}, [x0], #32
fmax v25.4s, v3.4s, v16.4s
fmax v26.4s, v4.4s, v16.4s

fmin v27.4s, v25.4s, v17.4s
fmin v28.4s, v26.4s, v17.4s
st1 {v27.4s, v28.4s}, [x0], #32

L4:
cmp x1, #0
beq Relu6End
Loop4:
ld1 {v1.4s}, [x5], #16
fmax v1.4s, v1.4s, v16.4s

fmin v1.4s, v1.4s, v17.4s

subs x1, x1, #1
st1 {v1.4s}, [x0], #16
bne Loop4

Relu6End:

ret
#endif
