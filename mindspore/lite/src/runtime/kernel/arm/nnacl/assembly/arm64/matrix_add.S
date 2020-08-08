
#ifdef __aarch64__
    .text
    .align 5
    //.p2align 5,,15
    .global MatrixAdd
#ifndef __APPLE__
    .type MatrixAdd, %function
#endif



//void MatrixAdd(const float* matDataA, const float* matDataB, float* matDataC,
// size_t aStride, size_t bStride, size_t cStride, size_t width, size_t height)

//Auto: x0: matDataA, x1:matDataB, x2:matDatac,
//x3:aStride, x4:bStride, x5:cStride, x6:width, x7:height

MatrixAdd:
mov x12, #4 //sizeof(float)
mul x3, x12, x3
mul x4, x12, x4
mul x5, x12, x5

loopH:
mov x8, x0
mov x9, x1
mov x10, x2

mov x11, x6

loop16LineIn:
cmp x11, #4
blt L8
sub x11, x11, #4
ld1 {v0.4s, v1.4s}, [x0], #32
ld1 {v2.4s, v3.4s}, [x1], #32

fadd v4.4s, v0.4s, v2.4s
fadd v5.4s, v1.4s, v3.4s

ld1 {v6.4s, v7.4s}, [x0], #32
ld1 {v8.4s, v9.4s}, [x1], #32

cmp x11, #4
blt loop16LineOut

loop16:
st1 {v4.4s, v5.4s}, [x2], #32
fadd v10.4s, v6.4s, v8.4s
fadd v11.4s, v7.4s, v9.4s
ld1 {v0.4s, v1.4s}, [x0], #32
ld1 {v2.4s, v3.4s}, [x1], #32

st1 {v10.4s, v11.4s}, [x2], #32
fadd v4.4s, v0.4s, v2.4s
fadd v5.4s, v1.4s, v3.4s
ld1 {v6.4s, v7.4s}, [x0], #32
ld1 {v8.4s, v9.4s}, [x1], #32

sub x11, x11, #4
cmp x11, #4
bge loop16

loop16LineOut:
st1 {v4.4s, v5.4s}, [x2], #32
fadd v10.4s, v6.4s, v8.4s
fadd v11.4s, v7.4s, v9.4s
st1 {v10.4s, v11.4s}, [x2], #32


L8:
cmp x11, #2
blt L4
ld1 {v0.4s, v1.4s}, [x0], #32
ld1 {v2.4s, v3.4s}, [x1], #32
fadd v4.4s, v0.4s, v2.4s
fadd v5.4s, v1.4s, v3.4s
sub x11, x11, #2
st1 {v4.4s, v5.4s}, [x2], #32


cmp x11, #0
beq loop16EndLine

L4:
ld1 {v0.4s}, [x0], #16
ld1 {v1.4s}, [x1], #16
fadd v0.4s, v0.4s, v1.4s
sub x11, x11, #1
st1 {v0.4s}, [x2], #16
//bne L4

loop16EndLine:
add x0, x8, x3
add x1, x9, x4
add x2, x10, x5

subs x7, x7, #1
bne loopH

ret
#endif
