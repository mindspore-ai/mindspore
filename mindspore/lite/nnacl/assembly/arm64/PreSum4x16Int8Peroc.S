#ifdef __aarch64__
#include "nnacl/assembly_global.h"

    .text
    .align 5
    //.p2align 5,,15
    .global PreSum4x16Int8Peroc
#ifndef __APPLE__
    .type PreSum4x16Int8Peroc, %function
#endif

//void PreSum4x16Int8Peroc(const int8_t *src, int32_t *sum, int32_t *zp, size_t hw4, size_t ic16, int32_t oc_div4,
//                            size_t oc_res4, size_t stride);

// x0 src
// x1 sum
// x2 zp
// w3 hw4
// w4 ic16
// w5 oc_div4
// w6 oc_res4
// w7 stride

asm_function PreSum4x16Int8Peroc
 mov w8, #0

RowLoop:
  cmp w8, w3
  beq End
  add w8, w8, #4
  dup v16.4s, wzr
  mov w9, #0
  mov x16, x2

Sum:
  cmp w9, w4
  beq Mul
  add w9, w9, #16

  ld1 {v0.16b}, [x0], #16
  ld1 {v1.16b}, [x0], #16
  ld1 {v2.16b}, [x0], #16
  ld1 {v3.16b}, [x0], #16

  saddlp v4.8h, v0.16b
  saddlp v5.8h, v1.16b
  saddlp v6.8h, v2.16b
  saddlp v7.8h, v3.16b
  saddlp v0.4S, v4.8h
  saddlp v1.4S, v5.8h
  saddlp v2.4S, v6.8h
  saddlp v3.4S, v7.8h
  addv s4, v0.4S
  addv s5, v1.4S
  addv s6, v2.4S
  addv s7, v3.4S
  mov v0.s[0], v4.s[0]
  mov v0.s[1], v5.s[0]
  mov v0.s[2], v6.s[0]
  mov v0.s[3], v7.s[0]

  add v16.4s, v16.4s, v0.4s
  b Sum

Mul:
  mov x12, x1
  add x1, x1, #64
  mov w9, #0

  dup v1.4s, v16.s[0]
  dup v2.4s, v16.s[1]
  dup v3.4s, v16.s[2]
  dup v4.4s, v16.s[3]

WriteOc4:
  cmp w9, w5
  beq OcRes4
  add w9, w9, #4
  ld1 {v5.4s}, [x16], #16

  mul v16.4s, v5.4s, v1.4s
  mul v17.4s, v5.4s, v2.4s
  mul v18.4s, v5.4s, v3.4s
  mul v19.4s, v5.4s, v4.4s
  st1 {v16.4s}, [x12], #16
  st1 {v17.4s}, [x12], #16
  st1 {v18.4s}, [x12], #16
  st1 {v19.4s}, [x12], #16
  add x12, x12, x7
  b WriteOc4

OcRes4:
  cmp w6, #0
  beq RowLoop
  dup v15.4s, wzr
  cmp w6, #1
  beq OcRes4_1
  cmp w6, #2
  beq OcRes4_2
  cmp w6, #3
  beq OcRes4_3

OcRes4_1:
  ld1 {v15.s}[0], [x16]
  b OcRes4End

OcRes4_2:
  ld1 {v15.d}[0], [x16]
  b OcRes4End

OcRes4_3:
  ld1 {v15.d}[0], [x16]
  add x16, x16, #8
  ld1 {v15.s}[2], [x16]
  b OcRes4End

OcRes4End:
  mul v16.4s, v15.4s, v1.4s
  mul v17.4s, v15.4s, v2.4s
  mul v18.4s, v15.4s, v3.4s
  mul v19.4s, v15.4s, v4.4s
  st1 {v16.4s}, [x12], #16
  st1 {v17.4s}, [x12], #16
  st1 {v18.4s}, [x12], #16
  st1 {v19.4s}, [x12], #16
  b RowLoop

End:
  ret
#endif