#ifdef __aarch64__
#include "nnacl/assembly_global.h"
    .text
    .align 5
    //.p2align 5,,15
    .global PreSum4x16Int8Pert
#ifndef __APPLE__
    .type PreSum4x16Int8Pert, %function
#endif

// void PreSum4x16Int8Pert(const int8_t *src, int32_t *dst, size_t row4, size_t col16, int32_t filter_zp);

// x0 src
// x1 dst
// w2 row4
// w3 co16
// w4 filter_zp

asm_function PreSum4x16Int8Pert
  dup v17.4s, w4
  mov w5, #0

RowLoop:
  cmp w5, w2
  beq End
  add w5, w5, #4
  dup v16.4s, wzr
  mov w6, #0

CalLoop:
  cmp w6, w3
  beq Write
  add w6, w6, #16

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
  b CalLoop

Write:
  mul v16.4s, v16.4s, v17.4s
  st1 {v16.4s}, [x1], #16
  beq RowLoop

End:
  ret
#endif