#ifdef __aarch64__
#include "nnacl/assembly_global.h"
    .text
    .align 5
    .global MatMulOptR4Int8Neon64
#ifndef __APPLE__
    .type MatMulOptR4Int8Neon64, %function
#endif

//void MatMulOptR4Int8Neon64(const int8_t *a, const int8_t *b, int *dst, int row4, int col4, int deep16,
//                          const int *input_sum, const int *bias)

// x0: a(left matrix ptr)
// x1: b(right matrix ptr)
// x2: out ptr
// w3: row4
// w4: col4
// w5: deep16
// x6: a_sums
// x7: bias

asm_function MatMulOptR4Int8Neon64
  sub sp, sp, #128
  st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
  st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64

  mov w15, #0       // b col index
  mov w16, #0       // a row index
  mov w17, #4       // sizeof(int8)*4
  mul w12, w5, w17  // the stride of a/b: sizeof(int8)*4*deep16

L1:
  cmp w15, w4      
  beq End1

  mov w16, #0     // reset a row index
  mov x17, x0     // reload a ptr
  mov x13, x6     // reload a_sums ptr 
L2:
  cmp w16, w3
  beq End2

  mov x18, x1     // reload b ptr
  mov x10, x7    // reload bias ptr
  mov w11, w5     // reload depth
  dup v16.4s, wzr
  dup v17.4s, wzr
  dup v18.4s, wzr
  dup v19.4s, wzr
  dup v20.4s, wzr
  dup v21.4s, wzr
  dup v22.4s, wzr
  dup v23.4s, wzr
  dup v24.4s, wzr
  dup v25.4s, wzr
  dup v26.4s, wzr
  dup v27.4s, wzr
  dup v28.4s, wzr
  dup v29.4s, wzr
  dup v30.4s, wzr
  dup v31.4s, wzr
L3:
  cmp w11, #0
  beq End3

  ld1 {v0.16b}, [x17], #16
  ld1 {v1.16b}, [x17], #16
  ld1 {v2.16b}, [x17], #16
  ld1 {v3.16b}, [x17], #16
  ld1 {v4.16b}, [x18], #16
  ld1 {v5.16b}, [x18], #16
  ld1 {v6.16b}, [x18], #16
  ld1 {v7.16b}, [x18], #16

  sdot v16.4s, v4.16b, v0.16b
  sdot v17.4s, v5.16b, v0.16b
  sdot v18.4s, v6.16b, v0.16b
  sdot v19.4s, v7.16b, v0.16b
  sdot v20.4s, v4.16b, v1.16b
  sdot v21.4s, v5.16b, v1.16b
  sdot v22.4s, v6.16b, v1.16b
  sdot v23.4s, v7.16b, v1.16b
  sdot v24.4s, v4.16b, v2.16b
  sdot v25.4s, v5.16b, v2.16b
  sdot v26.4s, v6.16b, v2.16b
  sdot v27.4s, v7.16b, v2.16b
  sdot v28.4s, v4.16b, v3.16b
  sdot v29.4s, v5.16b, v3.16b
  sdot v30.4s, v6.16b, v3.16b
  sdot v31.4s, v7.16b, v3.16b
  subs w11, w11, #16  // depth + 16
  b L3

End3:
  addp v16.4s, v16.4s, v17.4s
  addp v18.4s, v18.4s, v19.4s
  addp v20.4s, v20.4s, v21.4s
  addp v22.4s, v22.4s, v23.4s
  addp v24.4s, v24.4s, v25.4s
  addp v26.4s, v26.4s, v27.4s
  addp v28.4s, v28.4s, v29.4s
  addp v30.4s, v30.4s, v31.4s

  addp v16.4s, v16.4s, v18.4s
  addp v17.4s, v20.4s, v22.4s
  addp v18.4s, v24.4s, v26.4s
  addp v19.4s, v28.4s, v30.4s

  // Add (Bias+Depth*Za*Zb-Za*Bsums)
  ld1 {v15.4s}, [x10], #16  
  add v16.4s, v16.4s, v15.4s
  add v17.4s, v17.4s, v15.4s
  add v18.4s, v18.4s, v15.4s
  add v19.4s, v19.4s, v15.4s

  // Subtract (Asums*Zb)
  ld1 {v14.4s}, [x13], #16
  dup v20.4s, v14.s[0]
  dup v21.4s, v14.s[1]
  dup v22.4s, v14.s[2]
  dup v23.4s, v14.s[3]
  sub v16.4s, v16.4s, v20.4s
  sub v17.4s, v17.4s, v21.4s
  sub v18.4s, v18.4s, v22.4s
  sub v19.4s, v19.4s, v23.4s

  st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
  add w16, w16, #4      // a row index + 4
  b L2

End2:
  add w15, w15, #4      // b col index + 4
  add x1, x1, x12       // b ptr + stride
  add x7, x7, #16       // bias ptr + stride
  b L1

End1:
  sub sp, sp, #128
  ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
  ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
  ret
#endif
