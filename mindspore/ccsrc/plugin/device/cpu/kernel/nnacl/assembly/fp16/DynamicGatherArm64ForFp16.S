#ifdef ENABLE_ARM64
#include "nnacl/assembly_global.h"
.text
.align 5

// void DynamicGatherArm64ForFp16(const int8_t *src, float16_t *output, int count_16, int zp, float16_t scale);
// x0: src(left matrix ptr)
// x1: output(right matrix ptr)
// w2: count_16
// w3: zp
// w4: scale

asm_function DynamicGatherArm64ForFp16
    mov x5, x0          // reload src
    mov x6, x1          // reload out
    mov w7, w2          // reload count_16
    dup v1.4s, w3       // zp
    dup v2.4s, v0.s[0]  // scale

    LoopCount:
        ld1 {v0.16b}, [x5], #16

        sxtl v3.8h, v0.8b
        sxtl2 v4.8h, v0.16b

        sxtl v16.4s, v3.4h
        sxtl2 v17.4s, v3.8h
        sxtl v18.4s, v4.4h
        sxtl2 v19.4s, v4.8h

        sub v16.4s, v16.4s, v1.4s
        scvtf v16.4s,v16.4s
        fmul v16.4s, v16.4s, v2.4s
        sub v17.4s, v17.4s, v1.4s
        scvtf v17.4s,v17.4s
        fmul v17.4s, v17.4s, v2.4s
        sub v18.4s, v18.4s, v1.4s
        scvtf v18.4s,v18.4s
        fmul v18.4s, v18.4s, v2.4s
        sub v19.4s, v19.4s, v1.4s
        scvtf v19.4s,v19.4s
        fmul v19.4s, v19.4s, v2.4s

        fcvtn v16.4h, v16.4s
        fcvtn v17.4h, v17.4s
        fcvtn v18.4h, v18.4s
        fcvtn v19.4h, v19.4s

        st1 {v16.4h, v17.4h, v18.4h, v19.4h}, [x6], #32
        subs w7, w7, #16
    bgt LoopCount
ret

#endif