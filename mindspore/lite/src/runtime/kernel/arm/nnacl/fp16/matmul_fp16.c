/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nnacl/fp16/matmul_fp16.h"

void MatMulFp16(const float16_t *a, const float16_t *b, float16_t *c, const float16_t *bias, ActType act_type,
                int depth, int row, int col, int stride, bool write_nhwc) {
  MatmulFp16Neon64(a, b, c, bias, (int)act_type, depth, row, col, stride, write_nhwc);
}

void RowMajor2Col8MajorFp16(float16_t *src_ptr, float16_t *dst_ptr, size_t row, size_t col) {
  size_t row16 = row / C16NUM * C16NUM;
  size_t col8 = col / C8NUM * C8NUM;
  float16_t *src_r = src_ptr;
  float16_t *dst_r = dst_ptr;

  size_t ri = 0;
  for (; ri < row16; ri += C16NUM) {
    size_t ci = 0;
    for (; ci < col8; ci += C8NUM) {
      float16_t *src_c = src_r + ci;
      float16_t *dst_c = dst_r + ci * C16NUM;

      /* 16*8 row-major to col-major */
#ifdef ENABLE_ARM64
      size_t stride = col * 2;
      asm volatile(
        "mov x10, %[src_c]\n"
        "mov x11, %[dst_c]\n"

        "ld1 {v0.8h}, [x10], %[stride]\n"
        "ld1 {v1.8h}, [x10], %[stride]\n"
        "ld1 {v2.8h}, [x10], %[stride]\n"
        "ld1 {v3.8h}, [x10], %[stride]\n"
        "ld1 {v4.8h}, [x10], %[stride]\n"
        "ld1 {v5.8h}, [x10], %[stride]\n"
        "ld1 {v6.8h}, [x10], %[stride]\n"
        "ld1 {v7.8h}, [x10], %[stride]\n"

        "zip1 v16.8h, v0.8h, v1.8h\n"
        "zip1 v17.8h, v2.8h, v3.8h\n"
        "zip1 v18.8h, v4.8h, v5.8h\n"
        "zip1 v19.8h, v6.8h, v7.8h\n"

        "ld1 {v8.8h}, [x10], %[stride]\n"
        "ld1 {v9.8h}, [x10], %[stride]\n"
        "ld1 {v10.8h}, [x10], %[stride]\n"
        "ld1 {v11.8h}, [x10], %[stride]\n"
        "ld1 {v12.8h}, [x10], %[stride]\n"
        "ld1 {v13.8h}, [x10], %[stride]\n"
        "ld1 {v14.8h}, [x10], %[stride]\n"
        "ld1 {v15.8h}, [x10], %[stride]\n"

        "trn1 v20.4s, v16.4s, v17.4s\n"
        "trn2 v21.4s, v16.4s, v17.4s\n"
        "trn1 v22.4s, v18.4s, v19.4s\n"
        "trn2 v23.4s, v18.4s, v19.4s\n"

        "trn1 v24.2d, v20.2d, v22.2d\n"
        "trn2 v25.2d, v20.2d, v22.2d\n"
        "trn1 v26.2d, v21.2d, v23.2d\n"
        "trn2 v27.2d, v21.2d, v23.2d\n"

        "zip1 v16.8h, v8.8h, v9.8h\n"
        "zip1 v17.8h, v10.8h, v11.8h\n"
        "zip1 v18.8h, v12.8h, v13.8h\n"
        "zip1 v19.8h, v14.8h, v15.8h\n"

        "trn1 v20.4s, v16.4s, v17.4s\n"
        "trn2 v21.4s, v16.4s, v17.4s\n"
        "trn1 v22.4s, v18.4s, v19.4s\n"
        "trn2 v23.4s, v18.4s, v19.4s\n"

        "trn1 v28.2d, v20.2d, v22.2d\n"
        "trn2 v29.2d, v20.2d, v22.2d\n"
        "trn1 v30.2d, v21.2d, v23.2d\n"
        "trn2 v31.2d, v21.2d, v23.2d\n"

        "st1 {v24.8h}, [x11], #16\n"
        "st1 {v28.8h}, [x11], #16\n"
        "st1 {v26.8h}, [x11], #16\n"
        "st1 {v30.8h}, [x11], #16\n"
        "st1 {v25.8h}, [x11], #16\n"
        "st1 {v29.8h}, [x11], #16\n"
        "st1 {v27.8h}, [x11], #16\n"
        "st1 {v31.8h}, [x11], #16\n"

        "zip2 v16.8h, v0.8h, v1.8h\n"
        "zip2 v17.8h, v2.8h, v3.8h\n"
        "zip2 v18.8h, v4.8h, v5.8h\n"
        "zip2 v19.8h, v6.8h, v7.8h\n"

        "trn1 v20.4s, v16.4s, v17.4s\n"
        "trn2 v21.4s, v16.4s, v17.4s\n"
        "trn1 v22.4s, v18.4s, v19.4s\n"
        "trn2 v23.4s, v18.4s, v19.4s\n"

        "trn1 v24.2d, v20.2d, v22.2d\n"
        "trn2 v25.2d, v20.2d, v22.2d\n"
        "trn1 v26.2d, v21.2d, v23.2d\n"
        "trn2 v27.2d, v21.2d, v23.2d\n"

        "zip2 v16.8h, v8.8h, v9.8h\n"
        "zip2 v17.8h, v10.8h, v11.8h\n"
        "zip2 v18.8h, v12.8h, v13.8h\n"
        "zip2 v19.8h, v14.8h, v15.8h\n"

        "trn1 v20.4s, v16.4s, v17.4s\n"
        "trn2 v21.4s, v16.4s, v17.4s\n"
        "trn1 v22.4s, v18.4s, v19.4s\n"
        "trn2 v23.4s, v18.4s, v19.4s\n"

        "trn1 v28.2d, v20.2d, v22.2d\n"
        "trn2 v29.2d, v20.2d, v22.2d\n"
        "trn1 v30.2d, v21.2d, v23.2d\n"
        "trn2 v31.2d, v21.2d, v23.2d\n"

        "st1 {v24.8h}, [x11], #16\n"
        "st1 {v28.8h}, [x11], #16\n"
        "st1 {v26.8h}, [x11], #16\n"
        "st1 {v30.8h}, [x11], #16\n"
        "st1 {v25.8h}, [x11], #16\n"
        "st1 {v29.8h}, [x11], #16\n"
        "st1 {v27.8h}, [x11], #16\n"
        "st1 {v31.8h}, [x11], #16\n"
        :
        : [ dst_c ] "r"(dst_c), [ src_c ] "r"(src_c), [ stride ] "r"(stride)
        : "x10", "x11", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
          "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
          "v30", "v31");
#else
      for (int tr = 0; tr < C16NUM; tr++) {
        for (int tc = 0; tc < C8NUM; tc++) {
          dst_c[tc * C16NUM + tr] = src_c[tr * col + tc];
        }
      }
#endif
    }
    for (; ci < col; ci++) {
      float16_t *src_c = src_r + ci;
      float16_t *dst_c = dst_r + ci * C16NUM;
      for (size_t i = 0; i < C16NUM; i++) {
        dst_c[i] = src_c[i * col];
      }
    }
    src_r += C16NUM * col;
    dst_r += C16NUM * col;
  }
  for (; ri < row; ri++) {
    for (size_t i = 0; i < col; i++) {
      dst_r[i * C16NUM] = src_r[i];
    }
    src_r += col;
    dst_r += 1;
  }
  return;
}
