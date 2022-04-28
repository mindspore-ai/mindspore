#ifdef ENABLE_ARM64
/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "nnacl/fp32/pack_fp32_opt.h"
#include "nnacl/op_base.h"

void RowMajor2Col12MajorOptCore(const float *src_c, float *dst_c, size_t stride, int64_t row, int64_t col) {
  if (row <= 0 || col <= 0) {
    return;
  }
  size_t stride_byte = stride * sizeof(float);
  size_t stride_unit = stride * (C12NUM - 1);
  int64_t r = 0;
  for (; r <= row - C12NUM; r += C12NUM) {
    int64_t c = 0;
    for (; c <= col - C4NUM; c += C4NUM) {
      asm volatile(
        "mov x9, %[src_c]\n"
        "mov x10, %[dst_c]\n"

        "ld1 {v0.4s}, [x9], %[stride_byte]\n"
        "ld1 {v1.4s}, [x9], %[stride_byte]\n"
        "ld1 {v2.4s}, [x9], %[stride_byte]\n"
        "ld1 {v3.4s}, [x9], %[stride_byte]\n"

        "ld1 {v4.4s}, [x9], %[stride_byte]\n"
        "ld1 {v5.4s}, [x9], %[stride_byte]\n"
        "ld1 {v6.4s}, [x9], %[stride_byte]\n"
        "ld1 {v7.4s}, [x9], %[stride_byte]\n"

        "zip1 v12.4s, v0.4s, v1.4s\n"
        "zip2 v13.4s, v0.4s, v1.4s\n"
        "zip1 v14.4s, v2.4s, v3.4s\n"
        "zip2 v15.4s, v2.4s, v3.4s\n"

        "ld1 {v8.4s}, [x9], %[stride_byte]\n"
        "ld1 {v9.4s}, [x9], %[stride_byte]\n"
        "ld1 {v10.4s}, [x9], %[stride_byte]\n"
        "ld1 {v11.4s}, [x9], %[stride_byte]\n"

        "zip1 v16.4s, v4.4s, v5.4s\n"
        "zip2 v17.4s, v4.4s, v5.4s\n"
        "zip1 v18.4s, v6.4s, v7.4s\n"
        "zip2 v19.4s, v6.4s, v7.4s\n"

        "trn1 v20.2d, v12.2d, v14.2d\n"
        "trn2 v23.2d, v12.2d, v14.2d\n"
        "trn1 v26.2d, v13.2d, v15.2d\n"
        "trn2 v29.2d, v13.2d, v15.2d\n"

        "trn1 v21.2d, v16.2d, v18.2d\n"
        "trn2 v24.2d, v16.2d, v18.2d\n"
        "trn1 v27.2d, v17.2d, v19.2d\n"
        "trn2 v30.2d, v17.2d, v19.2d\n"

        "zip1 v12.4s, v8.4s, v9.4s\n"
        "zip2 v13.4s, v8.4s, v9.4s\n"
        "zip1 v14.4s, v10.4s, v11.4s\n"
        "zip2 v15.4s, v10.4s, v11.4s\n"

        "trn1 v22.2d, v12.2d, v14.2d\n"
        "trn2 v25.2d, v12.2d, v14.2d\n"
        "trn1 v28.2d, v13.2d, v15.2d\n"
        "trn2 v31.2d, v13.2d, v15.2d\n"

        "st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x10], #64\n"
        "st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x10], #64\n"
        "st1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x10], #64\n"

        :
        : [ dst_c ] "r"(dst_c), [ src_c ] "r"(src_c), [ stride_byte ] "r"(stride_byte)
        : "memory", "x9", "x10", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
          "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
          "v29", "v30", "v31");
      dst_c += C48NUM;
      src_c += C4NUM;
    }
    for (; c < col; ++c) {
      for (int i = 0; i < C12NUM; ++i) {
        dst_c[i] = src_c[i * stride];
      }
      ++src_c;
      dst_c += C12NUM;
    }
    src_c += stride_unit;
  }
  for (; r < row; ++r) {
    for (int c = 0; c < col; ++c) {
      dst_c[c * C12NUM] = src_c[c];
    }
    src_c += stride;
    ++dst_c;
  }
}

void RowMajor2Row12MajorOptCore(const float *src_c, float *dst_c, size_t stride, int64_t row, int64_t col) {
  if (row <= 0 || col <= 0) {
    return;
  }
  size_t stride_byte = stride * sizeof(float);
  int64_t c = 0;
  for (; c <= col - C12NUM; c += C12NUM) {
    asm volatile(
      "mov x9, %[src_c]\n"
      "mov x10, %[dst_c]\n"
      "mov x11, %[row]\n"
      "1:\n"
      "ld1 {v0.4s, v1.4s, v2.4s}, [x9], %[stride_byte]\n"
      "st1 {v0.4s, v1.4s, v2.4s}, [x10], #48\n"
      "subs x11, x11, #1\n"
      "bgt 1b\n"
      :
      : [ dst_c ] "r"(dst_c), [ src_c ] "r"(src_c), [ stride_byte ] "r"(stride_byte), [ row ] "r"(row)
      : "cc", "memory", "x9", "x10", "x11", "v0", "v1", "v2");
    dst_c += row * C12NUM;
    src_c += C12NUM;
  }
  int64_t c_remain = col - c;
  if (c_remain == 0) {
    return;
  }
  for (int64_t r = 0; r < row; ++r) {
    for (c = 0; c < c_remain; ++c) {
      dst_c[r * C12NUM + c] = src_c[c];
    }
    src_c += stride;
  }
}

void RowMajor2Col12MajorOpt(const float *src_ptr, float *dst_ptr, int64_t row, int64_t col, int64_t start,
                            int64_t end) {
  int64_t bundle_row = UP_DIV(row, C12NUM);
  int64_t unit_num_per_batch = bundle_row * col;
  if (unit_num_per_batch == 0) {
    return;
  }
  int64_t start_batch = start / unit_num_per_batch;
  int64_t end_batch = end / unit_num_per_batch;
  int64_t start_remain = start % unit_num_per_batch;
  int64_t end_remain = end % unit_num_per_batch;
  if (col == 0) {
    return;
  }
  int64_t start_row = start_remain / col;
  int64_t end_row = end_remain / col;
  int64_t start_col = start_remain % col;
  int64_t end_col = end_remain % col;
  const float *src = src_ptr + start_batch * row * col + start_row * C12NUM * col + start_col;
  float *dst = dst_ptr + start * C12NUM;
  int64_t row_num = C12NUM;
  if (start_row * C12NUM + C12NUM > row) {
    row_num -= (start_row * C12NUM + C12NUM - row);
  }
  if (start_batch == end_batch) {
    if (start_row == end_row) {
      RowMajor2Col12MajorOptCore(src, dst, col, row_num, end_col - start_col);
      return;
    }
    RowMajor2Col12MajorOptCore(src, dst, col, C12NUM, col - start_col);
    src += C12NUM * col - start_col;
    dst += (col - start_col) * C12NUM;
    ++start_row;
    if (start_row < end_row) {
      row_num = (end_row - start_row) * C12NUM;
      RowMajor2Col12MajorOptCore(src, dst, col, row_num, col);
      src += row_num * col;
      dst += row_num * col;
    }
    row_num = C12NUM;
    if (end_row * C12NUM + C12NUM > row) {
      row_num -= (end_row * C12NUM + C12NUM - row);
    }
    RowMajor2Col12MajorOptCore(src, dst, col, row_num, end_col);
    return;
  }
  RowMajor2Col12MajorOptCore(src, dst, col, row_num, col - start_col);
  src += row_num * col - start_col;
  dst += (col - start_col) * C12NUM;
  row_num = row - start_row * C12NUM - C12NUM;
  if (row_num > 0) {
    RowMajor2Col12MajorOptCore(src, dst, col, row_num, col);
    src += row_num * col;
    dst += UP_DIV(row_num, C12NUM) * C12NUM * col;
  }
  ++start_batch;
  for (; start_batch < end_batch; ++start_batch) {
    RowMajor2Col12MajorOptCore(src, dst, col, row, col);
    src += row * col;
    dst += bundle_row * C12NUM * col;
  }
  if (end_row > 0) {
    row_num = end_row * C12NUM;
    RowMajor2Col12MajorOptCore(src, dst, col, row_num, col);
    src += row_num * col;
    dst += row_num * col;
  }
  row_num = C12NUM;
  if (end_row * C12NUM + C12NUM > row) {
    row_num -= (end_row * C12NUM + C12NUM - row);
  }
  RowMajor2Col12MajorOptCore(src, dst, col, row_num, end_col);
}

void RowMajor2Row12MajorOpt(const float *src_ptr, float *dst_ptr, int64_t row, int64_t col, int64_t start,
                            int64_t end) {
  int64_t bundle_col = UP_DIV(col, C12NUM);
  int64_t unit_num_per_batch = bundle_col * row;
  if (unit_num_per_batch == 0) {
    return;
  }
  int64_t start_batch = start / unit_num_per_batch;
  int64_t end_batch = end / unit_num_per_batch;
  int64_t start_remain = start % unit_num_per_batch;
  int64_t end_remain = end % unit_num_per_batch;
  if (row == 0) {
    return;
  }
  int64_t start_row = start_remain % row;
  int64_t end_row = end_remain % row;
  int64_t start_col = start_remain / row;
  int64_t end_col = end_remain / row;
  const float *src = src_ptr + start_batch * row * col + start_row * col + start_col * C12NUM;
  float *dst = dst_ptr + start * C12NUM;
  int64_t col_num = C12NUM;
  if (start_col * C12NUM + C12NUM > col) {
    col_num -= (start_col * C12NUM + C12NUM - col);
  }
  if (start_batch == end_batch) {
    if (start_col == end_col) {
      RowMajor2Row12MajorOptCore(src, dst, col, end_row - start_row, col_num);
      return;
    }
    RowMajor2Row12MajorOptCore(src, dst, col, row - start_row, col_num);
    src += C12NUM - start_row * col;
    dst += (row - start_row) * C12NUM;
    ++start_col;
    if (start_col < end_col) {
      col_num = (end_col - start_col) * C12NUM;
      RowMajor2Row12MajorOptCore(src, dst, col, row, col_num);
      src += col_num;
      dst += row * col_num;
    }
    col_num = C12NUM;
    if (end_col * C12NUM + C12NUM > col) {
      col_num -= (end_col * C12NUM + C12NUM - col);
    }
    RowMajor2Row12MajorOptCore(src, dst, col, end_row, col_num);
    return;
  }
  RowMajor2Row12MajorOptCore(src, dst, col, row - start_row, col_num);
  src += col_num - start_row * col;
  dst += (row - start_row) * C12NUM;
  col_num = col - start_col * C12NUM - C12NUM;
  if (col_num > 0) {
    RowMajor2Row12MajorOptCore(src, dst, col, row, col_num);
    src += col_num;
    dst += UP_DIV(col_num, C12NUM) * C12NUM * row;
  }
  src += (row - 1) * col;
  ++start_batch;
  for (; start_batch < end_batch; ++start_batch) {
    RowMajor2Row12MajorOptCore(src, dst, col, row, col);
    src += row * col;
    dst += bundle_col * C12NUM * row;
  }
  if (end_col > 0) {
    col_num = end_col * C12NUM;
    RowMajor2Row12MajorOptCore(src, dst, col, row, col_num);
    src += col_num;
    dst += row * col_num;
  }
  col_num = C12NUM;
  if (end_col * C12NUM + C12NUM > col) {
    col_num -= (end_col * C12NUM + C12NUM - col);
  }
  RowMajor2Row12MajorOptCore(src, dst, col, end_row, col_num);
}
#endif
