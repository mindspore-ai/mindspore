/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "nnacl/fp32/splice_fp32.h"
void SpliceFp32(const float *src_data, int src_row, int src_col, const SpliceParameter *splice_parameter,
                float *dst_data, int dst_row, int dst_col) {
  int row_offset = splice_parameter->src_to_dst_row_offset_;
  for (int r = 0; r < dst_row; ++r) {
    for (int off = 0; off < splice_parameter->context_dim_; ++off) {
      int r_off = r + row_offset + splice_parameter->context_[off];
      r_off = MSMAX(r_off, 0);
      r_off = MSMIN(r_off, src_row - 1);
      const float *tmp_src_data = src_data + r_off * src_col;
      float *tmp_dst_data = dst_data + r * dst_col;
      memcpy(tmp_dst_data + off * src_col, tmp_src_data, src_col * sizeof(float));
    }
  }
}
